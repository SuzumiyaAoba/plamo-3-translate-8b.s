#include "../runtime/metal.h"
#include "../runtime/disk_cache.h"
#include "../runtime/packing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

extern void plamo_bf16_matmul(const struct plamo_matmul *);
static uint64_t available_bytes = UINT64_C(128) * 1024 * 1024;
static uint64_t available(void) { return available_bytes; }
#define CHECK(ok) do { if (!(ok)) { fprintf(stderr, "Metal check failed at %d: %s\n", __LINE__, #ok); exit(1); } } while (0)

static int kernels(int require_gpu) {
    struct plamo_metal_config config = {.memory = PLAMO_METAL_AUTO, .reserve = 64 * 1024 * 1024, .available = available};
    char error[1024];
    struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error);
    if (!metal) {
        int skip = !require_gpu && !strcmp(error, "Metal requires an available Apple Silicon GPU");
        fprintf(stderr, "Metal tests %s: %s\n", skip ? "SKIPPED" : "FAILED", error);
        return skip ? 77 : 1;
    }
    const size_t shapes[][3] = {{0, 0, 0}, {0, 128, 9}, {4, 0, 4}, {1, 1, 1}, {3, 7, 2},
        {5, 12, 17}, {35, 65, 9}, {129, 513, 3}, {259, 1027, 9},
        {259, 1027, 65}, {1024, 1024, 21}, {1024, 4096, 1}, {2053, 1536, 8}, {1025, 16384, 9}, {1056, 16384, 8}};
    double maximum_error = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
        size_t rows = shapes[s][0], cols = shapes[s][1], tokens = shapes[s][2], xs = cols + 7, ys = rows + 9;
        if (rows == 1056 && cols == 16384) xs = 2 * cols;
        uint16_t *w = malloc((rows * cols + 1) * 2);
        float *x = malloc((tokens * xs + 1) * 4), *reference = malloc((tokens * ys + 1) * 4), *actual = malloc((tokens * ys + 1) * 4);
        CHECK(w && x && reference && actual);
        uint32_t seed = 13;
        for (size_t i = 0; i < rows * cols; ++i) { seed = seed * 1664525u + 1013904223u; w[i] = (seed & 0x8000) | 0x3c00 | (seed & 0x3ff); }
        for (size_t i = 0; i < tokens * xs; ++i) x[i] = ((int)(i % 83) - 41) / 47.f;
        for (size_t i = 0; i < tokens * ys + 1; ++i) reference[i] = actual[i] = -123456.f;
        for (size_t i = 0; i < tokens; ++i) plamo_bf16_matvec_serial(w, x + i * xs, reference + i * ys, rows, cols);
        struct plamo_matmul p = {w, x, actual, rows, cols, tokens, xs, ys};
        struct plamo_metal_info info;
        plamo_metal_get_info(metal, &info);
        if (rows * cols * 2 <= info.cache.staging_bytes) CHECK(plamo_metal_linear(metal, &p));
        else plamo_bf16_matmul(&p);
        for (size_t i = 0; i < tokens * ys + 1; ++i) {
            double delta = fabs((double)actual[i] - reference[i]);
            if (delta > maximum_error) maximum_error = delta;
            if (!isfinite(actual[i]) || delta > 0.0002 * (1 + fabs(reference[i]))) {
                fprintf(stderr, "shape %zu,%zu,%zu index %zu: %.9g vs %.9g (error %.6g)\n", rows, cols, tokens, i, actual[i], reference[i], delta);
                return 1;
            }
        }
        free(w); free(x); free(reference); free(actual);
    }
    struct plamo_metal_info info;
    plamo_metal_get_info(metal, &info);
    CHECK(info.dispatches > 0 && !info.errors && !info.cpu_fallbacks);
    printf("Metal %s: PASS (%s; %llu dispatches; strides, tails, empty shapes, row windows; max abs error %.6g)\n", info.precision, info.device, (unsigned long long)info.dispatches, maximum_error);
    plamo_metal_close(metal);
    return 0;
}

extern void plamo_rmsnorm(const float *, const uint16_t *, float *, uint64_t, float, float);
extern void plamo_swiglu(const float *, float *, uint64_t);
static void fused_mlp(void) {
    // Odd widths exercise guarded loads, non-vector strides and norm offsets.
    enum { H = 129, N = 257, MAX_TOKENS = 65, WORDS = 3 * H * N };
    uint16_t *data = malloc(WORDS * 2), pre[H], post[H];
    float *input = malloc(H * MAX_TOKENS * 4), *expected = malloc(H * MAX_TOKENS * 4), *actual = malloc(H * MAX_TOKENS * 4);
    float norm[H], branch[H], scratch[2 * N];
    CHECK(data && input && expected && actual);
    uint32_t seed = 123;
    for (size_t i = 0; i < WORDS; ++i) { seed = seed * 1664525u + 1013904223u; data[i] = (seed & 0x8000) | 0x3c00 | (seed & 0x7f); }
    for (size_t i = 0; i < H; ++i) { pre[i] = 0x3a80 + i % 64; post[i] = 0x3b00 + i % 128; }
    for (size_t i = 0; i < H * MAX_TOKENS; ++i) input[i] = ((int)(i % 103) - 51) / 17.f;
    FILE *file = tmpfile();
    CHECK(file && fwrite(data, 2, WORDS, file) == WORDS && !fflush(file));
    int64_t fd = fileno(file);
    void *base = mmap(NULL, WORDS * 2, PROT_READ, MAP_PRIVATE, (int)fd, 0);
    CHECK(base != MAP_FAILED);
    void *gu = base, *down = (uint16_t *)base + 2 * H * N;
    struct plamo_mapping map = {base, WORDS * 2};
    const uint64_t gate_shape[] = {2 * N, H}, down_shape[] = {H, N};
    struct plamo_weight_spec schema[] = {{"gu", 2, gate_shape, &gu}, {"down", 2, down_shape, &down}, {0}};
    struct plamo_metal_config config = {.memory = PLAMO_METAL_RESIDENT, .maps = &map, .fds = &fd, .count = 1,
        .schema = schema, .reserve = 64 * 1024 * 1024, .available = available};
    char error[1024];
    struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error);
    if (!metal) fprintf(stderr, "fused MLP setup: %s\n", error);
    CHECK(metal);
    struct mlp p = {.pre = pre, .post = post, .gu = gu, .down = down, .hidden = H, .intermediate = N};
    const unsigned counts[] = {0, 1, 3, 4, 8, 21, MAX_TOKENS};
    for (size_t c = 0; c < sizeof counts / sizeof *counts; ++c) {
        unsigned tokens = counts[c];
        memcpy(actual, input, H * MAX_TOKENS * 4);
        memcpy(expected, input, H * MAX_TOKENS * 4);
        for (unsigned t = 0; t < tokens; ++t) {
            plamo_rmsnorm(input + t * H, pre, norm, H, .000001f, 1.f);
            plamo_bf16_matvec_serial(gu, norm, scratch, 2 * N, H);
            plamo_swiglu(scratch, scratch, N);
            plamo_bf16_matvec_serial(down, scratch, branch, H, N);
            plamo_rmsnorm(branch, post, branch, H, .000001f, .08944271909999159f);
            for (unsigned i = 0; i < H; ++i) expected[t * H + i] += branch[i];
        }
        struct plamo_metal_info before, after;
        plamo_metal_get_info(metal, &before);
        CHECK(plamo_metal_mlp(metal, &p, actual, tokens));
        plamo_metal_get_info(metal, &after);
        CHECK(after.command_buffers - before.command_buffers == (tokens != 0));
        CHECK(after.fused_mlps - before.fused_mlps == (tokens != 0));
        for (size_t i = 0; i < H * MAX_TOKENS; ++i)
            CHECK(isfinite(actual[i]) && fabs(actual[i] - expected[i]) <= .00002 * (1 + fabs(expected[i])));
    }
    plamo_metal_close(metal);
    config.memory = PLAMO_METAL_AUTO;
    metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal);
    memcpy(actual, input, H * MAX_TOKENS * 4);
    CHECK(!plamo_metal_mlp(metal, &p, actual, 21));
    CHECK(!memcmp(actual, input, H * MAX_TOKENS * 4));
    struct plamo_matmul g = {gu, input, scratch, 2 * N, H, 1, H, 2 * N};
    struct plamo_matmul d = {down, scratch, branch, H, N, 1, N, H};
    CHECK(plamo_metal_linear(metal, &g) && plamo_metal_linear(metal, &d));
    CHECK(plamo_metal_mlp(metal, &p, actual, 21));
    available_bytes = 0; plamo_metal_maintain(metal);
    memcpy(actual, input, H * MAX_TOKENS * 4);
    CHECK(!plamo_metal_mlp(metal, &p, actual, 21));
    CHECK(!memcmp(actual, input, H * MAX_TOKENS * 4));
    plamo_metal_close(metal);
    available_bytes = UINT64_C(128) * 1024 * 1024;
    munmap(base, WORDS * 2); fclose(file);
    free(data); free(input); free(expected); free(actual);
    puts("Metal fused MLP: PASS (FP32 CPU reference, batches 0..65, single command, cache miss/eviction leaves state unchanged)");
}

static void cache(void) {
    enum { ROWS = 8193, COLS = 1024, BYTES = ROWS * COLS * 2 };
    uint16_t *data = malloc(BYTES);
    float *input = malloc(COLS * 4), *reference = malloc(ROWS * 4), *actual = malloc(ROWS * 4);
    CHECK(data && input && reference && actual);
    for (size_t i = 0; i < ROWS * COLS; ++i) data[i] = 0x3c00 + i % 1024;
    for (size_t i = 0; i < COLS; ++i) input[i] = ((int)(i % 31) - 15) / 31.f;
    FILE *file = tmpfile();
    CHECK(file && fwrite(data, 1, BYTES, file) == BYTES && !fflush(file));
    int64_t fd = fileno(file);
    void *base = mmap(NULL, BYTES, PROT_READ, MAP_PRIVATE, (int)fd, 0);
    CHECK(base != MAP_FAILED);
    struct plamo_mapping map = {base, BYTES};
    const uint64_t shape[] = {ROWS, COLS};
    struct plamo_weight_spec schema[] = {{"matrix", 2, shape, &base}, {"alias", 2, shape, &base}, {0}};
    struct plamo_metal_config config = {.memory = PLAMO_METAL_RESIDENT, .maps = &map, .fds = &fd, .count = 1,
        .schema = schema, .reserve = 64 * 1024 * 1024, .available = available};
    char error[1024];
    struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error);
    CHECK(metal);
    struct plamo_metal_info info;
    plamo_metal_get_info(metal, &info);
    CHECK(info.cache.cached_bytes == BYTES);
    struct plamo_matmul p = {base, input, actual, ROWS, COLS, 1, COLS, ROWS};
    uint64_t reads = plamo_io_stats[1];
    uint64_t commands = info.command_buffers;
    CHECK(plamo_metal_linear(metal, &p));
    plamo_metal_get_info(metal, &info);
    CHECK(info.command_buffers == commands + 1);
    CHECK(plamo_io_stats[1] == reads);
    // Cached subranges need not be aligned for vector loads.
    p.weights = (const uint16_t *)base + 1; p.rows = ROWS - 1;
    CHECK(plamo_metal_linear(metal, &p));
    plamo_bf16_matvec_serial(p.weights, input, reference, p.rows, COLS);
    for (size_t i = 0; i < p.rows; ++i) CHECK(fabs(actual[i] - reference[i]) < 0.0002 * (1 + fabs(reference[i])));
    plamo_metal_close(metal);
    config.memory = PLAMO_METAL_AUTO;
    metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal);
    p.weights = base; p.rows = ROWS;
    CHECK(plamo_metal_linear(metal, &p));
    reads = plamo_io_stats[1];
    CHECK(plamo_metal_linear(metal, &p) && plamo_io_stats[1] == reads);
    available_bytes = 0; plamo_metal_maintain(metal);
    plamo_metal_get_info(metal, &info); CHECK(!info.cache.cached_bytes);
    available_bytes = UINT64_C(128) * 1024 * 1024; plamo_metal_maintain(metal);
    CHECK(plamo_metal_linear(metal, &p) && plamo_io_stats[1] > reads);
    available_bytes = 0; plamo_metal_maintain(metal);
    CHECK(!ftruncate((int)fd, BYTES / 2));
    CHECK(!plamo_metal_linear(metal, &p));
    plamo_metal_get_info(metal, &info); CHECK(!info.cache.cached_bytes && info.errors == 1);
    CHECK(pwrite((int)fd, data, BYTES, 0) == BYTES);
    available_bytes = UINT64_C(128) * 1024 * 1024; plamo_metal_maintain(metal);
    CHECK(plamo_metal_linear(metal, &p));
    plamo_metal_close(metal);
    munmap(base, BYTES); fclose(file); free(data); free(input); free(reference); free(actual);
    puts("Metal cache: PASS (shared BF16 residency, aliases, reuse, pressure shrink/grow, partial read recovery, restart)");
}

static void preload(void) {
    enum { MIB = 1024 * 1024, BYTES = 4 * MIB };
    uint16_t *data = malloc(BYTES);
    CHECK(data);
    for (size_t i = 0; i < BYTES / 2; ++i) data[i] = 0x3c00 + i % 127;
    FILE *file = tmpfile();
    CHECK(file && fwrite(data, 1, BYTES, file) == BYTES && !fflush(file));
    int64_t fd = fileno(file);
    void *base = mmap(NULL, BYTES, PROT_READ, MAP_PRIVATE, (int)fd, 0);
    CHECK(base != MAP_FAILED);
    struct plamo_mapping map = {base, BYTES};
    available_bytes = 11 * MIB;
    struct plamo_metal_config config = {.memory = PLAMO_METAL_AUTO, .maps = &map, .fds = &fd, .count = 1,
        .packing = PLAMO_PACK_RAW, .reserve = 8 * MIB, .available = available};
    char error[1024];
    struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal);
    uint64_t reads = plamo_io_stats[0];
    for (unsigned i = 0; i < 4; ++i) CHECK(plamo_metal_preload(metal, (char *)base + i * MIB, MIB));
    struct plamo_metal_info info;
    plamo_metal_get_info(metal, &info);
    CHECK(info.cache.cached_bytes == 2 * MIB && plamo_io_stats[0] - reads == 2 * MIB);
    CHECK(info.dispatches == 2 && info.command_buffers == 2 && !info.errors);
    CHECK(plamo_metal_warm_cache(metal));
    plamo_metal_get_info(metal, &info);
    CHECK(info.dispatches == 2 && info.command_buffers == 2 && !info.errors);
    CHECK(plamo_io_stats[0] - reads == 2 * MIB);
    CHECK(plamo_metal_preload(metal, base, MIB) && plamo_io_stats[0] - reads == 2 * MIB);
    available_bytes = 0; plamo_metal_maintain(metal);
    reads = plamo_io_stats[0];
    CHECK(plamo_metal_preload(metal, base, MIB) && plamo_io_stats[0] == reads);
    plamo_metal_get_info(metal, &info); CHECK(!info.cache.cached_bytes);
    uint64_t commands = info.command_buffers;
    CHECK(plamo_metal_warm_cache(metal));
    plamo_metal_get_info(metal, &info); CHECK(info.command_buffers == commands);
    available_bytes = 11 * MIB; plamo_metal_maintain(metal);
    CHECK(!ftruncate((int)fd, MIB / 2));
    CHECK(!plamo_metal_preload(metal, base, MIB));
    plamo_metal_get_info(metal, &info); CHECK(!info.cache.cached_bytes && info.errors == 1);
    CHECK(pwrite((int)fd, data, BYTES, 0) == BYTES);
    CHECK(plamo_metal_preload(metal, base, MIB));
    plamo_metal_close(metal);
    config.memory = PLAMO_METAL_STREAM;
    metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal);
    reads = plamo_io_stats[0];
    CHECK(plamo_metal_preload(metal, base, MIB) && plamo_io_stats[0] == reads);
    plamo_metal_get_info(metal, &info); CHECK(!info.cache.cached_bytes);
    plamo_metal_close(metal);
    available_bytes = UINT64_C(128) * MIB;
    munmap(base, BYTES); fclose(file); free(data);
    puts("Metal preload: PASS (bounded retention without staging reads, incremental GPU pages, idempotence, pressure, failed read, stream opt-out)");
}

static void disk_cache(void) {
    enum { MIB = 1024 * 1024, CHUNKS = 3, COLS = 2048, ROWS = 256, TOKENS = 21 };
    uint16_t *original = malloc(CHUNKS * MIB);
    float *input = malloc(COLS * TOKENS * 4), *expected = malloc(2 * ROWS * TOKENS * 4), *actual = malloc(2 * ROWS * TOKENS * 4);
    CHECK(original && input && expected && actual);
    for (size_t i = 0; i < CHUNKS * MIB / 2; ++i) original[i] = 0x3f80 | (i & 127) | ((i & 128) ? 0x8000 : 0);
    for (size_t i = 0; i < COLS * TOKENS; ++i) input[i] = ((int)(i % 103) - 51) / 31.f;
    FILE *source_file = tmpfile(); CHECK(source_file && fwrite(original, 1, CHUNKS * MIB, source_file) == CHUNKS * MIB && !fflush(source_file));
    int64_t fd = fileno(source_file);
    void *base = mmap(NULL, CHUNKS * MIB, PROT_READ, MAP_PRIVATE, (int)fd, 0); CHECK(base != MAP_FAILED);
    struct plamo_mapping map = {base, CHUNKS * MIB};
    const unsigned counts[] = {1, 4, 9, TOKENS};
    for (unsigned packed = 0; packed < 2; ++packed) {
        char path[] = "/tmp/plamo-metal-disk-XXXXXX", error[1024];
        int cache_fd = mkstemp(path); CHECK(cache_fd >= 0);
        uint64_t payload = packed ? MIB / 2 + (MIB / 2 / PLAMO_BF16_BLOCK) * 8 : MIB;
        uint64_t stored = (payload + PLAMO_DISK_ALIGNMENT - 1) / PLAMO_DISK_ALIGNMENT * PLAMO_DISK_ALIGNMENT;
        uint64_t file_bytes = PLAMO_DISK_ALIGNMENT + CHUNKS * stored;
        unsigned char *file = calloc(1, (size_t)file_bytes); CHECK(file);
        struct plamo_disk_header *h = (void *)file;
        memcpy(h->magic, PLAMO_DISK_MAGIC, 8); h->version = 1; h->alignment = PLAMO_DISK_ALIGNMENT;
        h->entries = CHUNKS; h->sources = 1; h->file_bytes = file_bytes; h->source_bytes[0] = CHUNKS * MIB; strcpy(h->revision, "synthetic");
        struct plamo_disk_entry *entries = (void *)(h + 1);
        for (unsigned chunk = 0; chunk < CHUNKS; ++chunk) {
            entries[chunk] = (struct plamo_disk_entry){0, packed, chunk * MIB, MIB, PLAMO_DISK_ALIGNMENT + chunk * stored, payload, ROWS, COLS};
            unsigned char *out = file + entries[chunk].offset;
            const uint16_t *in = original + chunk * MIB / 2;
            if (!packed) memcpy(out, in, MIB);
            else {
                unsigned blocks = MIB / 2 / PLAMO_BF16_BLOCK;
                struct plamo_bf16_block *headers = (void *)out;
                for (unsigned b = 0; b < blocks; ++b) {
                    headers[b] = (struct plamo_bf16_block){blocks * 8 + b * PLAMO_BF16_BLOCK, 127};
                    for (unsigned j = 0; j < PLAMO_BF16_BLOCK; ++j) {
                        uint16_t v = in[b * PLAMO_BF16_BLOCK + j];
                        out[headers[b].offset + j] = (v & 127) | ((v >> 8) & 128);
                    }
                }
            }
        }
        CHECK(write(cache_fd, file, (size_t)file_bytes) == (ssize_t)file_bytes); close(cache_fd); free(file);
        available_bytes = 13 * MIB;
        struct plamo_metal_config config = {.memory = PLAMO_METAL_AUTO, .packing = packed ? PLAMO_PACK_LOSSLESS : PLAMO_PACK_RAW,
            .maps = &map, .fds = &fd, .count = 1, .reserve = 8 * MIB, .available = available, .revision = "synthetic"};
        struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal);
        for (unsigned c = 0; c < CHUNKS; ++c) CHECK(plamo_metal_preload(metal, (char *)base + c * MIB, MIB));
        // Store each shape's reference independently before reopening the backend.
        float *gold[4];
        for (unsigned n = 0; n < 4; ++n) {
            gold[n] = malloc((size_t)ROWS * 2 * counts[n] * 4); CHECK(gold[n]);
            struct plamo_matmul p = {base, input, gold[n], ROWS * 2, COLS, counts[n], COLS, ROWS * 2}; CHECK(plamo_metal_linear(metal, &p));
        }
        plamo_metal_close(metal); config.disk_cache_path = path; config.packing = PLAMO_PACK_AUTO;
        metal = plamo_metal_open(&config, error, sizeof error); if (!metal) fprintf(stderr, "%s\n", error); CHECK(metal);
        CHECK(!unlink(path)); // Both bulk reads and borrowed GPU pages own the old archive lifetime.
        uint64_t reads = plamo_io_stats[0];
        for (unsigned c = 0; c < CHUNKS; ++c) CHECK(plamo_metal_preload(metal, (char *)base + c * MIB, MIB));
        struct plamo_metal_info info; plamo_metal_get_info(metal, &info);
        CHECK(info.disk_cache_hits == CHUNKS && info.disk_cache_storage_bytes == CHUNKS * stored && info.cache.cached_bytes == CHUNKS * MIB);
        CHECK(!info.disk_cache_mapped_bytes);
        CHECK(plamo_io_stats[0] == reads + CHUNKS * stored && !info.errors);
        for (unsigned n = 0; n < 4; ++n) {
            struct plamo_matmul p = {base, input, actual, ROWS * 2, COLS, counts[n], COLS, ROWS * 2}; CHECK(plamo_metal_linear(metal, &p));
            CHECK(!memcmp(actual, gold[n], (size_t)ROWS * 2 * counts[n] * 4)); if (n && n != 2) free(gold[n]);
        }
        struct plamo_matmul alias = {(uint16_t *)base + 8 * COLS, input, actual, 17, COLS, 1, COLS, 17};
        CHECK(plamo_metal_linear(metal, &alias));
        plamo_bf16_matvec_serial(alias.weights, input, expected, 17, COLS);
        for (unsigned i = 0; i < 17; ++i) CHECK(fabsf(actual[i] - expected[i]) < 0.0002f * (1 + fabsf(expected[i])));
        available_bytes = 0; plamo_metal_maintain(metal); plamo_metal_get_info(metal, &info);
        CHECK(!info.cache.cached_bytes && !info.cache.storage_bytes);
        uint64_t staged_reads = plamo_io_stats[0];
        struct plamo_matmul staged = {base, input, actual, ROWS * 2, COLS, 1, COLS, ROWS * 2};
        CHECK(plamo_metal_linear(metal, &staged));
        CHECK(!memcmp(actual, gold[0], (size_t)ROWS * 2 * 4)); free(gold[0]);
        plamo_bf16_matvec_serial(base, input, expected, ROWS * 2, COLS);
        for (unsigned i = 0; i < ROWS * 2; ++i) CHECK(fabsf(actual[i] - expected[i]) < 0.0002f * (1 + fabsf(expected[i])));
        plamo_metal_get_info(metal, &info);
        CHECK(!info.cache.cached_bytes && info.disk_cache_hits == CHUNKS + (packed ? 2 : 0));
        CHECK(info.disk_cache_mapped_bytes == (packed ? 2 * stored : 0));
        CHECK(plamo_io_stats[0] - staged_reads == (packed ? 0 : 2 * MIB));
        uint64_t decode_hits = info.disk_cache_hits, prefill_reads = plamo_io_stats[0];
        struct plamo_matmul prompt = {base, input, actual, ROWS * 2, COLS, counts[2], COLS, ROWS * 2};
        CHECK(plamo_metal_linear(metal, &prompt));
        CHECK(!memcmp(actual, gold[2], (size_t)ROWS * 2 * counts[2] * 4)); free(gold[2]);
        plamo_metal_get_info(metal, &info);
        CHECK(info.disk_cache_hits == decode_hits && plamo_io_stats[0] - prefill_reads == 2 * MIB);
        available_bytes = 13 * MIB; plamo_metal_maintain(metal);
        for (unsigned c = 0; c < CHUNKS; ++c) CHECK(plamo_metal_preload(metal, (char *)base + c * MIB, MIB));
        plamo_metal_get_info(metal, &info); CHECK(info.disk_cache_hits == 2 * CHUNKS + (packed ? 2 : 0) && !info.errors);
        plamo_metal_close(metal);
    }
    munmap(base, CHUNKS * MIB); fclose(source_file); free(original); free(input); free(expected); free(actual);
    available_bytes = UINT64_C(128) * 1024 * 1024;
    puts("Metal disk cache: PASS (raw/packed bulk read, bitwise FP32, aliases, multiple chunks, eviction/reload)");
}

static void packed_cache(void) {
    enum { MIB = 1024 * 1024, ROWS = 1025, COLS = 1024, TOKENS = 65, YS = ROWS + 9, CASES = 12 };
    const size_t bytes = ROWS * COLS * 2, elements = TOKENS * YS;
    uint16_t *data = malloc(bytes);
    float *x = malloc(TOKENS * (COLS + 7) * 4), *y = malloc(elements * 4), *reference = malloc(CASES * elements * 4);
    CHECK(data && x && y && reference);
    uint32_t seed = 31;
    for (size_t i = 0; i < ROWS * COLS; ++i) {
        seed = seed * 1664525u + 1013904223u;
        data[i] = (uint16_t)((seed & 0x8000) | ((112 + ((seed >> 23) & 15)) << 7) | (seed & 127));
    }
    for (size_t i = 0; i < TOKENS * (COLS + 7); ++i) x[i] = ((int)(i % 97) - 48) / 47.f;
    FILE *file = tmpfile(); CHECK(file && fwrite(data, 1, bytes, file) == bytes && !fflush(file));
    int64_t fd = fileno(file);
    void *base = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, (int)fd, 0); CHECK(base != MAP_FAILED);
    struct plamo_mapping map = {base, bytes};
    const unsigned token_counts[] = {1, 4, 8, TOKENS}, offsets[] = {0, 1, 4};
    for (unsigned format = 0; format < 2; ++format) {
        available_bytes = 13 * MIB;
        struct plamo_metal_config config = {.memory = PLAMO_METAL_AUTO,
            .packing = format ? PLAMO_PACK_LOSSLESS : PLAMO_PACK_RAW,
            .maps = &map, .fds = &fd, .count = 1, .reserve = 8 * MIB, .available = available};
        char error[1024];
        struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal);
        for (size_t start = 0; start < bytes; start += MIB) {
            size_t n = bytes - start; if (n > MIB) n = MIB;
            CHECK(plamo_metal_preload(metal, (const char *)base + start, n));
        }
        uint64_t reads = plamo_io_stats[0];
        unsigned test = 0;
        for (size_t o = 0; o < sizeof offsets / sizeof *offsets; ++o) for (size_t t = 0; t < 4; ++t, ++test) {
            for (size_t i = 0; i < elements; ++i) y[i] = -123456.f;
            struct plamo_matmul p = {(const uint16_t *)base + offsets[o], x, y,
                offsets[o] ? 255 : ROWS, COLS, token_counts[t], COLS + 7, YS};
            CHECK(plamo_metal_linear(metal, &p));
            if (!format) memcpy(reference + test * elements, y, elements * 4);
            else CHECK(!memcmp(reference + test * elements, y, elements * 4));
        }
        struct plamo_metal_info info; plamo_metal_get_info(metal, &info);
        CHECK(plamo_io_stats[0] == reads && !info.errors && !info.cpu_fallbacks);
        CHECK(info.cache.cached_bytes == bytes && info.cache.storage_bytes + info.cache.staging_bytes <= info.cache.budget_bytes);
        if (format) {
            CHECK(info.packed_weight_bytes == 2 * MIB && info.cache.storage_bytes < info.cache.cached_bytes);
            available_bytes = 1; plamo_metal_maintain(metal); plamo_metal_get_info(metal, &info);
            CHECK(!info.packed_weight_bytes && !info.cache.cached_bytes && !info.cache.storage_bytes);
            // An analyzed immutable range's exact size remains known after
            // eviction. Do not reread/reanalyze it if only the theoretical
            // minimum representation would fit the recovered budget.
            available_bytes = 8 * MIB + 640 * 1024; plamo_metal_maintain(metal);
            uint64_t commands = info.command_buffers;
            for (unsigned retry = 0; retry < 3; ++retry) CHECK(plamo_metal_preload(metal, base, MIB));
            plamo_metal_get_info(metal, &info);
            CHECK(plamo_io_stats[0] == reads && info.command_buffers == commands && !info.cache.cached_bytes);
            available_bytes = 13 * MIB; plamo_metal_maintain(metal);
            for (size_t start = 0; start < bytes; start += MIB) {
                size_t count = bytes - start; if (count > MIB) count = MIB;
                CHECK(plamo_metal_preload(metal, (const char *)base + start, count));
            }
            for (size_t i = 0; i < elements; ++i) y[i] = -123456.f;
            struct plamo_matmul p = {base, x, y, ROWS, COLS, 1, COLS + 7, YS};
            CHECK(plamo_metal_linear(metal, &p) && !memcmp(reference, y, elements * 4));
            plamo_metal_get_info(metal, &info);
            CHECK(info.cache.cached_bytes == bytes && info.packed_weight_bytes == 2 * MIB && !info.errors);
            CHECK(info.cache.storage_bytes + info.cache.staging_bytes <= info.cache.budget_bytes);
        }
        else CHECK(!info.packed_weight_bytes);
        plamo_metal_close(metal);
    }
    available_bytes = UINT64_C(128) * MIB;
    munmap(base, bytes); fclose(file); free(data); free(x); free(y); free(reference);
    puts("Metal lossless cache: PASS (bitwise FP32 parity with raw GPU, matvec/vector/matrix, aliases, pressure eviction/reload, physical bounds, no repeated analysis of known non-fitting ranges)");
}

static float unpack_bf16(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float value; memcpy(&value, &word, 4); return value;
}

static void draft_quantization(void) {
    // The auxiliary model alone uses Q4. Check the GPU encoder and all three
    // projection families against an independent CPU quantize/dequantize.
    const unsigned columns[] = {65, 256};
    for (unsigned shape = 0; shape < 2; ++shape) {
        unsigned rows = 133, cols = columns[shape], words = rows * cols;
        uint16_t *data = malloc(words * 2);
        float *dequantized = calloc(words, 4);
        CHECK(data && dequantized);
        uint32_t seed = 123;
        for (unsigned i = 0; i < words; ++i) {
            seed = 1664525u * seed + 1013904223u;
            data[i] = i < 32 ? 0 : (seed & 0x8000) | (0x3b00 + ((seed >> 16) % 1024));
        }
        for (unsigned first = 0; first < words; first += 32) {
            unsigned end = first + 32 < words ? first + 32 : words;
            float maximum = 0;
            for (unsigned i = first; i < end; ++i) maximum = fmaxf(maximum, fabsf(unpack_bf16(data[i])));
            float scale = maximum / 7.f;
            for (unsigned i = first; i < end; ++i) {
                float q = scale ? nearbyintf(unpack_bf16(data[i]) / scale) : 0;
                dequantized[i] = fminf(7, fmaxf(-8, q)) * scale;
            }
        }
        FILE *file = tmpfile();
        CHECK(file && fwrite(data, 2, words, file) == words && !fflush(file));
        int64_t fd = fileno(file);
        void *base = mmap(NULL, words * 2, PROT_READ, MAP_PRIVATE, (int)fd, 0);
        CHECK(base != MAP_FAILED);
        struct plamo_mapping map = {base, words * 2};
        uint64_t dims[] = {rows, cols};
        struct plamo_weight_spec schema[] = {{"draft", 2, dims, &base}, {0}};
        struct plamo_metal_config config = {.memory = PLAMO_METAL_RESIDENT, .packing = PLAMO_PACK_DRAFT_Q4,
            .maps = &map, .fds = &fd, .count = 1, .schema = schema, .reserve = 64 * 1024 * 1024, .available = available};
        char error[1024];
        struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error);
        CHECK(metal);
        const unsigned counts[] = {1, 4, 8, 21};
        for (unsigned c = 0; c < sizeof counts / sizeof *counts; ++c) {
            unsigned tokens = counts[c], xs = cols + 7, ys = rows + 3;
            float *x = malloc(tokens * xs * 4), *y = malloc(tokens * ys * 4);
            CHECK(x && y);
            for (unsigned i = 0; i < tokens * xs; ++i) x[i] = ((int)(i % 97) - 48) / 53.f;
            for (unsigned i = 0; i < tokens * ys; ++i) y[i] = -123456;
            struct plamo_matmul p = {base, x, y, rows, cols, tokens, xs, ys};
            CHECK(plamo_metal_linear(metal, &p));
            for (unsigned t = 0; t < tokens; ++t) for (unsigned r = 0; r < ys; ++r) {
                double expected = -123456;
                if (r < rows) {
                    expected = 0;
                    for (unsigned k = 0; k < cols; ++k) expected += (double)dequantized[r * cols + k] * x[t * xs + k];
                }
                CHECK(fabs(y[t * ys + r] - expected) < 2e-4 * (1 + fabs(expected)));
            }
            free(x); free(y);
        }
        struct plamo_metal_info info; plamo_metal_get_info(metal, &info);
        CHECK(!strcmp(info.weight_storage, "q4-draft") && !info.errors && !info.cpu_fallbacks);
        CHECK(info.cache.storage_bytes < words * 2);
        plamo_metal_close(metal);
        munmap(base, words * 2); fclose(file); free(data); free(dequantized);
    }
    puts("Metal draft Q4: PASS (CPU dequantization reference, zero blocks, tails, strided matvec/vector/matrix, reduced storage)");
}

int main(int argc, char **argv) {
    int require_gpu = argc == 2 && !strcmp(argv[1], "--require-gpu");
    int result = kernels(require_gpu);
    if (result) return result == 77 ? 0 : result;
    cache();
    fused_mlp();
    preload();
    packed_cache();
    disk_cache();
    draft_quantization();
    return 0;
}
