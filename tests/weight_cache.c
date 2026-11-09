#include "../runtime/weight_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MIB (UINT64_C(1024) * 1024)
#define CHECK(condition) do { if (!(condition)) { fprintf(stderr, "weight cache check failed at line %d: %s\n", __LINE__, #condition); exit(1); } } while (0)
extern const void *plamo_weight_acquire(const void *, uint64_t);
extern void plamo_bf16_matvec(const uint16_t *, const float *, float *, uint64_t, uint64_t);
struct matmul { const uint16_t *w; const float *x; float *y; uint64_t rows, cols, tokens, xs, ys; };
extern void plamo_bf16_matmul(const struct matmul *);
static uint64_t available_bytes;
static uint64_t available(void) { return available_bytes; }

static struct plamo_weight_cache *open_cache(struct plamo_mapping *map, int64_t *fd) {
    char error[256];
    available_bytes = 16 * MIB;
    struct plamo_weight_cache *c = plamo_weight_cache_open(map, fd, 1, 8 * MIB, available, error, sizeof error);
    if (!c) fprintf(stderr, "%s\n", error);
    CHECK(c);
    struct plamo_weight_cache_info info;
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.budget_bytes == 8 * MIB && info.max_budget_bytes == 8 * MIB && info.staging_bytes == MIB && !info.cached_bytes);
    available_bytes = 13 * MIB;
    plamo_weight_cache_rebalance(c);
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.budget_bytes == 5 * MIB);
    return c;
}

static void cache_tests(struct plamo_mapping *map, int64_t *fd, const unsigned char *expected) {
    struct plamo_weight_cache *c = open_cache(map, fd);
    struct plamo_weight_cache_info info;
    const void *saved[4];
    uint64_t before = plamo_io_stats[1];
    for (unsigned i = 0; i < 6; ++i) {
        const char *source = (char *)map->base + 37 + i * MIB;
        const void *p = plamo_weight_cache_acquire(source, MIB, c);
        CHECK(p != source && !memcmp(p, expected + 37 + i * MIB, MIB));
        if (i < 4) saved[i] = p;
    }
    CHECK(plamo_io_stats[1] - before == 6);
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.cached_bytes == 4 * MIB && info.cached_bytes + info.staging_bytes <= info.budget_bytes);
    for (unsigned i = 0; i < 4; ++i) {
        const char *source = (char *)map->base + 37 + i * MIB;
        CHECK(plamo_weight_cache_acquire(source, MIB, c) == saved[i]);
        CHECK(!memcmp(saved[i], expected + 37 + i * MIB, MIB));
        CHECK(plamo_weight_cache_acquire(source + 13, MIB - 13, c) == (char *)saved[i] + 13);
    }
    CHECK(plamo_io_stats[1] - before == 6); // Full scans did not evict the retained subset.
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.hits == 8 && info.hit_bytes == 8 * MIB - 4 * 13);

    available_bytes = 6 * MIB; // Reclaim 2 MiB while retaining the first two entries.
    plamo_weight_cache_rebalance(c);
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.budget_bytes == 3 * MIB && info.cached_bytes == 2 * MIB);
    CHECK(plamo_weight_cache_acquire((char *)map->base + 37, MIB, c) == saved[0]);
    CHECK(plamo_weight_cache_acquire((char *)map->base + 37 + MIB, MIB, c) == saved[1]);
    const char *dropped = (char *)map->base + 37 + 2 * MIB;
    const void *p = plamo_weight_cache_acquire(dropped, MIB, c);
    CHECK(p != dropped && !memcmp(p, expected + 37 + 2 * MIB, MIB));
    CHECK(plamo_io_stats[1] - before == 7);

    available_bytes = 14 * MIB;
    plamo_weight_cache_rebalance(c);
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.budget_bytes == 8 * MIB && info.cached_bytes == 2 * MIB);
    available_bytes = UINT64_MAX; // Reclamation cannot inflate the startup ceiling.
    plamo_weight_cache_rebalance(c);
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.budget_bytes == info.max_budget_bytes && info.budget_bytes == 8 * MIB);
    available_bytes = 14 * MIB;
    p = plamo_weight_cache_acquire(dropped, MIB, c);
    CHECK(p != dropped && !memcmp(p, expected + 37 + 2 * MIB, MIB));
    CHECK(plamo_weight_cache_acquire(dropped, MIB, c) == p);
    CHECK(plamo_io_stats[1] - before == 8); // A discarded entry was reloaded exactly once.

    available_bytes = 0;
    // A borrowed tail must stay valid even when the next original acquire will shrink.
    CHECK(plamo_weight_cache_acquire((char *)p + 7, MIB - 7, c) == (char *)p + 7);
    CHECK(!memcmp(p, expected + 37 + 2 * MIB, MIB));
    plamo_weight_cache_rebalance(c);
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.budget_bytes == MIB && !info.cached_bytes);
    p = plamo_weight_cache_acquire(map->base, MIB, c);
    CHECK(p != map->base && !memcmp(p, expected, MIB));
    CHECK(plamo_weight_cache_acquire(expected, MIB, c) == expected);
    CHECK(plamo_weight_cache_acquire(map->base, 0, c) == map->base);
    CHECK(plamo_weight_cache_acquire((char *)map->base + map->bytes - 3, MIB, c) == (char *)map->base + map->bytes - 3);
    plamo_weight_cache_close(c);

    c = open_cache(map, fd);
    CHECK(ftruncate((int)*fd, (off_t)(MIB / 2)) == 0);
    before = plamo_io_stats[3];
    CHECK(plamo_weight_cache_acquire(map->base, MIB, c) == map->base);
    plamo_weight_cache_get_info(c, &info);
    CHECK(!info.cached_bytes && plamo_io_stats[3] == before + 1);
    CHECK(pwrite((int)*fd, expected, (size_t)map->bytes, 0) == (ssize_t)map->bytes);
    p = plamo_weight_cache_acquire(map->base, MIB, c);
    CHECK(p != map->base && !memcmp(p, expected, MIB));
    CHECK(plamo_weight_cache_acquire(map->base, MIB, c) == p);
    plamo_weight_cache_close(c);
}

static void kernel_tests(struct plamo_mapping *map, int64_t *fd) {
    struct plamo_weight_cache *c = open_cache(map, fd);
    enum { ROWS = 257, COLS = 4096, TOKENS = 9, XS = COLS + 7, YS = ROWS + 9 };
    float *input = malloc(TOKENS * XS * sizeof(float));
    float reference[TOKENS * YS], actual[TOKENS * YS];
    CHECK(input);
    for (size_t i = 0; i < TOKENS * XS; ++i) input[i] = (float)((int)(i % 31) - 15) / 31;
    for (size_t i = 0; i < TOKENS * YS; ++i) reference[i] = actual[i] = -123456.f;
    struct matmul m = {map->base, input, reference, ROWS, COLS, TOKENS, XS, YS};
    plamo_io_policy = 0;
    plamo_bf16_matmul(&m);
    plamo_weight_context = c;
    plamo_weight_provider = plamo_weight_cache_acquire;
    plamo_weight_chunk_bytes = MIB;
    m.y = actual;
    plamo_bf16_matmul(&m); // Multiple row tiles, 8x3/row/token tails, and padded strides.
    CHECK(!memcmp(reference, actual, sizeof actual));
    for (unsigned i = 0; i < TOKENS; ++i)
        plamo_bf16_matvec(map->base, input + i * XS, actual + i * YS, ROWS, COLS);
    CHECK(!memcmp(reference, actual, sizeof actual));
    struct plamo_weight_cache_info info;
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.hits && info.cached_bytes == 2 * MIB && info.misses == 2);
    plamo_weight_provider = NULL;
    plamo_weight_context = NULL;
    plamo_weight_chunk_bytes = 0;
    plamo_weight_cache_close(c);
    free(input);
}

static void preload_tests(struct plamo_mapping *map, int64_t *fd, const unsigned char *expected) {
    char error[256];
    available_bytes = 13 * MIB;
    struct plamo_weight_cache *c = plamo_weight_cache_open(map, fd, 1, 8 * MIB, available, error, sizeof error);
    CHECK(c);
    uint64_t before = plamo_io_stats[0];
    for (unsigned i = 0; i < 8; ++i)
        CHECK(plamo_weight_cache_preload(c, (char *)map->base + i * MIB, MIB));
    struct plamo_weight_cache_info info;
    plamo_weight_cache_get_info(c, &info);
    CHECK(info.cached_bytes == 4 * MIB && plamo_io_stats[0] - before == 4 * MIB);
    for (unsigned i = 0; i < 4; ++i) {
        const void *source = (char *)map->base + i * MIB;
        const void *cached = plamo_weight_cache_acquire(source, MIB, c);
        CHECK(cached != source && !memcmp(cached, expected + i * MIB, MIB));
        CHECK(plamo_weight_cache_preload(c, source, MIB));
    }
    CHECK(plamo_io_stats[0] - before == 4 * MIB);
    CHECK(plamo_weight_cache_preload(c, map->base, 0));
    CHECK(plamo_weight_cache_preload(c, expected, MIB)); // Non-model data is not read.
    CHECK(plamo_weight_cache_preload(c, map->base, 2 * MIB)); // Outside the read window.
    CHECK(plamo_io_stats[0] - before == 4 * MIB);
    available_bytes = 0;
    plamo_weight_cache_rebalance(c);
    before = plamo_io_stats[0];
    CHECK(plamo_weight_cache_preload(c, map->base, MIB));
    plamo_weight_cache_get_info(c, &info);
    CHECK(!info.cached_bytes && plamo_io_stats[0] == before);
    plamo_weight_cache_close(c);

    available_bytes = 13 * MIB;
    c = plamo_weight_cache_open(map, fd, 1, 8 * MIB, available, error, sizeof error);
    CHECK(c);
    CHECK(!ftruncate((int)*fd, MIB / 2));
    CHECK(!plamo_weight_cache_preload(c, map->base, MIB));
    plamo_weight_cache_get_info(c, &info);
    CHECK(!info.cached_bytes);
    CHECK(pwrite((int)*fd, expected, (size_t)map->bytes, 0) == (ssize_t)map->bytes);
    CHECK(plamo_weight_cache_preload(c, map->base, MIB));
    const void *cached = plamo_weight_cache_acquire(map->base, MIB, c);
    CHECK(cached != map->base && !memcmp(cached, expected, MIB));
    plamo_weight_cache_close(c);
}

int main(void) {
    size_t bytes = 8 * MIB + 128;
    uint16_t *expected = malloc(bytes);
    CHECK(expected);
    for (size_t i = 0; i < bytes / 2; ++i) expected[i] = (uint16_t)(0x3e00 + i % 127);
    FILE *file = tmpfile();
    CHECK(file && fwrite(expected, 1, bytes, file) == bytes && !fflush(file));
    int64_t fd = fileno(file);
    void *base = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, (int)fd, 0);
    CHECK(base != MAP_FAILED);
    struct plamo_mapping map = {base, bytes};
    cache_tests(&map, &fd, (unsigned char *)expected);
    kernel_tests(&map, &fd);
    preload_tests(&map, &fd, (unsigned char *)expected);
    munmap(base, bytes); fclose(file); free(expected);
    puts("adaptive weights: PASS (bounded preloading without staging reads, reuse, pressure shrink/grow, short-read recovery, bitwise tiled kernels)");
    return 0;
}
