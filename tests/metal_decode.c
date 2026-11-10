#include "../runtime/metal.h"
#include "../runtime/disk_cache.h"
#include "../runtime/packing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "Decode check failed at %d: %s\n", __LINE__, #x); exit(1); } } while (0)
extern int (*plamo_forward_kernel)(struct model *, uint64_t, void *);
extern int (*plamo_prefill_kernel)(struct model *, const uint32_t *, uint64_t, float *, void *);
extern void (*plamo_matvec_kernel)(const uint16_t *, const float *, float *, uint64_t, uint64_t, void *);
extern void (*plamo_matmul_kernel)(const struct plamo_matmul *, void *);
extern int (*plamo_mlp_kernel)(const struct mlp *, float *, uint64_t, void *);
struct hooks {
    __typeof__(plamo_forward_kernel) forward;
    __typeof__(plamo_prefill_kernel) prefill;
    __typeof__(plamo_matvec_kernel) matvec;
    __typeof__(plamo_matmul_kernel) matmul;
    __typeof__(plamo_mlp_kernel) mlp;
};
static struct hooks cpu_begin(void) {
    struct hooks saved = {plamo_forward_kernel, plamo_prefill_kernel, plamo_matvec_kernel, plamo_matmul_kernel, plamo_mlp_kernel};
    plamo_forward_kernel = NULL; plamo_prefill_kernel = NULL;
    plamo_matvec_kernel = NULL; plamo_matmul_kernel = NULL; plamo_mlp_kernel = NULL;
    return saved;
}
static void cpu_end(struct hooks saved) {
    plamo_forward_kernel = saved.forward; plamo_prefill_kernel = saved.prefill;
    plamo_matvec_kernel = saved.matvec; plamo_matmul_kernel = saved.matmul; plamo_mlp_kernel = saved.mlp;
}
static int cpu_prefill(struct model *m, const uint32_t *ids, uint64_t n, float *states) {
    struct hooks saved = cpu_begin(); int ok = plamo_prefill(m, ids, n, states); cpu_end(saved); return ok;
}
static int cpu_forward(struct model *m, uint64_t token) {
    struct hooks saved = cpu_begin(); int ok = plamo_forward(m, token); cpu_end(saved); return ok;
}
static uint64_t available_bytes = UINT64_C(128) * 1024 * 1024;
static uint64_t available(void) { return available_bytes; }
struct allocation { void *base; size_t bytes; };
struct fixture {
    struct model model;
    struct layer layer[2]; struct layer *layers[2];
    struct mixer mixer[2]; struct qkv qkv[2]; struct mlp mlp[2];
    struct cache cache[2]; struct rotation rotation[2]; struct gqa gqa[2];
    float scales[2], frequencies[64];
    struct allocation allocations[40]; size_t count;
};

static float *buffer(struct fixture *f, size_t count) {
    size_t bytes = (count * 4 + 16383) / 16384 * 16384;
    void *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(base != MAP_FAILED && f->count < 40);
    f->allocations[f->count++] = (struct allocation){base, bytes};
    return base;
}
static const uint16_t *take(const uint16_t **p, size_t words) { const uint16_t *out = *p; *p += words; return out; }
static void initialize(struct fixture *f, const uint16_t *weights, size_t h, size_t n, size_t dim, size_t capacity) {
    memset(f, 0, sizeof *f);
    struct model *m = &f->model;
    size_t qh = 4, kh = 2, a = qh * dim, q = (qh + 2 * kh) * dim;
    m->vocab = 17; m->hidden = h; m->count = 2; m->layers = f->layers; m->scales = f->scales;
    m->embedding = take(&weights, m->vocab * h); m->head = take(&weights, m->vocab * h); m->norm = take(&weights, h);
    m->state = buffer(f, h); m->normalized = buffer(f, h); m->logits = buffer(f, m->vocab);
    for (size_t j = 0; j < dim / 2; ++j) f->frequencies[j] = powf(0.0001f, (float)j / (dim / 2));
    for (size_t i = 0; i < 2; ++i) {
        struct layer *l = &f->layer[i]; f->layers[i] = l;
        l->state = m->state; l->hidden = h; l->pre = take(&weights, h); l->post = take(&weights, h);
        l->normalized = buffer(f, h); l->branch = buffer(f, h); l->mixer = &f->mixer[i]; l->mlp = &f->mlp[i];
        struct mixer *mix = l->mixer; mix->qkv = &f->qkv[i]; mix->cache = &f->cache[i];
        mix->rotation = &f->rotation[i]; mix->gqa = &f->gqa[i]; mix->out = l->branch; mix->hidden = h;
        struct qkv *qq = mix->qkv; qq->input = l->normalized; qq->out = buffer(f, q);
        qq->weight = take(&weights, q * h); qq->qn = take(&weights, dim); qq->kn = take(&weights, dim);
        qq->hidden = h; qq->qh = qh; qq->kh = kh; qq->dim = dim;
        mix->weight = take(&weights, h * a);
        struct cache *c = mix->cache; c->capacity = i ? capacity : 3; c->width = kh * dim; c->sliding = !i;
        c->k = buffer(f, c->capacity * c->width); c->v = buffer(f, c->capacity * c->width);
        struct rotation *r = mix->rotation; r->cache = c; r->out = buffer(f, c->capacity * c->width);
        r->freq = f->frequencies; r->cos = buffer(f, dim / 2); r->sin = buffer(f, dim / 2); r->heads = kh; r->dim = dim;
        struct gqa *g = mix->gqa; g->out = buffer(f, a); g->scores = buffer(f, c->capacity); g->dim = dim; g->qh = qh; g->kh = kh;
        struct mlp *p = l->mlp; p->state = m->state; p->pre = take(&weights, h); p->post = take(&weights, h);
        p->gu = take(&weights, 2 * n * h); p->down = take(&weights, h * n);
        p->normalized = buffer(f, h); p->branch = buffer(f, h); p->scratch = buffer(f, 2 * n); p->hidden = h; p->intermediate = n;
        f->scales[i] = i ? 1.5545177f : 1.0f;
    }
}
static void destroy(struct fixture *f) {
    for (size_t i = 0; i < f->count; ++i) CHECK(!munmap(f->allocations[i].base, f->allocations[i].bytes));
}
static void preload(struct plamo_metal *metal, struct model *m) {
    for (size_t i = 0; i < m->count; ++i) {
        struct layer *l = m->layers[i]; struct qkv *q = l->mixer->qkv; struct mlp *p = l->mlp;
        CHECK(plamo_metal_preload(metal, q->weight, (q->qh + 2 * q->kh) * q->dim * q->hidden * 2));
        CHECK(plamo_metal_preload(metal, l->mixer->weight, m->hidden * q->qh * q->dim * 2));
        CHECK(plamo_metal_preload(metal, p->gu, 2 * p->intermediate * p->hidden * 2));
        CHECK(plamo_metal_preload(metal, p->down, p->intermediate * p->hidden * 2));
    }
    CHECK(plamo_metal_preload(metal, m->head, m->vocab * m->hidden * 2));
}
static void equal_floats(const float *a, const float *b, size_t count, const char *name) {
    double error = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            fprintf(stderr, "%s[%zu]: %.9g vs %.9g\n", name, i, a[i], b[i]); exit(1);
        }
        double delta = (double)a[i] - b[i]; error += delta * delta; norm += (double)b[i] * b[i];
        if (fabs(delta) > maximum) maximum = fabs(delta);
    }
    // GPU sqrt/division and CPU sqrt/division can differ by a few FP32 ulps.
    // Use the same complete-vector relative bound as the real-model probes.
    CHECK(sqrt(error / (norm ? norm : 1)) <= 0.0001 && maximum <= 0.00001);
}
static void compare(struct fixture *a, struct fixture *b) {
    equal_floats(a->model.state, b->model.state, a->model.hidden, "state");
    equal_floats(a->model.logits, b->model.logits, a->model.vocab, "logits");
    for (size_t i = 0; i < 2; ++i) {
        struct cache *ac = &a->cache[i], *bc = &b->cache[i]; CHECK(ac->count == bc->count);
        size_t words = ac->count * ac->width;
        equal_floats(ac->k, bc->k, words, "unrotated K"); equal_floats(ac->v, bc->v, words, "V");
        equal_floats(a->rotation[i].out, b->rotation[i].out, words, "rotated K");
        CHECK(a->rotation[i].processed_count == b->rotation[i].processed_count);
        CHECK(a->rotation[i].scale_bits == b->rotation[i].scale_bits);
    }
}
static void reset(struct fixture *f) {
    for (size_t i = 0; i < 2; ++i) {
        f->cache[i].count = 0; f->rotation[i].processed_count = 0; f->rotation[i].scale_bits = 0;
    }
}
static void prefills(struct plamo_metal *metal, struct fixture *a, struct fixture *b) {
    const size_t counts[] = {1, 2, 3, 4, 7, 8, 9, 17, 33, 129, 4096};
    const size_t prefixes[] = {0, 2, 6};
    size_t h = a->model.hidden;
    for (size_t pi = 0; pi < sizeof prefixes / sizeof *prefixes; ++pi)
        for (size_t ni = 0; ni < sizeof counts / sizeof *counts; ++ni) {
            size_t prefix = prefixes[pi], n = counts[ni];
            if (n + prefix >= a->cache[1].capacity) continue;
            if (n == 4096 && prefix != 2) continue;
            reset(a); reset(b);
            for (size_t j = 0; j < prefix; ++j) {
                CHECK(cpu_forward(&a->model, j % 17)); CHECK(cpu_forward(&b->model, j % 17));
            }
            // Rebuild stale prefix rotations; a repeated request also reuses
            // the same allocated GPU views/workspace with shorter prefixes.
            for (size_t j = 0; j < 2; ++j) a->rotation[j].processed_count = 0;
            uint32_t ids[4096]; for (size_t j = 0; j < n; ++j) ids[j] = (j * 7 + 3) % 17;
            float *actual = calloc(n * h, 4), *expected = calloc(n * h, 4); CHECK(actual && expected);
            struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
            CHECK(plamo_prefill(&a->model, ids, n, actual)); plamo_metal_get_info(metal, &after);
            CHECK(after.command_buffers == before.command_buffers + 1 && after.fused_prefills == before.fused_prefills + 1);
            CHECK(cpu_prefill(&b->model, ids, n, expected));
            equal_floats(actual, expected, n * h, "all prompt states"); compare(a, b);
            CHECK(plamo_forward(&a->model, 7)); CHECK(cpu_forward(&b->model, 7)); compare(a, b);
            free(actual); free(expected);
        }
    reset(a); reset(b);
    uint32_t ids[] = {2, 5, 9};
    float *actual = calloc(3 * h, 4), *expected = calloc(3 * h, 4); CHECK(actual && expected);
    available_bytes = 1; plamo_metal_maintain(metal);
    CHECK(plamo_metal_prefill(metal, &a->model, ids, 3, actual) == 0); compare(a, b);
    for (size_t j = 0; j < 3 * h; ++j) CHECK(actual[j] == 0);
    CHECK(plamo_prefill(&a->model, ids, 3, actual)); CHECK(cpu_prefill(&b->model, ids, 3, expected));
    equal_floats(actual, expected, 3 * h, "prefill pressure fallback"); compare(a, b);
    available_bytes = UINT64_C(128) * 1024 * 1024; plamo_metal_maintain(metal); preload(metal, &a->model);
    free(actual); free(expected); reset(a); reset(b);
}
static uint64_t fixture_hash(const struct fixture *);
static void verification(struct plamo_metal *metal, struct fixture *a, struct fixture *b) {
    const unsigned sizes[] = {1, 2, 3, 4, 6, 8, 12, 16};
    uint32_t prefix[] = {2, 5}, ids[PLAMO_VERIFY_MAX], next[PLAMO_VERIFY_MAX], gold_next[PLAMO_VERIFY_MAX];
    for (unsigned i = 0; i < PLAMO_VERIFY_MAX; ++i) ids[i] = (i * 7 + 3) % 17;
    size_t h = a->model.hidden, v = a->model.vocab;
    float *states = malloc(PLAMO_VERIFY_MAX * h * 4), *logits = malloc(PLAMO_VERIFY_MAX * v * 4);
    float *initial = malloc(2 * h * 4); CHECK(states && logits && initial);
    for (unsigned n = 0; n < sizeof sizes / sizeof *sizes; ++n) {
        unsigned count = sizes[n]; reset(b);
        CHECK(plamo_metal_prefill(metal, &b->model, prefix, 2, initial) == 1);
        for (unsigned i = 0; i < count; ++i) {
            CHECK(plamo_metal_forward(metal, &b->model, ids[i]) == 1);
            memcpy(states + i * h, b->model.state, h * 4); memcpy(logits + i * v, b->model.logits, v * 4);
            gold_next[i] = (uint32_t)plamo_argmax(b->model.logits, v);
        }
        reset(a); CHECK(plamo_metal_prefill(metal, &a->model, prefix, 2, initial) == 1);
        uint64_t untouched = fixture_hash(a);
        CHECK(plamo_metal_verify(metal, &a->model, ids, count, next) == 1 && fixture_hash(a) == untouched);
        CHECK(!plamo_metal_commit(metal, &a->model, 1) && fixture_hash(a) == untouched);
        for (unsigned i = 0; i < count; ++i) {
            CHECK(next[i] == gold_next[i] && plamo_metal_commit(metal, &a->model, i) == 1);
            equal_floats(a->model.state, states + i * h, h, "verified state");
            equal_floats(a->model.logits, logits + i * v, v, "verified logits");
        }
        compare(a, b); CHECK(!plamo_metal_commit(metal, &a->model, count));
        if (count >= 4) {
            reset(b); CHECK(plamo_metal_prefill(metal, &b->model, prefix, 2, initial) == 1);
            for (unsigned i = 0; i < 3; ++i) CHECK(plamo_metal_forward(metal, &b->model, ids[i]) == 1);
            CHECK(plamo_metal_forward(metal, &b->model, 12) == 1);
            reset(a); CHECK(plamo_metal_prefill(metal, &a->model, prefix, 2, initial) == 1);
            CHECK(plamo_metal_verify(metal, &a->model, ids, count, next) == 1);
            for (unsigned i = 0; i < 3; ++i) CHECK(plamo_metal_commit(metal, &a->model, i) == 1);
            CHECK(plamo_metal_forward(metal, &a->model, 12) == 1 && !plamo_metal_commit(metal, &a->model, 3)); compare(a, b);
        }
    }
    free(states); free(logits); free(initial); reset(a); reset(b);
    puts("Metal verification: PASS (1..16 FP32 candidates, unchanged canonical state before commit, sequential-prefix commit, sliding KV, rejected suffix, ordinary-decode recovery)");
}
static int run(size_t h, size_t n, size_t dim, size_t capacity, int require_gpu) {
    size_t qh = 4, kh = 2, words = 35 * h + 2 * (4 * h + (qh + 2 * kh) * dim * h + 2 * dim + h * qh * dim + 3 * h * n);
    uint16_t *data = malloc(words * 2); CHECK(data);
    uint32_t seed = 93;
    for (size_t i = 0; i < words; ++i) { seed = seed * 1664525u + 1013904223u; data[i] = 0x3c00 | (seed & 0xff) | (seed & 0x8000); }
    FILE *file = tmpfile(); CHECK(file && fwrite(data, 2, words, file) == words && !fflush(file)); free(data);
    int64_t fd = fileno(file); void *base = mmap(NULL, words * 2, PROT_READ, MAP_PRIVATE, (int)fd, 0); CHECK(base != MAP_FAILED);
    struct fixture a, b; initialize(&a, base, h, n, dim, capacity); initialize(&b, base, h, n, dim, capacity);
    struct plamo_mapping map = {base, words * 2};
    struct plamo_metal_config config = {.memory = PLAMO_METAL_AUTO, .maps = &map, .fds = &fd, .count = 1,
        .packing = n > 1000 ? PLAMO_PACK_LOSSLESS : PLAMO_PACK_RAW, .reserve = 64 * 1024 * 1024, .available = available};
    char error[1024]; struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error);
    if (!metal) {
        int skip = !require_gpu && !strcmp(error, "Metal requires an available Apple Silicon GPU");
        fprintf(stderr, "Decode tests %s: %s\n", skip ? "SKIPPED" : "FAILED", error); return skip ? 77 : 1;
    }
    CHECK(plamo_metal_forward(metal, &a.model, 2) == 0); compare(&a, &b);
    uint32_t ids[] = {2, 5, 9}; float *untouched = calloc(3 * h, 4); CHECK(untouched);
    CHECK(plamo_metal_prefill(metal, &a.model, ids, 3, untouched) == 0); compare(&a, &b);
    ids[2] = 17;
    CHECK(plamo_metal_prefill(metal, &a.model, ids, 3, untouched) == -1); compare(&a, &b);
    for (size_t j = 0; j < 3 * h; ++j) CHECK(untouched[j] == 0);
    free(untouched);
    preload(metal, &a.model); CHECK(plamo_metal_prepare_model(metal, &a.model));
    prefills(metal, &a, &b);
    verification(metal, &a, &b);
    struct plamo_metal_info initial; plamo_metal_get_info(metal, &initial);
    CHECK((initial.packed_weight_bytes != 0) == (n > 1000));
    for (size_t step = 0; step < capacity && step < 140; ++step) {
        if (step == 9) {
            // CPU batch writes directly into the GPU-visible canonical KV.
            uint32_t ids[] = {2, 5, 9}; float *as = calloc(3 * h, 4), *bs = calloc(3 * h, 4); CHECK(as && bs);
            CHECK(plamo_prefill(&a.model, ids, 3, as)); CHECK(cpu_prefill(&b.model, ids, 3, bs));
            equal_floats(as, bs, 3 * h, "prefill"); free(as); free(bs); compare(&a, &b);
            for (size_t i = 0; i < 2; ++i) {
                a.cache[i].count = b.cache[i].count = 0;
                a.rotation[i].processed_count = b.rotation[i].processed_count = 0;
            }
        }
        if (step == 17) {
            for (size_t i = 0; i < 2; ++i) a.rotation[i].processed_count = b.rotation[i].processed_count = 0;
        }
        CHECK(cpu_forward(&b.model, (step * 7 + 2) % 17));
        struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
        CHECK(plamo_forward(&a.model, (step * 7 + 2) % 17)); plamo_metal_get_info(metal, &after);
        CHECK(after.command_buffers == before.command_buffers + 1 && after.fused_decodes == before.fused_decodes + 1);
        CHECK(!after.errors && !after.cpu_fallbacks); compare(&a, &b);
        if (step == 27) {
            available_bytes = 1; plamo_metal_maintain(metal);
            CHECK(plamo_metal_forward(metal, &a.model, 3) == 0); compare(&a, &b);
            // The ordinary assembly path can resume from GPU-produced KV.
            CHECK(plamo_forward(&a.model, 3));
            CHECK(cpu_forward(&b.model, 3)); compare(&a, &b);
            available_bytes = UINT64_C(128) * 1024 * 1024; plamo_metal_maintain(metal); preload(metal, &a.model);
        }
    }
    uint64_t count = a.cache[1].count;
    CHECK(!plamo_forward(&a.model, 17) && a.cache[1].count == count); compare(&a, &b);
    struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
    a.cache[1].count = b.cache[1].count = capacity;
    CHECK(!plamo_forward(&a.model, 2)); plamo_metal_get_info(metal, &after);
    CHECK(a.cache[1].count == capacity && after.command_buffers == before.command_buffers); compare(&a, &b);
    plamo_metal_close(metal); CHECK(!plamo_forward_kernel && !plamo_prefill_kernel);
    destroy(&a); destroy(&b); CHECK(!munmap(base, words * 2)); CHECK(!fclose(file));
    return 0;
}

static uint64_t plentiful(void) { return UINT64_C(128) * 1024 * 1024; }
static uint64_t page_round(uint64_t n) { return (n + 16383) / 16384 * 16384; }
static void archive(const char *path, const struct fixture *f, const void *base, size_t bytes, int complete) {
    struct { const uint16_t *data; uint32_t rows, columns; } ranges[9];
    unsigned count = 0;
    for (unsigned i = 0; i < 2; ++i) {
        const struct layer *l = &f->layer[i]; const struct qkv *q = &f->qkv[i]; const struct mlp *p = &f->mlp[i];
        ranges[count++] = (__typeof__(ranges[0])){q->weight, (q->qh + 2 * q->kh) * q->dim, q->hidden};
        ranges[count++] = (__typeof__(ranges[0])){l->mixer->weight, l->hidden, q->qh * q->dim};
        ranges[count++] = (__typeof__(ranges[0])){p->gu, 2 * p->intermediate, p->hidden};
        ranges[count++] = (__typeof__(ranges[0])){p->down, p->hidden, p->intermediate};
    }
    if (complete) ranges[count++] = (__typeof__(ranges[0])){f->model.head, f->model.vocab, f->model.hidden};
    FILE *file = fopen(path, "wb"); CHECK(file);
    unsigned char metadata[PLAMO_DISK_ALIGNMENT] = {0};
    struct plamo_disk_header *h = (void *)metadata;
    memcpy(h->magic, PLAMO_DISK_MAGIC, 8); h->version = 1; h->alignment = PLAMO_DISK_ALIGNMENT;
    h->entries = count; h->sources = 1; h->source_bytes[0] = bytes; strcpy(h->revision, "file-graph");
    struct plamo_disk_entry *entries = (void *)(h + 1);
    uint64_t offset = sizeof metadata;
    for (unsigned i = 0; i < count; ++i) {
        uint64_t values = (uint64_t)ranges[i].rows * ranges[i].columns, blocks = (values + 511) / 512;
        unsigned char *out = calloc(1, (size_t)page_round(blocks * (8 + 1024) + 16)); CHECK(out);
        struct plamo_bf16_block *headers = (void *)out;
        for (uint64_t b = 0; b < blocks; ++b) {
            unsigned lo = 255, hi = 0, bits = 0;
            for (uint64_t j = b * 512; j < values && j < (b + 1) * 512; ++j) {
                unsigned e = (ranges[i].data[j] >> 7) & 255; if (e < lo) lo = e; if (e > hi) hi = e;
            }
            while ((1u << bits) <= hi - lo) ++bits;
            headers[b].exponent = lo | bits << 8;
        }
        uint64_t length = plamo_bf16_pack_offsets(headers, values); CHECK(length);
        for (uint64_t j = 0; j < values; ++j) {
            const struct plamo_bf16_block *b = &headers[j / 512]; uint16_t v = ranges[i].data[j];
            unsigned bits = b->exponent >> 8, delta = ((v >> 7) & 255) - (b->exponent & 255), k = j % 512;
            out[b->offset + k] = (v & 127) | ((v >> 8) & 128);
            for (unsigned bit = 0; bit < bits; ++bit)
                out[b->offset + 512 + (k * bits + bit) / 8] |= ((delta >> bit) & 1) << ((k * bits + bit) % 8);
        }
        entries[i] = (struct plamo_disk_entry){0, 1, (const char *)ranges[i].data - (const char *)base,
            values * 2, offset, length, ranges[i].rows, ranges[i].columns};
        CHECK(!fseek(file, (long)offset, SEEK_SET) && fwrite(out, 1, (size_t)page_round(length), file) == page_round(length));
        offset += page_round(length); free(out);
    }
    h->file_bytes = offset;
    CHECK(!fseek(file, 0, SEEK_SET) && fwrite(metadata, 1, sizeof metadata, file) == sizeof metadata && !fclose(file));
}
static void compare_bits(const struct fixture *a, const struct fixture *b) {
    CHECK(!memcmp(a->model.state, b->model.state, a->model.hidden * 4));
    CHECK(!memcmp(a->model.logits, b->model.logits, a->model.vocab * 4));
    for (unsigned i = 0; i < 2; ++i) {
        const struct cache *ac = &a->cache[i], *bc = &b->cache[i]; CHECK(ac->count == bc->count);
        size_t bytes = ac->capacity * ac->width * 4;
        CHECK(!memcmp(ac->k, bc->k, bytes) && !memcmp(ac->v, bc->v, bytes));
        CHECK(!memcmp(a->rotation[i].out, b->rotation[i].out, bytes));
        CHECK(a->rotation[i].processed_count == b->rotation[i].processed_count && a->rotation[i].scale_bits == b->rotation[i].scale_bits);
    }
}
static uint64_t fixture_hash(const struct fixture *f) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
#define HASH(data, bytes) do { const unsigned char *p = (const void *)(data); \
    for (size_t i = 0; i < (bytes); ++i) hash = (hash ^ p[i]) * UINT64_C(0x100000001b3); } while (0)
    HASH(f->model.state, f->model.hidden * 4); HASH(f->model.logits, f->model.vocab * 4);
    for (unsigned i = 0; i < 2; ++i) {
        const struct cache *c = &f->cache[i]; const struct rotation *r = &f->rotation[i];
        size_t bytes = c->capacity * c->width * 4;
        HASH(c->k, bytes); HASH(c->v, bytes); HASH(r->out, bytes);
        HASH(&c->count, sizeof c->count); HASH(&r->processed_count, sizeof r->processed_count); HASH(&r->scale_bits, sizeof r->scale_bits);
    }
#undef HASH
    return hash;
}
static void mapped_graph(void) {
    const size_t h = 128, n = 257, dim = 128, capacity = 160;
    size_t words = 35 * h + 2 * (4 * h + 8 * dim * h + 2 * dim + h * 4 * dim + 3 * h * n);
    uint16_t *data = malloc(words * 2); CHECK(data);
    for (size_t i = 0; i < words; ++i) data[i] = 0x3c00 | (i * 93 & 255) | ((i & 1) ? 0x8000 : 0);
    FILE *file = tmpfile(); CHECK(file && fwrite(data, 2, words, file) == words && !fflush(file)); free(data);
    int64_t fd = fileno(file); void *base = mmap(NULL, words * 2, PROT_READ, MAP_PRIVATE, (int)fd, 0); CHECK(base != MAP_FAILED);
    struct plamo_mapping map = {base, words * 2};
    struct fixture a, b; initialize(&a, base, h, n, dim, capacity); initialize(&b, base, h, n, dim, capacity);
    struct plamo_metal_config config = {.memory = PLAMO_METAL_AUTO, .maps = &map, .fds = &fd, .count = 1,
        .packing = PLAMO_PACK_RAW, .reserve = 64 * 1024 * 1024, .available = plentiful, .revision = "file-graph"};
    char error[1024], path[] = "/tmp/plamo-file-graph-XXXXXX";
    int cache_fd = mkstemp(path); CHECK(cache_fd >= 0 && !close(cache_fd)); archive(path, &a, base, words * 2, 1);
    struct plamo_metal *reference = plamo_metal_open(&config, error, sizeof error); CHECK(reference); preload(reference, &b.model);
    const unsigned counts[] = {1, 9, 33}; uint64_t expected_hashes[3][9]; float expected_states[3][33 * 128];
    // A process owns one backend. Save the complete raw-GPU reference before
    // reopening it with file-backed weights, preserving every intermediate hash.
    for (unsigned phase = 0; phase < 3; ++phase) {
        reset(&b); unsigned count = counts[phase]; uint32_t ids[33];
        for (unsigned i = 0; i < count; ++i) ids[i] = (i * 7 + 2) % 17;
        CHECK(plamo_metal_prefill(reference, &b.model, ids, count, expected_states[phase]) == 1);
        expected_hashes[phase][0] = fixture_hash(&b);
        for (unsigned step = 0; step < 8; ++step) {
            CHECK(plamo_metal_forward(reference, &b.model, step % 17) == 1);
            expected_hashes[phase][step + 1] = fixture_hash(&b);
        }
    }
    plamo_metal_close(reference);
    config.packing = PLAMO_PACK_LOSSLESS; config.available = available; config.disk_cache_path = path;
    available_bytes = plentiful();
    struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal && !unlink(path));
    for (unsigned phase = 0; phase < 3; ++phase) {
        reset(&a); available_bytes = phase ? plentiful() : 1; plamo_metal_maintain(metal);
        if (phase == 2) {
            const struct qkv *q = &a.qkv[0]; CHECK(plamo_metal_preload(metal, q->weight, 8 * q->dim * h * 2));
        }
        unsigned count = counts[phase];
        uint32_t ids[33]; for (unsigned i = 0; i < count; ++i) ids[i] = (i * 7 + 2) % 17;
        float actual[33 * 128];
        struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
        CHECK(plamo_metal_prefill(metal, &a.model, ids, count, actual) == 1);
        CHECK(!memcmp(actual, expected_states[phase], count * h * 4) && fixture_hash(&a) == expected_hashes[phase][0]);
        plamo_metal_get_info(metal, &after);
        CHECK(after.command_buffers == before.command_buffers + 1 && after.disk_cache_mapped_bytes > before.disk_cache_mapped_bytes);
        uint64_t mappings = after.disk_cache_mapping_bytes;
        CHECK((mappings > 0) == (phase > 0) && (after.cache.cached_bytes > 0) == (phase == 2));
        for (unsigned step = 0; step < 8; ++step) {
            CHECK(plamo_metal_forward(metal, &a.model, step % 17) == 1);
            CHECK(fixture_hash(&a) == expected_hashes[phase][step + 1]);
        }
        plamo_metal_get_info(metal, &after); CHECK(after.disk_cache_mapping_bytes == mappings && !after.errors);
        available_bytes = 1; plamo_metal_maintain(metal); plamo_metal_get_info(metal, &after);
        CHECK(!after.disk_cache_mapping_bytes && !after.cache.cached_bytes);
    }
    compare_bits(&a, &b);
    plamo_metal_close(metal); destroy(&a); destroy(&b);
    initialize(&a, base, h, n, dim, capacity); initialize(&b, base, h, n, dim, capacity);
    strcpy(path, "/tmp/plamo-file-graph-XXXXXX"); cache_fd = mkstemp(path); CHECK(cache_fd >= 0 && !close(cache_fd));
    archive(path, &a, base, words * 2, 0); available_bytes = plentiful();
    metal = plamo_metal_open(&config, error, sizeof error); CHECK(metal && !unlink(path));
    struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
    uint32_t ids[] = {2, 5, 9}; float untouched[3 * 128] = {0};
    CHECK(plamo_metal_forward(metal, &a.model, 2) == 0 && plamo_metal_prefill(metal, &a.model, ids, 3, untouched) == 0);
    compare_bits(&a, &b); for (unsigned i = 0; i < 3 * 128; ++i) CHECK(!untouched[i]);
    plamo_metal_get_info(metal, &after); CHECK(after.command_buffers == before.command_buffers && !after.disk_cache_mapping_bytes);
    plamo_metal_close(metal); destroy(&a); destroy(&b); CHECK(!munmap(base, words * 2) && !fclose(file));
    available_bytes = plentiful();
    puts("Metal file graph: PASS (bitwise raw-GPU states/logits/KV, cached/uncached views, mixed private/file weights, unlink lifetime, pressure release, atomic missing-range fallback)");
}
int main(int argc, char **argv) {
    int require = argc == 2 && !strcmp(argv[1], "--require-gpu");
    int result = run(7, 11, 6, 4100, require); if (result) return result == 77 ? 0 : result;
    result = run(128, 257, 128, 19, require); if (result) return result == 77 ? 0 : result;
    result = run(128, 4097, 128, 19, require); if (result) return result == 77 ? 0 : result;
    mapped_graph();
    puts("Metal decode/prefill: PASS (single command, full CPU reference, all prompt states/logits/KV, sliding batches, existing prefixes, odd widths, growth past 128 rows, stale rotations, request reset, pressure fallback/recovery)");
    return 0;
}
