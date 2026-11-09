#include "../runtime/metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "Decode check failed at %d: %s\n", __LINE__, #x); exit(1); } } while (0)
extern int (*plamo_forward_kernel)(struct model *, uint64_t, void *);
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
        .packing = PLAMO_PACK_RAW, .reserve = 64 * 1024 * 1024, .available = available};
    char error[1024]; struct plamo_metal *metal = plamo_metal_open(&config, error, sizeof error);
    if (!metal) {
        int skip = !require_gpu && !strcmp(error, "Metal requires an available Apple Silicon GPU");
        fprintf(stderr, "Decode tests %s: %s\n", skip ? "SKIPPED" : "FAILED", error); return skip ? 77 : 1;
    }
    CHECK(plamo_metal_forward(metal, &a.model, 2) == 0); compare(&a, &b);
    preload(metal, &a.model); CHECK(plamo_metal_prepare_model(metal, &a.model));
    for (size_t step = 0; step < capacity; ++step) {
        if (step == 9) {
            // CPU batch writes directly into the GPU-visible canonical KV.
            uint32_t ids[] = {2, 5, 9}; float *as = calloc(3 * h, 4), *bs = calloc(3 * h, 4); CHECK(as && bs);
            CHECK(plamo_prefill(&a.model, ids, 3, as)); CHECK(plamo_prefill(&b.model, ids, 3, bs));
            equal_floats(as, bs, 3 * h, "prefill"); free(as); free(bs); compare(&a, &b);
            for (size_t i = 0; i < 2; ++i) {
                a.cache[i].count = b.cache[i].count = 0;
                a.rotation[i].processed_count = b.rotation[i].processed_count = 0;
            }
        }
        if (step == 17) {
            for (size_t i = 0; i < 2; ++i) a.rotation[i].processed_count = b.rotation[i].processed_count = 0;
        }
        int (*hook)(struct model *, uint64_t, void *) = plamo_forward_kernel;
        plamo_forward_kernel = NULL; CHECK(plamo_forward(&b.model, (step * 7 + 2) % 17)); plamo_forward_kernel = hook;
        struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
        CHECK(plamo_forward(&a.model, (step * 7 + 2) % 17)); plamo_metal_get_info(metal, &after);
        CHECK(after.command_buffers == before.command_buffers + 1 && after.fused_decodes == before.fused_decodes + 1);
        CHECK(!after.errors && !after.cpu_fallbacks); compare(&a, &b);
        if (step == 27) {
            available_bytes = 1; plamo_metal_maintain(metal);
            CHECK(plamo_metal_forward(metal, &a.model, 3) == 0); compare(&a, &b);
            // The ordinary assembly path can resume from GPU-produced KV.
            CHECK(plamo_forward(&a.model, 3));
            plamo_forward_kernel = NULL; CHECK(plamo_forward(&b.model, 3)); plamo_forward_kernel = hook; compare(&a, &b);
            available_bytes = UINT64_C(128) * 1024 * 1024; plamo_metal_maintain(metal); preload(metal, &a.model);
        }
    }
    uint64_t count = a.cache[1].count;
    CHECK(!plamo_forward(&a.model, 17) && a.cache[1].count == count); compare(&a, &b);
    struct plamo_metal_info before, after; plamo_metal_get_info(metal, &before);
    a.cache[1].count = b.cache[1].count = capacity;
    CHECK(!plamo_forward(&a.model, 2)); plamo_metal_get_info(metal, &after);
    CHECK(a.cache[1].count == capacity && after.command_buffers == before.command_buffers); compare(&a, &b);
    plamo_metal_close(metal); CHECK(!plamo_forward_kernel);
    destroy(&a); destroy(&b); CHECK(!munmap(base, words * 2)); CHECK(!fclose(file));
    return 0;
}
int main(int argc, char **argv) {
    int require = argc == 2 && !strcmp(argv[1], "--require-gpu");
    int result = run(7, 11, 6, 140, require); if (result) return result == 77 ? 0 : result;
    result = run(128, 257, 128, 19, require); if (result) return result == 77 ? 0 : result;
    puts("Metal decode: PASS (single command, CPU attention reference, shared KV, sliding eviction, growth past 128 rows, stale rotations, batched prefill, request reset, pressure fallback/recovery)");
    return 0;
}
