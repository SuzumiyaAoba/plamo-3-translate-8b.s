#include "prompt_cache.h"
#include <stdlib.h>
#include <string.h>

struct plamo_prompt_cache {
    const struct model *model;
    uint64_t tokens, bytes, layers;
    uint32_t ids[PLAMO_PROMPT_PREFIX_LIMIT];
    struct { uint64_t width, scale; } layer[128];
    float kv[];
};

struct plamo_prompt_cache *plamo_prompt_cache_capture(const struct model *model, const uint32_t *ids, uint64_t tokens) {
    if (!model || !ids || !tokens || tokens > PLAMO_PROMPT_PREFIX_LIMIT || !model->count || model->count > 128) return NULL;
    uint64_t bytes = sizeof(struct plamo_prompt_cache), limit = UINT64_C(16) * 1024 * 1024;
    for (uint64_t i = 0; i < model->count; ++i) {
        const struct mixer *m = model->layers[i]->mixer;
        const struct cache *c = m->cache;
        if (!c->width || c->count != tokens || c->capacity < tokens || m->rotation->cache != c ||
            m->rotation->processed_count != tokens || !c->k || !c->v || !m->rotation->out ||
            c->width > (limit - bytes) / (tokens * 12)) return NULL;
        bytes += c->width * tokens * 12;
    }
    struct plamo_prompt_cache *snapshot = malloc((size_t)bytes);
    if (!snapshot) return NULL;
    snapshot->model = model; snapshot->layers = model->count; snapshot->tokens = tokens; snapshot->bytes = bytes;
    memcpy(snapshot->ids, ids, (size_t)tokens * 4);
    float *out = snapshot->kv;
    for (uint64_t i = 0; i < model->count; ++i) {
        const struct mixer *m = model->layers[i]->mixer;
        uint64_t words = m->cache->width * tokens;
        snapshot->layer[i].width = m->cache->width;
        snapshot->layer[i].scale = m->rotation->scale_bits;
        memcpy(out, m->cache->k, (size_t)words * 4); out += words;
        memcpy(out, m->cache->v, (size_t)words * 4); out += words;
        memcpy(out, m->rotation->out, (size_t)words * 4); out += words;
    }
    return snapshot;
}

uint64_t plamo_prompt_cache_restore(const struct plamo_prompt_cache *snapshot, struct model *model,
                                    const uint32_t *ids, uint64_t tokens) {
    if (!snapshot || snapshot->model != model || !ids || tokens <= snapshot->tokens ||
        model->count != snapshot->layers || memcmp(ids, snapshot->ids, (size_t)snapshot->tokens * 4)) return 0;
    // Validate all destinations before changing any of them. A model change
    // or a smaller window is an ordinary miss, never a partial restoration.
    for (uint64_t i = 0; i < model->count; ++i) {
        const struct mixer *m = model->layers[i]->mixer;
        if (m->cache->width != snapshot->layer[i].width || m->cache->capacity < snapshot->tokens ||
            m->rotation->cache != m->cache || !m->cache->k || !m->cache->v || !m->rotation->out) return 0;
    }
    const float *in = snapshot->kv;
    for (uint64_t i = 0; i < model->count; ++i) {
        struct mixer *m = model->layers[i]->mixer;
        uint64_t words = m->cache->width * snapshot->tokens;
        memcpy(m->cache->k, in, (size_t)words * 4); in += words;
        memcpy(m->cache->v, in, (size_t)words * 4); in += words;
        memcpy(m->rotation->out, in, (size_t)words * 4); in += words;
        m->cache->count = m->rotation->processed_count = snapshot->tokens;
        m->rotation->scale_bits = snapshot->layer[i].scale;
    }
    return snapshot->tokens;
}

uint64_t plamo_prompt_cache_tokens(const struct plamo_prompt_cache *snapshot) { return snapshot ? snapshot->tokens : 0; }
uint64_t plamo_prompt_cache_bytes(const struct plamo_prompt_cache *snapshot) { return snapshot ? snapshot->bytes : 0; }
void plamo_prompt_cache_free(struct plamo_prompt_cache *snapshot) { free(snapshot); }
