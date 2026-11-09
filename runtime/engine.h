#ifndef PLAMO_ENGINE_H
#define PLAMO_ENGINE_H
#include "resident.h"
#include "weight_cache.h"
#include "parallel.h"
#include "metal.h"

enum plamo_memory_mode { PLAMO_MEMORY_AUTO, PLAMO_MEMORY_RESIDENT, PLAMO_MEMORY_STREAM };
enum plamo_backend { PLAMO_BACKEND_AUTO, PLAMO_BACKEND_CPU, PLAMO_BACKEND_METAL, PLAMO_BACKEND_METAL_FP32 };
struct plamo_engine;
struct plamo_engine_info {
    const char *memory_mode;
    const char *backend;
    char memory_note[256];
    uint64_t weight_bytes, preloaded_weight_bytes, reclaimable_bytes, physical_bytes;
    uint64_t prefix_tokens[2], prefix_bytes;
    struct plamo_weight_cache_info cache;
    struct plamo_metal_info gpu;
    unsigned threads;
    double load_seconds, preload_seconds, prefix_seconds;
};
struct plamo_result {
    uint64_t prompt_tokens, completion_tokens, decode_steps, staged_bytes, read_calls;
    uint64_t cache_hits, cache_hit_bytes, cache_misses;
    uint64_t decode_staged_bytes, decode_read_calls, decode_command_buffers, decode_gpu_dispatches;
    uint64_t prefill_command_buffers, prefill_gpu_dispatches;
    uint64_t cached_prompt_tokens;
    uint64_t prefill_logits_hash, prefill_state_hash, logits_hash, state_hash;
    double tokenize_seconds, prefill_seconds, decode_seconds, first_token_seconds, total_seconds;
    double decode_gpu_seconds, prefill_gpu_seconds;
    const char *finish_reason;
};
struct plamo_callbacks {
    void (*prefill)(uint64_t prompt_tokens, void *context);
    int (*token)(uint32_t id, const char *bytes, size_t count, void *context);
    void *context;
};

/* One worker owns one engine for its process lifetime. Requests are serialized. */
struct plamo_engine *plamo_engine_open(enum plamo_memory_mode, int tokenizer_only, unsigned threads, enum plamo_backend,
                                     struct plamo_engine_info *, char *, size_t);
void plamo_engine_get_info(struct plamo_engine *, struct plamo_engine_info *);
void plamo_engine_maintain_memory(struct plamo_engine *);
/* API startup: fill the adaptive cache, prepare GPU resources and snapshot
   the two fixed translation prefixes before readiness. Bounded, idempotent;
   logical request state is reset afterwards. Single-shot CLI calls omit the
   prefix snapshots and prepare only their weights as part of the request. */
int plamo_engine_prepare(struct plamo_engine *, char *, size_t);
uint64_t plamo_engine_tokenize(struct plamo_engine *, const char *, unsigned direction,
                              const uint32_t **ids, char *, size_t);
int plamo_engine_translate(struct plamo_engine *, const char *, unsigned direction,
                          unsigned max_tokens, const struct plamo_callbacks *,
                          struct plamo_result *, char *, size_t);
#endif
