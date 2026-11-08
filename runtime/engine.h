#ifndef PLAMO_ENGINE_H
#define PLAMO_ENGINE_H
#include "resident.h"

enum plamo_memory_mode { PLAMO_MEMORY_AUTO, PLAMO_MEMORY_RESIDENT, PLAMO_MEMORY_STREAM };
struct plamo_engine;
struct plamo_engine_info {
    const char *memory_mode;
    char memory_note[256];
    uint64_t weight_bytes, preloaded_weight_bytes, reclaimable_bytes, physical_bytes;
    double load_seconds;
};
struct plamo_result {
    uint64_t prompt_tokens, completion_tokens, decode_steps, staged_bytes, read_calls;
    uint64_t prefill_logits_hash, prefill_state_hash, logits_hash, state_hash;
    double tokenize_seconds, prefill_seconds, decode_seconds, first_token_seconds, total_seconds;
    const char *finish_reason;
};
struct plamo_callbacks {
    void (*prefill)(uint64_t prompt_tokens, void *context);
    int (*token)(uint32_t id, const char *bytes, size_t count, void *context);
    void *context;
};

/* One worker owns one engine for its process lifetime. Requests are serialized. */
struct plamo_engine *plamo_engine_open(enum plamo_memory_mode, int tokenizer_only,
                                     struct plamo_engine_info *, char *, size_t);
uint64_t plamo_engine_tokenize(struct plamo_engine *, const char *, unsigned direction,
                              const uint32_t **ids, char *, size_t);
int plamo_engine_translate(struct plamo_engine *, const char *, unsigned direction,
                          unsigned max_tokens, const struct plamo_callbacks *,
                          struct plamo_result *, char *, size_t);
#endif
