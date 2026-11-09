#ifndef PLAMO_PROMPT_CACHE_H
#define PLAMO_PROMPT_CACHE_H
#include "model.h"

/* Immutable snapshots of small, fixed template prefixes. No user text is
   retained. A missing snapshot, token mismatch or empty remaining suffix means
   zero reused tokens, leaving the full prompt available to ordinary prefill. */
enum { PLAMO_PROMPT_PREFIX_LIMIT = 64 };
struct plamo_prompt_cache;
struct plamo_prompt_cache *plamo_prompt_cache_capture(const struct model *, const uint32_t *, uint64_t tokens);
uint64_t plamo_prompt_cache_restore(const struct plamo_prompt_cache *, struct model *, const uint32_t *, uint64_t tokens);
uint64_t plamo_prompt_cache_tokens(const struct plamo_prompt_cache *);
uint64_t plamo_prompt_cache_bytes(const struct plamo_prompt_cache *);
void plamo_prompt_cache_free(struct plamo_prompt_cache *);
#endif
