#ifndef PLAMO_WEIGHT_CACHE_H
#define PLAMO_WEIGHT_CACHE_H
#include "model.h"

struct plamo_weight_cache;
struct plamo_weight_cache_info {
    uint64_t budget_bytes, max_budget_bytes, cached_bytes, staging_bytes;
    uint64_t hits, hit_bytes, misses, resizes;
    uint64_t storage_bytes;
};

/* A single, serially used arena with a ceiling fixed from startup availability.
   Only the budgeted prefix is accessible; untouched space costs no weight
   storage. OS compression must never inflate the ceiling. available() is sampled at
   most four times a second while acquiring weights, and explicitly while idle.
   The source mappings/FDs must remain valid until close. */
struct plamo_weight_cache *plamo_weight_cache_open(const struct plamo_mapping *, const int64_t *,
                                                  size_t, uint64_t reserve, uint64_t (*available)(void),
                                                  char *, size_t);
void plamo_weight_cache_close(struct plamo_weight_cache *);
void plamo_weight_cache_rebalance(struct plamo_weight_cache *);
void plamo_weight_cache_get_info(const struct plamo_weight_cache *, struct plamo_weight_cache_info *);
/* Retain a mapped range only if it fits the current budget and read window.
   Unretainable/non-model ranges are skipped without I/O. Zero means a read or
   protection failure; complete retained ranges remain usable on failure. */
int plamo_weight_cache_preload(struct plamo_weight_cache *, const void *, uint64_t bytes);
/* Compatible with the assembly weight provider. A returned pointer is borrowed
   until the next acquire of an original file range or an explicit rebalance.
   Reacquiring a tail within the arena never moves or overwrites its parent. */
const void *plamo_weight_cache_acquire(const void *, uint64_t, void *);
#endif
