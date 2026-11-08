#ifndef PLAMO_RESIDENT_H
#define PLAMO_RESIDENT_H
#include "model.h"

#define PLAMO_MEMORY_RESERVE (UINT64_C(2) * 1024 * 1024 * 1024)
/* Reclaimable estimate: free + file-backed pages, never double-count speculative pages. */
uint64_t plamo_reclaimable_memory(void);
uint64_t plamo_physical_memory(void);
/* Copies exact file bytes into private, read-only mappings. No weight conversion.
   On failure all output mappings are released; inputs and their FDs are retained.
   reserve=0 disables the capacity guard for small, model-free unit tests. */
int plamo_resident_copy(const struct plamo_mapping *, const int64_t *, size_t,
                        uint64_t reserve, struct plamo_mapping *, char *, size_t);
void plamo_resident_release(struct plamo_mapping *, size_t);
/* Validate every tensor before changing any pointer. The byte offset is retained. */
int plamo_rebase_weights(struct plamo_weight_spec *, const struct plamo_mapping *,
                         const struct plamo_mapping *, size_t);
#endif
