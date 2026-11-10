#ifndef PLAMO_DRAFT_H
#define PLAMO_DRAFT_H
#include "model.h"

enum { PLAMO_DRAFT_MAX = 16, PLAMO_DRAFT_MAGIC = 0x504c4431 };
struct plamo_draft;
/* Optional, process-isolated proposer. Missing files, protocol failures and
   child termination simply disable speculation; the target keeps generating. */
struct plamo_draft *plamo_draft_open(void);
void plamo_draft_close(struct plamo_draft *);
int plamo_draft_reset(struct plamo_draft *, const char *, unsigned direction);
unsigned plamo_draft_propose(struct plamo_draft *, uint32_t first, unsigned limit, uint32_t *);
int plamo_draft_accept(struct plamo_draft *, unsigned consumed);

/* Save only the rows that a bounded speculative suffix could evict. Restoring
   a prefix leaves weights/workspace alone and invalidates derived rotations. */
struct plamo_draft_checkpoint;
struct plamo_draft_checkpoint *plamo_draft_checkpoint(struct model *, unsigned limit);
int plamo_draft_restore(struct plamo_draft_checkpoint *, unsigned processed, unsigned accepted);
void plamo_draft_checkpoint_free(struct plamo_draft_checkpoint *);
#endif
