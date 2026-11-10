#ifndef PLAMO_COMPUTE_H
#define PLAMO_COMPUTE_H
#include "model.h"
#include "parallel.h"

/* Assembly dispatch ABI. A single backend owns these hooks while inference is
   idle; the scalar/serial assembly entries remain the fallback when unset. */
extern void *plamo_compute_context;
extern void (*plamo_matvec_kernel)(const uint16_t *, const float *, float *, uint64_t, uint64_t, void *);
extern void (*plamo_matmul_kernel)(const struct plamo_matmul *, void *);
extern int (*plamo_mlp_kernel)(const struct mlp *, float *, uint64_t, void *);
extern int (*plamo_forward_kernel)(struct model *, uint64_t, void *);
extern int (*plamo_prefill_kernel)(struct model *, const uint32_t *, uint64_t, float *, void *);
#endif
