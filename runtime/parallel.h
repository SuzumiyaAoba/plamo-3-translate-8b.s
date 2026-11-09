#ifndef PLAMO_PARALLEL_H
#define PLAMO_PARALLEL_H
#include <stdint.h>

struct plamo_matmul {
    const uint16_t *weights;
    const float *input;
    float *output;
    uint64_t rows, columns, tokens, input_stride, output_stride;
};
_Static_assert(sizeof(struct plamo_matmul) == 64, "assembly matmul ABI");

struct plamo_parallel;
/* One scheduler per process, installed while inference is idle. Zero selects
   performance cores (at most four if that information is unavailable). A null
   result is a working single-threaded fallback, including allocation failure. */
struct plamo_parallel *plamo_parallel_open(unsigned threads);
unsigned plamo_parallel_threads(const struct plamo_parallel *);
void plamo_parallel_close(struct plamo_parallel *);

/* Compute-only assembly entries: weights must stay valid through the call. */
void plamo_bf16_matvec_serial(const uint16_t *, const float *, float *, uint64_t, uint64_t);
void plamo_bf16_matmul_core(const struct plamo_matmul *);
#endif
