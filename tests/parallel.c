#include "../runtime/parallel.h"
#include "../runtime/model.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void plamo_bf16_matvec(const uint16_t *, const float *, float *, uint64_t, uint64_t);
extern void plamo_bf16_matmul(const struct plamo_matmul *);
extern uint64_t plamo_matmul_tile_bytes;
extern void *plamo_compute_context;
static pthread_t owner;
static unsigned acquisitions;

static const void *acquire(const void *source, uint64_t bytes, void *context) {
    (void)bytes; (void)context;
    /* A provider may rebalance/unmap cached pages: only the caller may use it. */
    if (!pthread_equal(pthread_self(), owner)) abort();
    ++acquisitions;
    return source;
}

#define check(ok) do { if (!(ok)) { fprintf(stderr, "parallel check failed at line %d: %s\n", __LINE__, #ok); abort(); } } while (0)

int main(void) {
    const size_t shapes[][3] = {
        {0, 0, 0}, {0, 128, 17}, {128, 0, 17}, {1, 1, 1}, {3, 7, 9},
        {8, 12, 8}, {97, 33, 3}, {383, 1024, 1}, {385, 1027, 9},
        {769, 1024, 17}, {4097, 256, 33}
    };
    owner = pthread_self();
    check(!plamo_parallel_open(1));
    for (unsigned threads = 2; threads <= 4; threads += 2) {
        struct plamo_parallel *pool = plamo_parallel_open(threads);
        check(plamo_parallel_threads(pool) == threads);
        check(!plamo_parallel_open(2));
        for (size_t k = 0; k < sizeof shapes / sizeof *shapes; ++k) {
            size_t r = shapes[k][0], c = shapes[k][1], t = shapes[k][2], xs = c + 7, ys = r + 9;
            uint16_t *w = malloc((r * c + 1) * 2);
            float *x = malloc((t * xs + 1) * 4), *a = malloc((t * ys + 1) * 4), *b = malloc((t * ys + 1) * 4);
            check(w && x && a && b);
            uint32_t seed = 19;
            for (size_t i = 0; i < r * c; ++i) {
                seed = seed * 1664525u + 1013904223u;
                w[i] = (seed & 0x8000) | 0x3800 | (seed & 0x7ff);
            }
            for (size_t i = 0; i < t * xs; ++i) x[i] = ((int)(i % 83) - 41) / 13.f;
            for (size_t i = 0; i < t * ys + 1; ++i) b[i] = -123456.f;
            for (size_t i = 0; i < t; ++i) plamo_bf16_matvec_serial(w, x + i * xs, b + i * ys, r, c);
            struct plamo_matmul m = {w, x, a, r, c, t, xs, ys};
            const uint64_t windows[] = {0, 256 * 1024};
            plamo_weight_provider = acquire;
            for (size_t j = 0; j < sizeof windows / sizeof *windows; ++j) {
                plamo_weight_chunk_bytes = windows[j];
                /* Force cache tiling even with modest test matrices. */
                plamo_matmul_tile_bytes = 64 * 1024;
                for (unsigned repeat = 0; repeat < 3; ++repeat) {
                    for (size_t i = 0; i < t * ys + 1; ++i) a[i] = -123456.f;
                    plamo_bf16_matmul(&m);
                    check(!memcmp(a, b, (t * ys + 1) * 4));
                    for (size_t i = 0; i < t; ++i) plamo_bf16_matvec(w, x + i * xs, a + i * ys, r, c);
                    check(!memcmp(a, b, (t * ys + 1) * 4));
                }
            }
            plamo_weight_provider = NULL;
            free(w); free(x); free(a); free(b);
        }
        plamo_parallel_close(pool);
        check(plamo_parallel_threads(NULL) == 1);
        check(!plamo_compute_context);
    }
    check(acquisitions > 0);
    puts("parallel assembly: PASS (2/4 threads, bitwise results, strides, tails, row windows, caller-only weight acquisition, restart)");
    return 0;
}
