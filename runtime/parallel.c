#include "parallel.h"
#include <pthread.h>
#include <stdlib.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#include <unistd.h>

#define MAX_THREADS 64
/* Rows are independent; aligning to 24 preserves the 3/4/8-row fast paths. */
#define ROW_GROUP 24
#define MIN_OPERATIONS (UINT64_C(256) * 1024)

struct worker { struct plamo_parallel *pool; pthread_t thread; unsigned index; };
struct plamo_parallel {
    pthread_mutex_t mutex;
    pthread_cond_t start, done;
    struct worker workers[MAX_THREADS - 1];
    struct plamo_matmul job;
    uint64_t generation;
    unsigned threads, pending;
    int stopping, matvec;
};

extern void (*plamo_matvec_kernel)(const uint16_t *, const float *, float *, uint64_t, uint64_t, void *);
extern void (*plamo_matmul_kernel)(const struct plamo_matmul *, void *);
extern void *plamo_compute_context;

static void compute(const struct plamo_matmul *m, int matvec) {
    if (matvec) plamo_bf16_matvec_serial(m->weights, m->input, m->output, m->rows, m->columns);
    else plamo_bf16_matmul_core(m);
}

static void slice(struct plamo_parallel *p, unsigned index) {
    struct plamo_matmul m = p->job;
    uint64_t groups = (m.rows + ROW_GROUP - 1) / ROW_GROUP;
    uint64_t first = groups * index / p->threads * ROW_GROUP;
    uint64_t end = groups * (index + 1) / p->threads * ROW_GROUP;
    if (end > m.rows) end = m.rows;
    if (first >= end) return;
    m.weights += first * m.columns;
    m.output += first;
    m.rows = end - first;
    compute(&m, p->matvec);
}

static void *worker_main(void *context) {
    struct worker *w = context;
    struct plamo_parallel *p = w->pool;
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    uint64_t seen = 0;
    pthread_mutex_lock(&p->mutex);
    for (;;) {
        while (!p->stopping && seen == p->generation) pthread_cond_wait(&p->start, &p->mutex);
        if (p->stopping) break;
        seen = p->generation;
        pthread_mutex_unlock(&p->mutex);
        slice(p, w->index);
        pthread_mutex_lock(&p->mutex);
        if (!--p->pending) pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->mutex);
    return NULL;
}

static void run(struct plamo_parallel *p, const struct plamo_matmul *m, int matvec) {
    /* Avoid waking workers for small matrices, head normalization and tails. */
    uint64_t operations = m->rows * m->columns;
    if (m->rows < p->threads * ROW_GROUP || !m->tokens || !m->columns ||
        operations < MIN_OPERATIONS / m->tokens) {
        compute(m, matvec);
        return;
    }
    pthread_mutex_lock(&p->mutex);
    p->job = *m;
    p->matvec = matvec;
    p->pending = p->threads - 1;
    ++p->generation;
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mutex);
    slice(p, 0);
    pthread_mutex_lock(&p->mutex);
    while (p->pending) pthread_cond_wait(&p->done, &p->mutex);
    pthread_mutex_unlock(&p->mutex);
}

static void matvec(const uint16_t *w, const float *x, float *y, uint64_t rows, uint64_t columns, void *context) {
    struct plamo_matmul m = {w, x, y, rows, columns, 1, columns, rows};
    run(context, &m, 1);
}

static void matmul(const struct plamo_matmul *m, void *context) { run(context, m, 0); }

static unsigned automatic_threads(void) {
    unsigned cores = 0;
    size_t bytes = sizeof cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &bytes, NULL, 0) || !cores) {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        cores = online > 0 && online < 4 ? (unsigned)online : 4;
    }
    return cores > MAX_THREADS ? MAX_THREADS : cores;
}

struct plamo_parallel *plamo_parallel_open(unsigned threads) {
    if (plamo_compute_context) return NULL;
    if (!threads) threads = automatic_threads();
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    if (threads <= 1) return NULL;
    struct plamo_parallel *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    if (pthread_mutex_init(&p->mutex, NULL)) { free(p); return NULL; }
    if (pthread_cond_init(&p->start, NULL)) { pthread_mutex_destroy(&p->mutex); free(p); return NULL; }
    if (pthread_cond_init(&p->done, NULL)) {
        pthread_cond_destroy(&p->start); pthread_mutex_destroy(&p->mutex); free(p); return NULL;
    }
    p->threads = 1;
    for (unsigned i = 1; i < threads; ++i) {
        struct worker *w = &p->workers[i - 1];
        w->pool = p; w->index = i;
        if (pthread_create(&w->thread, NULL, worker_main, w)) break;
        ++p->threads;
    }
    if (p->threads == 1) { plamo_parallel_close(p); return NULL; }
    plamo_compute_context = p;
    plamo_matvec_kernel = matvec;
    plamo_matmul_kernel = matmul;
    return p;
}

unsigned plamo_parallel_threads(const struct plamo_parallel *p) { return p ? p->threads : 1; }

void plamo_parallel_close(struct plamo_parallel *p) {
    if (!p) return;
    if (plamo_compute_context == p) {
        plamo_matvec_kernel = NULL;
        plamo_matmul_kernel = NULL;
        plamo_compute_context = NULL;
    }
    pthread_mutex_lock(&p->mutex);
    p->stopping = 1;
    pthread_cond_broadcast(&p->start);
    pthread_mutex_unlock(&p->mutex);
    for (unsigned i = 1; i < p->threads; ++i) pthread_join(p->workers[i - 1].thread, NULL);
    pthread_cond_destroy(&p->done);
    pthread_cond_destroy(&p->start);
    pthread_mutex_destroy(&p->mutex);
    free(p);
}
