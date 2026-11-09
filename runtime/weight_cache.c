#include "weight_cache.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MIB (UINT64_C(1024) * 1024)
#define STAGING_LIMIT (64 * MIB)
#define MIN_READ MIB
#define MAX_ENTRIES 4096

struct entry { uintptr_t source; uint64_t offset, bytes, end; };
struct plamo_weight_cache {
    const struct plamo_mapping *maps;
    const int64_t *fds;
    size_t count, entries;
    uint64_t reserve, page, maximum, capacity, staging, staging_used, used, cached;
    uint64_t hits, hit_bytes, misses, resizes;
    uint64_t (*available)(void);
    double checked_at;
    void *arena;
    struct entry table[MAX_ENTRIES];
};

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t round_up(uint64_t n, uint64_t page) { return (n + page - 1) / page * page; }

/* Include only bytes already populated, never the unused virtual capacity.
   Otherwise each sample would count unused space as newly available memory. */
static uint64_t budget(const struct plamo_weight_cache *c, uint64_t available) {
    uint64_t owned = c->used + c->staging_used;
    uint64_t total = available > UINT64_MAX - owned ? UINT64_MAX : available + owned;
    uint64_t bytes = total > c->reserve ? total - c->reserve : 0;
    if (bytes > c->maximum) bytes = c->maximum;
    bytes = bytes / c->page * c->page;
    /* Keep one small read window even when the safety reserve is exhausted. */
    return bytes < c->staging ? c->staging : bytes;
}

static int resize(struct plamo_weight_cache *c, uint64_t capacity) {
    if (capacity == c->capacity) return 1;
    size_t keep = c->entries;
    while (keep && c->staging + c->table[keep - 1].end > capacity) --keep;
    uint64_t used = keep ? c->table[keep - 1].end : 0;
    uint64_t end = c->staging + used;
    if (capacity < c->capacity) {
        /* Replacing only the discarded tail releases physical pages immediately
           and preserves the reserved addresses for a later growth. */
        void *tail = (char *)c->arena + end;
        size_t bytes = (size_t)(c->capacity - end);
        if (mmap(tail, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == MAP_FAILED)
            return 0;
        for (size_t i = keep; i < c->entries; ++i) c->cached -= c->table[i].bytes;
        c->entries = keep;
        c->used = used;
    }
    uint64_t start = capacity < c->capacity ? end : c->capacity;
    if (capacity > start && mprotect((char *)c->arena + start, (size_t)(capacity - start), PROT_READ | PROT_WRITE)) {
        /* The retained cache and staging are still valid after a failed growth. */
        if (capacity < c->capacity) c->capacity = end;
        return 0;
    }
    c->capacity = capacity;
    ++c->resizes;
    return 1;
}

void plamo_weight_cache_rebalance(struct plamo_weight_cache *c) {
    uint64_t target = budget(c, c->available());
    c->checked_at = now();
    uint64_t difference = target > c->capacity ? target - c->capacity : c->capacity - target;
    /* Hysteresis avoids remapping for small changes in OS accounting. Shrinking
       below populated data and reaching the minimum window are always honored. */
    if (difference >= c->staging || target < c->staging + c->used || target == c->staging)
        (void)resize(c, target);
}

struct plamo_weight_cache *plamo_weight_cache_open(const struct plamo_mapping *maps, const int64_t *fds,
                                                  size_t count, uint64_t reserve, uint64_t (*available)(void),
                                                  char *error, size_t cap) {
    uint64_t total = 0;
    long page = sysconf(_SC_PAGESIZE);
    if (!maps || !fds || !count || count > PLAMO_MAX_WEIGHT_FILES || !available || page <= 0) goto invalid;
    for (size_t i = 0; i < count; ++i) {
        if (!maps[i].base || !maps[i].bytes || fds[i] < 0 || fds[i] > INT_MAX ||
            maps[i].bytes > SIZE_MAX - STAGING_LIMIT - (uint64_t)page - total) goto invalid;
        total += maps[i].bytes;
    }
    struct plamo_weight_cache *c = calloc(1, sizeof *c);
    if (!c) { snprintf(error, cap, "cannot allocate weight cache metadata"); return NULL; }
    c->maps = maps; c->fds = fds; c->count = count; c->reserve = reserve;
    c->page = (uint64_t)page; c->available = available;
    uint64_t free_bytes = available();
    uint64_t spare = free_bytes > reserve ? free_bytes - reserve : 0;
    /* Leave most of a small budget available for retained weights as well. */
    c->staging = spare / 8;
    if (c->staging > STAGING_LIMIT) c->staging = STAGING_LIMIT;
    c->staging = c->staging / c->page * c->page;
    if (c->staging < MIN_READ) c->staging = round_up(MIN_READ, c->page);
    c->maximum = round_up(total, c->page) + c->staging;
    uint64_t ceiling = spare / c->page * c->page;
    if (ceiling < c->staging) ceiling = c->staging;
    /* A fixed startup ceiling prevents OS compression/reclamation caused by our
       own allocations from repeatedly increasing the apparent spare capacity. */
    if (c->maximum > ceiling) c->maximum = ceiling;
    c->arena = mmap(NULL, (size_t)c->maximum, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (c->arena == MAP_FAILED) {
        snprintf(error, cap, "cannot reserve weight cache address space: %s", strerror(errno));
        free(c); return NULL;
    }
    if (!resize(c, budget(c, free_bytes))) {
        snprintf(error, cap, "cannot allocate weight cache window: %s", strerror(errno));
        munmap(c->arena, (size_t)c->maximum); free(c); return NULL;
    }
    c->checked_at = now();
    for (size_t i = 0; i < count; ++i) (void)fcntl((int)fds[i], F_NOCACHE, 1);
    return c;
invalid:
    snprintf(error, cap, "invalid weight cache source mapping or file descriptor");
    return NULL;
}

void plamo_weight_cache_close(struct plamo_weight_cache *c) {
    if (!c) return;
    for (size_t i = 0; i < c->count; ++i) (void)fcntl((int)c->fds[i], F_NOCACHE, 0);
    munmap(c->arena, (size_t)c->maximum);
    free(c);
}

void plamo_weight_cache_get_info(const struct plamo_weight_cache *c, struct plamo_weight_cache_info *out) {
    *out = (struct plamo_weight_cache_info){c->capacity, c->maximum, c->cached, c->staging,
        c->hits, c->hit_bytes, c->misses, c->resizes, c->used};
}

static const void *acquire(struct plamo_weight_cache *c, const void *source, uint64_t bytes, int retain_only) {
    uintptr_t address = (uintptr_t)source, base = (uintptr_t)c->arena;
    if (!bytes) return source;
    /* Nested matmul tails already borrow this arena. Do not rebalance here. */
    if (address >= base && address - base < c->capacity && bytes <= c->capacity - (address - base)) return source;
    size_t file = 0;
    uint64_t offset = 0;
    for (; file < c->count; ++file) {
        base = (uintptr_t)c->maps[file].base;
        if (address < base || address - base >= c->maps[file].bytes) continue;
        offset = address - base;
        if (bytes > c->maps[file].bytes - offset) return source;
        break;
    }
    if (file == c->count) return source;
    if (now() - c->checked_at >= 0.25) plamo_weight_cache_rebalance(c);
    for (size_t i = 0; i < c->entries; ++i) {
        const struct entry *e = &c->table[i];
        if (address >= e->source && address - e->source < e->bytes && bytes <= e->bytes - (address - e->source)) {
            ++c->hits; c->hit_bytes += bytes;
            return (char *)c->arena + c->staging + e->offset + (address - e->source);
        }
    }
    if (bytes < MIN_READ || bytes > c->staging || c->fds[file] < 0 || c->fds[file] > INT_MAX) return source;
    uint64_t stored = round_up(bytes, c->page);
    /* Keep a stable subset across full sequential scans. Evicting on every miss
       would give zero reuse when the model is larger than the cache. */
    int cache = c->entries < MAX_ENTRIES && stored <= c->capacity - c->staging - c->used;
    if (retain_only && !cache) return source;
    ++c->misses;
    char *destination = (char *)c->arena + (cache ? c->staging + c->used : 0);
    uint64_t done = 0;
    while (done < bytes) {
        size_t n = (size_t)(bytes - done);
        if (n > 8 * MIB) n = 8 * MIB;
        ++plamo_io_stats[1];
        ssize_t read_bytes = pread((int)c->fds[file], destination + done, n, (off_t)(offset + done));
        if (read_bytes < 0 && errno == EINTR) continue;
        if (read_bytes <= 0) {
            ++plamo_io_stats[3];
            /* Failed/short reads never publish an entry with incomplete bytes. */
            if (cache) {
                (void)mmap(destination, (size_t)stored, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
            } else if (done && round_up(done, c->page) > c->staging_used) {
                c->staging_used = round_up(done, c->page);
            }
            return retain_only ? NULL : source;
        }
        done += (uint64_t)read_bytes;
        plamo_io_stats[0] += (uint64_t)read_bytes;
    }
    if (cache) {
        /* Cached ranges have their own pages and become read-only. */
        if (!mprotect(destination, (size_t)stored, PROT_READ)) {
            c->table[c->entries++] = (struct entry){address, c->used, bytes, c->used + stored};
            c->used += stored;
            c->cached += bytes;
        } else if (retain_only) return NULL;
    } else if (stored > c->staging_used) c->staging_used = stored;
    return destination;
}

const void *plamo_weight_cache_acquire(const void *source, uint64_t bytes, void *context) {
    return acquire(context, source, bytes, 0);
}

int plamo_weight_cache_preload(struct plamo_weight_cache *c, const void *source, uint64_t bytes) {
    return !bytes || acquire(c, source, bytes, 1) != NULL;
}
