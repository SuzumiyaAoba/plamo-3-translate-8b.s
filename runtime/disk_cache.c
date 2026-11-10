#include "disk_cache.h"
#include "packing.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct plamo_disk_cache {
    int fd;
    void *mapping;
    size_t bytes, count;
    const struct plamo_mapping *sources;
    const struct plamo_disk_header *header;
    const struct plamo_disk_entry *entries;
};
static uint64_t page_round(uint64_t n) { return (n + PLAMO_DISK_ALIGNMENT - 1) / PLAMO_DISK_ALIGNMENT * PLAMO_DISK_ALIGNMENT; }
static int valid_packing(const struct plamo_disk_entry *e, const void *data) {
    if (!e->packed) return e->bytes == e->raw_bytes;
    uint64_t blocks = (e->raw_bytes / 2 + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    uint64_t offset = (blocks * 8 + 15) & ~UINT64_C(15);
    if (offset > e->bytes) return 0;
    const struct plamo_bf16_block *headers = data;
    for (uint64_t b = 0; b < blocks; ++b) {
        uint32_t bits = headers[b].exponent >> 8;
        if (bits > 8 || headers[b].offset != offset) return 0;
        offset += PLAMO_BF16_BLOCK + bits * (PLAMO_BF16_BLOCK / 8);
        if (offset > e->bytes) return 0;
    }
    return offset == e->bytes;
}
struct plamo_disk_cache *plamo_disk_cache_open(const char *path, const char *revision,
    const struct plamo_mapping *sources, size_t count, char *error, size_t cap) {
    struct plamo_disk_cache *c = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(error, cap, "cannot open Metal weight cache: %s", strerror(errno)); return NULL; }
    struct stat st;
    if (fstat(fd, &st) || st.st_size < (off_t)sizeof(struct plamo_disk_header) || (uint64_t)st.st_size > SIZE_MAX) {
        close(fd); snprintf(error, cap, "invalid Metal weight cache size"); return NULL;
    }
    c = calloc(1, sizeof *c);
    if (!c) { close(fd); snprintf(error, cap, "cannot allocate Metal weight cache descriptor"); return NULL; }
    c->fd = fd; c->bytes = (size_t)st.st_size;
    c->mapping = mmap(NULL, c->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    (void)fcntl(fd, F_NOCACHE, 1);
    if (c->mapping == MAP_FAILED) { c->mapping = NULL; goto invalid; }
    const struct plamo_disk_header *h = c->header = c->mapping;
    long page = sysconf(_SC_PAGESIZE);
    if (memcmp(h->magic, PLAMO_DISK_MAGIC, 8) || h->version != 1 || h->alignment != PLAMO_DISK_ALIGNMENT ||
        page <= 0 || PLAMO_DISK_ALIGNMENT % page || h->file_bytes != c->bytes ||
        !h->entries || h->entries > PLAMO_DISK_MAX_ENTRIES || !sources || !count || count > PLAMO_MAX_WEIGHT_FILES ||
        h->sources != count || !revision || h->revision[63] || strcmp(h->revision, revision)) goto invalid;
    for (size_t i = 0; i < count; ++i)
        if (!sources[i].base || sources[i].bytes != h->source_bytes[i]) goto invalid;
    uint64_t offset = page_round(sizeof *h + (uint64_t)h->entries * sizeof(struct plamo_disk_entry));
    if (offset > c->bytes) goto invalid;
    c->entries = (const struct plamo_disk_entry *)(h + 1);
    for (uint32_t i = 0; i < h->entries; ++i) {
        const struct plamo_disk_entry *e = &c->entries[i];
        if (e->source >= count || e->packed > 1 || !e->rows || !e->columns ||
            e->rows > PLAMO_DISK_RAW_LIMIT / 2 / e->columns ||
            e->raw_bytes != (uint64_t)e->rows * e->columns * 2 ||
            e->source_offset > sources[e->source].bytes || e->raw_bytes > sources[e->source].bytes - e->source_offset ||
            !e->bytes || e->bytes > 2 * PLAMO_DISK_RAW_LIMIT || e->offset != offset ||
            page_round(e->bytes) > c->bytes - offset) goto invalid;
        for (uint32_t j = 0; j < i; ++j) {
            const struct plamo_disk_entry *p = &c->entries[j];
            if (p->source == e->source && e->source_offset < p->source_offset + p->raw_bytes &&
                p->source_offset < e->source_offset + e->raw_bytes) goto invalid;
        }
        if (!valid_packing(e, (const char *)c->mapping + offset)) goto invalid;
        offset += page_round(e->bytes);
    }
    if (offset != c->bytes) goto invalid;
    c->sources = sources; c->count = count;
    return c;
invalid:
    plamo_disk_cache_close(c);
    snprintf(error, cap, "invalid or incompatible Metal weight cache; regenerate it with prepare_weights.py");
    return NULL;
}
const void *plamo_disk_cache_get(const struct plamo_disk_cache *c, const void *source,
    uint64_t bytes, unsigned packed, uint64_t *storage_bytes) {
    if (!c || !bytes) return NULL;
    uintptr_t address = (uintptr_t)source;
    for (size_t s = 0; s < c->count; ++s) {
        uintptr_t base = (uintptr_t)c->sources[s].base;
        if (address < base || address - base >= c->sources[s].bytes) continue;
        uint64_t offset = address - base;
        for (uint32_t i = 0; i < c->header->entries; ++i) {
            const struct plamo_disk_entry *e = &c->entries[i];
            if (e->source == s && e->source_offset == offset && e->raw_bytes == bytes && e->packed == packed) {
                *storage_bytes = page_round(e->bytes);
                return (const char *)c->mapping + e->offset;
            }
        }
        break;
    }
    return NULL;
}
int plamo_disk_cache_read(const struct plamo_disk_cache *c, const void *range,
    void *destination, uint64_t bytes, uint64_t io[4]) {
    if (!c || !range || !destination || !bytes) return 0;
    uintptr_t base = (uintptr_t)c->mapping, address = (uintptr_t)range;
    if (address < base || address - base >= c->bytes || bytes > c->bytes - (address - base)) return 0;
    uint64_t offset = address - base, done = 0;
    while (done < bytes) {
        size_t count = (size_t)(bytes - done); if (count > PLAMO_DISK_RAW_LIMIT) count = PLAMO_DISK_RAW_LIMIT;
        if (io) ++io[1];
        ssize_t n = pread(c->fd, (char *)destination + done, count, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (io) ++io[3]; return 0; }
        done += (uint64_t)n; if (io) io[0] += (uint64_t)n;
    }
    return 1;
}
void plamo_disk_cache_close(struct plamo_disk_cache *c) {
    if (!c) return;
    if (c->mapping) munmap(c->mapping, c->bytes);
    close(c->fd);
    free(c);
}
