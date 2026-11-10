#ifndef PLAMO_DISK_CACHE_H
#define PLAMO_DISK_CACHE_H
#include "model.h"

/* Little-endian, page-aligned copies of the existing raw/packed GPU buffers.
   Offsets inside a packed buffer remain relative to that buffer, not the file. */
enum { PLAMO_DISK_ALIGNMENT = 16384, PLAMO_DISK_MAX_ENTRIES = 4096,
       PLAMO_DISK_RAW_LIMIT = 64 * 1024 * 1024 };
#define PLAMO_DISK_MAGIC "PLAMOWC1"
struct plamo_disk_header {
    char magic[8];
    uint32_t version, alignment, entries, sources;
    uint64_t file_bytes, source_bytes[PLAMO_MAX_WEIGHT_FILES];
    char revision[64];
    unsigned char reserved[32];
};
struct plamo_disk_entry {
    uint32_t source, packed;
    uint64_t source_offset, raw_bytes, offset, bytes;
    uint32_t rows, columns;
};
_Static_assert(sizeof(struct plamo_disk_header) == 640, "disk cache header ABI");
_Static_assert(sizeof(struct plamo_disk_entry) == 48, "disk cache directory ABI");

struct plamo_disk_cache;
/* A requested cache must match the compiled revision and original file sizes.
   Validate all packed-header offsets before exposing any data to the GPU.
   The mapping stays immutable/alive until close, including after rename/unlink. */
struct plamo_disk_cache *plamo_disk_cache_open(const char *path, const char *revision,
    const struct plamo_mapping *, size_t count, char *error, size_t capacity);
/* Exact chunk lookup. Cache misses retain the ordinary reader and packer.
   storage_bytes includes page padding; no allocation or file read is issued. */
const void *plamo_disk_cache_get(const struct plamo_disk_cache *, const void *source,
    uint64_t bytes, unsigned packed, uint64_t *storage_bytes);
/* Bulk-read a returned range directly into a Metal shared allocation. Keep
   file paging out of the GPU's first command. io uses the weight-I/O counters
   {bytes, pread calls, advice calls, failures}; it may be NULL. */
int plamo_disk_cache_read(const struct plamo_disk_cache *, const void *range,
    void *destination, uint64_t storage_bytes, uint64_t io[4]);
void plamo_disk_cache_close(struct plamo_disk_cache *);
#endif
