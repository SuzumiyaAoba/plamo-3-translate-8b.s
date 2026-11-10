#include "../runtime/disk_cache.h"
#include "../runtime/packing.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char path[] = "/tmp/plamo-disk-cache-XXXXXX", error[256];
    int fd = mkstemp(path); assert(fd >= 0);
    const size_t bytes = 2 * PLAMO_DISK_ALIGNMENT;
    unsigned char *data = calloc(1, bytes); assert(data);
    uint16_t original[512] = {0};
    struct plamo_mapping source = {original, sizeof original};
    struct plamo_disk_header *h = (void *)data;
    memcpy(h->magic, PLAMO_DISK_MAGIC, 8); h->version = 1; h->alignment = PLAMO_DISK_ALIGNMENT;
    h->entries = h->sources = 1; h->source_bytes[0] = sizeof original; h->file_bytes = bytes;
    strcpy(h->revision, "synthetic");
    struct plamo_disk_entry *e = (void *)(h + 1);
    *e = (struct plamo_disk_entry){0, 1, 0, sizeof original, PLAMO_DISK_ALIGNMENT, 528, 16, 32};
    struct plamo_bf16_block *block = (void *)(data + e->offset);
    *block = (struct plamo_bf16_block){16, 127};
    unsigned char *valid = malloc(bytes); assert(valid); memcpy(valid, data, bytes);
    for (unsigned test = 0; test < 13; ++test) {
        memcpy(data, valid, bytes);
        switch (test) {
        case 1: h->magic[0] = 0; break;
        case 2: h->revision[0] = 'X'; break;
        case 3: h->source_bytes[0]++; break;
        case 4: e->offset++; break;
        case 5: e->bytes--; break;
        case 6: e->source = 1; break;
        case 7: e->source_offset = UINT64_MAX; break;
        case 8: e->raw_bytes++; break;
        case 9: block->exponent = 9 << 8; break;
        case 10: block->offset = UINT32_MAX; break;
        case 11: h->entries = PLAMO_DISK_MAX_ENTRIES + 1; break;
        case 12: h->file_bytes--; break;
        }
        assert(pwrite(fd, data, bytes, 0) == (ssize_t)bytes);
        struct plamo_disk_cache *c = plamo_disk_cache_open(path, "synthetic", &source, 1, error, sizeof error);
        if (test) { assert(!c); continue; }
        assert(c);
        uint64_t storage = 0;
        const void *payload = plamo_disk_cache_get(c, original, sizeof original, 1, &storage);
        assert(payload && storage == PLAMO_DISK_ALIGNMENT && !memcmp(payload, data + PLAMO_DISK_ALIGNMENT, 528));
        unsigned char loaded[PLAMO_DISK_ALIGNMENT]; uint64_t io[4] = {0};
        assert(plamo_disk_cache_read(c, payload, loaded, storage, io));
        assert(!memcmp(loaded, data + PLAMO_DISK_ALIGNMENT, storage) && io[0] == storage && io[1] == 1 && !io[3]);
        assert(!plamo_disk_cache_read(c, payload, loaded, bytes, io));
        assert(!plamo_disk_cache_get(c, original, sizeof original, 0, &storage));
        assert(!plamo_disk_cache_get(c, original + 1, sizeof original - 2, 1, &storage));
        plamo_disk_cache_close(c);
    }
    // The mapping owns its file lifetime; an atomic replacement/unlink is safe.
    assert(pwrite(fd, valid, bytes, 0) == (ssize_t)bytes);
    struct plamo_disk_cache *c = plamo_disk_cache_open(path, "synthetic", &source, 1, error, sizeof error); assert(c);
    assert(!unlink(path)); close(fd);
    uint64_t storage;
    const void *payload = plamo_disk_cache_get(c, original, sizeof original, 1, &storage); assert(payload);
    unsigned char loaded[PLAMO_DISK_ALIGNMENT];
    assert(plamo_disk_cache_read(c, payload, loaded, storage, NULL));
    assert(!memcmp(loaded, valid + PLAMO_DISK_ALIGNMENT, storage));
    plamo_disk_cache_close(c); plamo_disk_cache_close(NULL);
    free(data); free(valid);
    puts("disk cache: PASS (mapping lifetime, lookup, revision, ranges, packed headers)");
}
