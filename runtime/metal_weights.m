#include "metal_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "packing.h"
#include <errno.h>
#include <limits.h>

void plamo_metal_release_mappings(struct plamo_metal *m) {
    for (size_t i = 0; i < m->mapped_count; ++i) {
        plamo_metal_residency_remove(m, m->mapped[i].buffer);
        [m->mapped[i].buffer release];
    }
    plamo_metal_residency_commit(m);
    m->mapped_count = 0;
    m->info.disk_cache_mapping_bytes = 0;
}

id<MTLBuffer> plamo_metal_mapped_weights(struct plamo_metal *m, const void *data, uint64_t bytes) {
    for (size_t i = 0; i < m->mapped_count; ++i)
        if (m->mapped[i].data == data && m->mapped[i].bytes == bytes) return m->mapped[i].buffer;
    id<MTLBuffer> buffer = [m->device newBufferWithBytesNoCopy:(void *)data length:(NSUInteger)bytes
                                                   options:MTLResourceStorageModeShared deallocator:nil];
    if (!buffer) return nil;
    // Keep file-backed GPU resources resident just like private weights.
    // Maintenance drops them before evicting private copies under pressure.
    if (m->cache_mappings && m->mapped_count < MAX_ENTRIES) {
        m->mapped[m->mapped_count++] = (struct mapped_weight){data, bytes, buffer};
        m->info.disk_cache_mapping_bytes += bytes;
        plamo_metal_residency_add(m, buffer);
        return buffer;
    }
    return [buffer autorelease];
}

void plamo_metal_maintain(struct plamo_metal *m) {
    m->checked_at = plamo_metal_now();
    if (m->config.memory != PLAMO_METAL_AUTO) return;
    uint64_t available = m->config.available(), owned = m->used + m->staging_bytes;
    m->cache_mappings = available >= m->config.reserve;
    if (!m->cache_mappings) plamo_metal_release_mappings(m);
    uint64_t total = available > UINT64_MAX - owned ? UINT64_MAX : available + owned;
    uint64_t target = total > m->config.reserve ? total - m->config.reserve : 0;
    if (target > m->maximum) target = m->maximum;
    if (target < m->staging_bytes) target = m->staging_bytes;
    if (target == m->capacity) return;
    // All command buffers are complete before memory maintenance is called.
    size_t previous_entries = m->entries;
    while (m->entries && m->used + m->staging_bytes > target) {
        struct entry *e = &m->table[--m->entries];
        m->used -= e->allocated;
        m->info.cache.cached_bytes -= e->bytes;
        if (e->packed) m->info.packed_weight_bytes -= e->bytes;
        plamo_metal_residency_remove(m, e->buffer);
        [e->buffer release];
    }
    // Release evicted pages immediately, including while no GPU work is queued.
    if (m->entries != previous_entries) plamo_metal_residency_commit(m);
    m->capacity = target;
    ++m->info.cache.resizes;
}

static int file_range(struct plamo_metal *m, uintptr_t address, uint64_t bytes, size_t *file, uint64_t *offset) {
    for (size_t i = 0; i < m->config.count; ++i) {
        uintptr_t base = (uintptr_t)m->config.maps[i].base;
        if (address < base || address - base >= m->config.maps[i].bytes) continue;
        *offset = address - base;
        if (bytes > m->config.maps[i].bytes - *offset) return 0;
        *file = i;
        return 1;
    }
    return 0;
}

static id<MTLBuffer> pack_cache(struct plamo_metal *, uint64_t, uint64_t, unsigned *, uint64_t *);
static id<MTLBuffer> quantize_cache(struct plamo_metal *, uint64_t, uint64_t);

static void retain_weights(struct plamo_metal *m, uintptr_t address, uint64_t bytes, uint64_t allocation,
                           id<MTLBuffer> buffer, unsigned prepared, unsigned packed) {
    // Growing the private cache can supersede previously borrowed ranges.
    if (m->mapped_count) plamo_metal_release_mappings(m);
    m->table[m->entries++] = (struct entry){address, bytes, allocation, buffer, prepared, packed};
    m->used += allocation;
    m->info.cache.cached_bytes += bytes;
    if (packed) m->info.packed_weight_bytes += bytes;
    // Newly encoded packed buffers were registered by pack_cache already.
    if (!prepared) plamo_metal_residency_add(m, buffer);
}

id<MTLBuffer> plamo_metal_weights(struct plamo_metal *m, const void *source, uint64_t bytes, NSUInteger *buffer_offset,
                              int retain_only, unsigned *packed, int decode) {
    uintptr_t address = (uintptr_t)source;
    *buffer_offset = 0;
    *packed = 0;
    if (m->config.memory == PLAMO_METAL_AUTO && plamo_metal_now() - m->checked_at >= 0.25) plamo_metal_maintain(m);
    for (size_t i = 0; i < m->entries; ++i) {
        const struct entry *e = &m->table[i];
        if (address >= e->source && address - e->source < e->bytes && bytes <= e->bytes - (address - e->source) &&
            (!e->packed || !((address - e->source) & 1))) {
            ++m->info.cache.hits;
            m->info.cache.hit_bytes += bytes;
            *buffer_offset = (NSUInteger)(address - e->source);
            *packed = e->packed;
            return e->buffer;
        }
    }
    if (bytes > m->staging_bytes) return nil;
    size_t file = 0;
    uint64_t offset = 0;
    int mapped = file_range(m, address, bytes, &file, &offset);
    uint64_t allocation = plamo_metal_aligned(bytes);
    uint64_t remaining = m->capacity - m->staging_bytes - m->used;
    if (mapped && m->config.memory != PLAMO_METAL_STREAM && !m->draft_quantized) {
        uint64_t stored = 0;
        const void *data = plamo_disk_cache_get(m->disk_cache, source, bytes, m->use_packing, &stored);
        int keep = stored <= remaining && m->entries < MAX_ENTRIES;
        if (data && retain_only && !keep) return nil;
        // Borrow immutable file pages only for one-token projections. Prompt
        // matrix tiles keep raw staging; retained weights keep private copies.
        int stage = decode && m->use_packing && !retain_only && stored <= m->staging_bytes;
        if (data && stage && !keep) {
            id<MTLBuffer> view = plamo_metal_mapped_weights(m, data, stored);
            if (view) {
                *packed = m->use_packing;
                ++m->info.cache.misses;
                ++m->info.disk_cache_hits;
                m->info.disk_cache_weight_bytes += bytes;
                m->info.disk_cache_storage_bytes += stored;
                m->info.disk_cache_mapped_bytes += stored;
                // Uncached wrappers live through this projection's GPU wait;
                // cached wrappers are released on pressure or backend close.
                return view;
            }
        }
        if (data && (keep || stage)) {
            id<MTLBuffer> cached = keep ? [m->device newBufferWithLength:(NSUInteger)stored options:MTLResourceStorageModeShared] : m->staging;
            int loaded = cached && plamo_disk_cache_read(m->disk_cache, data, cached.contents, stored, plamo_io_stats);
            if (loaded) {
                *packed = m->use_packing;
                if (keep) retain_weights(m, address, bytes, stored, cached, 0, *packed);
                ++m->info.cache.misses;
                ++m->info.disk_cache_hits;
                m->info.disk_cache_weight_bytes += bytes;
                m->info.disk_cache_storage_bytes += stored;
                return cached;
            }
            if (keep) [cached release];
        }
    }
    uint64_t blocks = (bytes / 2 + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    uint64_t minimum = plamo_metal_aligned(((blocks * 8 + 15) & ~UINT64_C(15)) + blocks * PLAMO_BF16_BLOCK);
    int compress = mapped && m->use_packing && !(bytes & 1) && bytes >= MIB && m->config.memory != PLAMO_METAL_STREAM &&
                   m->entries < MAX_ENTRIES && minimum <= remaining;
    uint64_t quantized_bytes = plamo_metal_aligned(((bytes / 2 + 31) / 32) * 20);
    int quantize = mapped && m->draft_quantized && m->entries < MAX_ENTRIES && quantized_bytes <= remaining;
    size_t known = 0;
    if (compress) {
        for (; known < m->packing_sizes; ++known) {
            const struct packing_size *s = &m->sizes[known];
            if (s->source != address || s->bytes != bytes) continue;
            compress = s->packed && s->allocated <= remaining;
            break;
        }
    }
    // Only immutable model ranges are retained; borrowed synthetic buffers
    // can be changed/freed by their caller immediately after the projection.
    int retain = mapped && m->config.memory != PLAMO_METAL_STREAM && m->entries < MAX_ENTRIES &&
                 allocation <= remaining;
    if (retain_only && !retain && !compress && !quantize) return nil;
    id<MTLBuffer> buffer = retain && !compress && !quantize ? [m->device newBufferWithLength:(NSUInteger)allocation options:MTLResourceStorageModeShared] : nil;
    if (retain_only && !compress && !quantize && !buffer) return nil;
    if (!buffer) { retain = 0; buffer = m->staging; }
    if (mapped) {
        ++m->info.cache.misses;
        uint64_t done = 0;
        while (done < bytes) {
            size_t count = (size_t)(bytes - done);
            if (count > 8 * MIB) count = 8 * MIB;
            ++plamo_io_stats[1];
            ssize_t n = pread((int)m->config.fds[file], (char *)buffer.contents + done, count, (off_t)(offset + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                ++plamo_io_stats[3];
                if (retain && buffer != m->staging) [buffer release];
                plamo_metal_problem(m, "cannot read original BF16 bytes into a Metal buffer");
                return nil;
            }
            done += (uint64_t)n;
            plamo_io_stats[0] += (uint64_t)n;
        }
    } else {
        memcpy(buffer.contents, source, (size_t)bytes);
    }
    if (quantize) {
        id<MTLBuffer> result = quantize_cache(m, bytes, quantized_bytes);
        if (!result) return retain_only || m->info.errors ? nil : m->staging;
        buffer = result; *packed = 2; allocation = quantized_bytes; retain = 1;
    }
    if (compress) {
        uint64_t failures = m->info.errors;
        unsigned format = 0;
        uint64_t required = 0;
        id<MTLBuffer> result = pack_cache(m, bytes, remaining, &format, &required);
        if (m->info.errors == failures && required && known == m->packing_sizes && known < MAX_ENTRIES)
            m->sizes[m->packing_sizes++] = (struct packing_size){address, bytes, required, format};
        if (!result) {
            if (retain_only || m->info.errors != failures) return nil;
            return m->staging;
        }
        buffer = result;
        *packed = format;
        allocation = required;
        retain = 1;
    }
    if (retain) {
        retain_weights(m, address, bytes, allocation, buffer, *packed, *packed);
    }
    return buffer;
}

static int prepare_buffer(struct plamo_metal *, id<MTLBuffer>);

int plamo_metal_preload(struct plamo_metal *m, const void *source, uint64_t bytes) {
    if (!bytes) return 1;
    @autoreleasepool {
        uint64_t errors = m->info.errors;
        NSUInteger offset;
        unsigned packed;
        id<MTLBuffer> buffer = plamo_metal_weights(m, source, bytes, &offset, 1, &packed, 0);
        return errors == m->info.errors && (!buffer || prepare_buffer(m, buffer));
    }
}

/* Analyze and encode on the GPU. The original read window is never duplicated
   as a whole FP32 model. Only block headers cross back for a short prefix sum. */
static id<MTLBuffer> pack_cache(struct plamo_metal *m, uint64_t bytes, uint64_t remaining,
                                unsigned *packed, uint64_t *allocated) {
    uint32_t values = (uint32_t)(bytes / 2);
    uint64_t blocks = (values + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    if (!plamo_metal_ensure_buffer(m, &m->pack_headers, blocks * sizeof(struct plamo_bf16_block))) return nil;
    id<MTLCommandBuffer> command = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) { plamo_metal_problem(m, "cannot analyze lossless BF16 cache blocks"); return nil; }
    [encoder setComputePipelineState:m->analyze];
    [encoder setBuffer:m->staging offset:0 atIndex:0];
    [encoder setBuffer:m->pack_headers offset:0 atIndex:1];
    [encoder setBytes:&values length:sizeof values atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
    if (!plamo_metal_complete(m, command, encoder)) return nil;
    uint64_t length = plamo_bf16_pack_offsets(m->pack_headers.contents, values);
    if (!length) { plamo_metal_problem(m, "invalid BF16 lossless block analysis"); return nil; }
    int use = m->config.packing == PLAMO_PACK_LOSSLESS || plamo_metal_aligned(length) < plamo_metal_aligned(bytes);
    *packed = use;
    *allocated = plamo_metal_aligned(use ? length : bytes);
    if (*allocated > remaining) return nil;
    id<MTLBuffer> buffer = [m->device newBufferWithLength:(NSUInteger)*allocated options:MTLResourceStorageModeShared];
    if (!buffer) return nil;
    if (!use) { memcpy(buffer.contents, m->staging.contents, (size_t)bytes); return buffer; }
    plamo_metal_residency_add(m, buffer);
    command = [m->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
        plamo_metal_problem(m, "cannot encode lossless BF16 cache blocks");
        plamo_metal_residency_remove(m, buffer); plamo_metal_residency_commit(m); [buffer release]; return nil;
    }
    [encoder setComputePipelineState:m->pack];
    [encoder setBuffer:m->staging offset:0 atIndex:0];
    [encoder setBuffer:m->pack_headers offset:0 atIndex:1];
    [encoder setBuffer:buffer offset:0 atIndex:2];
    [encoder setBytes:&values length:sizeof values atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
    if (!plamo_metal_complete(m, command, encoder)) {
        plamo_metal_residency_remove(m, buffer); plamo_metal_residency_commit(m); [buffer release]; return nil;
    }
    return buffer;
}

static id<MTLBuffer> quantize_cache(struct plamo_metal *m, uint64_t bytes, uint64_t allocation) {
    id<MTLBuffer> buffer = [m->device newBufferWithLength:(NSUInteger)allocation options:MTLResourceStorageModeShared];
    if (!buffer) return nil;
    plamo_metal_residency_add(m, buffer);
    id<MTLCommandBuffer> command = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
    uint32_t values = (uint32_t)(bytes / 2);
    if (!command || !e) { plamo_metal_problem(m, "cannot create proposal quantization command"); goto fail; }
    [e setComputePipelineState:m->quantize4];
    [e setBuffer:m->staging offset:0 atIndex:0]; [e setBuffer:buffer offset:0 atIndex:1];
    [e setBytes:&values length:sizeof values atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((values + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    ++m->info.dispatches;
    if (plamo_metal_complete(m, command, e)) return buffer;
fail:
    plamo_metal_residency_remove(m, buffer); plamo_metal_residency_commit(m); [buffer release]; return nil;
}

static int prepare_buffer(struct plamo_metal *m, id<MTLBuffer> buffer) {
    @autoreleasepool {
        struct entry *entry = NULL;
        for (size_t i = 0; i < m->entries; ++i)
            if (m->table[i].buffer == buffer) { entry = &m->table[i]; break; }
        if (!entry || entry->prepared) return 1;
        long page_bytes = sysconf(_SC_PAGESIZE);
        if (page_bytes <= 0 || page_bytes > UINT32_MAX) return plamo_metal_problem(m, "cannot determine GPU preload page size");
        uint64_t length = entry->packed ? entry->allocated : entry->bytes;
        uint64_t pages = (length + page_bytes - 1) / page_bytes;
        if (!plamo_metal_ensure_buffer(m, &m->output, pages * 4)) return 0;
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return plamo_metal_problem(m, "cannot prepare retained GPU pages");
        [encoder setComputePipelineState:m->touch];
        [encoder setBuffer:m->output offset:0 atIndex:1];
        uint32_t params[] = {(uint32_t)length, (uint32_t)page_bytes};
        [encoder setBuffer:entry->buffer offset:0 atIndex:0];
        [encoder setBytes:params length:sizeof params atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((pages + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ++m->info.dispatches;
        if (!plamo_metal_complete(m, command, encoder)) return 0;
        entry->prepared = 1;
        return 1;
    }
}

int plamo_metal_warm_cache(struct plamo_metal *m) {
    plamo_metal_maintain(m);
    for (size_t i = 0; i < m->entries; ++i)
        if (!prepare_buffer(m, m->table[i].buffer)) return 0;
    return 1;
}

int plamo_metal_resolve_projection(struct plamo_metal *m, const uint16_t *w, uint64_t rows,
                               uint64_t columns, struct projection *out, int mapped) {
    uint64_t row_bytes = columns * 2, window = m->staging_bytes / row_bytes;
    if (window >= 8) window = window / 8 * 8;
    if (!window) return 0;
    out->count = 0;
    for (uint64_t first = 0; first < rows; first += window) {
        if (out->count == sizeof out->tiles / sizeof *out->tiles) return 0;
        uint64_t count = rows - first;
        if (count > window) count = window;
        uintptr_t address = (uintptr_t)w + first * row_bytes;
        const struct entry *found = NULL;
        for (size_t i = 0; i < m->entries; ++i) {
            const struct entry *e = &m->table[i];
            if (address >= e->source && address - e->source < e->bytes && count * row_bytes <= e->bytes - (address - e->source) &&
                (!e->packed || !((address - e->source) & 1))) {
                found = e; break;
            }
        }
        if (found) {
            out->tiles[out->count++] = (struct tile){found->buffer, address - found->source, first, count, 0, found->packed, NULL};
        } else {
            if (!mapped || !m->use_packing) return 0;
            uint64_t stored = 0;
            const void *data = plamo_disk_cache_get(m->disk_cache, (const void *)address, count * row_bytes, 1, &stored);
            if (!data || stored > m->staging_bytes) return 0;
            out->tiles[out->count++] = (struct tile){nil, 0, first, count, stored, 1, data};
        }
    }
    return 1;
}

void plamo_metal_encode_projection(struct plamo_metal *m, id<MTLComputeCommandEncoder> encoder, const struct projection *p,
                               id<MTLBuffer> x, uint64_t xs, id<MTLBuffer> y, uint64_t ys,
                               uint64_t columns, uint64_t tokens) {
    for (size_t i = 0; i < p->count; ++i) {
        const struct tile *t = &p->tiles[i];
        plamo_metal_encode_linear(m, encoder, t->buffer, t->offset, t->packed, x, xs, y, t->first * 4, ys, t->rows, columns, tokens);
        uint64_t bytes = t->rows * columns * 2;
        if (t->mapped_bytes) {
            ++m->info.cache.misses;
            ++m->info.disk_cache_hits;
            m->info.disk_cache_weight_bytes += bytes;
            m->info.disk_cache_storage_bytes += t->mapped_bytes;
            m->info.disk_cache_mapped_bytes += t->mapped_bytes;
        } else {
            ++m->info.cache.hits;
            m->info.cache.hit_bytes += bytes;
        }
    }
}

int plamo_metal_preload_schema(struct plamo_metal *m) {
    for (const struct plamo_weight_spec *s = m->config.schema; s && s->name; ++s) {
        if (s->rank != 2 || !s->slot || !s->shape[0] || !s->shape[1]) continue;
        uint64_t row_bytes = s->shape[1] * 2;
        uint64_t rows = m->staging_bytes / row_bytes;
        if (rows >= 8) rows = rows / 8 * 8;
        if (!rows) return plamo_metal_problem(m, "a BF16 tensor row exceeds the Metal read window");
        for (uint64_t first = 0; first < s->shape[0]; first += rows) {
            uint64_t count = s->shape[0] - first;
            if (count > rows) count = rows;
            const void *source = (const char *)*s->slot + first * row_bytes;
            NSUInteger offset;
            unsigned packed;
            id<MTLBuffer> buffer = plamo_metal_weights(m, source, count * row_bytes, &offset, 0, &packed, 0);
            if (!buffer || buffer == m->staging) return plamo_metal_problem(m, "BF16 matrices do not fit in the Metal residency budget");
            if (!prepare_buffer(m, buffer)) return 0;
        }
    }
    return 1;
}
