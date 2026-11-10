#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal.h"
#include "metal_kernels.h"
#include "packing.h"
#include "disk_cache.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MIB (UINT64_C(1024) * 1024)
#define MAX_ENTRIES 4096
struct entry { uintptr_t source; uint64_t bytes, allocated; id<MTLBuffer> buffer; unsigned prepared, packed; };
struct packing_size { uintptr_t source; uint64_t bytes, allocated; unsigned packed; };
struct tile { const struct entry *entry; NSUInteger offset; uint64_t first, rows; };
struct projection { struct tile tiles[128]; size_t count; };
struct view { id<MTLBuffer> buffer; void *source; NSUInteger offset; uint64_t bytes; };
struct decode_layer {
    struct projection qkv, attention, gate, down;
    struct view k, v, rotated;
    NSUInteger pre, post, mlp_pre, mlp_post, qn, kn;
    size_t rope;
};
struct rope_table { const float *frequency; uint64_t dim, positions; uint32_t scale; id<MTLBuffer> buffer; };
struct plamo_metal {
    struct plamo_metal_config config;
    struct plamo_metal_info info;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> matvec[2], vector[2], matmul[2][4], norm, swiglu, residual, touch, analyze, pack;
    id<MTLComputePipelineState> decode_keys, decode_qkv, decode_attention, decode_residual;
    id<MTLComputePipelineState> prefill_qkv, prefill_attention, prefill_store;
    id<MTLBuffer> staging, input, output;
    id<MTLBuffer> state, normalized, gate, branch, norms;
    id<MTLBuffer> pack_headers;
    id<MTLBuffer> decode_norms, qkv, attention, scores, logits;
    struct model *model;
    struct decode_layer *layers;
    struct rope_table *ropes;
    struct projection head;
    uint64_t layer_count, rope_count;
    NSUInteger final_norm;
    id residency;
    unsigned residency_dirty, use_packing;
    uint64_t maximum, capacity, used, staging_bytes;
    size_t entries;
    size_t packing_sizes;
    double checked_at;
    struct entry table[MAX_ENTRIES];
    struct packing_size sizes[MAX_ENTRIES];
    struct plamo_disk_cache *disk_cache;
};
extern void (*plamo_matvec_kernel)(const uint16_t *, const float *, float *, uint64_t, uint64_t, void *);
extern void (*plamo_matmul_kernel)(const struct plamo_matmul *, void *);
extern int (*plamo_mlp_kernel)(const struct mlp *, float *, uint64_t, void *);
extern int (*plamo_forward_kernel)(struct model *, uint64_t, void *);
extern int (*plamo_prefill_kernel)(struct model *, const uint32_t *, uint64_t, float *, void *);
extern void plamo_rope_table(const float *, uint64_t, uint64_t, float *, float *, float);
extern void *plamo_compute_context;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t aligned(uint64_t bytes) { return (bytes + 16383) / 16384 * 16384; }

static void residency_add(struct plamo_metal *m, id<MTLBuffer> buffer) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        [m->residency addAllocation:buffer]; m->residency_dirty = 1;
    }
#else
    (void)m; (void)buffer;
#endif
}

static void residency_remove(struct plamo_metal *m, id<MTLBuffer> buffer) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        [m->residency removeAllocation:buffer]; m->residency_dirty = 1;
    }
#else
    (void)m; (void)buffer;
#endif
}

static void residency_commit(struct plamo_metal *m) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (m->residency_dirty) { [m->residency commit]; m->residency_dirty = 0; }
    }
#else
    (void)m;
#endif
}

static int problem(struct plamo_metal *m, const char *message) {
    ++m->info.errors;
    snprintf(m->info.note, sizeof m->info.note, "%s", message);
    return 0;
}

void plamo_metal_maintain(struct plamo_metal *m) {
    m->checked_at = now();
    if (m->config.memory != PLAMO_METAL_AUTO) return;
    uint64_t available = m->config.available(), owned = m->used + m->staging_bytes;
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
        residency_remove(m, e->buffer);
        [e->buffer release];
    }
    // Release evicted pages immediately, including while no GPU work is queued.
    if (m->entries != previous_entries) residency_commit(m);
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

static void retain_weights(struct plamo_metal *m, uintptr_t address, uint64_t bytes, uint64_t allocation,
                           id<MTLBuffer> buffer, unsigned prepared, unsigned packed) {
    m->table[m->entries++] = (struct entry){address, bytes, allocation, buffer, prepared, packed};
    m->used += allocation;
    m->info.cache.cached_bytes += bytes;
    if (packed) m->info.packed_weight_bytes += bytes;
    // Newly encoded packed buffers were registered by pack_cache already.
    if (!prepared) residency_add(m, buffer);
}

static id<MTLBuffer> weights(struct plamo_metal *m, const void *source, uint64_t bytes, NSUInteger *buffer_offset,
                              int retain_only, unsigned *packed) {
    uintptr_t address = (uintptr_t)source;
    *buffer_offset = 0;
    *packed = 0;
    if (m->config.memory == PLAMO_METAL_AUTO && now() - m->checked_at >= 0.25) plamo_metal_maintain(m);
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
    uint64_t allocation = aligned(bytes);
    uint64_t remaining = m->capacity - m->staging_bytes - m->used;
    if (mapped && m->config.memory != PLAMO_METAL_STREAM) {
        uint64_t stored = 0;
        const void *data = plamo_disk_cache_get(m->disk_cache, source, bytes, m->use_packing, &stored);
        int keep = stored <= remaining && m->entries < MAX_ENTRIES;
        if (data && retain_only && !keep) return nil;
        if (data && keep) {
            id<MTLBuffer> cached = [m->device newBufferWithLength:(NSUInteger)stored options:MTLResourceStorageModeShared];
            if (cached && plamo_disk_cache_read(m->disk_cache, data, cached.contents, stored, plamo_io_stats)) {
                *packed = m->use_packing;
                retain_weights(m, address, bytes, stored, cached, 0, *packed);
                ++m->info.cache.misses;
                ++m->info.disk_cache_hits;
                m->info.disk_cache_weight_bytes += bytes;
                m->info.disk_cache_storage_bytes += stored;
                return cached;
            }
            [cached release];
        }
    }
    uint64_t blocks = (bytes / 2 + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    uint64_t minimum = aligned(((blocks * 8 + 15) & ~UINT64_C(15)) + blocks * PLAMO_BF16_BLOCK);
    int compress = mapped && m->use_packing && !(bytes & 1) && bytes >= MIB && m->config.memory != PLAMO_METAL_STREAM &&
                   m->entries < MAX_ENTRIES && minimum <= remaining;
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
    if (retain_only && !retain && !compress) return nil;
    id<MTLBuffer> buffer = retain && !compress ? [m->device newBufferWithLength:(NSUInteger)allocation options:MTLResourceStorageModeShared] : nil;
    if (retain_only && !compress && !buffer) return nil;
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
                problem(m, "cannot read original BF16 bytes into a Metal buffer");
                return nil;
            }
            done += (uint64_t)n;
            plamo_io_stats[0] += (uint64_t)n;
        }
    } else {
        memcpy(buffer.contents, source, (size_t)bytes);
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
        id<MTLBuffer> buffer = weights(m, source, bytes, &offset, 1, &packed);
        return errors == m->info.errors && (!buffer || prepare_buffer(m, buffer));
    }
}

static int ensure_buffer(struct plamo_metal *m, id<MTLBuffer> *buffer, uint64_t bytes) {
    if (*buffer && (*buffer).length >= bytes) return 1;
    if (bytes > m->device.maxBufferLength) return problem(m, "Metal activation buffer exceeds the device limit");
    id<MTLBuffer> replacement = [m->device newBufferWithLength:(NSUInteger)aligned(bytes) options:MTLResourceStorageModeShared];
    if (!replacement) return problem(m, "cannot allocate Metal activation buffer");
    int replacing = *buffer != nil;
    if (*buffer) residency_remove(m, *buffer);
    [*buffer release];
    *buffer = replacement;
    residency_add(m, replacement);
    if (replacing) residency_commit(m);
    return 1;
}

static void encode_linear(struct plamo_metal *m, id<MTLComputeCommandEncoder> encoder,
                          id<MTLBuffer> w, NSUInteger offset, unsigned packed, id<MTLBuffer> x, uint64_t xs,
                          id<MTLBuffer> y, NSUInteger output_offset, uint64_t ys,
                          uint64_t rows, uint64_t columns, uint64_t tokens) {
    struct { uint32_t rows, columns, tokens, xs, ys, aligned_weights, weight_offset; } params = {
        (uint32_t)rows, (uint32_t)columns, (uint32_t)tokens, (uint32_t)xs, (uint32_t)ys,
        !(offset % 8), packed ? (uint32_t)(offset / 2) : 0
    };
    int batched = tokens >= 8, vector = tokens >= 4 && !batched;
    // Keep the minimum number of weight scans, then choose the narrowest
    // compiled tile: e.g. 69 tokens use 3 x 24 rather than 3 x 32.
    uint64_t panels = (tokens + 31) / 32;
    uint64_t tile_tokens = ((tokens + panels - 1) / panels + 7) / 8 * 8;
    [encoder setComputePipelineState:batched ? m->matmul[packed][tile_tokens / 8 - 1] : vector ? m->vector[packed] : m->matvec[packed]];
    [encoder setBuffer:w offset:packed ? 0 : offset atIndex:0];
    [encoder setBuffer:x offset:0 atIndex:1];
    [encoder setBuffer:y offset:output_offset atIndex:2];
    [encoder setBytes:&params length:sizeof params atIndex:3];
    MTLSize groups = batched ? MTLSizeMake((rows + 31) / 32, (tokens + tile_tokens - 1) / tile_tokens, 1) :
                              MTLSizeMake((rows + 3) / 4, vector ? (tokens + 7) / 8 : tokens, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
}

static int complete(struct plamo_metal *m, id<MTLCommandBuffer> command, id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding];
    // Batch initial weight registrations and new activation buffers. No GPU
    // work is submitted before the complete set of changes is committed.
    residency_commit(m);
    [command commit];
    [command waitUntilCompleted];
    ++m->info.command_buffers;
    if (command.status != MTLCommandBufferStatusCompleted)
        return problem(m, command.error ? command.error.localizedDescription.UTF8String : "Metal command failed");
    m->info.gpu_seconds += command.GPUEndTime - command.GPUStartTime;
    return 1;
}

/* Analyze and encode on the GPU. The original read window is never duplicated
   as a whole FP32 model. Only block headers cross back for a short prefix sum. */
static id<MTLBuffer> pack_cache(struct plamo_metal *m, uint64_t bytes, uint64_t remaining,
                                unsigned *packed, uint64_t *allocated) {
    uint32_t values = (uint32_t)(bytes / 2);
    uint64_t blocks = (values + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    if (!ensure_buffer(m, &m->pack_headers, blocks * sizeof(struct plamo_bf16_block))) return nil;
    id<MTLCommandBuffer> command = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) { problem(m, "cannot analyze lossless BF16 cache blocks"); return nil; }
    [encoder setComputePipelineState:m->analyze];
    [encoder setBuffer:m->staging offset:0 atIndex:0];
    [encoder setBuffer:m->pack_headers offset:0 atIndex:1];
    [encoder setBytes:&values length:sizeof values atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
    if (!complete(m, command, encoder)) return nil;
    uint64_t length = plamo_bf16_pack_offsets(m->pack_headers.contents, values);
    if (!length) { problem(m, "invalid BF16 lossless block analysis"); return nil; }
    int use = m->config.packing == PLAMO_PACK_LOSSLESS || aligned(length) < aligned(bytes);
    *packed = use;
    *allocated = aligned(use ? length : bytes);
    if (*allocated > remaining) return nil;
    id<MTLBuffer> buffer = [m->device newBufferWithLength:(NSUInteger)*allocated options:MTLResourceStorageModeShared];
    if (!buffer) return nil;
    if (!use) { memcpy(buffer.contents, m->staging.contents, (size_t)bytes); return buffer; }
    residency_add(m, buffer);
    command = [m->queue commandBuffer];
    encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
        problem(m, "cannot encode lossless BF16 cache blocks");
        residency_remove(m, buffer); residency_commit(m); [buffer release]; return nil;
    }
    [encoder setComputePipelineState:m->pack];
    [encoder setBuffer:m->staging offset:0 atIndex:0];
    [encoder setBuffer:m->pack_headers offset:0 atIndex:1];
    [encoder setBuffer:buffer offset:0 atIndex:2];
    [encoder setBytes:&values length:sizeof values atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
    if (!complete(m, command, encoder)) {
        residency_remove(m, buffer); residency_commit(m); [buffer release]; return nil;
    }
    return buffer;
}

static int prepare_buffer(struct plamo_metal *m, id<MTLBuffer> buffer) {
    @autoreleasepool {
        struct entry *entry = NULL;
        for (size_t i = 0; i < m->entries; ++i)
            if (m->table[i].buffer == buffer) { entry = &m->table[i]; break; }
        if (!entry || entry->prepared) return 1;
        long page_bytes = sysconf(_SC_PAGESIZE);
        if (page_bytes <= 0 || page_bytes > UINT32_MAX) return problem(m, "cannot determine GPU preload page size");
        uint64_t length = entry->packed ? entry->allocated : entry->bytes;
        uint64_t pages = (length + page_bytes - 1) / page_bytes;
        if (!ensure_buffer(m, &m->output, pages * 4)) return 0;
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return problem(m, "cannot prepare retained GPU pages");
        [encoder setComputePipelineState:m->touch];
        [encoder setBuffer:m->output offset:0 atIndex:1];
        uint32_t params[] = {(uint32_t)length, (uint32_t)page_bytes};
        [encoder setBuffer:entry->buffer offset:0 atIndex:0];
        [encoder setBytes:params length:sizeof params atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((pages + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        ++m->info.dispatches;
        if (!complete(m, command, encoder)) return 0;
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

static int resident_projection(struct plamo_metal *, const uint16_t *, uint64_t, uint64_t, struct projection *);
static void encode_projection(struct plamo_metal *, id<MTLComputeCommandEncoder>, const struct projection *,
                               id<MTLBuffer>, uint64_t, id<MTLBuffer>, uint64_t, uint64_t, uint64_t);

int plamo_metal_linear(struct plamo_metal *m, const struct plamo_matmul *p) {
    if (!p->rows || !p->tokens) return 1;
    if (!p->columns) {
        for (uint64_t t = 0; t < p->tokens; ++t) memset(p->output + t * p->output_stride, 0, (size_t)p->rows * 4);
        return 1;
    }
    if (p->rows > UINT32_MAX || p->columns > UINT32_MAX || p->tokens > UINT32_MAX ||
        p->input_stride < p->columns || p->output_stride < p->rows ||
        p->rows > SIZE_MAX / 2 / p->columns ||
        p->rows > SIZE_MAX / 4 / p->tokens || p->columns > SIZE_MAX / 4 / p->tokens) return 0;
    @autoreleasepool {
        struct projection projection;
        int whole = p->rows * p->columns * 2 > m->staging_bytes;
        if (whole) {
            if (m->config.memory == PLAMO_METAL_AUTO && now() - m->checked_at >= 0.25) plamo_metal_maintain(m);
            if (!resident_projection(m, p->weights, p->rows, p->columns, &projection)) {
                uint64_t window = m->staging_bytes / (p->columns * 2);
                if (window >= 8) window = window / 8 * 8;
                if (!window) return 0;
                // Streamed ranges finish before the read window is reused.
                // Resident projections instead encode every tile together.
                for (uint64_t first = 0; first < p->rows; first += window) {
                    struct plamo_matmul part = *p;
                    part.weights += first * p->columns;
                    part.output += first;
                    part.rows = p->rows - first;
                    if (part.rows > window) part.rows = window;
                    if (!plamo_metal_linear(m, &part)) return 0;
                }
                return 1;
            }
        }
        if (!ensure_buffer(m, &m->input, p->tokens * p->columns * 4) ||
            !ensure_buffer(m, &m->output, p->tokens * p->rows * 4)) return 0;
        NSUInteger offset = 0;
        unsigned packed = 0;
        id<MTLBuffer> w = whole ? nil : weights(m, p->weights, p->rows * p->columns * 2, &offset, 0, &packed);
        if (!whole && !w) return 0;
        for (uint64_t t = 0; t < p->tokens; ++t)
            memcpy((float *)m->input.contents + t * p->columns, p->input + t * p->input_stride, (size_t)p->columns * 4);
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return problem(m, "cannot create a Metal compute command");
        if (whole) encode_projection(m, encoder, &projection, m->input, p->columns, m->output, p->rows, p->columns, p->tokens);
        else encode_linear(m, encoder, w, offset, packed, m->input, p->columns, m->output, 0, p->rows,
                           p->rows, p->columns, p->tokens);
        if (!complete(m, command, encoder)) return 0;
        for (uint64_t t = 0; t < p->tokens; ++t)
            memcpy(p->output + t * p->output_stride, (float *)m->output.contents + t * p->rows, (size_t)p->rows * 4);
        return 1;
    }
}

// Preflight without I/O, allocation or eviction: a fused command must never
// refer to the reusable staging buffer or to an entry removed mid-command.
static int resident_projection(struct plamo_metal *m, const uint16_t *w, uint64_t rows,
                                uint64_t columns, struct projection *out) {
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
        if (!found) return 0;
        out->tiles[out->count++] = (struct tile){found, address - found->source, first, count};
    }
    return 1;
}

static void encode_projection(struct plamo_metal *m, id<MTLComputeCommandEncoder> encoder, const struct projection *p,
                               id<MTLBuffer> x, uint64_t xs, id<MTLBuffer> y, uint64_t ys,
                               uint64_t columns, uint64_t tokens) {
    for (size_t i = 0; i < p->count; ++i) {
        const struct tile *t = &p->tiles[i];
        encode_linear(m, encoder, t->entry->buffer, t->offset, t->entry->packed, x, xs, y, t->first * 4, ys, t->rows, columns, tokens);
        ++m->info.cache.hits;
        m->info.cache.hit_bytes += t->rows * columns * 2;
    }
}

static void release_view(struct plamo_metal *m, struct view *v) {
    if (v->buffer) { residency_remove(m, v->buffer); [v->buffer release]; }
    *v = (struct view){0};
}

// Map only a growing prefix of the existing anonymous model arena. The full
// global KV capacity is virtual (262K rows); making it all resident would
// displace model weights. No copy or alternate logical KV cache is needed.
static int cache_view(struct plamo_metal *m, struct view *v, void *source, uint64_t rows,
                       uint64_t width, uint64_t capacity) {
    if (!source || !rows || !width || rows > capacity || rows > SIZE_MAX / 4 / width) return 0;
    uint64_t bytes = rows * width * 4;
    if (v->source == source && v->bytes >= bytes) return 1;
    uint64_t size = 128;
    while (size < rows && size <= UINT64_MAX / 2) size *= 2;
    if (size > capacity) size = capacity;
    if (size > SIZE_MAX / 4 / width) return 0;
    bytes = size * width * 4;
    uintptr_t address = (uintptr_t)source;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0;
    uint64_t page = (uint64_t)page_size, offset = address % page;
    if (bytes > UINT64_MAX - offset - page) return 0;
    uint64_t length = (bytes + offset + page - 1) / page * page;
    if (length > m->device.maxBufferLength) return 0;
    mach_vm_address_t base = address - offset, region = base;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    kern_return_t status = mach_vm_region(mach_task_self(), &region, &region_size, VM_REGION_BASIC_INFO_64,
                                         (vm_region_info_t)&info, &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (status != KERN_SUCCESS || region > base || base - region > region_size ||
        length > region_size - (base - region) ||
        (info.protection & (VM_PROT_READ | VM_PROT_WRITE)) != (VM_PROT_READ | VM_PROT_WRITE)) return 0;
    id<MTLBuffer> replacement = [m->device newBufferWithBytesNoCopy:(void *)base length:(NSUInteger)length
                                                          options:MTLResourceStorageModeShared deallocator:nil];
    if (!replacement) return 0;
    release_view(m, v);
    *v = (struct view){replacement, source, (NSUInteger)offset, bytes};
    residency_add(m, replacement);
    residency_commit(m);
    return 1;
}

static void release_decode(struct plamo_metal *m) {
    for (uint64_t i = 0; i < m->layer_count; ++i) {
        release_view(m, &m->layers[i].k); release_view(m, &m->layers[i].v); release_view(m, &m->layers[i].rotated);
    }
    for (uint64_t i = 0; i < m->rope_count; ++i) {
        residency_remove(m, m->ropes[i].buffer); [m->ropes[i].buffer release];
    }
    free(m->layers); free(m->ropes);
    m->layers = NULL; m->ropes = NULL; m->model = NULL; m->layer_count = m->rope_count = 0;
    residency_commit(m);
}

static int prepare_rope(struct plamo_metal *m, struct rope_table *r, uint64_t positions) {
    if (positions <= r->positions) return 1;
    uint64_t rows = 128;
    while (rows < positions) rows *= 2;
    if (rows > UINT32_MAX || r->dim > SIZE_MAX / 4 / rows) return 0;
    uint64_t previous = r->positions;
    id<MTLBuffer> replacement = [m->device newBufferWithLength:(NSUInteger)aligned(rows * r->dim * 4)
                                                      options:MTLResourceStorageModeShared];
    if (!replacement) return problem(m, "cannot allocate exact RoPE table");
    if (previous) memcpy(replacement.contents, r->buffer.contents, (size_t)(previous * r->dim * 4));
    float scale; memcpy(&scale, &r->scale, 4);
    for (uint64_t row = previous; row < rows; ++row) {
        float *p = (float *)replacement.contents + row * r->dim;
        plamo_rope_table(r->frequency, row, r->dim / 2, p, p + r->dim / 2, scale);
    }
    if (r->buffer) { residency_remove(m, r->buffer); [r->buffer release]; }
    r->buffer = replacement; r->positions = rows;
    residency_add(m, replacement);
    return 1;
}

int plamo_metal_prepare_model(struct plamo_metal *m, struct model *model) {
    if (m->config.memory == PLAMO_METAL_STREAM || m->model == model) return 1;
    if (!model || !model->count || model->count > 128 || !model->hidden || model->hidden > UINT32_MAX ||
        !model->vocab || model->vocab > UINT32_MAX) return 0;
    @autoreleasepool {
        release_decode(m);
        m->layers = calloc((size_t)model->count, sizeof *m->layers);
        m->ropes = calloc((size_t)model->count, sizeof *m->ropes);
        if (!m->layers || !m->ropes) { release_decode(m); return problem(m, "cannot allocate decode descriptors"); }
        m->layer_count = model->count;
        uint64_t norm_bytes = 0, qkv = 0, attention = 0, gate = 0, heads = 0;
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            const struct rotation *r = l->mixer->rotation;
            if (l->hidden != model->hidden || q->hidden != model->hidden || l->mlp->hidden != model->hidden ||
                !q->kh || !q->qh || q->kh > UINT32_MAX || q->qh > UINT32_MAX || q->qh % q->kh ||
                !q->dim || q->dim % 2 || q->dim > UINT32_MAX / (q->qh + 2 * q->kh) ||
                !c->capacity || c->capacity > UINT32_MAX || c->width != q->kh * q->dim ||
                r->cache != c || r->heads != q->kh || r->dim != q->dim || !l->mlp->intermediate ||
                l->mlp->intermediate > UINT32_MAX / 2) { release_decode(m); return 0; }
            struct decode_layer *d = &m->layers[i];
            NSUInteger *offsets[] = {&d->pre, &d->post, &d->mlp_pre, &d->mlp_post, &d->qn, &d->kn};
            for (size_t j = 0; j < 6; ++j) {
                *offsets[j] = (NSUInteger)norm_bytes;
                norm_bytes += ((j < 4 ? model->hidden : q->dim) * 2 + 15) & ~UINT64_C(15);
            }
            uint32_t scale; memcpy(&scale, &model->scales[i], 4);
            size_t rope = 0;
            for (; rope < m->rope_count; ++rope) {
                struct rope_table *t = &m->ropes[rope];
                if (t->dim == q->dim && t->scale == scale && !memcmp(t->frequency, r->freq, (size_t)(q->dim * 2))) break;
            }
            if (rope == m->rope_count) m->ropes[m->rope_count++] = (struct rope_table){r->freq, q->dim, 0, scale, nil};
            d->rope = rope;
            // API accepts up to 4096 prompt + 4096 output tokens. Grow further
            // only if another native caller actually needs a later position.
            if (!prepare_rope(m, &m->ropes[rope], c->capacity < 8192 ? c->capacity : 8192) ||
                !cache_view(m, &d->k, c->k, 1, c->width, c->capacity) ||
                !cache_view(m, &d->v, c->v, 1, c->width, c->capacity) ||
                !cache_view(m, &d->rotated, r->out, 1, c->width, c->capacity)) { release_decode(m); return 0; }
            uint64_t n = (q->qh + 2 * q->kh) * q->dim;
            if (n > qkv) qkv = n;
            if (q->qh * q->dim > attention) attention = q->qh * q->dim;
            if (q->qh > heads) heads = q->qh;
            if (l->mlp->intermediate * 2 > gate) gate = l->mlp->intermediate * 2;
        }
        m->final_norm = (NSUInteger)norm_bytes;
        norm_bytes += model->hidden * 2;
        if (!ensure_buffer(m, &m->decode_norms, norm_bytes) ||
            !ensure_buffer(m, &m->state, model->hidden * 4) ||
            !ensure_buffer(m, &m->normalized, model->hidden * 4) ||
            !ensure_buffer(m, &m->branch, model->hidden * 4) ||
            !ensure_buffer(m, &m->gate, gate * 4) ||
            !ensure_buffer(m, &m->qkv, qkv * 4) ||
            !ensure_buffer(m, &m->attention, attention * 4) ||
            !ensure_buffer(m, &m->logits, model->vocab * 4) ||
            !ensure_buffer(m, &m->scores, 128 * heads * 4)) { release_decode(m); return 0; }
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct decode_layer *d = &m->layers[i];
            const uint16_t *weights[] = {l->pre, l->post, l->mlp->pre, l->mlp->post, l->mixer->qkv->qn, l->mixer->qkv->kn};
            const NSUInteger offsets[] = {d->pre, d->post, d->mlp_pre, d->mlp_post, d->qn, d->kn};
            for (size_t j = 0; j < 6; ++j)
                memcpy((char *)m->decode_norms.contents + offsets[j], weights[j], (size_t)((j < 4 ? model->hidden : l->mixer->qkv->dim) * 2));
        }
        memcpy((char *)m->decode_norms.contents + m->final_norm, model->norm, (size_t)(model->hidden * 2));
        m->model = model;
        return 1;
    }
}

static void encode_norm(struct plamo_metal *m, id<MTLComputeCommandEncoder> e, NSUInteger offset,
                         uint64_t hidden, uint64_t tokens, NSUInteger state_offset) {
    uint32_t p[] = {(uint32_t)hidden, 0, (uint32_t)tokens};
    [e setComputePipelineState:m->norm];
    [e setBuffer:m->state offset:state_offset atIndex:0];
    [e setBuffer:m->decode_norms offset:offset atIndex:1];
    [e setBuffer:m->normalized offset:0 atIndex:2];
    [e setBytes:p length:sizeof p atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
}

static void encode_residual(struct plamo_metal *m, id<MTLComputeCommandEncoder> e, NSUInteger offset,
                             uint64_t hidden, uint64_t tokens, int mlp) {
    uint32_t p[] = {(uint32_t)hidden, 0, (uint32_t)tokens};
    [e setComputePipelineState:mlp ? m->residual : m->decode_residual];
    [e setBuffer:m->state offset:0 atIndex:0];
    [e setBuffer:m->decode_norms offset:offset atIndex:1];
    [e setBuffer:m->branch offset:0 atIndex:2];
    [e setBytes:p length:sizeof p atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    ++m->info.dispatches;
}

int plamo_metal_forward(struct plamo_metal *m, struct model *model, uint64_t token) {
    if (token >= model->vocab) return -1;
    if (m->config.memory == PLAMO_METAL_STREAM) return 0;
    @autoreleasepool {
        plamo_metal_maintain(m);
        if (!plamo_metal_prepare_model(m, model)) return -1;
        uint64_t h = model->hidden, scores = 0;
        // Preflight the whole graph before touching embedding, KV or state.
        // No weight acquisition/eviction takes place during command encoding.
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            struct decode_layer *d = &m->layers[i];
            if (c->count > c->capacity || (c->count == c->capacity && !c->sliding)) return -1;
            if (!resident_projection(m, q->weight, (q->qh + 2 * q->kh) * q->dim, h, &d->qkv) ||
                !resident_projection(m, l->mixer->weight, h, q->qh * q->dim, &d->attention) ||
                !resident_projection(m, l->mlp->gu, l->mlp->intermediate * 2, h, &d->gate) ||
                !resident_projection(m, l->mlp->down, h, l->mlp->intermediate, &d->down)) return 0;
            uint64_t n = c->count < c->capacity ? c->count + 1 : c->capacity;
            if (n * q->qh > scores) scores = n * q->qh;
            if (!prepare_rope(m, &m->ropes[d->rope], n) ||
                !cache_view(m, &d->k, c->k, n, c->width, c->capacity) ||
                !cache_view(m, &d->v, c->v, n, c->width, c->capacity) ||
                !cache_view(m, &d->rotated, l->mixer->rotation->out, n, c->width, c->capacity)) return -1;
        }
        if (!resident_projection(m, model->head, model->vocab, h, &m->head)) return 0;
        if (!ensure_buffer(m, &m->scores, scores * 4)) return -1;
        // A previous batched prefill can have resized these reusable buffers;
        // using a prefix is safe. Embedding expansion preserves every BF16 bit.
        uint32_t *state = m->state.contents;
        for (uint64_t i = 0; i < h; ++i) state[i] = (uint32_t)model->embedding[token * h + i] << 16;
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
        if (!command || !e) { problem(m, "cannot create complete Metal decode command"); return -1; }
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            const struct rotation *r = l->mixer->rotation;
            const struct mlp *mlp = l->mlp;
            const struct decode_layer *d = &m->layers[i];
            id<MTLBuffer> table = m->ropes[d->rope].buffer;
            uint32_t p[] = {(uint32_t)q->qh, (uint32_t)q->kh, (uint32_t)q->dim,
                            (uint32_t)(c->count < c->capacity ? c->count + 1 : c->capacity)};
            uint32_t shift = c->count == c->capacity;
            encode_norm(m, e, d->pre, h, 1, 0);
            encode_projection(m, e, &d->qkv, m->normalized, h, m->qkv, (q->qh + 2 * q->kh) * q->dim, h, 1);
            if (shift || r->processed_count != c->count || r->scale_bits != m->ropes[d->rope].scale) {
                [e setComputePipelineState:m->decode_keys];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:0];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:1];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:2];
                [e setBuffer:table offset:0 atIndex:3];
                [e setBytes:p length:sizeof p atIndex:4];
                [e setBytes:&shift length:sizeof shift atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            [e setComputePipelineState:m->decode_qkv];
            [e setBuffer:m->qkv offset:0 atIndex:0];
            [e setBuffer:m->decode_norms offset:d->qn atIndex:1];
            [e setBuffer:m->decode_norms offset:d->kn atIndex:2];
            [e setBuffer:d->k.buffer offset:d->k.offset atIndex:3];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:4];
            [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:5];
            [e setBuffer:table offset:0 atIndex:6];
            [e setBytes:p length:sizeof p atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(q->qh + q->kh, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e setComputePipelineState:m->decode_attention];
            [e setBuffer:m->qkv offset:0 atIndex:0];
            [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:1];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
            [e setBuffer:m->attention offset:0 atIndex:3];
            [e setBuffer:m->scores offset:0 atIndex:4];
            [e setBytes:p length:sizeof p atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(q->qh, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            encode_projection(m, e, &d->attention, m->attention, q->qh * q->dim, m->branch, h, q->qh * q->dim, 1);
            encode_residual(m, e, d->post, h, 1, 0);
            encode_norm(m, e, d->mlp_pre, h, 1, 0);
            encode_projection(m, e, &d->gate, m->normalized, h, m->gate, 2 * mlp->intermediate, h, 1);
            uint32_t mp[] = {(uint32_t)h, (uint32_t)mlp->intermediate, 1};
            [e setComputePipelineState:m->swiglu];
            [e setBuffer:m->gate offset:0 atIndex:0];
            [e setBytes:mp length:sizeof mp atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake((mlp->intermediate + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            encode_projection(m, e, &d->down, m->gate, 2 * mlp->intermediate, m->branch, h, mlp->intermediate, 1);
            encode_residual(m, e, d->mlp_post, h, 1, 1);
            m->info.dispatches += 3;
        }
        encode_norm(m, e, m->final_norm, h, 1, 0);
        encode_projection(m, e, &m->head, m->normalized, h, m->logits, model->vocab, h, 1);
        if (!complete(m, command, e)) return -1;
        memcpy(model->state, m->state.contents, (size_t)(h * 4));
        memcpy(model->normalized, m->normalized.contents, (size_t)(h * 4));
        memcpy(model->logits, m->logits.contents, (size_t)(model->vocab * 4));
        for (uint64_t i = 0; i < model->count; ++i) {
            struct mixer *mix = model->layers[i]->mixer;
            if (mix->cache->count < mix->cache->capacity) ++mix->cache->count;
            mix->rotation->processed_count = mix->cache->count;
            mix->rotation->scale_bits = m->ropes[m->layers[i].rope].scale;
        }
        ++m->info.fused_decodes;
        m->info.fused_mlps += model->count;
        return 1;
    }
}

int plamo_metal_prefill(struct plamo_metal *m, struct model *model, const uint32_t *ids,
                         uint64_t tokens, float *states) {
    if (!model || !ids || !states || !tokens) return -1;
    if (m->config.memory == PLAMO_METAL_STREAM || tokens > 4096) return 0;
    for (uint64_t t = 0; t < tokens; ++t) if (ids[t] >= model->vocab) return -1;
    @autoreleasepool {
        plamo_metal_maintain(m);
        if (!plamo_metal_prepare_model(m, model)) return -1;
        uint64_t h = model->hidden, qkv = 0, attention = 0, gate = 0, score_row = 0;
        // Resolve every retained tile before any state/KV change. Maintenance
        // cannot evict a projection while this single command is encoded.
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            struct decode_layer *d = &m->layers[i];
            if (c->count > c->capacity || (!c->sliding && tokens > c->capacity - c->count)) return -1;
            if (c->count > UINT32_MAX - tokens) return 0;
            if (!resident_projection(m, q->weight, (q->qh + 2 * q->kh) * q->dim, h, &d->qkv) ||
                !resident_projection(m, l->mixer->weight, h, q->qh * q->dim, &d->attention) ||
                !resident_projection(m, l->mlp->gu, l->mlp->intermediate * 2, h, &d->gate) ||
                !resident_projection(m, l->mlp->down, h, l->mlp->intermediate, &d->down)) return 0;
            uint64_t count = c->count + tokens;
            if (count > c->capacity) count = c->capacity;
            if (count * q->qh > score_row) score_row = count * q->qh;
            if (!prepare_rope(m, &m->ropes[d->rope], count) ||
                !cache_view(m, &d->k, c->k, count, c->width, c->capacity) ||
                !cache_view(m, &d->v, c->v, count, c->width, c->capacity) ||
                !cache_view(m, &d->rotated, l->mixer->rotation->out, count, c->width, c->capacity)) return -1;
            uint64_t width = (q->qh + 2 * q->kh) * q->dim;
            if (width > qkv) qkv = width;
            if (q->qh * q->dim > attention) attention = q->qh * q->dim;
            if (2 * l->mlp->intermediate > gate) gate = 2 * l->mlp->intermediate;
        }
        if (!resident_projection(m, model->head, model->vocab, h, &m->head)) return 0;
        // Cap attention scratch at 16 MiB (or a single score row for a larger
        // native context). Dispatches reuse it in query tiles, without waits.
        if (score_row > m->device.maxBufferLength / 4) return 0;
        uint64_t queries = 4 * MIB / score_row;
        if (!queries) queries = 1;
        if (queries > tokens) queries = tokens;
        if (!ensure_buffer(m, &m->state, tokens * h * 4) ||
            !ensure_buffer(m, &m->normalized, tokens * h * 4) ||
            !ensure_buffer(m, &m->branch, tokens * h * 4) ||
            !ensure_buffer(m, &m->gate, tokens * gate * 4) ||
            !ensure_buffer(m, &m->qkv, tokens * qkv * 4) ||
            !ensure_buffer(m, &m->attention, tokens * attention * 4) ||
            !ensure_buffer(m, &m->scores, queries * score_row * 4)) return -1;
        uint32_t *state = m->state.contents;
        for (uint64_t t = 0; t < tokens; ++t)
            for (uint64_t col = 0; col < h; ++col)
                state[t * h + col] = (uint32_t)model->embedding[(uint64_t)ids[t] * h + col] << 16;
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
        if (!command || !e) { problem(m, "cannot create complete Metal prefill command"); return -1; }
        for (uint64_t i = 0; i < model->count; ++i) {
            const struct layer *l = model->layers[i];
            const struct qkv *q = l->mixer->qkv;
            const struct cache *c = l->mixer->cache;
            const struct rotation *r = l->mixer->rotation;
            const struct mlp *mlp = l->mlp;
            const struct decode_layer *d = &m->layers[i];
            id<MTLBuffer> table = m->ropes[d->rope].buffer;
            uint32_t p[] = {(uint32_t)q->qh, (uint32_t)q->kh, (uint32_t)q->dim, (uint32_t)c->count,
                            (uint32_t)tokens, (uint32_t)c->capacity, 0};
            encode_norm(m, e, d->pre, h, tokens, 0);
            encode_projection(m, e, &d->qkv, m->normalized, h, m->qkv, (q->qh + 2 * q->kh) * q->dim, h, tokens);
            if (c->count && c->count + tokens <= c->capacity &&
                (r->processed_count != c->count || r->scale_bits != m->ropes[d->rope].scale)) {
                uint32_t dp[] = {p[0], p[1], p[2], p[3] + 1}, shift = 0;
                [e setComputePipelineState:m->decode_keys];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:0];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:1];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:2];
                [e setBuffer:table offset:0 atIndex:3];
                [e setBytes:dp length:sizeof dp atIndex:4];
                [e setBytes:&shift length:sizeof shift atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            [e setComputePipelineState:m->prefill_qkv];
            [e setBuffer:m->qkv offset:0 atIndex:0];
            [e setBuffer:m->decode_norms offset:d->qn atIndex:1];
            [e setBuffer:m->decode_norms offset:d->kn atIndex:2];
            [e setBuffer:d->k.buffer offset:d->k.offset atIndex:3];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:4];
            [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:5];
            [e setBuffer:table offset:0 atIndex:6];
            [e setBytes:p length:sizeof p atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(q->qh + q->kh, tokens, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            [e setComputePipelineState:m->prefill_attention];
            [e setBuffer:m->qkv offset:0 atIndex:0];
            [e setBuffer:d->k.buffer offset:d->k.offset atIndex:1];
            [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
            [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:3];
            [e setBuffer:table offset:0 atIndex:4];
            [e setBuffer:m->attention offset:0 atIndex:5];
            [e setBuffer:m->scores offset:0 atIndex:6];
            for (uint64_t first = 0; first < tokens; first += queries) {
                uint64_t count = tokens - first;
                if (count > queries) count = queries;
                p[6] = (uint32_t)first;
                [e setBytes:p length:sizeof p atIndex:7];
                [e dispatchThreadgroups:MTLSizeMake(q->qh, count, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            if (c->count + tokens > c->capacity) {
                [e setComputePipelineState:m->prefill_store];
                [e setBuffer:m->qkv offset:0 atIndex:0];
                [e setBuffer:d->k.buffer offset:d->k.offset atIndex:1];
                [e setBuffer:d->v.buffer offset:d->v.offset atIndex:2];
                [e setBuffer:d->rotated.buffer offset:d->rotated.offset atIndex:3];
                [e setBuffer:table offset:0 atIndex:4];
                [e setBytes:p length:sizeof p atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake((c->width / 2 + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                ++m->info.dispatches;
            }
            encode_projection(m, e, &d->attention, m->attention, q->qh * q->dim, m->branch, h, q->qh * q->dim, tokens);
            encode_residual(m, e, d->post, h, tokens, 0);
            encode_norm(m, e, d->mlp_pre, h, tokens, 0);
            encode_projection(m, e, &d->gate, m->normalized, h, m->gate, 2 * mlp->intermediate, h, tokens);
            uint32_t mp[] = {(uint32_t)h, (uint32_t)mlp->intermediate, (uint32_t)tokens};
            [e setComputePipelineState:m->swiglu];
            [e setBuffer:m->gate offset:0 atIndex:0];
            [e setBytes:mp length:sizeof mp atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake((mlp->intermediate + 127) / 128, tokens, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            ++m->info.dispatches;
            encode_projection(m, e, &d->down, m->gate, 2 * mlp->intermediate, m->branch, h, mlp->intermediate, tokens);
            encode_residual(m, e, d->mlp_post, h, tokens, 1);
        }
        encode_norm(m, e, m->final_norm, h, 1, (tokens - 1) * h * 4);
        encode_projection(m, e, &m->head, m->normalized, h, m->logits, model->vocab, h, 1);
        if (!complete(m, command, e)) return -1;
        memcpy(states, m->state.contents, (size_t)(tokens * h * 4));
        memcpy(model->state, (float *)m->state.contents + (tokens - 1) * h, (size_t)(h * 4));
        memcpy(model->normalized, m->normalized.contents, (size_t)(h * 4));
        memcpy(model->logits, m->logits.contents, (size_t)(model->vocab * 4));
        for (uint64_t i = 0; i < model->count; ++i) {
            struct mixer *mix = model->layers[i]->mixer;
            mix->cache->count += tokens;
            if (mix->cache->count > mix->cache->capacity) mix->cache->count = mix->cache->capacity;
            mix->rotation->processed_count = mix->cache->count;
            mix->rotation->scale_bits = m->ropes[m->layers[i].rope].scale;
        }
        ++m->info.fused_prefills;
        m->info.fused_mlps += model->count;
        return 1;
    }
}

int plamo_metal_mlp(struct plamo_metal *m, const struct mlp *p, float *states, uint64_t tokens) {
    if (!tokens) return 1;
    uint64_t h = p->hidden, n = p->intermediate;
    uint64_t norm_stride = (h * 2 + 15) & ~UINT64_C(15);
    if (!h || !n || h > UINT32_MAX || n > UINT32_MAX / 2 || tokens > UINT32_MAX ||
        tokens > SIZE_MAX / 4 / h || tokens > SIZE_MAX / 8 / n || h > SIZE_MAX / 4 / n) return 0;
    @autoreleasepool {
        if (m->config.memory == PLAMO_METAL_AUTO && now() - m->checked_at >= 0.25) plamo_metal_maintain(m);
        struct projection gate, down;
        if (!resident_projection(m, p->gu, 2 * n, h, &gate) ||
            !resident_projection(m, p->down, h, n, &down)) return 0;
        if (!ensure_buffer(m, &m->state, tokens * h * 4) ||
            !ensure_buffer(m, &m->normalized, tokens * h * 4) ||
            !ensure_buffer(m, &m->gate, tokens * n * 8) ||
            !ensure_buffer(m, &m->branch, tokens * h * 4) ||
            !ensure_buffer(m, &m->norms, norm_stride + h * 2)) return 0;
        memcpy(m->state.contents, states, (size_t)(tokens * h * 4));
        memcpy(m->norms.contents, p->pre, (size_t)(h * 2));
        memcpy((char *)m->norms.contents + norm_stride, p->post, (size_t)(h * 2));
        struct { uint32_t hidden, intermediate, tokens; } params = {(uint32_t)h, (uint32_t)n, (uint32_t)tokens};
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return problem(m, "cannot create a fused Metal MLP command");
        [encoder setComputePipelineState:m->norm];
        [encoder setBuffer:m->state offset:0 atIndex:0];
        [encoder setBuffer:m->norms offset:0 atIndex:1];
        [encoder setBuffer:m->normalized offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        encode_projection(m, encoder, &gate, m->normalized, h, m->gate, 2 * n, h, tokens);
        [encoder setComputePipelineState:m->swiglu];
        [encoder setBuffer:m->gate offset:0 atIndex:0];
        [encoder setBytes:&params length:sizeof params atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake((n + 127) / 128, tokens, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        encode_projection(m, encoder, &down, m->gate, 2 * n, m->branch, h, n, tokens);
        [encoder setComputePipelineState:m->residual];
        [encoder setBuffer:m->state offset:0 atIndex:0];
        [encoder setBuffer:m->norms offset:norm_stride atIndex:1];
        [encoder setBuffer:m->branch offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        m->info.dispatches += 3;
        if (!complete(m, command, encoder)) return 0;
        memcpy(states, m->state.contents, (size_t)(tokens * h * 4));
        ++m->info.fused_mlps;
        return 1;
    }
}

static int mlp(const struct mlp *p, float *states, uint64_t tokens, void *context) {
    return plamo_metal_mlp(context, p, states, tokens);
}
static int forward(struct model *model, uint64_t token, void *context) {
    return plamo_metal_forward(context, model, token);
}
static int prefill(struct model *model, const uint32_t *ids, uint64_t tokens, float *states, void *context) {
    return plamo_metal_prefill(context, model, ids, tokens, states);
}

static void matmul(const struct plamo_matmul *p, void *context) {
    struct plamo_metal *m = context;
    if (!plamo_metal_linear(m, p)) { ++m->info.cpu_fallbacks; plamo_bf16_matmul_core(p); }
}
static void matvec(const uint16_t *w, const float *x, float *y, uint64_t rows, uint64_t columns, void *context) {
    struct plamo_matmul p = {w, x, y, rows, columns, 1, columns, rows};
    matmul(&p, context);
}
// Defer original-byte acquisition until the GPU module can choose a retained
// shared buffer or its shared staging buffer. Never create a CPU staging copy.
static const void *acquire(const void *source, uint64_t bytes, void *context) { (void)bytes; (void)context; return source; }

static int preload(struct plamo_metal *m) {
    for (const struct plamo_weight_spec *s = m->config.schema; s && s->name; ++s) {
        if (s->rank != 2 || !s->slot || !s->shape[0] || !s->shape[1]) continue;
        uint64_t row_bytes = s->shape[1] * 2;
        uint64_t rows = m->staging_bytes / row_bytes;
        if (rows >= 8) rows = rows / 8 * 8;
        if (!rows) return problem(m, "a BF16 tensor row exceeds the Metal read window");
        for (uint64_t first = 0; first < s->shape[0]; first += rows) {
            uint64_t count = s->shape[0] - first;
            if (count > rows) count = rows;
            const void *source = (const char *)*s->slot + first * row_bytes;
            NSUInteger offset;
            unsigned packed;
            id<MTLBuffer> buffer = weights(m, source, count * row_bytes, &offset, 0, &packed);
            if (!buffer || buffer == m->staging) return problem(m, "BF16 matrices do not fit in the Metal residency budget");
            if (!prepare_buffer(m, buffer)) return 0;
        }
    }
    return 1;
}

struct plamo_metal *plamo_metal_open(const struct plamo_metal_config *config, char *error, size_t cap) {
    @autoreleasepool {
        if (!config || !config->available || config->count > PLAMO_MAX_WEIGHT_FILES ||
            (unsigned)config->memory > PLAMO_METAL_STREAM || config->precision != PLAMO_METAL_FP32 ||
            (unsigned)config->packing > PLAMO_PACK_LOSSLESS ||
            plamo_compute_context || plamo_weight_provider) {
            snprintf(error, cap, "invalid Metal configuration or an existing compute owner"); return NULL;
        }
        struct plamo_metal *m = calloc(1, sizeof *m);
        if (!m) { snprintf(error, cap, "cannot allocate Metal state"); return NULL; }
        m->config = *config;
        m->device = MTLCreateSystemDefaultDevice();
        if (!m->device || !m->device.hasUnifiedMemory || ![m->device supportsFamily:MTLGPUFamilyApple7]) {
            snprintf(error, cap, "Metal requires an available Apple Silicon GPU"); goto fail;
        }
        snprintf(m->info.device, sizeof m->info.device, "%s", m->device.name.UTF8String);
        snprintf(m->info.precision, sizeof m->info.precision, "fp32");
        NSError *failure = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_3;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else {
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        }
#endif
        id<MTLLibrary> library = [m->device newLibraryWithSource:[NSString stringWithUTF8String:plamo_metal_source] options:options error:&failure];
        [options release];
        if (!library) { snprintf(error, cap, "Metal shader compilation failed: %s", failure.localizedDescription.UTF8String); goto fail; }
        id<MTLFunction> fn;
        for (unsigned packed = 0; packed < 2; ++packed) {
            fn = [library newFunctionWithName:packed ? @"plamo_matvec_packed" : @"plamo_matvec"];
            m->matvec[packed] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
            fn = [library newFunctionWithName:packed ? @"plamo_matmul_vector_packed" : @"plamo_matmul_vector"];
            m->vector[packed] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
            NSString *names[] = {@"plamo_matmul8", @"plamo_matmul16", @"plamo_matmul_short", @"plamo_matmul"};
            for (unsigned n = 0; n < 4; ++n) {
                fn = [library newFunctionWithName:packed ? [names[n] stringByAppendingString:@"_packed"] : names[n]];
                m->matmul[packed][n] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
            }
        }
        fn = [library newFunctionWithName:@"plamo_mlp_norm"];
        m->norm = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_mlp_swiglu"];
        m->swiglu = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_mlp_residual"];
        m->residual = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_cache_touch"];
        m->touch = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_pack_analyze"];
        m->analyze = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_pack_encode"];
        m->pack = [m->device newComputePipelineStateWithFunction:fn error:&failure];
        [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_keys"];
        m->decode_keys = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_qkv"];
        m->decode_qkv = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_attention"];
        m->decode_attention = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_decode_residual"];
        m->decode_residual = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_prefill_qkv"];
        m->prefill_qkv = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_prefill_attention"];
        m->prefill_attention = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_prefill_store"];
        m->prefill_store = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        [library release];
        m->queue = [m->device newCommandQueue];
        int pipelines = m->norm && m->swiglu && m->residual && m->touch && m->analyze && m->pack && m->queue &&
                        m->decode_keys && m->decode_qkv && m->decode_attention && m->decode_residual &&
                        m->prefill_qkv && m->prefill_attention && m->prefill_store;
        for (unsigned packed = 0; packed < 2; ++packed) {
            pipelines = pipelines && m->matvec[packed] && m->vector[packed] &&
                m->matvec[packed].threadExecutionWidth == 32 && m->vector[packed].threadExecutionWidth == 32 &&
                m->matvec[packed].maxTotalThreadsPerThreadgroup >= 128 && m->vector[packed].maxTotalThreadsPerThreadgroup >= 128;
            for (unsigned n = 0; n < 4; ++n) {
                id<MTLComputePipelineState> p = m->matmul[packed][n];
                pipelines = pipelines && p && p.threadExecutionWidth == 32 && p.maxTotalThreadsPerThreadgroup >= 128 &&
                            p.staticThreadgroupMemoryLength <= m->device.maxThreadgroupMemoryLength;
            }
        }
        if (!pipelines) {
            snprintf(error, cap, "Metal compute pipeline is unavailable: %s", failure ? failure.localizedDescription.UTF8String : "unsupported limits"); goto fail;
        }
        uint64_t total = 0;
        for (size_t i = 0; i < config->count; ++i) {
            if (!config->maps || !config->fds || !config->maps[i].base || !config->maps[i].bytes ||
                config->fds[i] < 0 || config->fds[i] > INT_MAX || UINT64_MAX - total < config->maps[i].bytes) {
                snprintf(error, cap, "invalid Metal weight mapping"); goto fail;
            }
            total += config->maps[i].bytes;
        }
        uint64_t available = config->available();
        uint64_t spare = available > config->reserve ? available - config->reserve : 0;
        m->staging_bytes = spare / 8;
        if (m->staging_bytes > 64 * MIB) m->staging_bytes = 64 * MIB;
        m->staging_bytes = m->staging_bytes / 16384 * 16384;
        if (m->staging_bytes < MIB) m->staging_bytes = MIB;
        uint64_t device_budget = m->device.recommendedMaxWorkingSetSize;
        if (device_budget > 128 * MIB) device_budget -= 128 * MIB;
        m->maximum = spare < device_budget ? spare : device_budget;
        // Keep room for the page padding of individual retained ranges. The
        // budget is an upper bound, not an allocation of the whole amount.
        if (m->maximum < m->staging_bytes) m->maximum = m->staging_bytes;
        m->capacity = config->memory == PLAMO_METAL_STREAM ? m->staging_bytes : m->maximum;
        uint64_t working = config->working_weight_bytes ? config->working_weight_bytes : total;
        m->use_packing = config->memory != PLAMO_METAL_STREAM &&
            (config->packing == PLAMO_PACK_LOSSLESS ||
             (config->packing == PLAMO_PACK_AUTO && working > m->capacity - m->staging_bytes));
        if (config->memory == PLAMO_METAL_RESIDENT &&
            ((!m->use_packing && total > spare) || (m->use_packing && total / 2 > m->capacity - m->staging_bytes))) {
            snprintf(error, cap, "Metal BF16 preload requires weights plus the memory reserve"); goto fail;
        }
        m->staging = [m->device newBufferWithLength:(NSUInteger)m->staging_bytes options:MTLResourceStorageModeShared];
        if (!m->staging) { snprintf(error, cap, "cannot allocate a shared Metal weight window"); goto fail; }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor *descriptor = [[MTLResidencySetDescriptor alloc] init];
            descriptor.initialCapacity = MAX_ENTRIES + 10;
            m->residency = [m->device newResidencySetWithDescriptor:descriptor error:&failure];
            [descriptor release];
            if (!m->residency) { snprintf(error, cap, "cannot create Metal residency set: %s", failure ? failure.localizedDescription.UTF8String : "unavailable"); goto fail; }
            [m->queue addResidencySet:m->residency];
            residency_add(m, m->staging);
            residency_commit(m);
            [m->residency requestResidency];
            m->info.residency_sets = 1;
        }
#endif
        if (config->disk_cache_path && config->memory != PLAMO_METAL_STREAM) {
            m->disk_cache = plamo_disk_cache_open(config->disk_cache_path, config->revision, config->maps,
                                                config->count, m->info.note, sizeof m->info.note);
        }
        m->config.disk_cache_path = NULL; // The path is borrowed only during open.
        for (size_t i = 0; i < config->count; ++i) (void)fcntl((int)config->fds[i], F_NOCACHE, 1);
        m->checked_at = now();
        if (config->memory == PLAMO_METAL_RESIDENT && !preload(m)) { snprintf(error, cap, "%s", m->info.note); goto fail; }
        plamo_weight_context = m;
        plamo_weight_provider = acquire;
        // The GPU module owns row tiling and combines resident tiles into one
        // command. The assembly caller therefore submits complete projections.
        plamo_weight_chunk_bytes = 0;
        plamo_compute_context = m;
        plamo_matvec_kernel = matvec;
        plamo_matmul_kernel = matmul;
        plamo_mlp_kernel = mlp;
        plamo_forward_kernel = forward;
        plamo_prefill_kernel = prefill;
        return m;
fail:
        plamo_metal_close(m);
        return NULL;
    }
}

void plamo_metal_get_info(const struct plamo_metal *m, struct plamo_metal_info *out) {
    *out = m->info;
    out->cache.budget_bytes = m->capacity;
    out->cache.max_budget_bytes = m->maximum;
    out->cache.staging_bytes = m->staging_bytes;
    out->cache.storage_bytes = m->used;
    snprintf(out->weight_storage, sizeof out->weight_storage, "%s", m->info.packed_weight_bytes ? "lossless-bf16" : "bf16");
    out->allocated_bytes = m->device.currentAllocatedSize;
    out->recommended_working_set_bytes = m->device.recommendedMaxWorkingSetSize;
}

void plamo_metal_close(struct plamo_metal *m) {
    if (!m) return;
    if (plamo_compute_context == m) {
        plamo_matvec_kernel = NULL; plamo_matmul_kernel = NULL; plamo_mlp_kernel = NULL; plamo_compute_context = NULL;
        plamo_forward_kernel = NULL;
        plamo_prefill_kernel = NULL;
        plamo_weight_provider = NULL; plamo_weight_context = NULL; plamo_weight_chunk_bytes = 0;
    }
    release_decode(m);
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (m->residency) {
            [m->queue removeResidencySet:m->residency];
            [m->residency endResidency];
            [m->residency removeAllAllocations]; [m->residency commit];
            [m->residency release];
        }
    }
#endif
    for (size_t i = 0; i < m->entries; ++i) [m->table[i].buffer release];
    [m->staging release]; [m->input release]; [m->output release];
    [m->state release]; [m->normalized release]; [m->gate release]; [m->branch release]; [m->norms release];
    for (unsigned packed = 0; packed < 2; ++packed) {
        [m->matvec[packed] release]; [m->vector[packed] release];
        for (unsigned n = 0; n < 4; ++n) [m->matmul[packed][n] release];
    }
    [m->norm release]; [m->swiglu release]; [m->residual release];
    [m->touch release];
    [m->pack_headers release]; [m->analyze release]; [m->pack release];
    [m->decode_keys release]; [m->decode_qkv release]; [m->decode_attention release]; [m->decode_residual release];
    [m->prefill_qkv release]; [m->prefill_attention release]; [m->prefill_store release];
    [m->decode_norms release]; [m->qkv release]; [m->attention release]; [m->scores release]; [m->logits release];
    [m->queue release]; [m->device release];
    plamo_disk_cache_close(m->disk_cache);
    for (size_t i = 0; i < m->config.count; ++i)
        if (m->config.fds && m->config.fds[i] >= 0 && m->config.fds[i] <= INT_MAX) (void)fcntl((int)m->config.fds[i], F_NOCACHE, 0);
    free(m);
}
