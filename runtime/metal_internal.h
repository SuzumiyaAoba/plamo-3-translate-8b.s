#ifndef PLAMO_METAL_INTERNAL_H
#define PLAMO_METAL_INTERNAL_H
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal.h"
#include "compute.h"
#include "disk_cache.h"
#include <time.h>

/* Backend-private state shared by device/projection, weight and model modules.
   Public callers only see metal.h. Graph encoding never acquires/evicts weights;
   every view must be resolved before state or KV can change. */
#define MIB (UINT64_C(1024) * 1024)
#define MAX_ENTRIES 4096
struct entry { uintptr_t source; uint64_t bytes, allocated; id<MTLBuffer> buffer; unsigned prepared, packed; };
struct packing_size { uintptr_t source; uint64_t bytes, allocated; unsigned packed; };
struct mapped_weight { const void *data; uint64_t bytes; id<MTLBuffer> buffer; };
struct tile { id<MTLBuffer> buffer; NSUInteger offset; uint64_t first, rows, mapped_bytes; unsigned packed; const void *mapped_data; };
struct projection { struct tile tiles[128]; size_t count; };
struct view { id<MTLBuffer> buffer; void *source; NSUInteger offset; uint64_t bytes; };
struct decode_layer {
    struct projection qkv, attention, gate, down;
    struct view k, v, rotated;
    id<MTLBuffer> proposal, proposal_rotated;
    uint64_t verify_prefix;
    NSUInteger pre, post, mlp_pre, mlp_post, qn, kn;
    size_t rope;
};
struct rope_table { const float *frequency; uint64_t dim, positions; uint32_t scale; id<MTLBuffer> buffer; };
struct plamo_metal {
    struct plamo_metal_config config;
    struct plamo_metal_info info;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> matvec[3], vector[3], matmul[3][4], norm, swiglu, residual, touch, analyze, pack, quantize4;
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
    unsigned residency_dirty, use_packing, cache_mappings, verify_encoding, draft_quantized;
    uint64_t verification_tokens, verification_index;
    uint64_t maximum, capacity, used, staging_bytes;
    size_t entries;
    size_t packing_sizes;
    size_t mapped_count;
    double checked_at;
    struct entry table[MAX_ENTRIES];
    struct packing_size sizes[MAX_ENTRIES];
    struct mapped_weight mapped[MAX_ENTRIES];
    struct plamo_disk_cache *disk_cache;
};

static inline double plamo_metal_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static inline uint64_t plamo_metal_aligned(uint64_t bytes) { return (bytes + 16383) / 16384 * 16384; }

void plamo_metal_residency_add(struct plamo_metal *, id<MTLBuffer>);
void plamo_metal_residency_remove(struct plamo_metal *, id<MTLBuffer>);
void plamo_metal_residency_commit(struct plamo_metal *);
int plamo_metal_problem(struct plamo_metal *, const char *);
int plamo_metal_ensure_buffer(struct plamo_metal *, id<MTLBuffer> *, uint64_t);
int plamo_metal_complete(struct plamo_metal *, id<MTLCommandBuffer>, id<MTLComputeCommandEncoder>);
id<MTLBuffer> plamo_metal_weights(struct plamo_metal *, const void *, uint64_t, NSUInteger *, int, unsigned *, int);
id<MTLBuffer> plamo_metal_mapped_weights(struct plamo_metal *, const void *, uint64_t);
void plamo_metal_release_mappings(struct plamo_metal *);
int plamo_metal_preload_schema(struct plamo_metal *);
int plamo_metal_resolve_projection(struct plamo_metal *, const uint16_t *, uint64_t, uint64_t, struct projection *, int);
void plamo_metal_encode_projection(struct plamo_metal *, id<MTLComputeCommandEncoder>, const struct projection *,
                                   id<MTLBuffer>, uint64_t, id<MTLBuffer>, uint64_t, uint64_t, uint64_t);
void plamo_metal_encode_linear(struct plamo_metal *, id<MTLComputeCommandEncoder>, id<MTLBuffer>, NSUInteger,
                              unsigned, id<MTLBuffer>, uint64_t, id<MTLBuffer>, NSUInteger, uint64_t,
                              uint64_t, uint64_t, uint64_t);
void plamo_metal_release_decode(struct plamo_metal *);
#endif
