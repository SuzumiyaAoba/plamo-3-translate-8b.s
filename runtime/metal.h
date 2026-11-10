#ifndef PLAMO_METAL_H
#define PLAMO_METAL_H
#include "parallel.h"
#include "weight_cache.h"

enum plamo_metal_memory { PLAMO_METAL_AUTO, PLAMO_METAL_RESIDENT, PLAMO_METAL_STREAM };
enum plamo_metal_precision { PLAMO_METAL_FP32 };
enum plamo_metal_packing { PLAMO_PACK_AUTO, PLAMO_PACK_RAW, PLAMO_PACK_LOSSLESS };
struct plamo_metal_config {
    enum plamo_metal_memory memory;
    enum plamo_metal_precision precision;
    enum plamo_metal_packing packing;
    uint64_t working_weight_bytes;
    const struct plamo_mapping *maps;
    const int64_t *fds;
    size_t count;
    const struct plamo_weight_spec *schema;
    uint64_t reserve;
    uint64_t (*available)(void);
    const char *disk_cache_path, *revision;
};
struct plamo_metal_info {
    char device[128], precision[16], weight_storage[24], note[256];
    struct plamo_weight_cache_info cache;
    uint64_t dispatches, command_buffers, fused_mlps, fused_decodes, fused_prefills, cpu_fallbacks, errors;
    uint64_t allocated_bytes, recommended_working_set_bytes;
    uint64_t packed_weight_bytes;
    uint64_t disk_cache_hits, disk_cache_weight_bytes, disk_cache_storage_bytes;
    uint64_t disk_cache_mapped_bytes;
    unsigned residency_sets;
    double gpu_seconds;
};
struct plamo_metal;
/* Installs one process-owned scheduler and weight provider after successful
   initialization. Missing devices and incomplete preloads leave CPU hooks alone. */
struct plamo_metal *plamo_metal_open(const struct plamo_metal_config *, char *, size_t);
void plamo_metal_close(struct plamo_metal *);
void plamo_metal_maintain(struct plamo_metal *);
void plamo_metal_get_info(const struct plamo_metal *, struct plamo_metal_info *);
/* Retain only ranges that fit and finish each range's first GPU page access
   before loading the next one. Skip others without staging reads. Zero reports
   a read/GPU failure; all original BF16 bytes are preserved. */
int plamo_metal_preload(struct plamo_metal *, const void *, uint64_t bytes);
/* Finish the GPU's first access to retained pages before readiness; no model
   operations or file reads. Skips an empty cache and honors memory pressure. */
int plamo_metal_warm_cache(struct plamo_metal *);
/* Allocate reusable decode workspace, copy small normalization weights and
   build exact CPU RoPE tables before readiness. The model retains its KV
   storage; GPU views share those same pages without a second KV cache. */
int plamo_metal_prepare_model(struct plamo_metal *, struct model *);
/* One complete FP32 decode, including attention/KV. 0 means a missing retained
   projection and leaves all model state untouched; -1 is an execution failure.
   On success the ordinary CPU state, logits and KV remain immediately usable. */
int plamo_metal_forward(struct plamo_metal *, struct model *, uint64_t token);
/* Complete layer-major prompt processing with one GPU submission. Shares the
   canonical KV and returns every prompt state plus final state/logits. Like
   forward, 0 declines without state changes and -1 must not be retried. */
int plamo_metal_prefill(struct plamo_metal *, struct model *, const uint32_t *, uint64_t tokens, float *states);
/* Synchronous dense projection; 0 leaves the caller free to use its CPU kernel.
   Input/output must not overlap. Arbitrary FP32 strides and BF16 row/column
   tails are supported. */
int plamo_metal_linear(struct plamo_metal *, const struct plamo_matmul *);
/* Executes a complete resident MLP in one command buffer. A cache miss leaves
   states untouched, so the ordinary projection path can populate the cache. */
int plamo_metal_mlp(struct plamo_metal *, const struct mlp *, float *states, uint64_t tokens);
#endif
