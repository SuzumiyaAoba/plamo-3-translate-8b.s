#include "metal_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "metal_kernels.h"
#include <fcntl.h>
#include <limits.h>

void plamo_metal_residency_add(struct plamo_metal *m, id<MTLBuffer> buffer) {
    if (!buffer) return;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        [m->residency addAllocation:buffer]; m->residency_dirty = 1;
    }
#else
    (void)m; (void)buffer;
#endif
}

void plamo_metal_residency_remove(struct plamo_metal *m, id<MTLBuffer> buffer) {
    if (!buffer) return;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        [m->residency removeAllocation:buffer]; m->residency_dirty = 1;
    }
#else
    (void)m; (void)buffer;
#endif
}

void plamo_metal_residency_commit(struct plamo_metal *m) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        if (m->residency_dirty) { [m->residency commit]; m->residency_dirty = 0; }
    }
#else
    (void)m;
#endif
}

int plamo_metal_problem(struct plamo_metal *m, const char *message) {
    ++m->info.errors;
    snprintf(m->info.note, sizeof m->info.note, "%s", message);
    return 0;
}

int plamo_metal_ensure_buffer(struct plamo_metal *m, id<MTLBuffer> *buffer, uint64_t bytes) {
    if (*buffer && (*buffer).length >= bytes) return 1;
    if (bytes > m->device.maxBufferLength) return plamo_metal_problem(m, "Metal activation buffer exceeds the device limit");
    id<MTLBuffer> replacement = [m->device newBufferWithLength:(NSUInteger)plamo_metal_aligned(bytes) options:MTLResourceStorageModeShared];
    if (!replacement) return plamo_metal_problem(m, "cannot allocate Metal activation buffer");
    int replacing = *buffer != nil;
    if (*buffer) plamo_metal_residency_remove(m, *buffer);
    [*buffer release];
    *buffer = replacement;
    plamo_metal_residency_add(m, replacement);
    if (replacing) plamo_metal_residency_commit(m);
    return 1;
}

void plamo_metal_encode_linear(struct plamo_metal *m, id<MTLComputeCommandEncoder> encoder,
                          id<MTLBuffer> w, NSUInteger offset, unsigned packed, id<MTLBuffer> x, uint64_t xs,
                          id<MTLBuffer> y, NSUInteger output_offset, uint64_t ys,
                          uint64_t rows, uint64_t columns, uint64_t tokens) {
    struct { uint32_t rows, columns, tokens, xs, ys, aligned_weights, weight_offset; } params = {
        (uint32_t)rows, (uint32_t)columns, (uint32_t)tokens, (uint32_t)xs, (uint32_t)ys,
        !(offset % 8), packed ? (uint32_t)(offset / 2) : 0
    };
    int batched = tokens >= 8 || (m->verify_encoding && tokens > 1), vector = tokens >= 4 && !batched;
    // Keep the minimum number of weight scans, then choose the narrowest
    // compiled tile: e.g. 69 tokens use 3 x 24 rather than 3 x 32.
    uint64_t panels = (tokens + 31) / 32;
    uint64_t tile_tokens = ((tokens + panels - 1) / panels + 7) / 8 * 8;
    if (m->split_k && batched && tokens <= 32 && columns >= 8192 && !(columns % 128) && rows <= 4096) {
        params.ys = (uint32_t)rows;
        [encoder setComputePipelineState:m->matmul_split[packed][tile_tokens / 8 - 1]];
        [encoder setBuffer:w offset:packed ? 0 : offset atIndex:0];
        [encoder setBuffer:x offset:0 atIndex:1];
        [encoder setBuffer:m->split offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 31) / 32, (tokens + tile_tokens - 1) / tile_tokens, 4) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        params.ys = (uint32_t)ys;
        [encoder setComputePipelineState:m->split_reduce];
        [encoder setBuffer:m->split offset:0 atIndex:0];
        [encoder setBuffer:y offset:output_offset atIndex:1];
        [encoder setBytes:&params length:sizeof params atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake((rows * tokens + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        m->info.dispatches += 2;
        return;
    }
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

int plamo_metal_complete(struct plamo_metal *m, id<MTLCommandBuffer> command, id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding];
    // Batch initial weight registrations and new activation buffers. No GPU
    // work is submitted before the complete set of changes is committed.
    plamo_metal_residency_commit(m);
    [command commit];
    [command waitUntilCompleted];
    ++m->info.command_buffers;
    if (command.status != MTLCommandBufferStatusCompleted)
        return plamo_metal_problem(m, command.error ? command.error.localizedDescription.UTF8String : "Metal command failed");
    m->info.gpu_seconds += command.GPUEndTime - command.GPUStartTime;
    return 1;
}

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
            if (m->config.memory == PLAMO_METAL_AUTO && plamo_metal_now() - m->checked_at >= 0.25) plamo_metal_maintain(m);
            if (!plamo_metal_resolve_projection(m, p->weights, p->rows, p->columns, &projection, 0)) {
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
        if (!plamo_metal_ensure_buffer(m, &m->input, p->tokens * p->columns * 4) ||
            !plamo_metal_ensure_buffer(m, &m->output, p->tokens * p->rows * 4)) return 0;
        NSUInteger offset = 0;
        unsigned packed = 0;
        id<MTLBuffer> w = whole ? nil : plamo_metal_weights(m, p->weights, p->rows * p->columns * 2, &offset, 0, &packed, p->tokens == 1);
        if (!whole && !w) return 0;
        for (uint64_t t = 0; t < p->tokens; ++t)
            memcpy((float *)m->input.contents + t * p->columns, p->input + t * p->input_stride, (size_t)p->columns * 4);
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return plamo_metal_problem(m, "cannot create a Metal compute command");
        if (whole) plamo_metal_encode_projection(m, encoder, &projection, m->input, p->columns, m->output, p->rows, p->columns, p->tokens);
        else plamo_metal_encode_linear(m, encoder, w, offset, packed, m->input, p->columns, m->output, 0, p->rows,
                           p->rows, p->columns, p->tokens);
        if (!plamo_metal_complete(m, command, encoder)) return 0;
        for (uint64_t t = 0; t < p->tokens; ++t)
            memcpy(p->output + t * p->output_stride, (float *)m->output.contents + t * p->rows, (size_t)p->rows * 4);
        return 1;
    }
}

// Resolve the complete graph before changing model state. Uncached file views
// live in the enclosing autorelease pool until its GPU command is complete.
// Never refer to staging or evict a retained entry during command encoding.
int plamo_metal_mlp(struct plamo_metal *m, const struct mlp *p, float *states, uint64_t tokens) {
    m->verification_tokens = 0;
    if (!tokens) return 1;
    uint64_t h = p->hidden, n = p->intermediate;
    uint64_t norm_stride = (h * 2 + 15) & ~UINT64_C(15);
    if (!h || !n || h > UINT32_MAX || n > UINT32_MAX / 2 || tokens > UINT32_MAX ||
        tokens > SIZE_MAX / 4 / h || tokens > SIZE_MAX / 8 / n || h > SIZE_MAX / 4 / n) return 0;
    @autoreleasepool {
        if (m->config.memory == PLAMO_METAL_AUTO && plamo_metal_now() - m->checked_at >= 0.25) plamo_metal_maintain(m);
        struct projection gate, down;
        if (!plamo_metal_resolve_projection(m, p->gu, 2 * n, h, &gate, 0) ||
            !plamo_metal_resolve_projection(m, p->down, h, n, &down, 0)) return 0;
        if (!plamo_metal_ensure_buffer(m, &m->state, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->normalized, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->gate, tokens * n * 8) ||
            !plamo_metal_ensure_buffer(m, &m->branch, tokens * h * 4) ||
            !plamo_metal_ensure_buffer(m, &m->norms, norm_stride + h * 2)) return 0;
        memcpy(m->state.contents, states, (size_t)(tokens * h * 4));
        memcpy(m->norms.contents, p->pre, (size_t)(h * 2));
        memcpy((char *)m->norms.contents + norm_stride, p->post, (size_t)(h * 2));
        struct { uint32_t hidden, intermediate, tokens; } params = {(uint32_t)h, (uint32_t)n, (uint32_t)tokens};
        id<MTLCommandBuffer> command = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return plamo_metal_problem(m, "cannot create a fused Metal MLP command");
        [encoder setComputePipelineState:m->norm];
        [encoder setBuffer:m->state offset:0 atIndex:0];
        [encoder setBuffer:m->norms offset:0 atIndex:1];
        [encoder setBuffer:m->normalized offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        plamo_metal_encode_projection(m, encoder, &gate, m->normalized, h, m->gate, 2 * n, h, tokens);
        [encoder setComputePipelineState:m->swiglu];
        [encoder setBuffer:m->gate offset:0 atIndex:0];
        [encoder setBytes:&params length:sizeof params atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake((n + 127) / 128, tokens, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        plamo_metal_encode_projection(m, encoder, &down, m->gate, 2 * n, m->branch, h, n, tokens);
        [encoder setComputePipelineState:m->residual];
        [encoder setBuffer:m->state offset:0 atIndex:0];
        [encoder setBuffer:m->norms offset:norm_stride atIndex:1];
        [encoder setBuffer:m->branch offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof params atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        m->info.dispatches += 3;
        if (!plamo_metal_complete(m, command, encoder)) return 0;
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

struct plamo_metal *plamo_metal_open(const struct plamo_metal_config *config, char *error, size_t cap) {
    @autoreleasepool {
        if (!config || !config->available || config->count > PLAMO_MAX_WEIGHT_FILES ||
            (unsigned)config->memory > PLAMO_METAL_STREAM || config->precision != PLAMO_METAL_FP32 ||
            (unsigned)config->packing > PLAMO_PACK_DRAFT_Q4 ||
            plamo_compute_context || plamo_weight_provider) {
            snprintf(error, cap, "invalid Metal configuration or an existing compute owner"); return NULL;
        }
        struct plamo_metal *m = calloc(1, sizeof *m);
        if (!m) { snprintf(error, cap, "cannot allocate Metal state"); return NULL; }
        m->config = *config;
        m->draft_quantized = config->packing == PLAMO_PACK_DRAFT_Q4 && config->memory != PLAMO_METAL_STREAM;
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
        for (unsigned packed = 0; packed < (m->draft_quantized ? 3u : 2u); ++packed) {
            NSString *suffix = packed == 2 ? @"_q4" : packed ? @"_packed" : @"";
            fn = [library newFunctionWithName:[@"plamo_matvec" stringByAppendingString:suffix]];
            m->matvec[packed] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
            fn = [library newFunctionWithName:[@"plamo_matmul_vector" stringByAppendingString:suffix]];
            m->vector[packed] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
            NSString *names[] = {@"plamo_matmul8", @"plamo_matmul16", @"plamo_matmul_short", @"plamo_matmul"};
            for (unsigned n = 0; n < 4; ++n) {
                fn = [library newFunctionWithName:[names[n] stringByAppendingString:suffix]];
                m->matmul[packed][n] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
                fn = [library newFunctionWithName:[NSString stringWithFormat:@"plamo_matmul%u_split%@", (n + 1) * 8, suffix]];
                m->matmul_split[packed][n] = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
            }
        }
        fn = [library newFunctionWithName:@"plamo_split_reduce"];
        m->split_reduce = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        const char *split_k = getenv("PLAMO_METAL_SPLIT_K");
        m->split_k = split_k && !strcmp(split_k, "4");
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
        if (m->draft_quantized) {
            const char *mse = getenv("PLAMO_DRAFT_Q4_MSE");
            fn = [library newFunctionWithName:mse && !strcmp(mse, "1") ? @"plamo_draft_quantize4_mse" : @"plamo_draft_quantize4"];
            m->quantize4 = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        }
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
        fn = [library newFunctionWithName:@"plamo_verify_scores"];
        m->verify_scores = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_verify_softmax"];
        m->verify_softmax = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        fn = [library newFunctionWithName:@"plamo_verify_values"];
        m->verify_values = [m->device newComputePipelineStateWithFunction:fn error:&failure]; [fn release];
        [library release];
        m->queue = [m->device newCommandQueue];
        int pipelines = (!m->draft_quantized || m->quantize4) && m->norm && m->swiglu && m->residual && m->touch && m->analyze && m->pack && m->queue &&
                        m->decode_keys && m->decode_qkv && m->decode_attention && m->decode_residual &&
                        m->prefill_qkv && m->prefill_attention && m->prefill_store &&
                        m->verify_scores && m->verify_softmax && m->verify_values;
        for (unsigned packed = 0; packed < (m->draft_quantized ? 3u : 2u); ++packed) {
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
        const char *raw_mappings = getenv("PLAMO_METAL_RAW_MMAP");
        m->raw_mappings = raw_mappings && !strcmp(raw_mappings, "1") &&
            config->memory != PLAMO_METAL_STREAM && !m->draft_quantized;
        if (m->raw_mappings) m->use_packing = 0;
        if (config->memory == PLAMO_METAL_RESIDENT &&
            ((!m->use_packing && !m->draft_quantized && total > spare) || (m->use_packing && total / 2 > m->capacity - m->staging_bytes))) {
            snprintf(error, cap, "Metal BF16 preload requires weights plus the memory reserve"); goto fail;
        }
        m->staging = [m->device newBufferWithLength:(NSUInteger)m->staging_bytes options:MTLResourceStorageModeShared];
        if (!m->staging) { snprintf(error, cap, "cannot allocate a shared Metal weight window"); goto fail; }
        if (m->split_k && !plamo_metal_ensure_buffer(m, &m->split, 2 * MIB)) { snprintf(error, cap, "cannot allocate split-K workspace"); goto fail; }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor *descriptor = [[MTLResidencySetDescriptor alloc] init];
            descriptor.initialCapacity = MAX_ENTRIES + 10;
            m->residency = [m->device newResidencySetWithDescriptor:descriptor error:&failure];
            [descriptor release];
            if (!m->residency) { snprintf(error, cap, "cannot create Metal residency set: %s", failure ? failure.localizedDescription.UTF8String : "unavailable"); goto fail; }
            [m->queue addResidencySet:m->residency];
            plamo_metal_residency_add(m, m->staging);
            plamo_metal_residency_commit(m);
            [m->residency requestResidency];
            m->info.residency_sets = 1;
        }
#endif
        if (config->disk_cache_path && config->memory != PLAMO_METAL_STREAM && !m->draft_quantized && !m->raw_mappings) {
            m->disk_cache = plamo_disk_cache_open(config->disk_cache_path, config->revision, config->maps,
                                                config->count, m->info.note, sizeof m->info.note);
            if (config->packing == PLAMO_PACK_AUTO && plamo_disk_cache_packed(m->disk_cache)) m->use_packing = 1;
        }
        m->config.disk_cache_path = NULL; // The path is borrowed only during open.
        for (size_t i = 0; i < config->count; ++i) (void)fcntl((int)config->fds[i], F_NOCACHE, 1);
        m->checked_at = plamo_metal_now();
        if (config->memory == PLAMO_METAL_RESIDENT && !m->raw_mappings && !plamo_metal_preload_schema(m)) { snprintf(error, cap, "%s", m->info.note); goto fail; }
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
    snprintf(out->weight_storage, sizeof out->weight_storage, "%s", m->draft_quantized ? "q4-draft" : m->raw_mappings ? "mapped-bf16" : m->info.packed_weight_bytes || m->info.disk_cache_mapped_bytes ? "lossless-bf16" : "bf16");
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
    plamo_metal_release_decode(m);
    plamo_metal_release_mappings(m);
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
    for (unsigned packed = 0; packed < (m->draft_quantized ? 3u : 2u); ++packed) {
        [m->matvec[packed] release]; [m->vector[packed] release];
        for (unsigned n = 0; n < 4; ++n) { [m->matmul[packed][n] release]; [m->matmul_split[packed][n] release]; }
    }
    [m->norm release]; [m->swiglu release]; [m->residual release];
    [m->touch release];
    [m->pack_headers release]; [m->analyze release]; [m->pack release]; [m->quantize4 release];
    [m->decode_keys release]; [m->decode_qkv release]; [m->decode_attention release]; [m->decode_residual release];
    [m->prefill_qkv release]; [m->prefill_attention release]; [m->prefill_store release];
    [m->verify_scores release]; [m->verify_softmax release]; [m->verify_values release];
    [m->decode_norms release]; [m->qkv release]; [m->attention release]; [m->scores release]; [m->logits release];
    [m->split release]; [m->split_reduce release];
    [m->queue release]; [m->device release];
    plamo_disk_cache_close(m->disk_cache);
    for (size_t i = 0; i < m->config.count; ++i)
        if (m->config.fds && m->config.fds[i] >= 0 && m->config.fds[i] <= INT_MAX) (void)fcntl((int)m->config.fds[i], F_NOCACHE, 0);
    free(m);
}
