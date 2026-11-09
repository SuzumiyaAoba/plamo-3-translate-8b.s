#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_layout_kernels.h"
#include "../runtime/packing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "layout probe failed at %d: %s\n", __LINE__, #x); exit(1); } } while (0)
struct linear { uint32_t rows, columns, tokens, xs, ys, aligned, offset; };
struct gpu {
    id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLLibrary> library;
    id<MTLComputePipelineState> reorder, analyze, pack;
};
struct variant {
    unsigned layout, packed; uint64_t bytes; double prepare_seconds;
    id<MTLBuffer> weights;
    id<MTLComputePipelineState> vector, matrix[4], unpack;
};
enum { LAYOUTS = 6, VARIANTS = LAYOUTS * 2, SAMPLES = 7 };
static unsigned physical_layout(unsigned layout) { return layout == 3 || layout == 4 ? layout - 2 : layout; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t align_page(uint64_t n) { return (n + 16383) / 16384 * 16384; }
static double complete(id<MTLCommandBuffer> command, id<MTLComputeCommandEncoder> encoder) {
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
    CHECK(command.status == MTLCommandBufferStatusCompleted);
    return command.GPUEndTime - command.GPUStartTime;
}
static id<MTLComputePipelineState> pipeline(struct gpu *g, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> fn = [g->library newFunctionWithName:name]; CHECK(fn);
    id<MTLComputePipelineState> p = [g->device newComputePipelineStateWithFunction:fn error:&error]; [fn release];
    if (!p) fprintf(stderr, "%s: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
    CHECK(p && p.threadExecutionWidth == 32); return p;
}
static id<MTLBuffer> reordered(struct gpu *g, id<MTLBuffer> source, uint32_t rows, uint32_t columns, unsigned layout) {
    layout = physical_layout(layout);
    if (!layout) return [source retain];
    id<MTLBuffer> out = [g->device newBufferWithLength:source.length options:MTLResourceStorageModeShared]; CHECK(out);
    uint32_t p[] = {rows, columns, layout == 1 ? 4 : layout == 5 ? 32 : 8, layout == 1 ? 8 : 32};
    id<MTLCommandBuffer> command = [g->queue commandBuffer]; id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
    [e setComputePipelineState:g->reorder]; [e setBuffer:source offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:p length:sizeof p atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(((uint64_t)rows * columns / 4 + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    complete(command, e); return out;
}
static id<MTLBuffer> compressed(struct gpu *g, id<MTLBuffer> source, uint32_t count) {
    uint64_t blocks = ((uint64_t)count + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    id<MTLBuffer> headers = [g->device newBufferWithLength:blocks * 8 options:MTLResourceStorageModeShared]; CHECK(headers);
    id<MTLCommandBuffer> command = [g->queue commandBuffer]; id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
    [e setComputePipelineState:g->analyze]; [e setBuffer:source offset:0 atIndex:0]; [e setBuffer:headers offset:0 atIndex:1];
    [e setBytes:&count length:sizeof count atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; complete(command, e);
    uint64_t bytes = plamo_bf16_pack_offsets(headers.contents, count); CHECK(bytes);
    id<MTLBuffer> out = [g->device newBufferWithLength:align_page(bytes) options:MTLResourceStorageModeShared]; CHECK(out);
    command = [g->queue commandBuffer]; e = [command computeCommandEncoder];
    [e setComputePipelineState:g->pack]; [e setBuffer:source offset:0 atIndex:0]; [e setBuffer:headers offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2]; [e setBytes:&count length:sizeof count atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; complete(command, e);
    [headers release]; return out;
}
static void pipelines(struct gpu *g, struct variant *v) {
    NSString *suffix = v->packed ? @"_packed" : @"";
    NSString *layout = physical_layout(v->layout) == 1 ? @"4x8" : v->layout == 5 ? @"32x32" : @"8x32";
    if (v->layout) {
        v->vector = pipeline(g, [NSString stringWithFormat:@"layout_%@vec_%@%@", v->layout > 2 ? @"simple_" : @"", layout, suffix]);
        for (unsigned n = 0; n < 4; ++n)
            v->matrix[n] = pipeline(g, [NSString stringWithFormat:@"layout_mm%u_%@%@", (n + 1) * 8, layout, suffix]);
    } else {
        v->vector = pipeline(g, [@"plamo_matvec" stringByAppendingString:suffix]);
        NSString *names[] = {@"plamo_matmul8", @"plamo_matmul16", @"plamo_matmul_short", @"plamo_matmul"};
        for (unsigned n = 0; n < 4; ++n) v->matrix[n] = pipeline(g, [names[n] stringByAppendingString:suffix]);
    }
    v->unpack = pipeline(g, [NSString stringWithFormat:@"layout_unpack_%@%@", v->layout ? layout : @"row", suffix]);
}
static void prepare(struct gpu *g, struct variant *v, id<MTLBuffer> source, uint32_t rows, uint32_t cols) {
    double start = now();
    id<MTLBuffer> reordered_weights = reordered(g, source, rows, cols, v->layout);
    v->weights = v->packed ? compressed(g, reordered_weights, rows * cols) : [reordered_weights retain];
    [reordered_weights release]; v->bytes = v->weights.length; v->prepare_seconds = now() - start;
}
static void verify_weights(struct gpu *g, struct variant *v, id<MTLBuffer> source, uint32_t rows, uint32_t cols) {
    struct linear p = {rows, cols, 1, cols, rows, 1, 0};
    id<MTLBuffer> out = [g->device newBufferWithLength:source.length options:MTLResourceStorageModeShared]; CHECK(out);
    id<MTLCommandBuffer> command = [g->queue commandBuffer]; id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
    [e setComputePipelineState:v->unpack]; [e setBuffer:v->weights offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
    [e setBytes:&p length:sizeof p atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(((uint64_t)rows * cols + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    complete(command, e); CHECK(!memcmp(source.contents, out.contents, source.length)); [out release];
}
static double run(struct gpu *g, struct variant *v, id<MTLBuffer> input, id<MTLBuffer> output,
                  uint32_t rows, uint32_t cols, uint32_t tokens, unsigned repeat, double *wall) {
    double start = now();
    struct linear p = {rows, cols, tokens, cols, rows, 1, 0};
    uint32_t panels = (tokens + 31) / 32, tile = ((tokens + panels - 1) / panels + 7) / 8 * 8;
    id<MTLComputePipelineState> state = tokens == 1 ? v->vector : v->matrix[tile / 8 - 1];
    unsigned threads = tokens == 1 && v->layout == 2 ? 256 : 128;
    unsigned row_tile = tokens == 1 ? (v->layout == 2 ? 8 : 4) : 32;
    MTLSize grid = MTLSizeMake(rows / row_tile, tokens == 1 ? tokens : (tokens + tile - 1) / tile, 1);
    id<MTLCommandBuffer> command = [g->queue commandBuffer]; id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
    [e setComputePipelineState:state]; [e setBuffer:v->weights offset:0 atIndex:0];
    [e setBuffer:input offset:0 atIndex:1]; [e setBuffer:output offset:0 atIndex:2]; [e setBytes:&p length:sizeof p atIndex:3];
    for (unsigned i = 0; i < repeat; ++i) [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    double seconds = complete(command, e); *wall = (now() - start) / repeat; return seconds / repeat;
}
static int compare(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
    int validate = argc == 4 && !strcmp(argv[3], "--validate");
    if (argc != 3 && !validate) { fprintf(stderr, "usage: bench-metal-layout SAMPLE.bf16 COLUMNS [--validate]\n"); return 64; }
    @autoreleasepool {
        struct gpu g = {0}; g.device = MTLCreateSystemDefaultDevice();
        if (!g.device || !g.device.hasUnifiedMemory || ![g.device supportsFamily:MTLGPUFamilyApple7]) { fprintf(stderr, "Apple Silicon GPU required\n"); return 77; }
        NSError *error = nil; MTLCompileOptions *options = [[MTLCompileOptions alloc] init]; options.languageVersion = MTLLanguageVersion2_3;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        g.library = [g.device newLibraryWithSource:[NSString stringWithUTF8String:plamo_metal_source] options:options error:&error];
        [options release]; if (!g.library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); CHECK(g.library);
        g.queue = [g.device newCommandQueue]; CHECK(g.queue);
        g.reorder = pipeline(&g, @"layout_reorder"); g.analyze = pipeline(&g, @"plamo_pack_analyze"); g.pack = pipeline(&g, @"plamo_pack_encode");
        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]]; CHECK(data.length && data.length < UINT32_MAX);
        uint32_t cols = (uint32_t)strtoul(argv[2], NULL, 10); CHECK(cols && cols % 256 == 0);
        uint32_t rows = (uint32_t)(data.length / 2 / cols); CHECK(rows && rows % 32 == 0 && (uint64_t)rows * cols * 2 == data.length);
        id<MTLBuffer> raw = [g.device newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared]; CHECK(raw);
        struct variant variants[VARIANTS] = {0};
        for (unsigned i = 0; i < VARIANTS; ++i) {
            variants[i].layout = i % LAYOUTS; variants[i].packed = i / LAYOUTS; pipelines(&g, &variants[i]);
            prepare(&g, &variants[i], raw, rows, cols); verify_weights(&g, &variants[i], raw, rows, cols);
        }
        // The inverse permutation and compressed path preserve all 65536 BF16
        // patterns, including nonfinite values. Matmul below uses real weights.
        id<MTLBuffer> bits = [g.device newBufferWithLength:65536 * 2 options:MTLResourceStorageModeShared]; CHECK(bits);
        for (unsigned i = 0; i < 65536; ++i) ((uint16_t *)bits.contents)[i] = (uint16_t)(i * 40503u + 17u);
        for (unsigned i = 0; i < VARIANTS; ++i) {
            struct variant test = variants[i]; prepare(&g, &test, bits, 256, 256); verify_weights(&g, &test, bits, 256, 256); [test.weights release];
        }
        [bits release]; fprintf(stderr, "BF16 layouts: PASS (full real sample and all 65536 bit patterns)\n");
        const uint32_t counts[] = {1, 8, 9, 21, 69, 81};
        id<MTLBuffer> input = [g.device newBufferWithLength:(uint64_t)cols * 81 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [g.device newBufferWithLength:(uint64_t)rows * 81 * 4 options:MTLResourceStorageModeShared]; CHECK(input && output);
        for (uint64_t i = 0; i < (uint64_t)cols * 81; ++i) ((float *)input.contents)[i] = ((int)(i % 103) - 51) / 31.f;
        void *reference = malloc((size_t)rows * 81 * 4); CHECK(reference);
        for (unsigned ci = 0; ci < sizeof counts / sizeof *counts; ++ci) {
            uint32_t tokens = counts[ci]; double unused;
            run(&g, &variants[0], input, output, rows, cols, tokens, 1, &unused); memcpy(reference, output.contents, (size_t)rows * tokens * 4);
            for (unsigned i = 0; i < VARIANTS; ++i) {
                run(&g, &variants[i], input, output, rows, cols, tokens, 1, &unused);
                if (memcmp(reference, output.contents, (size_t)rows * tokens * 4)) fprintf(stderr, "output mismatch layout=%u packed=%u tokens=%u\n", variants[i].layout, variants[i].packed, tokens);
                CHECK(!memcmp(reference, output.contents, (size_t)rows * tokens * 4));
            }
            if (validate) { fprintf(stderr, "FP32 layouts: PASS (%u tokens; raw/lossless, all layouts)\n", tokens); continue; }
            double pilot = run(&g, &variants[0], input, output, rows, cols, tokens, 5, &unused);
            CHECK(pilot > 0.0);
            unsigned repeat = (unsigned)fmin(128, fmax(5, ceil(0.03 / pilot)));
            for (unsigned warmup = 0; warmup < 3; ++warmup) for (unsigned i = 0; i < VARIANTS; ++i)
                @autoreleasepool { run(&g, &variants[i], input, output, rows, cols, tokens, repeat, &unused); }
            double gpu[VARIANTS][SAMPLES], wall[VARIANTS][SAMPLES];
            for (unsigned trial = 0; trial < SAMPLES; ++trial) for (unsigned position = 0; position < VARIANTS; ++position) {
                unsigned i = (position + trial) % VARIANTS;
                @autoreleasepool { gpu[i][trial] = run(&g, &variants[i], input, output, rows, cols, tokens, repeat, &wall[i][trial]); }
            }
            for (unsigned i = 0; i < VARIANTS; ++i) {
                qsort(gpu[i], SAMPLES, sizeof(double), compare); qsort(wall[i], SAMPLES, sizeof(double), compare);
                printf("{\"device\":\"%s\",\"rows\":%u,\"columns\":%u,\"tokens\":%u,\"layout\":%u,\"lossless\":%s,\"weight_bytes\":%llu,\"storage_bytes\":%llu,\"prepare_seconds\":%.9f,\"batch_repeats\":%u,\"gpu_seconds\":%.9f,\"gpu_q1_seconds\":%.9f,\"gpu_q3_seconds\":%.9f,\"wall_seconds\":%.9f,\"bf16_roundtrip\":true,\"output_bitwise_equal\":true}\n", g.device.name.UTF8String, rows, cols, tokens, variants[i].layout, variants[i].packed ? "true" : "false", (unsigned long long)data.length, (unsigned long long)variants[i].bytes, variants[i].prepare_seconds, repeat, gpu[i][SAMPLES / 2], gpu[i][SAMPLES / 4], gpu[i][SAMPLES * 3 / 4], wall[i][SAMPLES / 2]); fflush(stdout);
            }
        }
        free(reference); [raw release]; [input release]; [output release];
        for (unsigned i = 0; i < VARIANTS; ++i) {
            [variants[i].weights release]; [variants[i].vector release]; [variants[i].unpack release];
            for (unsigned n = 0; n < 4; ++n) [variants[i].matrix[n] release];
        }
        [g.reorder release]; [g.analyze release]; [g.pack release]; [g.queue release]; [g.library release]; [g.device release];
        return 0;
    }
}
