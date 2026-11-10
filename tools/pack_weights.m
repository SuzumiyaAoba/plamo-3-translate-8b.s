#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_kernels.h"
#include "../runtime/disk_cache.h"
#include "../runtime/packing.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "weight conversion failed at %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static uint64_t rounded(uint64_t n) { return (n + PLAMO_DISK_ALIGNMENT - 1) / PLAMO_DISK_ALIGNMENT * PLAMO_DISK_ALIGNMENT; }
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    NSError *error = nil; id<MTLFunction> fn = [library newFunctionWithName:name]; CHECK(fn);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error]; [fn release];
    if (!p) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); CHECK(p); return p;
}
static void complete(id<MTLCommandBuffer> c, id<MTLComputeCommandEncoder> e) {
    [e endEncoding]; [c commit]; [c waitUntilCompleted];
    if (c.status != MTLCommandBufferStatusCompleted) fprintf(stderr, "%s\n", c.error.localizedDescription.UTF8String);
    CHECK(c.status == MTLCommandBufferStatusCompleted);
}
static void read_all(int fd, void *data, size_t bytes, uint64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        size_t n = bytes - done; if (n > 8 * 1024 * 1024) n = 8 * 1024 * 1024;
        ssize_t got = pread(fd, (char *)data + done, n, (off_t)(offset + done));
        if (got < 0 && errno == EINTR) continue;
        CHECK(got > 0); done += got;
    }
}
static void write_all(int fd, const void *data, size_t bytes, uint64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, (const char *)data + done, bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        CHECK(n > 0); done += n;
    }
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: pack-metal-weights PLAN.json NEW_OUTPUT\n"); return 64; }
    @autoreleasepool {
        NSData *plan = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]]; CHECK(plan);
        NSError *error = nil;
        NSDictionary *p = [NSJSONSerialization JSONObjectWithData:plan options:0 error:&error]; CHECK([p isKindOfClass:NSDictionary.class]);
        NSArray *sources = p[@"sources"], *chunks = p[@"chunks"]; NSString *revision = p[@"revision"];
        CHECK(sources.count && sources.count <= PLAMO_MAX_WEIGHT_FILES && chunks.count && chunks.count <= PLAMO_DISK_MAX_ENTRIES);
        CHECK(revision.length && strlen(revision.UTF8String) < 64);
        unsigned packed = [p[@"packed"] boolValue];
        int fds[PLAMO_MAX_WEIGHT_FILES]; struct plamo_disk_header header = {0};
        memcpy(header.magic, PLAMO_DISK_MAGIC, 8); header.version = 1; header.alignment = PLAMO_DISK_ALIGNMENT;
        header.sources = (uint32_t)sources.count; header.entries = (uint32_t)chunks.count;
        strcpy(header.revision, revision.UTF8String);
        for (unsigned i = 0; i < sources.count; ++i) {
            fds[i] = open([sources[i][@"path"] fileSystemRepresentation], O_RDONLY); CHECK(fds[i] >= 0);
            struct stat st; CHECK(!fstat(fds[i], &st)); header.source_bytes[i] = [sources[i][@"bytes"] unsignedLongLongValue];
            CHECK(st.st_size > 0 && (uint64_t)st.st_size == header.source_bytes[i]);
            CHECK(!fcntl(fds[i], F_NOCACHE, 1));
        }
        struct plamo_disk_entry *entries = calloc(chunks.count, sizeof *entries); CHECK(entries);
        uint64_t offset = rounded(sizeof header + chunks.count * sizeof *entries), raw_total = 0;
        int output = open(argv[2], O_RDWR | O_CREAT | O_EXCL, 0600); CHECK(output >= 0);
        CHECK(!fcntl(output, F_NOCACHE, 1)); CHECK(!ftruncate(output, (off_t)offset));
        id<MTLDevice> device = MTLCreateSystemDefaultDevice(); CHECK(device && device.hasUnifiedMemory);
        NSString *shader = [[NSString stringWithUTF8String:plamo_metal_source] stringByAppendingString:
            @"\nkernel void disk_unpack(device const ushort *w [[buffer(0)]], device ushort *out [[buffer(1)]], constant Linear &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<p.columns) out[i]=weight_bits<true>(w,i,p); }\n"];
        id<MTLLibrary> library = [device newLibraryWithSource:shader options:nil error:&error];
        if (!library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); CHECK(library);
        id<MTLComputePipelineState> analyze = pipeline(device, library, @"plamo_pack_analyze"), encode = pipeline(device, library, @"plamo_pack_encode"), unpack = pipeline(device, library, @"disk_unpack");
        id<MTLCommandQueue> queue = [device newCommandQueue]; CHECK(queue);
        id<MTLBuffer> raw = [device newBufferWithLength:PLAMO_DISK_RAW_LIMIT options:MTLResourceStorageModeShared];
        id<MTLBuffer> decoded = [device newBufferWithLength:PLAMO_DISK_RAW_LIMIT options:MTLResourceStorageModeShared];
        id<MTLBuffer> headers = [device newBufferWithLength:PLAMO_DISK_RAW_LIMIT / 2 / PLAMO_BF16_BLOCK * 8 options:MTLResourceStorageModeShared]; CHECK(raw && decoded && headers);
        for (unsigned i = 0; i < chunks.count; ++i) @autoreleasepool {
            NSDictionary *chunk = chunks[i]; struct plamo_disk_entry *e = &entries[i];
            e->source = [chunk[@"source"] unsignedIntValue]; e->source_offset = [chunk[@"offset"] unsignedLongLongValue];
            e->rows = [chunk[@"rows"] unsignedIntValue]; e->columns = [chunk[@"columns"] unsignedIntValue]; e->packed = packed;
            CHECK(e->source < sources.count && e->columns && e->rows && e->rows <= PLAMO_DISK_RAW_LIMIT / 2 / e->columns);
            e->raw_bytes = (uint64_t)e->rows * e->columns * 2; e->offset = offset;
            CHECK(e->source_offset <= header.source_bytes[e->source] && e->raw_bytes <= header.source_bytes[e->source] - e->source_offset);
            read_all(fds[e->source], raw.contents, (size_t)e->raw_bytes, e->source_offset);
            id<MTLBuffer> data = raw;
            if (packed) {
                uint32_t count = (uint32_t)(e->raw_bytes / 2), blocks = (count + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
                id<MTLCommandBuffer> command = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
                [enc setComputePipelineState:analyze]; [enc setBuffer:raw offset:0 atIndex:0]; [enc setBuffer:headers offset:0 atIndex:1];
                [enc setBytes:&count length:sizeof count atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; complete(command, enc);
                e->bytes = plamo_bf16_pack_offsets(headers.contents, count); CHECK(e->bytes);
                data = [device newBufferWithLength:rounded(e->bytes) options:MTLResourceStorageModeShared]; CHECK(data);
                command = [queue commandBuffer]; enc = [command computeCommandEncoder];
                [enc setComputePipelineState:encode]; [enc setBuffer:raw offset:0 atIndex:0]; [enc setBuffer:headers offset:0 atIndex:1];
                [enc setBuffer:data offset:0 atIndex:2]; [enc setBytes:&count length:sizeof count atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                struct { uint32_t rows, columns, tokens, xs, ys, aligned, offset; } linear = {1, count, 1, count, 1, 1, 0};
                [enc setComputePipelineState:unpack]; [enc setBuffer:data offset:0 atIndex:0]; [enc setBuffer:decoded offset:0 atIndex:1];
                [enc setBytes:&linear length:sizeof linear atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake((count + 127) / 128, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; complete(command, enc);
                CHECK(!memcmp(raw.contents, decoded.contents, (size_t)e->raw_bytes));
            } else e->bytes = e->raw_bytes;
            write_all(output, data.contents, (size_t)e->bytes, offset);
            if (packed) [data release];
            raw_total += e->raw_bytes; offset += rounded(e->bytes);
            if ((i + 1) % 16 == 0 || i + 1 == chunks.count) fprintf(stderr, "converted %u/%lu chunks; original %.2f GB, stored %.2f GB\n", i + 1, (unsigned long)chunks.count, raw_total / 1e9, offset / 1e9);
        }
        header.file_bytes = offset; CHECK(!ftruncate(output, (off_t)offset));
        write_all(output, &header, sizeof header, 0); write_all(output, entries, chunks.count * sizeof *entries, sizeof header);
        CHECK(!fsync(output)); CHECK(!close(output));
        for (unsigned i = 0; i < sources.count; ++i) close(fds[i]);
        fprintf(stdout, "{\"chunks\":%lu,\"raw_bytes\":%llu,\"file_bytes\":%llu,\"packed\":%s,\"bf16_bitwise_verified\":true}\n", (unsigned long)chunks.count, (unsigned long long)raw_total, (unsigned long long)offset, packed ? "true" : "false");
        free(entries); [raw release]; [decoded release]; [headers release]; [queue release];
        [analyze release]; [encode release]; [unpack release]; [library release]; [device release];
    }
    return 0;
}
