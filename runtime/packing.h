#ifndef PLAMO_BF16_PACKING_H
#define PLAMO_BF16_PACKING_H
#include <stddef.h>
#include <stdint.h>

/* Lossless BF16 cache layout, shared with kernels.metal. Each block stores
   sign+mantissa bytes and fixed-width deltas from its minimum exponent. */
enum { PLAMO_BF16_BLOCK = 512 };
struct plamo_bf16_block { uint32_t offset, exponent; };
_Static_assert(sizeof(struct plamo_bf16_block) == 8, "Metal packing ABI");

/* GPU analysis supplies {0, minimum_exponent | exponent_bits << 8}. Fill
   byte offsets and return the exact buffer length. Zero rejects an invalid
   analysis or an unrepresentable buffer; no original weight is rounded. */
static inline uint64_t plamo_bf16_pack_offsets(struct plamo_bf16_block *blocks, uint64_t values) {
    if (!values || values > UINT64_MAX - (PLAMO_BF16_BLOCK - 1)) return 0;
    uint64_t count = (values + PLAMO_BF16_BLOCK - 1) / PLAMO_BF16_BLOCK;
    if (count > UINT32_MAX / sizeof *blocks) return 0;
    uint64_t bytes = (count * sizeof *blocks + 15) & ~UINT64_C(15);
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t bits = blocks[i].exponent >> 8;
        if (bits > 8 || bytes > UINT32_MAX) return 0;
        blocks[i].offset = (uint32_t)bytes;
        bytes += PLAMO_BF16_BLOCK + (PLAMO_BF16_BLOCK / 8) * bits;
    }
    return bytes <= UINT32_MAX ? bytes : 0;
}
#endif
