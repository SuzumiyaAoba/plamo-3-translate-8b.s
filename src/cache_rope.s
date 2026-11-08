.text
.p2align 2
// x0=descriptor {cache,rotated_keys,inv_freq,cos_scratch,sin_scratch,heads,dim},
// s0=concentration. Cache retains normalized unrotated keys. Rotates a copy
// using local positions 0..count-1, including after sliding eviction.
// Caller validates dim even, cache width=heads*dim, disjoint output/scratch.
.globl _plamo_cache_rope
_plamo_cache_rope:
    mov x1, #0
    b Lcrfrom

// Extended descriptor adds {processed_count,scale_bits} after dim. It owns a
// dedicated rotated buffer. Reuse previous rows until a sliding eviction/reset
// or scale change; in those cases recompute from original unrotated keys.
.globl _plamo_cache_rope_cached
_plamo_cache_rope_cached:
    stp x19, x30, [sp, #-32]!
    stp x20, x21, [sp, #16]
    mov x19, x0
    fmov w20, s0
    ldr x9, [x19]
    ldr x21, [x9, #24]
    mov x1, #0
    cbz x21, Lcrcached_store
    ldp x10, x11, [x19, #56]
    cmp x11, x20
    b.ne Lcrcached_compute
    add x12, x10, #1
    cmp x12, x21
    csel x1, x10, xzr, eq
Lcrcached_compute:
    bl Lcrfrom
Lcrcached_store:
    stp x21, x20, [x19, #56]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #32
    ret
Lcrfrom:
    stp x19, x30, [sp, #-80]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    str s0, [sp, #64]
    mov x19, x0
    ldr x9, [x19]
    ldr x20, [x9]
    ldr x21, [x9, #24]
    ldr x22, [x9, #32]
    ldr x23, [x19, #8]
    mov x24, x1
    mul x9, x24, x22
    add x20, x20, x9, lsl #2
    add x23, x23, x9, lsl #2
Lcrtoken:
    cmp x24, x21
    b.hs Lcrdone
    ldr x0, [x19, #16]
    mov x1, x24
    ldr x2, [x19, #48]
    lsr x2, x2, #1
    ldp x3, x4, [x19, #24]
    ldr s0, [sp, #64]
    bl _plamo_rope_table
    mov x0, x20
    mov x1, x23
    ldp x2, x3, [x19, #24]
    ldp x4, x5, [x19, #40]
    lsr x5, x5, #1
    bl _plamo_rope_copy
    add x20, x20, x22, lsl #2
    add x23, x23, x22, lsl #2
    add x24, x24, #1
    b Lcrtoken
Lcrdone:
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #80
    ret
