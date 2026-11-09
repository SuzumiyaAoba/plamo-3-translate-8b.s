// AAPCS64: x0=BF16 row-major weights, x1=FP32 input,
// x2=FP32 output, x3=rows, x4=columns. No alignment constraints.
.text
.p2align 2
.globl _plamo_bf16_matvec
_plamo_bf16_matvec:
    adrp x9, _plamo_weight_chunk_bytes@PAGE
    ldr x9, [x9, _plamo_weight_chunk_bytes@PAGEOFF]
    cbz x9, Lmv_whole
    cbz x4, Lmv_whole
    mul x10, x3, x4
    lsl x10, x10, #1
    cmp x10, x9
    b.hi Lmv_tiled
Lmv_whole:
    stp x29, x30, [sp, #-64]!
    stp x0, x1, [sp, #16]
    stp x2, x3, [sp, #32]
    str x4, [sp, #48]
    mul x1, x3, x4
    lsl x1, x1, #1
    bl _plamo_weight_acquire
    str x0, [sp, #16]
    ldp x0, x1, [sp, #16]
    ldp x2, x3, [sp, #32]
    ldr x4, [sp, #48]
    ldp x29, x30, [sp], #64
    b _plamo_bf16_matvec_core
// Row tiles bound the lifetime of every acquired range to the worker's read
// window, including heads larger than 1 GiB. Columns/reduction stay unchanged.
Lmv_tiled:
    lsl x10, x4, #1
    udiv x9, x9, x10
    cbz x9, Lmv_whole // A single unusually wide row uses its original mapping.
    cmp x9, #8
    b.lo Lmv_tile_rows
    and x9, x9, #0xfffffffffffffff8
Lmv_tile_rows:
    stp x19, x30, [sp, #-80]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x9
Lmv_tile_loop:
    cmp x22, x24
    csel x25, x22, x24, lo
    mov x0, x19
    mul x1, x25, x23
    lsl x1, x1, #1
    bl _plamo_weight_acquire
    mov x1, x20
    mov x2, x21
    mov x3, x25
    mov x4, x23
    bl _plamo_bf16_matvec_core
    mul x9, x25, x23
    add x19, x19, x9, lsl #1
    add x21, x21, x25, lsl #2
    subs x22, x22, x25
    b.ne Lmv_tile_loop
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #80
    ret
.globl _plamo_bf16_matvec_core
_plamo_bf16_matvec_core:
    // The provider has finished reading before workers borrow this range.
    adrp x9, _plamo_matvec_kernel@PAGE
    ldr x9, [x9, _plamo_matvec_kernel@PAGEOFF]
    cbz x9, _plamo_bf16_matvec_serial
    adrp x5, _plamo_compute_context@PAGE
    ldr x5, [x5, _plamo_compute_context@PAGEOFF]
    br x9
.globl _plamo_bf16_matvec_serial
_plamo_bf16_matvec_serial:
    // Eight independent rows, four K-vectors per loop. The accumulator
    // sequence inside every row is unchanged, including final reduction.
    cmp x3, #8
    b.lo Lmv4dispatch
    cbz x4, Lmv4dispatch
    tst x4, #3
    b.ne Lmv4dispatch
Lmv8block:
    lsl x7, x4, #1
    add x8, x0, x7
    add x9, x8, x7
    add x10, x9, x7
    add x11, x10, x7
    add x12, x11, x7
    add x13, x12, x7
    add x14, x13, x7
    mov x5, x1
    mov x6, x4
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    cmp x6, #16
    b.lo Lmv8tail
Lmv8k16:
    ldr q16, [x5], #16
    ldr d17, [x0], #8
    shll v17.4s, v17.4h, #16
    ldr d18, [x8], #8
    shll v18.4s, v18.4h, #16
    ldr d19, [x9], #8
    shll v19.4s, v19.4h, #16
    ldr d20, [x10], #8
    shll v20.4s, v20.4h, #16
    ldr d21, [x11], #8
    shll v21.4s, v21.4h, #16
    ldr d22, [x12], #8
    shll v22.4s, v22.4h, #16
    ldr d23, [x13], #8
    shll v23.4s, v23.4h, #16
    ldr d24, [x14], #8
    shll v24.4s, v24.4h, #16
    fmla v0.4s, v17.4s, v16.4s
    fmla v1.4s, v18.4s, v16.4s
    fmla v2.4s, v19.4s, v16.4s
    fmla v3.4s, v20.4s, v16.4s
    fmla v4.4s, v21.4s, v16.4s
    fmla v5.4s, v22.4s, v16.4s
    fmla v6.4s, v23.4s, v16.4s
    fmla v7.4s, v24.4s, v16.4s
    ldr q16, [x5], #16
    ldr d17, [x0], #8
    shll v17.4s, v17.4h, #16
    ldr d18, [x8], #8
    shll v18.4s, v18.4h, #16
    ldr d19, [x9], #8
    shll v19.4s, v19.4h, #16
    ldr d20, [x10], #8
    shll v20.4s, v20.4h, #16
    ldr d21, [x11], #8
    shll v21.4s, v21.4h, #16
    ldr d22, [x12], #8
    shll v22.4s, v22.4h, #16
    ldr d23, [x13], #8
    shll v23.4s, v23.4h, #16
    ldr d24, [x14], #8
    shll v24.4s, v24.4h, #16
    fmla v0.4s, v17.4s, v16.4s
    fmla v1.4s, v18.4s, v16.4s
    fmla v2.4s, v19.4s, v16.4s
    fmla v3.4s, v20.4s, v16.4s
    fmla v4.4s, v21.4s, v16.4s
    fmla v5.4s, v22.4s, v16.4s
    fmla v6.4s, v23.4s, v16.4s
    fmla v7.4s, v24.4s, v16.4s
    ldr q16, [x5], #16
    ldr d17, [x0], #8
    shll v17.4s, v17.4h, #16
    ldr d18, [x8], #8
    shll v18.4s, v18.4h, #16
    ldr d19, [x9], #8
    shll v19.4s, v19.4h, #16
    ldr d20, [x10], #8
    shll v20.4s, v20.4h, #16
    ldr d21, [x11], #8
    shll v21.4s, v21.4h, #16
    ldr d22, [x12], #8
    shll v22.4s, v22.4h, #16
    ldr d23, [x13], #8
    shll v23.4s, v23.4h, #16
    ldr d24, [x14], #8
    shll v24.4s, v24.4h, #16
    fmla v0.4s, v17.4s, v16.4s
    fmla v1.4s, v18.4s, v16.4s
    fmla v2.4s, v19.4s, v16.4s
    fmla v3.4s, v20.4s, v16.4s
    fmla v4.4s, v21.4s, v16.4s
    fmla v5.4s, v22.4s, v16.4s
    fmla v6.4s, v23.4s, v16.4s
    fmla v7.4s, v24.4s, v16.4s
    ldr q16, [x5], #16
    ldr d17, [x0], #8
    shll v17.4s, v17.4h, #16
    ldr d18, [x8], #8
    shll v18.4s, v18.4h, #16
    ldr d19, [x9], #8
    shll v19.4s, v19.4h, #16
    ldr d20, [x10], #8
    shll v20.4s, v20.4h, #16
    ldr d21, [x11], #8
    shll v21.4s, v21.4h, #16
    ldr d22, [x12], #8
    shll v22.4s, v22.4h, #16
    ldr d23, [x13], #8
    shll v23.4s, v23.4h, #16
    ldr d24, [x14], #8
    shll v24.4s, v24.4h, #16
    fmla v0.4s, v17.4s, v16.4s
    fmla v1.4s, v18.4s, v16.4s
    fmla v2.4s, v19.4s, v16.4s
    fmla v3.4s, v20.4s, v16.4s
    fmla v4.4s, v21.4s, v16.4s
    fmla v5.4s, v22.4s, v16.4s
    fmla v6.4s, v23.4s, v16.4s
    fmla v7.4s, v24.4s, v16.4s
    sub x6, x6, #16
    cmp x6, #16
    b.hs Lmv8k16
Lmv8tail:
    cbz x6, Lmv8reduce
Lmv8k4:
    ldr q16, [x5], #16
    ldr d17, [x0], #8
    shll v17.4s, v17.4h, #16
    ldr d18, [x8], #8
    shll v18.4s, v18.4h, #16
    ldr d19, [x9], #8
    shll v19.4s, v19.4h, #16
    ldr d20, [x10], #8
    shll v20.4s, v20.4h, #16
    ldr d21, [x11], #8
    shll v21.4s, v21.4h, #16
    ldr d22, [x12], #8
    shll v22.4s, v22.4h, #16
    ldr d23, [x13], #8
    shll v23.4s, v23.4h, #16
    ldr d24, [x14], #8
    shll v24.4s, v24.4h, #16
    fmla v0.4s, v17.4s, v16.4s
    fmla v1.4s, v18.4s, v16.4s
    fmla v2.4s, v19.4s, v16.4s
    fmla v3.4s, v20.4s, v16.4s
    fmla v4.4s, v21.4s, v16.4s
    fmla v5.4s, v22.4s, v16.4s
    fmla v6.4s, v23.4s, v16.4s
    fmla v7.4s, v24.4s, v16.4s
    subs x6, x6, #4
    b.ne Lmv8k4
Lmv8reduce:
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    faddp v1.4s, v1.4s, v1.4s
    faddp s1, v1.2s
    faddp v2.4s, v2.4s, v2.4s
    faddp s2, v2.2s
    faddp v3.4s, v3.4s, v3.4s
    faddp s3, v3.2s
    faddp v4.4s, v4.4s, v4.4s
    faddp s4, v4.2s
    faddp v5.4s, v5.4s, v5.4s
    faddp s5, v5.2s
    faddp v6.4s, v6.4s, v6.4s
    faddp s6, v6.2s
    faddp v7.4s, v7.4s, v7.4s
    faddp s7, v7.2s
    stp s0, s1, [x2], #8
    stp s2, s3, [x2], #8
    stp s4, s5, [x2], #8
    stp s6, s7, [x2], #8
    mov x0, x14
    sub x3, x3, #8
    cmp x3, #8
    b.hs Lmv8block
Lmv4dispatch:
    // Four output rows share each input load. Each row retains the original
    // accumulation order, so this path is bit-identical to the scalar-row path.
    cmp x3, #4
    b.lo Lmvslow
    cbz x4, Lmvslow
    tst x4, #3
    b.ne Lmvslow
Lmvblock:
    lsl x7, x4, #1
    add x8, x0, x7
    add x9, x8, x7
    add x10, x9, x7
    mov x5, x1
    mov x6, x4
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
Lmvcolumns:
    ldr q5, [x5], #16
    ldr d4, [x0], #8
    shll v4.4s, v4.4h, #16
    fmla v0.4s, v4.4s, v5.4s
    ldr d4, [x8], #8
    shll v4.4s, v4.4h, #16
    fmla v1.4s, v4.4s, v5.4s
    ldr d4, [x9], #8
    shll v4.4s, v4.4h, #16
    fmla v2.4s, v4.4s, v5.4s
    ldr d4, [x10], #8
    shll v4.4s, v4.4h, #16
    fmla v3.4s, v4.4s, v5.4s
    subs x6, x6, #4
    b.ne Lmvcolumns
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    faddp v1.4s, v1.4s, v1.4s
    faddp s1, v1.2s
    faddp v2.4s, v2.4s, v2.4s
    faddp s2, v2.2s
    faddp v3.4s, v3.4s, v3.4s
    faddp s3, v3.2s
    stp s0, s1, [x2], #8
    stp s2, s3, [x2], #8
    mov x0, x10
    sub x3, x3, #4
    cmp x3, #4
    b.hs Lmvblock
Lmvslow:
    cbz x3, 4f
1:  mov x5, x1
    mov x6, x4
    movi v0.4s, #0
2:  cmp x6, #4
    b.lo 3f
    ldr d1, [x0], #8
    shll v1.4s, v1.4h, #16
    ldr q2, [x5], #16
    fmla v0.4s, v1.4s, v2.4s
    sub x6, x6, #4
    b 2b
3:  faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    cbz x6, 6f
5:  ldrh w7, [x0], #2
    lsl w7, w7, #16
    fmov s1, w7
    ldr s2, [x5], #4
    fmadd s0, s1, s2, s0
    subs x6, x6, #1
    b.ne 5b
6:  str s0, [x2], #4
    subs x3, x3, #1
    b.ne 1b
4:  ret

// x0=input FP32, x1=weights BF16, x2=output FP32, x3=count
// s0=epsilon, s1=weight offset. Supports input=output; count=0 is a no-op.
// Accumulation and output stay FP32 (reference verification uses FP32).
.globl _plamo_rmsnorm
_plamo_rmsnorm:
    cbz x3, 4f
    mov x4, #0
    fmov s2, wzr
1:  ldr s3, [x0, x4, lsl #2]
    fmul s3, s3, s3
    fadd s2, s2, s3
    add x4, x4, #1
    cmp x4, x3
    b.lo 1b
    ucvtf s3, x3
    fdiv s2, s2, s3
    fadd s2, s2, s0
    fsqrt s2, s2
    fmov s3, #1.0
    fdiv s2, s3, s2
    mov x4, #0
    dup v1.4s, v1.s[0]
    dup v2.4s, v2.s[0]
Lnormvector:
    sub x5, x3, x4
    cmp x5, #4
    b.lo 2f
    add x5, x1, x4, lsl #1
    ldr d3, [x5]
    shll v3.4s, v3.4h, #16
    fadd v3.4s, v3.4s, v1.4s
    lsl x5, x4, #2
    ldr q4, [x0, x5]
    fmul v4.4s, v4.4s, v2.4s
    fmul v4.4s, v4.4s, v3.4s
    str q4, [x2, x5]
    add x4, x4, #4
    b Lnormvector
2:  cmp x4, x3
    b.hs 4f
    ldrh w5, [x1, x4, lsl #1]
    lsl w5, w5, #16
    fmov s3, w5
    fadd s3, s3, s1
    ldr s4, [x0, x4, lsl #2]
    fmul s4, s4, s2
    fmul s4, s4, s3
    str s4, [x2, x4, lsl #2]
    add x4, x4, #1
    cmp x4, x3
    b.lo 2b
4:  ret
