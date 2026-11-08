// BF16 W[rows,cols] times FP32 token rows: Y[token,row].
// x0 points to {weights,input,output,rows,cols,tokens,input_stride,output_stride}.
// Strides are FP32 elements, >= cols/rows; buffers disjoint. Zero rows/tokens
// are no-ops. Four/eight-token tiles preserve the matvec accumulation order.
.text
.p2align 4
.globl _plamo_bf16_matmul
_plamo_bf16_matmul:
    ldr x9, [x0, #40]
    cmp x9, #8
    b.lo _plamo_bf16_matmul4
    ldp x10, x11, [x0, #24]
    cbz x10, _plamo_bf16_matmul4
    cbz x11, _plamo_bf16_matmul4
    tst x11, #3
    b.ne _plamo_bf16_matmul4
    cmp x10, #3
    b.lo Lmm_smallrows
    b _plamo_bf16_matmul8x3
Lmm_smallrows:
    cmp x10, #2
    b.ne _plamo_bf16_matmul4
    b _plamo_bf16_matmul8
.globl _plamo_bf16_matmul4
_plamo_bf16_matmul4:
    stp x29, x30, [sp, #-96]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    ldp x19, x20, [x0]
    ldp x21, x22, [x0, #16]
    ldp x23, x24, [x0, #32]
    ldp x25, x26, [x0, #48]
    lsl x25, x25, #2
    lsl x26, x26, #2
    lsl x27, x23, #1
    cbz x22, Lmmdone
    cbz x24, Lmmdone
    mov x0, x19
    mul x1, x22, x23
    lsl x1, x1, #1
    bl _plamo_weight_acquire
    mov x19, x0
    cbz x23, Lmmfallback
    orr x9, x22, x23
    tst x9, #3
    b.ne Lmmfallback
Lmmtokens:
    cmp x24, #4
    b.lo Lmmfallback
    mov x0, x19
    mov x9, x21
    add x10, x9, x26
    add x11, x10, x26
    add x12, x11, x26
    mov x28, x22
Lmmrows:
    add x1, x0, x27
    add x2, x1, x27
    add x3, x2, x27
    mov x4, x20
    add x5, x4, x25
    add x6, x5, x25
    add x7, x6, x25
    mov x8, x23
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
Lmmk:
    ldr q24, [x4], #16
    ldr q25, [x5], #16
    ldr q26, [x6], #16
    ldr q27, [x7], #16
    ldr d28, [x0], #8
    shll v28.4s, v28.4h, #16
    ldr d29, [x1], #8
    shll v29.4s, v29.4h, #16
    ldr d30, [x2], #8
    shll v30.4s, v30.4h, #16
    ldr d31, [x3], #8
    shll v31.4s, v31.4h, #16
    fmla v0.4s, v28.4s, v24.4s
    fmla v1.4s, v29.4s, v24.4s
    fmla v2.4s, v30.4s, v24.4s
    fmla v3.4s, v31.4s, v24.4s
    fmla v4.4s, v28.4s, v25.4s
    fmla v5.4s, v29.4s, v25.4s
    fmla v6.4s, v30.4s, v25.4s
    fmla v7.4s, v31.4s, v25.4s
    fmla v16.4s, v28.4s, v26.4s
    fmla v17.4s, v29.4s, v26.4s
    fmla v18.4s, v30.4s, v26.4s
    fmla v19.4s, v31.4s, v26.4s
    fmla v20.4s, v28.4s, v27.4s
    fmla v21.4s, v29.4s, v27.4s
    fmla v22.4s, v30.4s, v27.4s
    fmla v23.4s, v31.4s, v27.4s
    subs x8, x8, #4
    b.ne Lmmk
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    faddp v1.4s, v1.4s, v1.4s
    faddp s1, v1.2s
    faddp v2.4s, v2.4s, v2.4s
    faddp s2, v2.2s
    faddp v3.4s, v3.4s, v3.4s
    faddp s3, v3.2s
    stp s0, s1, [x9], #8
    stp s2, s3, [x9], #8
    faddp v4.4s, v4.4s, v4.4s
    faddp s4, v4.2s
    faddp v5.4s, v5.4s, v5.4s
    faddp s5, v5.2s
    faddp v6.4s, v6.4s, v6.4s
    faddp s6, v6.2s
    faddp v7.4s, v7.4s, v7.4s
    faddp s7, v7.2s
    stp s4, s5, [x10], #8
    stp s6, s7, [x10], #8
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    faddp v17.4s, v17.4s, v17.4s
    faddp s17, v17.2s
    faddp v18.4s, v18.4s, v18.4s
    faddp s18, v18.2s
    faddp v19.4s, v19.4s, v19.4s
    faddp s19, v19.2s
    stp s16, s17, [x11], #8
    stp s18, s19, [x11], #8
    faddp v20.4s, v20.4s, v20.4s
    faddp s20, v20.2s
    faddp v21.4s, v21.4s, v21.4s
    faddp s21, v21.2s
    faddp v22.4s, v22.4s, v22.4s
    faddp s22, v22.2s
    faddp v23.4s, v23.4s, v23.4s
    faddp s23, v23.2s
    stp s20, s21, [x12], #8
    stp s22, s23, [x12], #8
    mov x0, x3
    subs x28, x28, #4
    b.ne Lmmrows
    add x20, x20, x25, lsl #2
    add x21, x21, x26, lsl #2
    sub x24, x24, #4
    b Lmmtokens
Lmmfallback:
    cbz x24, Lmmdone
    mov x0, x19
    mov x1, x20
    mov x2, x21
    mov x3, x22
    mov x4, x23
    bl _plamo_bf16_matvec_core
    add x20, x20, x25
    add x21, x21, x26
    sub x24, x24, #1
    b Lmmfallback
Lmmdone:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

// Register-argument convenience interface to the same kernel:
// x0=W,x1=X,x2=Y,x3=rows,x4=cols,x5=tokens,x6=xstride,x7=ystride.
.globl _plamo_bf16_linear
_plamo_bf16_linear:
    stp x29, x30, [sp, #-80]!
    stp x0, x1, [sp, #16]
    stp x2, x3, [sp, #32]
    stp x4, x5, [sp, #48]
    stp x6, x7, [sp, #64]
    add x0, sp, #16
    bl _plamo_bf16_matmul
    ldp x29, x30, [sp], #80
    ret
