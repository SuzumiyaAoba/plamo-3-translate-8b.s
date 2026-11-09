// Eight input tokens x two output rows, with the matvec summation order.
// Internal fast path: positive even rows, cols divisible by four, tokens>=8.
.text
.p2align 4
.globl _plamo_bf16_matmul8
_plamo_bf16_matmul8:
    adrp x1, _plamo_bf16_matmul8_core@PAGE
    add x1, x1, _plamo_bf16_matmul8_core@PAGEOFF
    b _plamo_bf16_matmul_acquire
.globl _plamo_bf16_matmul8_core
_plamo_bf16_matmul8_core:
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
Lmm8tokens:
    cmp x24, #8
    b.lo Lmm8tail
    mov x0, x19
    mov x28, x22
Lmm8rows:
    add x1, x0, x27
    mov x4, x20
    add x5, x4, x25
    add x6, x5, x25
    add x7, x6, x25
    add x10, x7, x25
    add x11, x10, x25
    add x12, x11, x25
    add x13, x12, x25
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
Lmm8k:
    ldr d28, [x0], #8
    shll v28.4s, v28.4h, #16
    ldr d29, [x1], #8
    shll v29.4s, v29.4h, #16
    ldr q24, [x4], #16
    ldr q25, [x5], #16
    ldr q26, [x6], #16
    ldr q27, [x7], #16
    fmla v0.4s, v28.4s, v24.4s
    fmla v1.4s, v29.4s, v24.4s
    fmla v2.4s, v28.4s, v25.4s
    fmla v3.4s, v29.4s, v25.4s
    fmla v4.4s, v28.4s, v26.4s
    fmla v5.4s, v29.4s, v26.4s
    fmla v6.4s, v28.4s, v27.4s
    fmla v7.4s, v29.4s, v27.4s
    ldr q24, [x10], #16
    ldr q25, [x11], #16
    ldr q26, [x12], #16
    ldr q27, [x13], #16
    fmla v16.4s, v28.4s, v24.4s
    fmla v17.4s, v29.4s, v24.4s
    fmla v18.4s, v28.4s, v25.4s
    fmla v19.4s, v29.4s, v25.4s
    fmla v20.4s, v28.4s, v26.4s
    fmla v21.4s, v29.4s, v26.4s
    fmla v22.4s, v28.4s, v27.4s
    fmla v23.4s, v29.4s, v27.4s
    subs x8, x8, #4
    b.ne Lmm8k
    sub x9, x22, x28
    add x9, x21, x9, lsl #2
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    faddp v1.4s, v1.4s, v1.4s
    faddp s1, v1.2s
    stp s0, s1, [x9]
    add x9, x9, x26
    faddp v2.4s, v2.4s, v2.4s
    faddp s2, v2.2s
    faddp v3.4s, v3.4s, v3.4s
    faddp s3, v3.2s
    stp s2, s3, [x9]
    add x9, x9, x26
    faddp v4.4s, v4.4s, v4.4s
    faddp s4, v4.2s
    faddp v5.4s, v5.4s, v5.4s
    faddp s5, v5.2s
    stp s4, s5, [x9]
    add x9, x9, x26
    faddp v6.4s, v6.4s, v6.4s
    faddp s6, v6.2s
    faddp v7.4s, v7.4s, v7.4s
    faddp s7, v7.2s
    stp s6, s7, [x9]
    add x9, x9, x26
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    faddp v17.4s, v17.4s, v17.4s
    faddp s17, v17.2s
    stp s16, s17, [x9]
    add x9, x9, x26
    faddp v18.4s, v18.4s, v18.4s
    faddp s18, v18.2s
    faddp v19.4s, v19.4s, v19.4s
    faddp s19, v19.2s
    stp s18, s19, [x9]
    add x9, x9, x26
    faddp v20.4s, v20.4s, v20.4s
    faddp s20, v20.2s
    faddp v21.4s, v21.4s, v21.4s
    faddp s21, v21.2s
    stp s20, s21, [x9]
    add x9, x9, x26
    faddp v22.4s, v22.4s, v22.4s
    faddp s22, v22.2s
    faddp v23.4s, v23.4s, v23.4s
    faddp s23, v23.2s
    stp s22, s23, [x9]
    add x9, x9, x26
    mov x0, x1
    subs x28, x28, #2
    b.ne Lmm8rows
    add x20, x20, x25, lsl #3
    add x21, x21, x26, lsl #3
    sub x24, x24, #8
    b Lmm8tokens
Lmm8tail:
    cbz x24, Lmm8done
    mov x0, x19
    mov x1, x20
    mov x2, x21
    mov x3, x22
    mov x4, x23
    mov x5, x24
    lsr x6, x25, #2
    lsr x7, x26, #2
    bl _plamo_bf16_linear_core
Lmm8done:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret
