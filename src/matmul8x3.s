// Eight tokens x three output rows. Same per-output FMA sequence as matvec.
// Callee-saved SIMD registers v8-v15 are preserved in full. Row tails dispatch
// to the smaller tiles; unsupported shapes use the general four-token path.
.text
.p2align 4
.globl _plamo_bf16_matmul8x3
_plamo_bf16_matmul8x3:
    adrp x1, _plamo_bf16_matmul8x3_core@PAGE
    add x1, x1, _plamo_bf16_matmul8x3_core@PAGEOFF
    b _plamo_bf16_matmul_acquire
.globl _plamo_bf16_matmul8x3_core
_plamo_bf16_matmul8x3_core:
    ldr x9, [x0, #40]
    cmp x9, #8
    b.lo Lmm83fallback
    ldp x10, x11, [x0, #24]
    cmp x10, #3
    b.lo Lmm83fallback
    cbz x11, Lmm83fallback
    tst x11, #3
    b.ne Lmm83fallback
    stp x29, x30, [sp, #-224]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp q8, q9, [sp, #96]
    stp q10, q11, [sp, #128]
    stp q12, q13, [sp, #160]
    stp q14, q15, [sp, #192]
    ldp x19, x20, [x0]
    ldp x21, x22, [x0, #16]
    ldp x23, x24, [x0, #32]
    ldp x25, x26, [x0, #48]
    lsl x25, x25, #2
    lsl x26, x26, #2
    lsl x27, x23, #1
Lmm83tokens:
    cmp x24, #8
    b.lo Lmm83tail
    mov x0, x19
    mov x28, x22
Lmm83rows:
    add x1, x0, x27
    add x2, x1, x27
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
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
Lmm83k:
    ldr d28, [x0], #8
    shll v28.4s, v28.4h, #16
    ldr d29, [x1], #8
    shll v29.4s, v29.4h, #16
    ldr d30, [x2], #8
    shll v30.4s, v30.4h, #16
    ldr q31, [x4], #16
    fmla v0.4s, v28.4s, v31.4s
    fmla v1.4s, v29.4s, v31.4s
    fmla v2.4s, v30.4s, v31.4s
    ldr q31, [x5], #16
    fmla v3.4s, v28.4s, v31.4s
    fmla v4.4s, v29.4s, v31.4s
    fmla v5.4s, v30.4s, v31.4s
    ldr q31, [x6], #16
    fmla v6.4s, v28.4s, v31.4s
    fmla v7.4s, v29.4s, v31.4s
    fmla v8.4s, v30.4s, v31.4s
    ldr q31, [x7], #16
    fmla v9.4s, v28.4s, v31.4s
    fmla v10.4s, v29.4s, v31.4s
    fmla v11.4s, v30.4s, v31.4s
    ldr q31, [x10], #16
    fmla v12.4s, v28.4s, v31.4s
    fmla v13.4s, v29.4s, v31.4s
    fmla v14.4s, v30.4s, v31.4s
    ldr q31, [x11], #16
    fmla v15.4s, v28.4s, v31.4s
    fmla v16.4s, v29.4s, v31.4s
    fmla v17.4s, v30.4s, v31.4s
    ldr q31, [x12], #16
    fmla v18.4s, v28.4s, v31.4s
    fmla v19.4s, v29.4s, v31.4s
    fmla v20.4s, v30.4s, v31.4s
    ldr q31, [x13], #16
    fmla v21.4s, v28.4s, v31.4s
    fmla v22.4s, v29.4s, v31.4s
    fmla v23.4s, v30.4s, v31.4s
    subs x8, x8, #4
    b.ne Lmm83k
    sub x9, x22, x28
    add x9, x21, x9, lsl #2
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    faddp v1.4s, v1.4s, v1.4s
    faddp s1, v1.2s
    faddp v2.4s, v2.4s, v2.4s
    faddp s2, v2.2s
    stp s0, s1, [x9]
    str s2, [x9, #8]
    add x9, x9, x26
    faddp v3.4s, v3.4s, v3.4s
    faddp s3, v3.2s
    faddp v4.4s, v4.4s, v4.4s
    faddp s4, v4.2s
    faddp v5.4s, v5.4s, v5.4s
    faddp s5, v5.2s
    stp s3, s4, [x9]
    str s5, [x9, #8]
    add x9, x9, x26
    faddp v6.4s, v6.4s, v6.4s
    faddp s6, v6.2s
    faddp v7.4s, v7.4s, v7.4s
    faddp s7, v7.2s
    faddp v8.4s, v8.4s, v8.4s
    faddp s8, v8.2s
    stp s6, s7, [x9]
    str s8, [x9, #8]
    add x9, x9, x26
    faddp v9.4s, v9.4s, v9.4s
    faddp s9, v9.2s
    faddp v10.4s, v10.4s, v10.4s
    faddp s10, v10.2s
    faddp v11.4s, v11.4s, v11.4s
    faddp s11, v11.2s
    stp s9, s10, [x9]
    str s11, [x9, #8]
    add x9, x9, x26
    faddp v12.4s, v12.4s, v12.4s
    faddp s12, v12.2s
    faddp v13.4s, v13.4s, v13.4s
    faddp s13, v13.2s
    faddp v14.4s, v14.4s, v14.4s
    faddp s14, v14.2s
    stp s12, s13, [x9]
    str s14, [x9, #8]
    add x9, x9, x26
    faddp v15.4s, v15.4s, v15.4s
    faddp s15, v15.2s
    faddp v16.4s, v16.4s, v16.4s
    faddp s16, v16.2s
    faddp v17.4s, v17.4s, v17.4s
    faddp s17, v17.2s
    stp s15, s16, [x9]
    str s17, [x9, #8]
    add x9, x9, x26
    faddp v18.4s, v18.4s, v18.4s
    faddp s18, v18.2s
    faddp v19.4s, v19.4s, v19.4s
    faddp s19, v19.2s
    faddp v20.4s, v20.4s, v20.4s
    faddp s20, v20.2s
    stp s18, s19, [x9]
    str s20, [x9, #8]
    add x9, x9, x26
    faddp v21.4s, v21.4s, v21.4s
    faddp s21, v21.2s
    faddp v22.4s, v22.4s, v22.4s
    faddp s22, v22.2s
    faddp v23.4s, v23.4s, v23.4s
    faddp s23, v23.2s
    stp s21, s22, [x9]
    str s23, [x9, #8]
    add x9, x9, x26
    mov x0, x2
    sub x28, x28, #3
    cmp x28, #3
    b.hs Lmm83rows
    cbz x28, Lmm83panel_done
    // Only one or two output rows remain for this eight-token panel.
    mov x1, x20
    sub x9, x22, x28
    add x2, x21, x9, lsl #2
    mov x3, x28
    mov x4, x23
    mov x5, #8
    lsr x6, x25, #2
    lsr x7, x26, #2
    bl _plamo_bf16_linear_core
Lmm83panel_done:
    add x20, x20, x25, lsl #3
    add x21, x21, x26, lsl #3
    sub x24, x24, #8
    b Lmm83tokens
Lmm83tail:
    cbz x24, Lmm83done
    mov x0, x19
    mov x1, x20
    mov x2, x21
    mov x3, x22
    mov x4, x23
    mov x5, x24
    lsr x6, x25, #2
    lsr x7, x26, #2
    bl _plamo_bf16_linear_core
Lmm83done:
    ldp q14, q15, [sp, #192]
    ldp q12, q13, [sp, #160]
    ldp q10, q11, [sp, #128]
    ldp q8, q9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #224
    ret

Lmm83fallback:
    b _plamo_bf16_matmul4_core
