.text
.p2align 2
.globl _plamo_bf16_matvec_4row
_plamo_bf16_matvec_4row:
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
