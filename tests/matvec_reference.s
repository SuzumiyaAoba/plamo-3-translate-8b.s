// AAPCS64: x0=BF16 row-major weights, x1=FP32 input,
// x2=FP32 output, x3=rows, x4=columns. No alignment constraints.
.text
.p2align 2
.globl _plamo_bf16_matvec_reference
_plamo_bf16_matvec_reference:
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
