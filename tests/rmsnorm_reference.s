.text
.p2align 2
// x0=input FP32, x1=weights BF16, x2=output FP32, x3=count
// s0=epsilon, s1=weight offset. Supports input=output; count=0 is a no-op.
// Accumulation and output stay FP32 (reference verification uses FP32).
.globl _plamo_rmsnorm_reference
_plamo_rmsnorm_reference:
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
2:  ldrh w5, [x1, x4, lsl #1]
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
