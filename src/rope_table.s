.text
.p2align 2
// x0=FP32 inv_freq[half],x1=position,x2=half,x3=cos output,x4=sin output,
// s0=concentration. FP32 position*frequency matches reference angle formation.
.globl _plamo_rope_table
_plamo_rope_table:
    cbz x2, Ltableret
    stp x19, x30, [sp, #-80]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    str s0, [sp, #48]
    ucvtf s1, x1
    str s1, [sp, #52]
    mov x19, x0
    mov x20, x2
    mov x21, x3
    mov x22, x4
Ltableloop:
    ldr s0, [x19], #4
    ldr s1, [sp, #52]
    fmul s0, s0, s1
    fcvt d0, s0
    add x0, sp, #56
    bl _plamo_sincos
    ldp s0, s1, [sp, #56]
    ldr s2, [sp, #48]
    fmul s0, s0, s2
    fmul s1, s1, s2
    str s0, [x22], #4
    str s1, [x21], #4
    subs x20, x20, #1
    b.ne Ltableloop
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #80
Ltableret:
    ret
