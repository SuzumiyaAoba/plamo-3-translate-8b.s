.text
.p2align 2
// In-place split-half rotary embedding, matching _rotate_half in PLaMo3.
// x0=FP32 heads[heads,2*half],x1=cos[half],x2=sin[half],
// x3=head count,x4=half dimension. Coefficients can include YaRN scaling.
// Zero dimensions are no-ops. Tables must not overlap writable heads.
.globl _plamo_rope_apply_reference
_plamo_rope_apply_reference:
    cbz x3, Lroperet
    cbz x4, Lroperet
Lropehead:
    add x5, x0, x4, lsl #2
    mov x6, #0
Lropepair:
    ldr s0, [x0, x6, lsl #2]
    ldr s1, [x5, x6, lsl #2]
    ldr s2, [x1, x6, lsl #2]
    ldr s3, [x2, x6, lsl #2]
    // Separate mul/add matches the reference expression's FP32 operations.
    fmul s4, s0, s2
    fmul s5, s1, s3
    fsub s4, s4, s5
    fmul s5, s1, s2
    fmul s6, s0, s3
    fadd s5, s5, s6
    str s4, [x0, x6, lsl #2]
    str s5, [x5, x6, lsl #2]
    add x6, x6, #1
    cmp x6, x4
    b.lo Lropepair
    add x0, x5, x4, lsl #2
    subs x3, x3, #1
    b.ne Lropehead
Lroperet:
    ret
