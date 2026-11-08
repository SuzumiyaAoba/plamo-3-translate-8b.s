.text
.p2align 2
// In-place split-half rotary embedding. x0=heads,x1=cos,x2=sin,
// x3=head count,x4=half dimension. Same FP32 mul/add order as the reference.
.globl _plamo_rope_apply
_plamo_rope_apply:
    mov x5, x4
    mov x4, x3
    mov x3, x2
    mov x2, x1
    mov x1, x0
    b _plamo_rope_copy

// Copy and rotate in one pass: x0=source,x1=destination,x2=cos,x3=sin,
// x4=head count,x5=half dimension. Source=destination supported, otherwise
// buffers must be disjoint. Zero dimensions are no-ops.
.globl _plamo_rope_copy
_plamo_rope_copy:
    cbz x4, Lroperet
    cbz x5, Lroperet
Lropehead:
    add x6, x0, x5, lsl #2
    add x7, x1, x5, lsl #2
    mov x8, #0
Lropevector:
    sub x9, x5, x8
    cmp x9, #4
    b.lo Lropepair
    lsl x9, x8, #2
    ldr q0, [x0, x9]
    ldr q1, [x6, x9]
    ldr q2, [x2, x9]
    ldr q3, [x3, x9]
    fmul v4.4s, v0.4s, v2.4s
    fmul v5.4s, v1.4s, v3.4s
    fsub v4.4s, v4.4s, v5.4s
    fmul v5.4s, v1.4s, v2.4s
    fmul v6.4s, v0.4s, v3.4s
    fadd v5.4s, v5.4s, v6.4s
    str q4, [x1, x9]
    str q5, [x7, x9]
    add x8, x8, #4
    b Lropevector
Lropepair:
    cmp x8, x5
    b.hs Lropehead_done
    ldr s0, [x0, x8, lsl #2]
    ldr s1, [x6, x8, lsl #2]
    ldr s2, [x2, x8, lsl #2]
    ldr s3, [x3, x8, lsl #2]
    fmul s4, s0, s2
    fmul s5, s1, s3
    fsub s4, s4, s5
    fmul s5, s1, s2
    fmul s6, s0, s3
    fadd s5, s5, s6
    str s4, [x1, x8, lsl #2]
    str s5, [x7, x8, lsl #2]
    add x8, x8, #1
    b Lropepair
Lropehead_done:
    add x0, x6, x5, lsl #2
    add x1, x7, x5, lsl #2
    subs x4, x4, #1
    b.ne Lropehead
Lroperet:
    ret
