.text
.p2align 2
// FP32 exp for finite inference inputs. Saturates below -87 to zero,
// above 88 to +inf. Range reduction, degree-7 Taylor on [-ln2/2,ln2/2].
// Leaf: clobbers x9,x10 and v1-v6 only.
.globl _plamo_exp_reference
_plamo_exp_reference:
    adrp x9, Lexpconst@PAGE
    add x9, x9, Lexpconst@PAGEOFF
    ldr s1, [x9]
    fcmp s0, s1
    b.lt Lexzero
    ldr s1, [x9, #4]
    fcmp s0, s1
    b.gt Lexinf
    ldr s1, [x9, #8]
    fmul s2, s0, s1
    frintn s2, s2
    fcvtzs w10, s2
    ldr s1, [x9, #12]
    fmsub s3, s2, s1, s0
    // fmsub computes r = x - n*ln2
    ldr s4, [x9, #16]
    ldr s5, [x9, #20]
    fmadd s4, s4, s3, s5
    ldr s5, [x9, #24]
    fmadd s4, s4, s3, s5
    ldr s5, [x9, #28]
    fmadd s4, s4, s3, s5
    ldr s5, [x9, #32]
    fmadd s4, s4, s3, s5
    ldr s5, [x9, #36]
    fmadd s4, s4, s3, s5
    fmov s5, #1.0
    fmadd s4, s4, s3, s5
    fmadd s4, s4, s3, s5
    add w10, w10, #127
    lsl w10, w10, #23
    fmov s6, w10
    fmul s0, s4, s6
    ret
Lexzero:
    fmov s0, wzr
    ret
Lexinf:
    mov w10, #0x7f800000
    fmov s0, w10
    ret

// In-place softmax, x0=FP32 array,x1=count. Finite inputs; zero count no-op.
.globl _plamo_softmax_reference
_plamo_softmax_reference:
    cbz x1, Lsoftret
    stp x19, x30, [sp, #-48]!
    stp x20, x21, [sp, #16]
    mov x19, x0
    mov x20, x1
    ldr s7, [x19]
    mov x21, #1
Lmax:
    cmp x21, x20
    b.hs Lexpstart
    ldr s0, [x19, x21, lsl #2]
    fmax s7, s7, s0
    add x21, x21, #1
    b Lmax
Lexpstart:
    str s7, [sp, #32]
    fmov s7, wzr
    str s7, [sp, #36]
    mov x21, #0
Lsoftloop:
    ldr s0, [x19, x21, lsl #2]
    ldr s7, [sp, #32]
    fsub s0, s0, s7
    bl _plamo_exp_reference
    str s0, [x19, x21, lsl #2]
    ldr s7, [sp, #36]
    fadd s7, s7, s0
    str s7, [sp, #36]
    add x21, x21, #1
    cmp x21, x20
    b.lo Lsoftloop
    mov x21, #0
Ldivide:
    ldr s0, [x19, x21, lsl #2]
    fdiv s0, s0, s7
    str s0, [x19, x21, lsl #2]
    add x21, x21, #1
    cmp x21, x20
    b.lo Ldivide
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #48
Lsoftret:
    ret

// x0=gate/up concatenated FP32 [2*n], x1=output[n], x2=n.
// Supports output aliasing the first half of input.
.globl _plamo_swiglu_reference
_plamo_swiglu_reference:
    cbz x2, Lgluret
    stp x19, x30, [sp, #-48]!
    stp x20, x21, [sp, #16]
    str x22, [sp, #32]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    add x22, x0, x2, lsl #2
Lgluloop:
    ldr s0, [x19]
    fneg s0, s0
    bl _plamo_exp_reference
    fmov s1, #1.0
    fadd s0, s0, s1
    ldr s1, [x19], #4
    fdiv s0, s1, s0
    ldr s1, [x22], #4
    fmul s0, s0, s1
    str s0, [x20], #4
    subs x21, x21, #1
    b.ne Lgluloop
    ldr x22, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #48
Lgluret:
    ret
.section __TEXT,__const
.p2align 2
Lexpconst:
.float -87.0,88.0,1.4426950408889634,0.6931471805599453
.float 0.0001984126984126984,0.001388888888888889,0.008333333333333333
.float 0.041666666666666664,0.16666666666666666,0.5
