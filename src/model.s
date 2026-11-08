.text
.p2align 2
// MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).
// x0=BF16 gate_up[2*intermediate,hidden], x1=BF16 down[hidden,intermediate],
// x2=input FP32[hidden],x3=output FP32[hidden],x4=scratch FP32[2*intermediate],
// x5=hidden,x6=intermediate. Scratch must not overlap weights/input/output.
// Input/output may alias; caller supplies validated, positive dimensions.
.globl _plamo_mlp
_plamo_mlp:
    stp x29, x30, [sp, #-80]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x19, x1
    mov x20, x3
    mov x21, x4
    mov x22, x5
    mov x23, x6
    mov x1, x2
    mov x2, x21
    lsl x3, x23, #1
    mov x4, x22
    bl _plamo_bf16_matvec
    mov x0, x21
    mov x1, x21
    mov x2, x23
    bl _plamo_swiglu
    mov x0, x19
    mov x1, x21
    mov x2, x20
    mov x3, x22
    mov x4, x23
    bl _plamo_bf16_matvec
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret

// Residual addition x0=destination,x1=branch,x2=count. Exact alias allowed.
.globl _plamo_residual
_plamo_residual:
    cbz x2, 2f
Lresidual4:
    cmp x2, #4
    b.lo 1f
    ldr q0, [x0]
    ldr q1, [x1], #16
    fadd v0.4s, v0.4s, v1.4s
    str q0, [x0], #16
    sub x2, x2, #4
    b Lresidual4
1:  cbz x2, 2f
    ldr s0, [x0]
    ldr s1, [x1], #4
    fadd s0, s0, s1
    str s0, [x0], #4
    subs x2, x2, #1
    b.ne 1b
2:  ret

// x0=MLP branch descriptor (all pointers and dimensions validated):
// {state,pre_weight,post_weight,gate_up,down,normalized,branch,scratch,hidden,intermediate}
// state[hidden] updated in place. normalized/branch[hidden], scratch[2*intermediate]
// are disjoint reusable FP32 buffers. BF16 weights, FP32 activations.
.globl _plamo_mlp_branch
_plamo_mlp_branch:
    stp x19, x30, [sp, #-16]!
    mov x19, x0
    ldp x0, x1, [x19]
    ldr x2, [x19, #40]
    ldr x3, [x19, #64]
    adrp x9, Lbranchconstants@PAGE
    add x9, x9, Lbranchconstants@PAGEOFF
    ldr s0, [x9]
    fmov s1, #1.0
    bl _plamo_rmsnorm
    ldp x0, x1, [x19, #24]
    ldp x2, x3, [x19, #40]
    ldr x4, [x19, #56]
    ldp x5, x6, [x19, #64]
    bl _plamo_mlp
    ldr x0, [x19, #48]
    ldr x1, [x19, #16]
    mov x2, x0
    ldr x3, [x19, #64]
    adrp x9, Lbranchconstants@PAGE
    add x9, x9, Lbranchconstants@PAGEOFF
    ldp s0, s1, [x9]
    bl _plamo_rmsnorm
    ldr x0, [x19]
    ldr x1, [x19, #48]
    ldr x2, [x19, #64]
    bl _plamo_residual
    ldp x19, x30, [sp], #16
    ret
.section __TEXT,__const
.p2align 2
Lbranchconstants:
.float 0.000001,0.08944271909999159
