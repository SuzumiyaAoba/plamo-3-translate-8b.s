.text
.p2align 2
// x0=descriptor {input,qkv_weight,q_norm,k_norm,output,hidden,qheads,kvheads,dim}.
// BF16 weights; output FP32[(qh+2*kh)*dim], laid out Q then K then V.
// Performs projection then per-head offset-1 RMSNorm on Q/K only.
// Scratch output disjoint from input/weights. Positive validated dimensions.
.globl _plamo_qkv
_plamo_qkv:
    stp x19, x30, [sp, #-64]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    mov x19, x0
    ldp x20, x21, [x19, #48]
    ldr x22, [x19, #64]
    ldr x0, [x19, #8]
    ldr x1, [x19]
    ldr x2, [x19, #32]
    add x3, x20, x21, lsl #1
    mul x3, x3, x22
    ldr x4, [x19, #40]
    bl _plamo_bf16_matvec
    ldr x0, [x19, #32]
    ldp x1, x2, [x19, #16]
    mov x3, x20
    mov x4, x21
    mov x5, x22
    bl _plamo_qk_norm
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #64
    ret

// x0=projected QKV,x1=q norm,x2=k norm,x3=qheads,x4=kvheads,x5=dim.
// Normalize Q/K heads in place. V stays untouched; buffers validated/nonempty.
.globl _plamo_qk_norm
_plamo_qk_norm:
    stp x19, x30, [sp, #-64]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, #0
Lqknormloop:
    mov x0, x19
    mov x1, x20
    cmp x25, x22
    csel x1, x20, x21, lo
    mov x2, x19
    mov x3, x24
    adrp x9, Lqepsilon@PAGE
    ldr s0, [x9, Lqepsilon@PAGEOFF]
    fmov s1, #1.0
    bl _plamo_rmsnorm
    add x19, x19, x24, lsl #2
    add x25, x25, #1
    add x9, x22, x23
    cmp x25, x9
    b.lo Lqknormloop
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #64
    ret
.section __TEXT,__const
.p2align 2
Lqepsilon: .float 0.000001
