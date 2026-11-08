.text
.p2align 2
// x0=mixer {qkv,cache,cache_rope,gqa,o_weight,output,hidden}; s0=RoPE scale.
.globl _plamo_mixer
_plamo_mixer:
    stp x19, x30, [sp, #-32]!
    str s0, [sp, #16]
    mov x19, x0
    ldr x0, [x19]
    bl _plamo_qkv
    ldr x9, [x19]
    ldr x1, [x9, #32]
    ldr x9, [x19, #24]
    ldr x2, [x9, #24]
    mov x0, x19
    ldr s0, [sp, #16]
    bl _plamo_mixer_attention
    cbz x0, Lmixer_return
    ldr x0, [x19, #32]
    ldr x9, [x19, #24]
    ldr x1, [x9, #24]
    ldr x2, [x19, #40]
    ldr x3, [x19, #48]
    ldp x9, x10, [x9, #48] // dim,qheads
    mul x4, x9, x10
    bl _plamo_bf16_matvec
    mov x0, #1
Lmixer_return:
    ldp x19, x30, [sp], #32
    ret

// Attention between projections. x0=mixer,x1=normalized QKV,
// x2=output[qheads*dim],s0=RoPE scale. Returns 1/0 (cache full).
// Stack-local GQA descriptor avoids retaining pointers to batch scratch.
.globl _plamo_mixer_attention
_plamo_mixer_attention:
    stp x19, x30, [sp, #-160]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    str s0, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    ldr x22, [x19, #8]
    ldr x23, [x19]
    ldr x24, [x19, #16]
    ldr x9, [x23, #48]
    ldr x10, [x23, #64]
    mul x9, x9, x10
    add x1, x20, x9, lsl #2
    ldr x9, [x22, #32]
    add x2, x1, x9, lsl #2
    mov x0, x22
    bl _plamo_cache_append
    cbz x0, Lma_return
    mov x0, x24
    ldr s0, [sp, #64]
    bl _plamo_cache_rope_cached
    mov x0, x20
    ldp x1, x2, [x24, #24]
    ldr x3, [x23, #48]
    ldr x4, [x23, #64]
    lsr x4, x4, #1
    bl _plamo_rope_apply
    ldr x25, [x19, #24]
    add x0, sp, #80
    ldr x9, [x24, #8]
    stp x20, x9, [x0]
    ldr x9, [x22, #8]
    stp x9, x21, [x0, #16]
    ldr x9, [x25, #32]
    ldr x10, [x22, #24]
    stp x9, x10, [x0, #32]
    ldp x9, x10, [x25, #48]
    stp x9, x10, [x0, #48]
    ldr x9, [x25, #64]
    str x9, [x0, #64]
    bl _plamo_gqa
    mov x0, #1
Lma_return:
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #160
    ret
