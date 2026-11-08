.text
.p2align 2
// x0=BF16 embedding[vocab,hidden],x1=token,x2=vocab,x3=hidden,
// x4=FP32 output[hidden]. Returns 1=success,0=out-of-range (output untouched).
// Caller validates allocation/dimension products at model load time.
.globl _plamo_embedding
_plamo_embedding:
    cmp x1, x2
    b.hs Lembbad
    mul x5, x1, x3
    add x0, x0, x5, lsl #1
    cbz x3, Lembok
Lembvec:
    cmp x3, #4
    b.lo Lembtail
    ldr d0, [x0], #8
    shll v0.4s, v0.4h, #16
    str q0, [x4], #16
    sub x3, x3, #4
    b Lembvec
Lembtail:
    cbz x3, Lembok
    ldrh w5, [x0], #2
    lsl w5, w5, #16
    str w5, [x4], #4
    sub x3, x3, #1
    b Lembtail
Lembok:
    mov x0, #1
    ret
Lembbad:
    mov x0, #0
    ret

// x0=FP32 logits,x1=count. Return first maximal index (PyTorch tie rule).
// Empty returns UINT64_MAX. NaN is rejected as UINT64_MAX, including late NaNs.
.globl _plamo_argmax
_plamo_argmax:
    cbz x1, Largbad
    ldr s0, [x0]
    fcmp s0, s0
    b.vs Largbad
    mov x2, #0
    mov x3, #1
Largloop:
    cmp x3, x1
    b.hs Largdone
    ldr s1, [x0, x3, lsl #2]
    fcmp s1, s1
    b.vs Largbad
    fcmp s1, s0
    b.le Largnext
    fmov s0, s1
    mov x2, x3
Largnext:
    add x3, x3, #1
    b Largloop
Largdone:
    mov x0, x2
    ret
Largbad:
    mov x0, #-1
    ret
