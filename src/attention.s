.text
.p2align 2
// Single-query/head scaled dot-product attention over a selected KV range.
// x0=query[d], x1=keys, x2=values, x3=output[d], x4=scores[n],
// x5=n (positive), x6=d (positive), x7=KV token stride in float elements.
// Input keys/query already normalized and rotated. Caller selects causal/window
// range and GQA head. Output must not alias query or KV; scores is scratch.
.globl _plamo_attention_head
_plamo_attention_head:
    stp x29, x30, [sp, #-96]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x6
    lsl x26, x7, #2
    ucvtf s0, x25
    fsqrt s0, s0
    fmov s1, #1.0
    fdiv s0, s1, s0
    str s0, [sp, #80]
    mov x8, #0
Lscore4:
    sub x9, x24, x8
    cmp x9, #4
    b.lo Lscore
    tst x25, #3
    b.ne Lscore
    mov x10, x20
    add x11, x10, x26
    add x12, x11, x26
    add x13, x12, x26
    mov x9, #0
    movi v0.4s, #0
Ldot4:
    lsl x14, x9, #2
    ldr q1, [x19, x14]
    ldr q2, [x10, x14]
    ldr q3, [x11, x14]
    ldr q4, [x12, x14]
    ldr q5, [x13, x14]
    // Transpose four keys: each accumulator lane is one key token.
    trn1 v6.4s, v2.4s, v3.4s
    trn2 v7.4s, v2.4s, v3.4s
    trn1 v16.4s, v4.4s, v5.4s
    trn2 v17.4s, v4.4s, v5.4s
    zip1 v2.2d, v6.2d, v16.2d
    zip1 v3.2d, v7.2d, v17.2d
    zip2 v4.2d, v6.2d, v16.2d
    zip2 v5.2d, v7.2d, v17.2d
    fmla v0.4s, v2.4s, v1.s[0]
    fmla v0.4s, v3.4s, v1.s[1]
    fmla v0.4s, v4.4s, v1.s[2]
    fmla v0.4s, v5.4s, v1.s[3]
    add x9, x9, #4
    cmp x9, x25
    b.lo Ldot4
    ldr s1, [sp, #80]
    dup v1.4s, v1.s[0]
    fmul v0.4s, v0.4s, v1.4s
    lsl x14, x8, #2
    str q0, [x23, x14]
    add x20, x20, x26, lsl #2
    add x8, x8, #4
    cmp x8, x24
    b.lo Lscore4
    b Lscores_done
Lscore:
    mov x9, #0
    fmov s0, wzr
Ldot:
    ldr s1, [x19, x9, lsl #2]
    ldr s2, [x20, x9, lsl #2]
    fmadd s0, s1, s2, s0
    add x9, x9, #1
    cmp x9, x25
    b.lo Ldot
    ldr s1, [sp, #80]
    fmul s0, s0, s1
    str s0, [x23, x8, lsl #2]
    add x20, x20, x26
    add x8, x8, #1
    cmp x8, x24
    b.lo Lscore
Lscores_done:
    mov x0, x23
    mov x1, x24
    bl _plamo_softmax
    mov x8, #0
Loutput4:
    sub x9, x25, x8
    cmp x9, #4
    b.lo Loutput
    mov x9, #0
    add x10, x21, x8, lsl #2
    movi v0.4s, #0
Lweighted4:
    ldr s1, [x23, x9, lsl #2]
    dup v1.4s, v1.s[0]
    ldr q2, [x10]
    fmla v0.4s, v1.4s, v2.4s
    add x10, x10, x26
    add x9, x9, #1
    cmp x9, x24
    b.lo Lweighted4
    lsl x11, x8, #2
    str q0, [x22, x11]
    add x8, x8, #4
    cmp x8, x25
    b.lo Loutput4
    b Lattention_done
Loutput:
    mov x9, #0
    add x10, x21, x8, lsl #2
    fmov s0, wzr
Lweighted:
    ldr s1, [x23, x9, lsl #2]
    ldr s2, [x10]
    fmadd s0, s1, s2, s0
    add x10, x10, x26
    add x9, x9, #1
    cmp x9, x24
    b.lo Lweighted
    str s0, [x22, x8, lsl #2]
    add x8, x8, #1
    cmp x8, x25
    b.lo Loutput
Lattention_done:
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

// Grouped-query attention, one query token. x0 points to descriptor:
// {query,keys,values,output,scores,tokens,head_dim,query_heads,kv_heads}
// FP32 token-major KV [tokens,kv_heads,head_dim]. Query/output [qh,dim].
// Scores[tokens] reused per head. Validated positive sizes; qh divisible by kvh.
// Keys/query already have RMSNorm/RoPE applied; caller selects window range.
.globl _plamo_gqa
_plamo_gqa:
    stp x29, x30, [sp, #-64]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x19, x0
    ldp x21, x22, [x19, #56] // qheads, kvheads
    udiv x23, x21, x22 // group size
    mov x20, #0
Lgqahead:
    ldr x6, [x19, #48]
    mul x24, x20, x6
    lsl x24, x24, #2
    ldr x0, [x19]
    add x0, x0, x24
    ldr x3, [x19, #24]
    add x3, x3, x24
    udiv x8, x20, x23
    mul x8, x8, x6
    lsl x8, x8, #2
    ldp x1, x2, [x19, #8]
    add x1, x1, x8
    add x2, x2, x8
    ldp x4, x5, [x19, #32]
    mul x7, x22, x6
    bl _plamo_attention_head
    add x20, x20, #1
    cmp x20, x21
    b.lo Lgqahead
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret
