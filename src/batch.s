.text
.p2align 2
// Contiguous rows of independent RMSNorm. x0=input,x1=BF16 weights,
// x2=output,x3=width,x4=rows,s0=epsilon,s1=offset. In-place supported.
.globl _plamo_rmsnorm_rows
_plamo_rmsnorm_rows:
    cbz x4, Lnr_ret
    stp x19, x30, [sp, #-64]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp s0, s1, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
Lnr_loop:
    mov x0, x19
    mov x1, x20
    mov x2, x21
    mov x3, x22
    ldp s0, s1, [sp, #48]
    bl _plamo_rmsnorm
    add x19, x19, x22, lsl #2
    add x21, x21, x22, lsl #2
    subs x23, x23, #1
    b.ne Lnr_loop
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #64
Lnr_ret:
    ret

// x0=gate_up[rows,2*width],x1=width,x2=rows. In-place in first half of each row.
.globl _plamo_swiglu_rows
_plamo_swiglu_rows:
    cbz x2, Lsr_ret
    stp x19, x30, [sp, #-32]!
    stp x20, x21, [sp, #16]
    mov x19, x0
    mov x20, x1
    mov x21, x2
Lsr_loop:
    mov x0, x19
    mov x1, x19
    mov x2, x20
    bl _plamo_swiglu
    add x19, x19, x20, lsl #3
    subs x21, x21, #1
    b.ne Lsr_loop
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #32
Lsr_ret:
    ret

// x0=layer,x1=states[tokens,hidden],x2=tokens,x3=workspace pointers:
// {normalized,QKV,attention,branch,gate_up}. s0=RoPE concentration.
// Every large linear projection shares weights across four input tokens.
// Causal Attention/cache updates stay in original token order. Descriptor
// buffers/pointers are not mutated; only states and KV caches are updated.
.globl _plamo_layer_batch
_plamo_layer_batch:
    stp x19, x30, [sp, #-112]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    stp x26, x27, [sp, #64]
    str x28, [sp, #80]
    str s0, [sp, #96]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    cbz x21, Lbatch_ok
    ldr x23, [x19, #56]
    ldr x24, [x19, #40]
    ldr x25, [x24]
    ldr x26, [x19, #48]
    ldp x9, x10, [x25, #48]
    ldr x11, [x25, #64]
    mul x28, x9, x11
    add x9, x9, x10, lsl #1
    mul x27, x9, x11
    // pre-Attention normalization
    mov x0, x20
    ldr x1, [x19, #8]
    ldr x2, [x22]
    mov x3, x23
    mov x4, x21
    adrp x9, Lbatchconstants@PAGE
    ldr s0, [x9, Lbatchconstants@PAGEOFF]
    fmov s1, #1.0
    bl _plamo_rmsnorm_rows
    // QKV projection
    ldr x0, [x25, #8]
    ldp x1, x2, [x22]
    mov x3, x27
    mov x4, x23
    mov x5, x21
    mov x6, x23
    mov x7, x27
    bl _plamo_bf16_linear
    str xzr, [sp, #104]
Lbatch_attention:
    ldr x9, [sp, #104]
    mul x10, x9, x27
    ldr x0, [x22, #8]
    add x0, x0, x10, lsl #2
    ldp x1, x2, [x25, #16]
    ldp x3, x4, [x25, #48]
    ldr x5, [x25, #64]
    bl _plamo_qk_norm
    ldr x9, [sp, #104]
    mul x10, x9, x27
    ldr x1, [x22, #8]
    add x1, x1, x10, lsl #2
    mul x10, x9, x28
    ldr x2, [x22, #16]
    add x2, x2, x10, lsl #2
    mov x0, x24
    ldr s0, [sp, #96]
    bl _plamo_mixer_attention
    cbz x0, Lbatch_return
    ldr x9, [sp, #104]
    add x9, x9, #1
    str x9, [sp, #104]
    cmp x9, x21
    b.lo Lbatch_attention
    // Attention output projection
    ldr x0, [x24, #32]
    ldp x1, x2, [x22, #16]
    mov x3, x23
    mov x4, x28
    mov x5, x21
    mov x6, x28
    mov x7, x23
    bl _plamo_bf16_linear
    // post-Attention norm, residual, pre-MLP norm
    ldr x0, [x22, #24]
    ldr x1, [x19, #16]
    mov x2, x0
    mov x3, x23
    mov x4, x21
    adrp x9, Lbatchconstants@PAGE
    add x9, x9, Lbatchconstants@PAGEOFF
    ldp s0, s1, [x9]
    bl _plamo_rmsnorm_rows
    mov x0, x20
    ldr x1, [x22, #24]
    mul x2, x21, x23
    bl _plamo_residual
    // A resident backend can keep the complete MLP on the GPU. It must leave
    // states untouched when declining, so streamed weights use the same path.
    adrp x9, _plamo_mlp_kernel@PAGE
    ldr x9, [x9, _plamo_mlp_kernel@PAGEOFF]
    cbz x9, Lbatch_mlp
    mov x0, x26
    mov x1, x20
    mov x2, x21
    adrp x3, _plamo_compute_context@PAGE
    ldr x3, [x3, _plamo_compute_context@PAGEOFF]
    blr x9
    cbnz x0, Lbatch_ok
Lbatch_mlp:
    mov x0, x20
    ldr x1, [x26, #8]
    ldr x2, [x22]
    mov x3, x23
    mov x4, x21
    adrp x9, Lbatchconstants@PAGE
    ldr s0, [x9, Lbatchconstants@PAGEOFF]
    fmov s1, #1.0
    bl _plamo_rmsnorm_rows
    // gate/up projection
    ldr x0, [x26, #24]
    ldr x1, [x22]
    ldr x2, [x22, #32]
    ldr x9, [x26, #72]
    lsl x3, x9, #1
    mov x4, x23
    mov x5, x21
    mov x6, x23
    mov x7, x3
    bl _plamo_bf16_linear
    ldr x0, [x22, #32]
    ldr x1, [x26, #72]
    mov x2, x21
    bl _plamo_swiglu_rows
    // down projection uses only the first half of each gate/up row
    ldr x0, [x26, #32]
    ldr x1, [x22, #32]
    ldr x2, [x22, #24]
    mov x3, x23
    ldr x4, [x26, #72]
    mov x5, x21
    lsl x6, x4, #1
    mov x7, x23
    bl _plamo_bf16_linear
    // post-MLP norm and residual
    ldr x0, [x22, #24]
    ldr x1, [x26, #16]
    mov x2, x0
    mov x3, x23
    mov x4, x21
    adrp x9, Lbatchconstants@PAGE
    add x9, x9, Lbatchconstants@PAGEOFF
    ldr s0, [x9]
    ldr s1, [x9, #8]
    bl _plamo_rmsnorm_rows
    mov x0, x20
    ldr x1, [x22, #24]
    mul x2, x21, x23
    bl _plamo_residual
Lbatch_ok:
    mov x0, #1
Lbatch_return:
    ldr x28, [sp, #80]
    ldp x26, x27, [sp, #64]
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #112
    ret
.section __TEXT,__const
.p2align 2
Lbatchconstants: .float 0.000001,0.2,0.08944271909999159
