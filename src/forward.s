.text
.p2align 2
// x0=model {embedding,vocab,hidden,state,layers,count,scales,final_norm,
//           normalized,lm_head,logits}, x1=token ID.
// layers=array of layer pointers, scales=FP32 RoPE concentrations[count].
// All allocations/dimensions/weights validated before use. Returns 1/0.
// On layer failure earlier layer caches may have advanced: caller must reset
// caches before retrying. Input token failure makes no state changes.
.globl _plamo_forward
_plamo_forward:
    adrp x9, _plamo_forward_kernel@PAGE
    ldr x9, [x9, _plamo_forward_kernel@PAGEOFF]
    cbz x9, Lforward_cpu
    stp x29, x30, [sp, #-32]!
    stp x0, x1, [sp, #16]
    adrp x2, _plamo_compute_context@PAGE
    ldr x2, [x2, _plamo_compute_context@PAGEOFF]
    blr x9
    cbz x0, Lforward_fallback
    // The backend uses -1 for an execution failure, 0 for an untouched
    // cache miss, and 1 for success. Never retry a partially executed graph.
    cmp x0, #1
    cset x0, eq
    ldp x29, x30, [sp], #32
    ret
Lforward_fallback:
    ldp x0, x1, [sp, #16]
    ldp x29, x30, [sp], #32
Lforward_cpu:
    mov x2, #1
    b Lforwardcommon
// Prefill-only entry: caches/state advance, normalized/logits stay untouched.
.globl _plamo_forward_hidden
_plamo_forward_hidden:
    mov x2, #0
Lforwardcommon:
    stp x19, x30, [sp, #-48]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    mov x23, x2
    mov x19, x0
    ldr x0, [x19]
    ldp x2, x3, [x19, #8]
    ldr x4, [x19, #24]
    bl _plamo_embedding
    cbz x0, Lforwardreturn
    mov x20, #0
    ldr x21, [x19, #40]
    ldr x22, [x19, #32]
Lforwardlayer:
    cmp x20, x21
    b.hs Lforwardfinal
    ldr x0, [x22, x20, lsl #3]
    ldr x9, [x19, #24]
    str x9, [x0]
    ldr x9, [x19, #48]
    ldr s0, [x9, x20, lsl #2]
    bl _plamo_layer
    cbz x0, Lforwardreturn
    add x20, x20, #1
    b Lforwardlayer
Lforwardfinal:
    cbz x23, Lforwardok
    ldr x0, [x19, #24]
    ldp x1, x2, [x19, #56]
    ldr x3, [x19, #16]
    adrp x9, Lforwardepsilon@PAGE
    ldr s0, [x9, Lforwardepsilon@PAGEOFF]
    fmov s1, #1.0
    bl _plamo_rmsnorm
    ldp x0, x2, [x19, #72]
    ldr x1, [x19, #64]
    ldp x3, x4, [x19, #8]
    bl _plamo_bf16_matvec
Lforwardok:
    mov x0, #1
Lforwardreturn:
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #48
    ret
.section __TEXT,__const
.p2align 2
Lforwardepsilon: .float 0.000001
