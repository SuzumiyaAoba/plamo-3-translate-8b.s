.text
.p2align 2
// Layer-major prompt processing: x0=model,x1=uint32 IDs,x2=count,
// x3=FP32 states[count,hidden]. Positive count, scratch disjoint from model.
// Returns 1/0. Leaves caches advanced and logits for the final prompt token.
// Model state is copied from final token to preserve forward interface.
.globl _plamo_prefill
_plamo_prefill:
    cbz x2, Lprefillearly
    stp x19, x30, [sp, #-176]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    stp x26, x27, [sp, #64]
    stp xzr, xzr, [sp, #80]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    ldr x23, [x19, #16]
    mov x24, #0
Lprefillemb:
    ldr x0, [x19]
    ldr w1, [x20, x24, lsl #2]
    ldr x2, [x19, #8]
    mov x3, x23
    mul x9, x24, x23
    add x4, x22, x9, lsl #2
    bl _plamo_embedding
    cbz x0, Lprefillreturn
    add x24, x24, #1
    cmp x24, x21
    b.lo Lprefillemb
    mov x20, #0
    ldr x26, [x19, #40]
    cbz x26, Lprefill_workspace_done
    cmp x21, #2
    b.lo Lprefill_workspace_done
    // A single workspace for every layer; dimensions are fixed per model.
    ldr x9, [x19, #32]
    ldr x9, [x9]
    ldr x10, [x9, #40]
    ldr x10, [x10]
    ldp x11, x12, [x10, #48]
    ldr x13, [x10, #64]
    mul x14, x11, x13
    str x14, [sp, #144] // attention width
    add x11, x11, x12, lsl #1
    mul x11, x11, x13
    str x11, [sp, #136] // QKV width
    ldr x9, [x9, #48]
    ldr x9, [x9, #72]
    lsl x9, x9, #1
    str x9, [sp, #152] // gate/up width
    add x9, x9, x14
    add x9, x9, x11
    add x9, x9, x23, lsl #1
    mul x9, x9, x21
    lsl x0, x9, #2
    str x0, [sp, #88]
    bl _plamo_allocate
    cbz x0, Lprefillreturn
    mov x20, x0
    str x0, [sp, #80]
    // normalized, QKV, attention, branch, gate/up
    str x0, [sp, #96]
    mul x9, x21, x23
    add x0, x0, x9, lsl #2
    str x0, [sp, #104]
    ldr x9, [sp, #136]
    mul x9, x9, x21
    add x0, x0, x9, lsl #2
    str x0, [sp, #112]
    ldr x9, [sp, #144]
    mul x9, x9, x21
    add x0, x0, x9, lsl #2
    str x0, [sp, #120]
    mul x9, x21, x23
    add x0, x0, x9, lsl #2
    str x0, [sp, #128]
Lprefill_workspace_done:
    mov x25, #0
Lprefilllayer:
    cmp x25, x26
    b.hs Lprefillfinal
    ldr x9, [x19, #32]
    ldr x27, [x9, x25, lsl #3]
    cbz x20, Lprefill_separate
    mov x0, x27
    mov x1, x22
    mov x2, x21
    add x3, sp, #96
    ldr x9, [x19, #48]
    ldr s0, [x9, x25, lsl #2]
    bl _plamo_layer_batch
    cbz x0, Lprefillreturn
    b Lprefill_nextlayer
Lprefill_separate:
    mov x24, #0
Lprefilltoken:
    mul x9, x24, x23
    add x9, x22, x9, lsl #2
    str x9, [x27]
    mov x0, x27
    ldr x9, [x19, #48]
    ldr s0, [x9, x25, lsl #2]
    bl _plamo_layer
    cbz x0, Lprefillreturn
    add x24, x24, #1
    cmp x24, x21
    b.lo Lprefilltoken
Lprefill_nextlayer:
    add x25, x25, #1
    b Lprefilllayer
Lprefillfinal:
    sub x9, x21, #1
    mul x9, x9, x23
    add x20, x22, x9, lsl #2
    ldr x21, [x19, #24]
    mov x9, #0
Lprefillcopy:
    ldr w10, [x20, x9, lsl #2]
    str w10, [x21, x9, lsl #2]
    add x9, x9, #1
    cmp x9, x23
    b.lo Lprefillcopy
    mov x0, x21
    ldp x1, x2, [x19, #56]
    mov x3, x23
    adrp x9, Lprefillepsilon@PAGE
    ldr s0, [x9, Lprefillepsilon@PAGEOFF]
    fmov s1, #1.0
    bl _plamo_rmsnorm
    ldp x0, x2, [x19, #72]
    ldr x1, [x19, #64]
    ldp x3, x4, [x19, #8]
    bl _plamo_bf16_matvec
    mov x0, #1
Lprefillreturn:
    str x0, [sp, #160]
    ldp x0, x1, [sp, #80]
    cbz x0, Lprefill_freed
    bl _plamo_unmap_file
Lprefill_freed:
    ldr x0, [sp, #160]
    ldp x26, x27, [sp, #64]
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #176
    ret
Lprefillearly:
    mov x0, #0
    ret
.section __TEXT,__const
.p2align 2
Lprefillepsilon: .float 0.000001
