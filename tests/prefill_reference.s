.text
.p2align 2
// Layer-major prompt processing: x0=model,x1=uint32 IDs,x2=count,
// x3=FP32 states[count,hidden]. Positive count, scratch disjoint from model.
// Returns 1/0. Leaves caches advanced and logits for the final prompt token.
// Model state is copied from final token to preserve forward interface.
.globl _plamo_prefill_reference
_plamo_prefill_reference:
    cbz x2, Lprefillearly
    stp x19, x30, [sp, #-80]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    stp x26, x27, [sp, #64]
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
    mov x25, #0
    ldr x26, [x19, #40]
Lprefilllayer:
    cmp x25, x26
    b.hs Lprefillfinal
    ldr x9, [x19, #32]
    ldr x27, [x9, x25, lsl #3]
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
    ldp x26, x27, [sp, #64]
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #80
    ret
Lprefillearly:
    mov x0, #0
    ret
.section __TEXT,__const
.p2align 2
Lprefillepsilon: .float 0.000001
