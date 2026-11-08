.text
.p2align 2
// x0=shards array {base,size},x1=count,x2=NUL name,x3=expected rank,
// x4=uint64 expected dimensions[rank]. Return BF16 data pointer or NULL.
// Rank <=4. All model setup uses this boundary to validate weight shapes.
.globl _plamo_resolve_weight
_plamo_resolve_weight:
    cmp x3, #4
    b.hi Lresolveearly
    stp x19, x30, [sp, #-128]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
Lresolveshard:
    cbz x20, Lresolvebad
    ldp x0, x1, [x19], #16
    mov x2, x21
    add x3, sp, #64
    bl _plamo_find_tensor
    sub x20, x20, #1
    cbz x0, Lresolveshard
    ldr x9, [sp, #80]
    cmp x9, x22
    b.ne Lresolvebad
    mov x10, #0
    add x11, sp, #88
Lresolvedim:
    cmp x10, x22
    b.hs Lresolveok
    ldr x12, [x11, x10, lsl #3]
    ldr x13, [x23, x10, lsl #3]
    cmp x12, x13
    b.ne Lresolvebad
    add x10, x10, #1
    b Lresolvedim
Lresolveok:
    ldr x0, [sp, #64]
    b Lresolvereturn
Lresolvebad:
    mov x0, #0
Lresolvereturn:
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #128
    ret
Lresolveearly:
    mov x0, #0
    ret
