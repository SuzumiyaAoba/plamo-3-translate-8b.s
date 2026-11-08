.text
.p2align 2
// x0=shards,x1=count,x2=schema {name,rank,dimensions,slot}[] terminated by NULL.
// Resolve all weights and validate shapes. Returns 1/0; failure clears all slots.
.globl _plamo_bind_weights
_plamo_bind_weights:
    stp x19, x30, [sp, #-48]!
    stp x20, x21, [sp, #16]
    str x22, [sp, #32]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x2
Lbindloop:
    ldr x2, [x21]
    cbz x2, Lbinddone
    mov x0, x19
    mov x1, x20
    ldp x3, x4, [x21, #8]
    bl _plamo_resolve_weight
    cbz x0, Lbindbad
    ldr x9, [x21, #24]
    str x0, [x9]
    add x21, x21, #32
    b Lbindloop
Lbindbad:
    ldr x9, [x22]
    cbz x9, Lbindfailed
    ldr x9, [x22, #24]
    str xzr, [x9]
    add x22, x22, #32
    b Lbindbad
Lbindfailed:
    mov x0, #0
    b Lbindreturn
Lbinddone:
    mov x0, #1
Lbindreturn:
    ldr x22, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #48
    ret
