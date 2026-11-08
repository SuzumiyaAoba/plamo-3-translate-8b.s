// Lookup a BF16 tensor in an original mapped safetensors file.
// x0=file,x1=size,x2=name (NUL terminated),x3=result (64 bytes).
// result={data,bytes,rank,dimensions[4],reserved}; 1=found,0=absent/invalid.
// Tensor names are raw ASCII. __metadata__ is a string-to-string object.
.text
.p2align 2
.macro find_next
    mov x0, x19
    mov x1, x20
    add x2, sp, #96
    bl _plamo_json_next
    ldr x19, [sp, #96]
.endm
.macro find_expect code
    find_next
    cmp x0, #\code
    b.ne Lffail
.endm
.globl _plamo_find_tensor
_plamo_find_tensor:
    stp x29, x30, [sp, #-240]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x21, x2
    mov x22, x3
    stp xzr, xzr, [x22]
    stp xzr, xzr, [x22, #16]
    stp xzr, xzr, [x22, #32]
    stp xzr, xzr, [x22, #48]
    add x2, sp, #128
    bl _plamo_safetensors_view
    cbz x0, Lffail
    ldp x19, x20, [sp, #128]
    add x20, x19, x20
    ldp x23, x24, [sp, #144]
    find_expect 123
    find_next
    cmp x0, #125
    b.eq Lffail
Lfentry:
    cmp x0, #256
    b.ne Lffail
    ldp x25, x26, [sp, #104]
    // Name match, including exact length.
    mov x27, #0
    mov x28, #1
Lfcompare:
    cmp x27, x26
    b.hs Lfcompareend
    ldrb w9, [x21, x27]
    cbz w9, Lfnomatch
    ldrb w10, [x25, x27]
    cmp w9, w10
    b.ne Lfnomatch
    add x27, x27, #1
    b Lfcompare
Lfcompareend:
    ldrb w9, [x21, x27]
    cbz w9, Lfcompared
Lfnomatch:
    mov x28, #0
Lfcompared:
    // Detect metadata key separately.
    cmp x26, #12
    b.ne Lftensor
    adrp x27, Lfmetadata@PAGE
    add x27, x27, Lfmetadata@PAGEOFF
    mov x26, #0
Lfmetakey:
    ldrb w9, [x25, x26]
    ldrb w10, [x27, x26]
    cmp w9, w10
    b.ne Lftensor
    add x26, x26, #1
    cmp x26, #12
    b.lo Lfmetakey
    find_expect 58
    find_expect 123
    find_next
    cmp x0, #125
    b.eq Lfseparator
Lfmetaitem:
    cmp x0, #256
    b.ne Lffail
    find_expect 58
    find_expect 256
    find_next
    cmp x0, #125
    b.eq Lfseparator
    cmp x0, #44
    b.ne Lffail
    find_next
    b Lfmetaitem
Lftensor:
    find_expect 58
    mov x0, x19
    mov x1, x20
    mov x2, x24
    add x3, sp, #160
    bl _plamo_tensor_descriptor
    cbz x0, Lffail
    ldr x19, [sp, #160]
    cbz x28, Lfseparator
    ldp x9, x10, [sp, #168]
    add x9, x23, x9
    stp x9, x10, [x22]
    ldp x9, x10, [sp, #184]
    stp x9, x10, [x22, #16]
    ldp x9, x10, [sp, #200]
    stp x9, x10, [x22, #32]
    ldr x9, [sp, #216]
    str x9, [x22, #48]
    mov x0, #1
    b Lfreturn
Lfseparator:
    find_next
    cmp x0, #44
    b.ne Lffail
    find_next
    b Lfentry
Lffail:
    mov x0, #0
Lfreturn:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #240
    ret
.section __TEXT,__const
Lfmetadata: .ascii "__metadata__"
