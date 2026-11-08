// Parse one BF16 tensor descriptor, field order independent.
// x0=JSON cursor, x1=end, x2=data bytes, x3=result (64 bytes):
// {next,offset,byte_length,rank,dimensions[4]}. Failure clears result.
.text
.p2align 2
.macro next_token
    mov x0, x19
    mov x1, x20
    add x2, sp, #96
    bl _plamo_json_next
    ldr x19, [sp, #96]
.endm
.macro expect code
    next_token
    cmp x0, #\code
    b.ne Ltfail
.endm
.globl _plamo_tensor_descriptor
_plamo_tensor_descriptor:
    stp x29, x30, [sp, #-192]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, #0 // seen fields
    mov x24, #1 // element count
    stp xzr, xzr, [x22]
    stp xzr, xzr, [x22, #16]
    stp xzr, xzr, [x22, #32]
    stp xzr, xzr, [x22, #48]
    expect 123
Ltfield:
    expect 256
    ldp x25, x26, [sp, #104] // start,length
    // Compare raw ASCII field names; escaped keys intentionally rejected.
    cmp x26, #5
    b.ne Ltoffkey
    ldr w27, [x25]
    mov w28, #0x7464
    movk w28, #0x7079, lsl #16 // dtyp
    cmp w27, w28
    b.ne Ltshapecheck
    ldrb w27, [x25, #4]
    cmp w27, #101
    b.ne Ltfail
    tbnz x23, #0, Ltfail
    orr x23, x23, #1
    expect 58
    expect 256
    ldp x25, x26, [sp, #104]
    cmp x26, #4
    b.ne Ltfail
    ldr w27, [x25]
    mov w28, #0x4642
    movk w28, #0x3631, lsl #16 // BF16
    cmp w27, w28
    b.ne Ltfail
    b Ltseparator
Ltshapecheck:
    mov w28, #0x6873
    movk w28, #0x7061, lsl #16 // shap
    cmp w27, w28
    b.ne Ltfail
    ldrb w27, [x25, #4]
    cmp w27, #101
    b.ne Ltfail
    tbnz x23, #1, Ltfail
    orr x23, x23, #2
    expect 58
    expect 91
    mov x25, #0
    next_token
    cmp x0, #93
    b.eq Ltshapedone
Ltshapeitem:
    cmp x0, #257
    b.ne Ltfail
    cmp x25, #4
    b.hs Ltfail
    ldr x26, [sp, #120]
    umulh x27, x24, x26
    cbnz x27, Ltfail
    mul x24, x24, x26
    add x27, x22, #32
    str x26, [x27, x25, lsl #3]
    add x25, x25, #1
    next_token
    cmp x0, #93
    b.eq Ltshapedone
    cmp x0, #44
    b.ne Ltfail
    next_token
    b Ltshapeitem
Ltshapedone:
    str x25, [x22, #24]
    b Ltseparator
Ltoffkey:
    cmp x26, #12
    b.ne Ltfail
    adrp x27, Loffsetkey@PAGE
    add x27, x27, Loffsetkey@PAGEOFF
    mov x28, #0
Ltkeyloop:
    ldrb w9, [x25, x28]
    ldrb w10, [x27, x28]
    cmp w9, w10
    b.ne Ltfail
    add x28, x28, #1
    cmp x28, #12
    b.lo Ltkeyloop
    tbnz x23, #2, Ltfail
    orr x23, x23, #4
    expect 58
    expect 91
    expect 257
    ldr x25, [sp, #120]
    expect 44
    expect 257
    ldr x26, [sp, #120]
    cmp x26, x25
    b.lo Ltfail
    cmp x26, x21
    b.hi Ltfail
    sub x26, x26, x25
    stp x25, x26, [x22, #8]
    expect 93
Ltseparator:
    next_token
    cmp x0, #44
    b.eq Ltfield
    cmp x0, #125
    b.ne Ltfail
    cmp x23, #7
    b.ne Ltfail
    tbnz x24, #63, Ltfail
    lsl x24, x24, #1
    ldr x25, [x22, #16]
    cmp x24, x25
    b.ne Ltfail
    str x19, [x22]
    mov x0, #1
    b Ltreturn
Ltfail:
    stp xzr, xzr, [x22]
    stp xzr, xzr, [x22, #16]
    stp xzr, xzr, [x22, #32]
    stp xzr, xzr, [x22, #48]
    mov x0, #0
Ltreturn:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #192
    ret
.section __TEXT,__const
Loffsetkey: .ascii "data_offsets"
