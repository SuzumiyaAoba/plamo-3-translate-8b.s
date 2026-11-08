.text
.p2align 2
// Same descriptor as encode_segment, codepoints contain U+EE00 boundaries.
// Ignore boundary markers and concatenate encodings of intervening segments.
// Caller provides scratch large enough for the longest segment (whole input
// capacity is sufficient). Original descriptor is never modified.
.globl _plamo_encode_marked
_plamo_encode_marked:
    stp x19, x30, [sp, #-160]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    ldp x9, x10, [x0]
    stp x9, x10, [sp, #80]
    ldp x9, x10, [x0, #16]
    stp x9, x10, [sp, #96]
    ldp x9, x10, [x0, #32]
    stp x9, x10, [sp, #112]
    ldp x9, x10, [x0, #48]
    stp x9, x10, [sp, #128]
    ldp x9, x10, [x0, #64]
    stp x9, x10, [sp, #144]
    ldp x19, x20, [x0, #8]
    mov x21, #0 // scan position
    mov x22, #0 // total output
    ldp x24, x25, [x0, #64]
Lmarkedstart:
    cmp x21, x20
    b.hs Lmarkeddone
    ldr w9, [x19, x21, lsl #2]
    mov w10, #0xee00
    cmp w9, w10
    b.ne Lmarkedsegment
    add x21, x21, #1
    b Lmarkedstart
Lmarkedsegment:
    mov x23, x21
Lmarkedscan:
    add x21, x21, #1
    cmp x21, x20
    b.hs Lmarkedcall
    ldr w9, [x19, x21, lsl #2]
    cmp w9, w10
    b.ne Lmarkedscan
Lmarkedcall:
    add x9, x19, x23, lsl #2
    sub x10, x21, x23
    stp x9, x10, [sp, #88]
    add x9, x24, x22, lsl #2
    sub x10, x25, x22
    stp x9, x10, [sp, #144]
    add x0, sp, #80
    bl _plamo_encode_segment
    cmn x0, #1
    b.eq Lmarkedreturn
    add x22, x22, x0
    b Lmarkedstart
Lmarkeddone:
    mov x0, x22
Lmarkedreturn:
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #160
    ret
