.text
.p2align 2
// x0=cursor,x1=end,x2=result {next,codepoint}. 1=valid,0=invalid/end.
// Strict UTF-8: rejects overlongs, surrogates, >U+10FFFF, truncation.
.globl _plamo_utf8_next
_plamo_utf8_next:
    stp xzr, xzr, [x2]
    cmp x0, x1
    b.hs Lu8bad
    ldrb w3, [x0], #1
    cmp w3, #128
    b.lo Lu8done
    cmp w3, #194
    b.lo Lu8bad
    cmp w3, #224
    b.lo Lu8two
    cmp w3, #240
    b.lo Lu8three
    cmp w3, #245
    b.hs Lu8bad
    and w3, w3, #7
    mov x4, #3
    mov w5, #0x10000
    b Lu8loop
Lu8three:
    and w3, w3, #15
    mov x4, #2
    mov w5, #0x800
    b Lu8loop
Lu8two:
    and w3, w3, #31
    mov x4, #1
    mov w5, #128
Lu8loop:
    cmp x0, x1
    b.hs Lu8bad
    ldrb w6, [x0], #1
    and w7, w6, #0xc0
    cmp w7, #0x80
    b.ne Lu8bad
    and w6, w6, #63
    lsl w3, w3, #6
    orr w3, w3, w6
    subs x4, x4, #1
    b.ne Lu8loop
    cmp w3, w5
    b.lo Lu8bad
    mov w6, #0xd800
    sub w6, w3, w6
    cmp w6, #0x800
    b.lo Lu8bad
    mov w6, #0x10ffff
    cmp w3, w6
    b.hi Lu8bad
Lu8done:
    stp x0, x3, [x2]
    mov x0, #1
    ret
Lu8bad:
    mov x0, #0
    ret

// x0=Unicode scalar,x1=uint32 byte-token IDs[256],x2=uint32 output[4].
// Return number of IDs written, 0=invalid scalar (output untouched).
.globl _plamo_byte_fallback
_plamo_byte_fallback:
    mov w3, #0x10ffff
    cmp x0, x3
    b.hi Lfallbackbad
    mov x3, #0xd800
    sub x3, x0, x3
    cmp x3, #0x800
    b.lo Lfallbackbad
    cmp x0, #128
    b.lo Lfallbackone
    cmp x0, #0x800
    b.lo Lfallbacktwo
    mov x3, #0x10000
    cmp x0, x3
    b.lo Lfallbackthree
    mov x4, #18
    mov w5, #0xf0
    mov x6, #4
    b Lfallbackfirst
Lfallbackthree:
    mov x4, #12
    mov w5, #0xe0
    mov x6, #3
    b Lfallbackfirst
Lfallbacktwo:
    mov x4, #6
    mov w5, #0xc0
    mov x6, #2
Lfallbackfirst:
    lsr x3, x0, x4
    orr x3, x3, x5
    ldr w7, [x1, x3, lsl #2]
    str w7, [x2], #4
Lfallbackcontinuation:
    sub x4, x4, #6
    lsr x3, x0, x4
    and x3, x3, #63
    orr x3, x3, #0x80
    ldr w7, [x1, x3, lsl #2]
    str w7, [x2], #4
    cbnz x4, Lfallbackcontinuation
    mov x0, x6
    ret
Lfallbackone:
    ldr w7, [x1, x0, lsl #2]
    str w7, [x2]
    mov x0, #1
    ret
Lfallbackbad:
    mov x0, #0
    ret
