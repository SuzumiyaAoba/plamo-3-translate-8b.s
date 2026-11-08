.text
.p2align 2
// x0=Unicode codepoints,x1=count,x2=uint8 boundary flags[count+1].
// OR hard boundaries around >=2 ASCII spaces or >=4 repeated characters.
// Python regex '.' excludes LF, reproduced here. Caller zeroes flags first.
// This stage does not recognize special tokens; those boundaries are separate.
.globl _plamo_repeat_boundaries
_plamo_repeat_boundaries:
    mov x3, #0
Lboundrun:
    cmp x3, x1
    b.hs Lbounddone
    ldr w4, [x0, x3, lsl #2]
    add x5, x3, #1
Lboundscan:
    cmp x5, x1
    b.hs Lboundclassify
    ldr w6, [x0, x5, lsl #2]
    cmp w6, w4
    b.ne Lboundclassify
    add x5, x5, #1
    b Lboundscan
Lboundclassify:
    sub x6, x5, x3
    cmp w4, #32
    b.ne Lboundrepeat
    cmp x6, #2
    b.hs Lboundmark
Lboundrepeat:
    cmp w4, #10
    b.eq Lboundnext
    cmp x6, #4
    b.lo Lboundnext
Lboundmark:
    mov w7, #1
    strb w7, [x2, x3]
    strb w7, [x2, x5]
Lboundnext:
    mov x3, x5
    b Lboundrun
Lbounddone:
    ret

// Mark Python regex (<\|plamo:[^|\s]{,64}\|>) matches, non-overlapping.
// x0=codepoints,x1=count,x2=flags[count+1]. OR into existing flags.
.globl _plamo_special_boundaries
_plamo_special_boundaries:
    mov x3, #0
    adrp x8, Lspecialprefix@PAGE
    add x8, x8, Lspecialprefix@PAGEOFF
Lspstart:
    sub x4, x1, x3
    cmp x4, #10
    b.lo Lspdone
    mov x4, #0
Lspprefix:
    add x5, x3, x4
    ldr w6, [x0, x5, lsl #2]
    ldrb w7, [x8, x4]
    cmp w6, w7
    b.ne Lspnext
    add x4, x4, #1
    cmp x4, #8
    b.lo Lspprefix
    add x4, x3, #8
    mov x9, #0
Lspbody:
    cmp x4, x1
    b.hs Lspnext
    ldr w6, [x0, x4, lsl #2]
    cmp w6, #124
    b.eq Lspclose
    cmp x9, #64
    b.hs Lspnext
    // Python str.isspace set, including C0 separators U+001C..U+001F.
    sub w7, w6, #9
    cmp w7, #4
    b.ls Lspnext
    sub w7, w6, #28
    cmp w7, #4
    b.ls Lspnext
    cmp w6, #0x85
    b.eq Lspnext
    cmp w6, #0xa0
    b.eq Lspnext
    mov w7, #0x1680
    cmp w6, w7
    b.eq Lspnext
    mov w7, #0x2000
    sub w7, w6, w7
    cmp w7, #10
    b.ls Lspnext
    mov w7, #0x2028
    cmp w6, w7
    b.eq Lspnext
    add w7, w7, #1
    cmp w6, w7
    b.eq Lspnext
    mov w7, #0x202f
    cmp w6, w7
    b.eq Lspnext
    mov w7, #0x205f
    cmp w6, w7
    b.eq Lspnext
    mov w7, #0x3000
    cmp w6, w7
    b.eq Lspnext
    add x4, x4, #1
    add x9, x9, #1
    b Lspbody
Lspclose:
    add x5, x4, #1
    cmp x5, x1
    b.hs Lspnext
    ldr w6, [x0, x5, lsl #2]
    cmp w6, #62
    b.ne Lspnext
    add x4, x4, #2
    mov w7, #1
    strb w7, [x2, x3]
    strb w7, [x2, x4]
    mov x3, x4
    b Lspstart
Lspnext:
    add x3, x3, #1
    b Lspstart
Lspdone:
    ret
.section __TEXT,__const
Lspecialprefix: .ascii "<|plamo:"
