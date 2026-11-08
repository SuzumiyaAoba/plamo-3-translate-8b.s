.text
.p2align 2
// x0=original JSONL,x1=end,x2=token ID,x3=byte output,x4=capacity,
// x5=Unicode scratch,x6=scratch capacity. Return bytes or UINT64_MAX.
// BYTE tokens emit their raw byte; caller concatenates bytes before UTF-8 display.
.globl _plamo_decode_token
_plamo_decode_token:
    stp x19, x30, [sp, #-128]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x6
Ldecskip:
    cbz x21, Ldecline
    cmp x19, x20
    b.hs Ldecbad
    ldrb w9, [x19], #1
    cmp w9, #10
    b.ne Ldecskip
    sub x21, x21, #1
    b Ldecskip
Ldecline:
    mov x1, x19
Ldecend:
    cmp x1, x20
    b.hs Ldecparse
    ldrb w9, [x1], #1
    cmp w9, #10
    b.ne Ldecend
Ldecparse:
    mov x0, x19
    mov x2, x24
    mov x3, x25
    add x4, sp, #80
    bl _plamo_vocab_row
    cbz x0, Ldecbad
    ldr x20, [sp, #80]
    ldr x9, [sp, #96]
    cmp x9, #1
    b.eq Ldecbyte
    mov x19, #0
    mov x21, #0
Ldecunicode:
    cmp x19, x20
    b.hs Ldecdone
    ldr w0, [x24, x19, lsl #2]
    adrp x1, Lidentity@PAGE
    add x1, x1, Lidentity@PAGEOFF
    add x2, sp, #104
    bl _plamo_byte_fallback
    cbz x0, Ldecbad
    sub x9, x23, x21
    cmp x0, x9
    b.hi Ldecbad
    mov x9, #0
    add x10, sp, #104
Ldecbytes:
    ldr w11, [x10, x9, lsl #2]
    strb w11, [x22, x21]
    add x21, x21, #1
    add x9, x9, #1
    cmp x9, x0
    b.lo Ldecbytes
    add x19, x19, #1
    b Ldecunicode
Ldecbyte:
    cbz x23, Ldecbad
    cmp x20, #6
    b.ne Ldecbad
    mov x19, #3
    mov w20, #0
Ldecbytehex:
    ldr w9, [x24, x19, lsl #2]
    sub w10, w9, #48
    cmp w10, #9
    b.ls Ldecbyteok
    orr w9, w9, #32
    sub w10, w9, #97
    cmp w10, #5
    b.hi Ldecbad
    add w10, w10, #10
Ldecbyteok:
    lsl w20, w20, #4
    orr w20, w20, w10
    add x19, x19, #1
    cmp x19, #5
    b.lo Ldecbytehex
    strb w20, [x22]
    mov x21, #1
Ldecdone:
    mov x0, x21
    b Ldecreturn
Ldecbad:
    mov x0, #-1
Ldecreturn:
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #128
    ret
.section __TEXT,__const
.p2align 2
Lidentity:
.long 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255
