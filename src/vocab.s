.text
.p2align 2
// Parse one JSONL vocabulary row [string,score,type].
// x0=start,x1=end (single line),x2=uint32 codepoints,x3=capacity,
// x4=result {length,int64 score,type}; types NORMAL=0,BYTE=1,CONTROL=2,UNKNOWN=3.
// Returns 1/0. Failure clears result; codepoint buffer may contain partial data.
.macro vn
    mov x0, x19
    mov x1, x20
    add x2, sp, #80
    bl _plamo_json_next
    ldr x19, [sp, #80]
.endm
.macro ve code
    vn
    cmp x0, #\code
    b.ne Lvbad
.endm
.globl _plamo_vocab_row
_plamo_vocab_row:
    stp x19, x30, [sp, #-128]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    stp xzr, xzr, [x23]
    str xzr, [x23, #16]
    ve 91
    ve 256
    ldp x0, x1, [sp, #88]
    add x1, x0, x1
    mov x2, x21
    mov x3, x22
    bl _plamo_json_string
    cmn x0, #1
    b.eq Lvbad
    mov x24, x0
    ve 44
    // Skip whitespace before the signed decimal score.
Lvspace:
    cmp x19, x20
    b.hs Lvbad
    ldrb w9, [x19]
    cmp w9, #32
    b.eq Lvskip
    cmp w9, #9
    b.ne Lvscore
Lvskip:
    add x19, x19, #1
    b Lvspace
Lvscore:
    mov x0, x19
    mov x1, x20
    add x2, sp, #80
    bl _plamo_parse_score
    cbz x0, Lvbad
    ldp x19, x25, [sp, #80]
    ve 44
    ve 256
    ldp x9, x10, [sp, #88]
    adrp x11, Lvtypes@PAGE
    add x11, x11, Lvtypes@PAGEOFF
    mov x12, #0
Lvtype:
    ldrb w13, [x11]
    cmp x10, x13
    b.ne Lvtypenext
    mov x14, #0
Lvtypechar:
    add x15, x11, #1
    ldrb w16, [x15, x14]
    ldrb w17, [x9, x14]
    cmp w16, w17
    b.ne Lvtypenext
    add x14, x14, #1
    cmp x14, x10
    b.lo Lvtypechar
    str x12, [sp, #112]
    b Lvtypeok
Lvtypenext:
    add x11, x11, #9
    add x12, x12, #1
    cmp x12, #4
    b.lo Lvtype
    b Lvbad
Lvtypeok:
    ve 93
    vn
    cbnz x0, Lvbad
    stp x24, x25, [x23]
    ldr x9, [sp, #112]
    str x9, [x23, #16]
    mov x0, #1
    b Lvreturn
Lvbad:
    mov x0, #0
Lvreturn:
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #128
    ret
.section __TEXT,__const
Lvtypes:
.byte 6
.ascii "NORMAL\0\0"
.byte 4
.ascii "BYTE\0\0\0\0"
.byte 7
.ascii "CONTROL\0"
.byte 7
.ascii "UNKNOWN\0"
