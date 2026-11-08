// Restricted JSON lexer for safetensors: strings, unsigned integers, punctuation.
// x0=cursor, x1=end, x2=token {next,start,length,value}.
// Returns ASCII punctuation, 256=string (raw escaped bytes), 257=uint,
// 0=end, -1=invalid. Grammar validation belongs to the header parser.
.text
.p2align 2
.globl _plamo_json_next
_plamo_json_next:
    stp xzr, xzr, [x2]
    stp xzr, xzr, [x2, #16]
Lspace:
    cmp x0, x1
    b.hs Leof
    ldrb w3, [x0]
    cmp w3, #32
    b.eq Lskip
    cmp w3, #9
    b.eq Lskip
    cmp w3, #10
    b.eq Lskip
    cmp w3, #13
    b.ne Ltoken
Lskip:
    add x0, x0, #1
    b Lspace
Ltoken:
    cmp w3, #34
    b.eq Lstring
    cmp w3, #123
    b.eq Lpunct
    cmp w3, #125
    b.eq Lpunct
    cmp w3, #91
    b.eq Lpunct
    cmp w3, #93
    b.eq Lpunct
    cmp w3, #58
    b.eq Lpunct
    cmp w3, #44
    b.eq Lpunct
    // Delegate checked integer parsing, preserving lexer output address.
    stp x19, x30, [sp, #-32]!
    stp x20, x21, [sp, #16]
    mov x19, x2
    mov x20, x0
    mov x21, x1
    bl _plamo_parse_u64
    cbz x0, Lnumfail
    ldp x3, x4, [x19]
    cmp x3, x21
    b.hs Lnumok
    ldrb w5, [x3]
    cmp w5, #32
    b.eq Lnumok
    cmp w5, #9
    b.eq Lnumok
    cmp w5, #10
    b.eq Lnumok
    cmp w5, #13
    b.eq Lnumok
    cmp w5, #44
    b.eq Lnumok
    cmp w5, #93
    b.eq Lnumok
    cmp w5, #125
    b.ne Lnumfail
Lnumok:
    sub x5, x3, x20
    stp x3, x20, [x19]
    stp x5, x4, [x19, #16]
    mov x0, #257
    b Lnumret
Lnumfail:
    stp xzr, xzr, [x19]
    mov x0, #-1
Lnumret:
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #32
    ret
Lpunct:
    add x4, x0, #1
    stp x4, x0, [x2]
    mov x4, #1
    str x4, [x2, #16]
    mov w0, w3
    ret
Lstring:
    add x3, x0, #1
    mov x4, x3
Lchar:
    cmp x4, x1
    b.hs Lbad
    ldrb w5, [x4], #1
    cmp w5, #34
    b.eq Ldone
    cmp w5, #32
    b.lo Lbad
    cmp w5, #92
    b.ne Lchar
    cmp x4, x1
    b.hs Lbad
    ldrb w5, [x4], #1
    cmp w5, #34
    b.eq Lchar
    cmp w5, #92
    b.eq Lchar
    cmp w5, #47
    b.eq Lchar
    cmp w5, #98
    b.eq Lchar
    cmp w5, #102
    b.eq Lchar
    cmp w5, #110
    b.eq Lchar
    cmp w5, #114
    b.eq Lchar
    cmp w5, #116
    b.eq Lchar
    cmp w5, #117
    b.ne Lbad
    mov x6, #4
Lhex:
    cmp x4, x1
    b.hs Lbad
    ldrb w5, [x4], #1
    sub w7, w5, #48
    cmp w7, #9
    b.ls Lhexok
    orr w5, w5, #32
    sub w7, w5, #97
    cmp w7, #5
    b.hi Lbad
Lhexok:
    subs x6, x6, #1
    b.ne Lhex
    b Lchar
Ldone:
    sub x5, x4, x3
    sub x5, x5, #1
    stp x4, x3, [x2]
    str x5, [x2, #16]
    mov x0, #256
    ret
Leof:
    str x0, [x2]
    mov x0, #0
    ret
Lbad:
    mov x0, #-1
    ret
