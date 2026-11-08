.text
.p2align 2
// Fixed-revision vocabulary score parser: [-]digits[.1-to-4-digits].
// x0=cursor,x1=end,x2=result {next,int64 score_times_10000}; return 1/0.
// No floating point rounding. Magnitude limited to 1000 (vocab max <100).
// Rejects exponents, excessive precision and leading zeros; delimiter required.
.globl _plamo_parse_score
_plamo_parse_score:
    stp xzr, xzr, [x2]
    cmp x0, x1
    b.hs Lscbad
    mov x3, #0
    ldrb w4, [x0]
    cmp w4, #45
    b.ne Lscunsigned
    mov x3, #1
    add x0, x0, #1
Lscunsigned:
    mov x5, #0
    mov x6, #0
    mov x7, #10
Lscinteger:
    cmp x0, x1
    b.hs Lscintdone
    ldrb w4, [x0]
    sub w8, w4, #48
    cmp w8, #9
    b.hi Lscintdone
    cbz x6, Lscdigit
    cbz x5, Lscbad
Lscdigit:
    madd x5, x5, x7, x8
    cmp x5, #1000
    b.hi Lscbad
    add x6, x6, #1
    add x0, x0, #1
    b Lscinteger
Lscintdone:
    cbz x6, Lscbad
    mov x6, #10000
    mul x5, x5, x6
    cmp x0, x1
    b.hs Lscfinish
    cmp w4, #46
    b.ne Lscdelimiter
    add x0, x0, #1
    mov x6, #1000
    mov x9, #0
Lscfraction:
    cmp x0, x1
    b.hs Lscfracdone
    ldrb w4, [x0]
    sub w8, w4, #48
    cmp w8, #9
    b.hi Lscfracdone
    cbz x6, Lscbad
    madd x5, x8, x6, x5
    udiv x6, x6, x7
    add x9, x9, #1
    add x0, x0, #1
    b Lscfraction
Lscfracdone:
    cbz x9, Lscbad
    cmp x0, x1
    b.hs Lscfinish
Lscdelimiter:
    cmp w4, #44
    b.eq Lscfinish
    cmp w4, #93
    b.eq Lscfinish
    cmp w4, #32
    b.eq Lscfinish
    cmp w4, #9
    b.eq Lscfinish
    cmp w4, #10
    b.eq Lscfinish
    cmp w4, #13
    b.ne Lscbad
Lscfinish:
    cmp x3, #0
    cneg x5, x5, ne
    stp x0, x5, [x2]
    mov x0, #1
    ret
Lscbad:
    mov x0, #0
    ret
