.text
.p2align 2
// Decode raw JSON string content (without surrounding quotes).
// x0=start,x1=end,x2=uint32 output,x3=capacity. Return count or UINT64_MAX.
// On failure output may contain a partial prefix. Strict Unicode scalars.
.globl _plamo_json_string
_plamo_json_string:
    stp x19, x30, [sp, #-80]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    str x24, [sp, #48]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, #0
Lstrloop:
    cmp x19, x20
    b.hs Lstrdone
    cmp x23, x22
    b.hs Lstrbad
    ldrb w4, [x19]
    cmp w4, #32
    b.lo Lstrbad
    cmp w4, #34
    b.eq Lstrbad
    cmp w4, #92
    b.eq Lstrescape
    mov x0, x19
    mov x1, x20
    add x2, sp, #64
    bl _plamo_utf8_next
    cbz x0, Lstrbad
    ldp x19, x4, [sp, #64]
    b Lstrstore
Lstrescape:
    add x19, x19, #1
    cmp x19, x20
    b.hs Lstrbad
    ldrb w4, [x19], #1
    cmp w4, #34
    b.eq Lstrstore
    cmp w4, #92
    b.eq Lstrstore
    cmp w4, #47
    b.eq Lstrstore
    cmp w4, #98
    b.ne Lstrf
    mov w4, #8
    b Lstrstore
Lstrf:
    cmp w4, #102
    b.ne Lstrn
    mov w4, #12
    b Lstrstore
Lstrn:
    cmp w4, #110
    b.ne Lstrr
    mov w4, #10
    b Lstrstore
Lstrr:
    cmp w4, #114
    b.ne Lstrt
    mov w4, #13
    b Lstrstore
Lstrt:
    cmp w4, #116
    b.ne Lstru
    mov w4, #9
    b Lstrstore
Lstru:
    cmp w4, #117
    b.ne Lstrbad
    bl Lstrhex
    cbz x0, Lstrbad
    mov w5, #0xd800
    sub w5, w4, w5
    cmp w5, #0x800
    b.hs Lstrstore
    cmp w5, #0x400
    b.hs Lstrbad
    mov w24, w5
    sub x5, x20, x19
    cmp x5, #6
    b.lo Lstrbad
    ldrb w5, [x19], #1
    cmp w5, #92
    b.ne Lstrbad
    ldrb w5, [x19], #1
    cmp w5, #117
    b.ne Lstrbad
    bl Lstrhex
    cbz x0, Lstrbad
    mov w5, #0xdc00
    sub w4, w4, w5
    cmp w4, #0x400
    b.hs Lstrbad
    add w4, w4, w24, lsl #10
    add w4, w4, #0x10000
Lstrstore:
    str w4, [x21, x23, lsl #2]
    add x23, x23, #1
    b Lstrloop
Lstrdone:
    mov x0, x23
    b Lstrreturn
Lstrbad:
    mov x0, #-1
Lstrreturn:
    ldr x24, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #80
    ret
// Internal: consume four hex digits from x19; result w4, success x0.
Lstrhex:
    sub x5, x20, x19
    cmp x5, #4
    b.lo Lstrhexbad
    mov w4, #0
    mov x6, #4
Lstrhexloop:
    ldrb w5, [x19], #1
    sub w7, w5, #48
    cmp w7, #9
    b.ls Lstrhexok
    orr w5, w5, #32
    sub w7, w5, #97
    cmp w7, #5
    b.hi Lstrhexbad
    add w7, w7, #10
Lstrhexok:
    lsl w4, w4, #4
    orr w4, w4, w7
    subs x6, x6, #1
    b.ne Lstrhexloop
    mov x0, #1
    ret
Lstrhexbad:
    mov x0, #0
    ret
