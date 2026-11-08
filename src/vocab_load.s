.text
.p2align 2
// x0=JSONL start,x1=end,x2=initialized empty trie,x3=codepoint scratch,
// x4=scratch capacity,x5=zeroed byte IDs[256].
// Returns row count or UINT64_MAX. Failure leaves partially built tables.
// Requires all 256 BYTE entries; BYTE IDs must be nonzero and unique.
.globl _plamo_vocab_load
_plamo_vocab_load:
    stp x19, x30, [sp, #-128]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    stp x26, x27, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, #0
Lvlloop:
    cmp x19, x20
    b.hs Lvldone
    mov x26, x19
Lvlfindend:
    cmp x26, x20
    b.hs Lvlrow
    ldrb w9, [x26], #1
    cmp w9, #10
    b.ne Lvlfindend
Lvlrow:
    mov x0, x19
    mov x1, x26
    mov x2, x22
    mov x3, x23
    add x4, sp, #96
    bl _plamo_vocab_row
    cbz x0, Lvlbad
    ldr x9, [sp, #112]
    cmp x9, #1
    b.eq Lvlbyte
    mov x0, x21
    mov x1, x22
    ldr x2, [sp, #96]
    mov x3, x25
    ldr x4, [sp, #104]
    bl _plamo_trie_insert
    cbz x0, Lvlbad
    b Lvlnext
Lvlbyte:
    ldr x9, [sp, #96]
    cmp x9, #6
    b.ne Lvlbad
    ldp w9, w10, [x22]
    cmp w9, #60
    b.ne Lvlbad
    cmp w10, #48
    b.ne Lvlbad
    ldr w9, [x22, #8]
    cmp w9, #120
    b.ne Lvlbad
    ldr w9, [x22, #20]
    cmp w9, #62
    b.ne Lvlbad
    mov x27, #0
    mov x10, #3
Lvlhex:
    ldr w9, [x22, x10, lsl #2]
    sub w11, w9, #48
    cmp w11, #9
    b.ls Lvlhexok
    orr w9, w9, #32
    sub w11, w9, #97
    cmp w11, #5
    b.hi Lvlbad
    add w11, w11, #10
Lvlhexok:
    lsl x27, x27, #4
    orr x27, x27, x11
    add x10, x10, #1
    cmp x10, #5
    b.lo Lvlhex
    cbz x25, Lvlbad
    ldr w9, [x24, x27, lsl #2]
    cbnz w9, Lvlbad
    str w25, [x24, x27, lsl #2]
Lvlnext:
    add x25, x25, #1
    mov x19, x26
    b Lvlloop
Lvldone:
    mov x9, #0
Lvlcheck:
    ldr w10, [x24, x9, lsl #2]
    cbz w10, Lvlbad
    add x9, x9, #1
    cmp x9, #256
    b.lo Lvlcheck
    mov x0, x25
    b Lvlreturn
Lvlbad:
    mov x0, #-1
Lvlreturn:
    ldp x26, x27, [sp, #64]
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #128
    ret
