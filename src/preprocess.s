.text
.p2align 2
// x0=descriptor {input,count,temp,temp_capacity,output,output_capacity,flags}
// flags capacity >= max(count,temp_capacity)+1; input/temp/output disjoint.
// Returns output codepoint count or UINT64_MAX. Output contains U+EE00 markers.
// Adjacent matches share one marker; equivalent for segment splitting.
.globl _plamo_preprocess
_plamo_preprocess:
    stp x19, x30, [sp, #-48]!
    stp x20, x21, [sp, #16]
    str x22, [sp, #32]
    mov x19, x0
    ldp x20, x21, [x19]
    ldr x22, [x19, #48]
    mov x9, #0
Lpclear1:
    strb wzr, [x22, x9]
    add x9, x9, #1
    cmp x9, x21
    b.ls Lpclear1
    mov x0, x20
    mov x1, x21
    mov x2, x22
    bl _plamo_special_boundaries
    mov x0, x20
    mov x1, x21
    mov x2, x22
    ldp x3, x4, [x19, #16]
    bl Lpexpand
    cmn x0, #1
    b.eq Lpreturn
    mov x21, x0
    ldr x20, [x19, #16]
    mov x9, #0
Lpclear2:
    strb wzr, [x22, x9]
    add x9, x9, #1
    cmp x9, x21
    b.ls Lpclear2
    mov x0, x20
    mov x1, x21
    mov x2, x22
    bl _plamo_repeat_boundaries
    mov x0, x20
    mov x1, x21
    mov x2, x22
    ldp x3, x4, [x19, #32]
    bl Lpexpand
Lpreturn:
    ldr x22, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #48
    ret
// x0=input,x1=count,x2=flags,x3=output,x4=capacity -> output count.
Lpexpand:
    mov x5, #0
    mov x6, #0
    mov w8, #0xee00
Lpexpandloop:
    ldrb w7, [x2, x5]
    cbz w7, Lpcopy
    cmp x6, x4
    b.hs Lpexpandbad
    str w8, [x3, x6, lsl #2]
    add x6, x6, #1
Lpcopy:
    cmp x5, x1
    b.hs Lpexpanddone
    cmp x6, x4
    b.hs Lpexpandbad
    ldr w7, [x0, x5, lsl #2]
    str w7, [x3, x6, lsl #2]
    add x6, x6, #1
    add x5, x5, #1
    b Lpexpandloop
Lpexpanddone:
    mov x0, x6
    ret
Lpexpandbad:
    mov x0, #-1
    ret
