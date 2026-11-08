.text
.p2align 2
// Validate safetensors envelope without touching tensor data.
// x0=file, x1=file length, x2=output {header,header_size,data,data_size}.
// Returns 1 on success, 0 on failure; failure clears the output.
.globl _plamo_safetensors_view
_plamo_safetensors_view:
    stp xzr, xzr, [x2]
    stp xzr, xzr, [x2, #16]
    cmp x1, #10
    b.lo 9f
    cbz x0, 9f
    ldr x3, [x0]
    sub x4, x1, #8
    cmp x3, x4
    b.hi 9f
    cmp x3, #2
    b.lo 9f
    // safetensors limits JSON headers to 100 MB.
    mov x5, #0xe100
    movk x5, #0x5f5, lsl #16
    cmp x3, x5
    b.hi 9f
    add x5, x0, #8
    ldrb w6, [x5]
    cmp w6, #123 // {
    b.ne 9f
    add x6, x5, x3
    sub x7, x6, #1
1:  ldrb w8, [x7]
    cmp w8, #32
    b.ne 2f
    sub x7, x7, #1
    cmp x7, x5
    b.hi 1b
    b 9f
2:  cmp w8, #125 // }
    b.ne 9f
    sub x4, x4, x3
    stp x5, x3, [x2]
    stp x6, x4, [x2, #16]
    mov x0, #1
    ret
9:  mov x0, #0
    ret

// Parse a decimal unsigned integer within [x0,x1).
// x2 points to {next,value}; output cleared on invalid input/overflow.
// JSON integer grammar: no leading zeros except the single digit zero.
.globl _plamo_parse_u64
_plamo_parse_u64:
    stp xzr, xzr, [x2]
    cmp x0, x1
    b.hs 9b
    mov x3, x0
    mov x4, #0
    mov x5, #10
1:  ldrb w6, [x3]
    sub w6, w6, #48
    cmp w6, #9
    b.hi 3f
    umulh x7, x4, x5
    cbnz x7, 9b
    mul x4, x4, x5
    adds x4, x4, x6
    b.cs 9b
    add x3, x3, #1
    cmp x3, x1
    b.hs 3f
    cbnz x4, 1b
    // A zero must be the only digit.
    ldrb w6, [x3]
    sub w6, w6, #48
    cmp w6, #9
    b.ls 9b
3:  cmp x3, x0
    b.eq 9b
    stp x3, x4, [x2]
    mov x0, #1
    ret
