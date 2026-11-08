.text
.p2align 2
// x0=cache {keys,values,capacity,count,width,sliding}; x1=new key,x2=new value.
// FP32 token-major data. Positive capacity/width, nonoverlapping incoming rows.
// Stores unrotated normalized keys, matching upstream cache-before-RoPE order.
// When full, sliding cache drops oldest row; global cache returns 0 unchanged.
// Returns 1=success,0=invalid dimensions/full global cache.
.globl _plamo_cache_append_reference
_plamo_cache_append_reference:
    ldp x3, x4, [x0]
    ldp x5, x6, [x0, #16]
    ldr x7, [x0, #32]
    cbz x5, Lcachebad
    cbz x7, Lcachebad
    cmp x6, x5
    b.hi Lcachebad
    b.lo Lcacheinsert
    ldr x8, [x0, #40]
    cbz x8, Lcachebad
    // Forward overlapping copy is safe: destination precedes source.
    sub x8, x5, #1
    mul x8, x8, x7
    mov x9, #0
    add x10, x3, x7, lsl #2
    add x11, x4, x7, lsl #2
Lcacheshift:
    cmp x9, x8
    b.hs Lcacheshifted
    ldr w12, [x10, x9, lsl #2]
    ldr w13, [x11, x9, lsl #2]
    str w12, [x3, x9, lsl #2]
    str w13, [x4, x9, lsl #2]
    add x9, x9, #1
    b Lcacheshift
Lcacheshifted:
    sub x6, x6, #1
Lcacheinsert:
    mul x8, x6, x7
    add x3, x3, x8, lsl #2
    add x4, x4, x8, lsl #2
    mov x9, #0
Lcachecopy:
    ldr w10, [x1, x9, lsl #2]
    ldr w11, [x2, x9, lsl #2]
    str w10, [x3, x9, lsl #2]
    str w11, [x4, x9, lsl #2]
    add x9, x9, #1
    cmp x9, x7
    b.lo Lcachecopy
    add x6, x6, #1
    str x6, [x0, #24]
    mov x0, #1
    ret
Lcachebad:
    mov x0, #0
    ret
