.text
.p2align 2
// Unicode-segment Unigram encoder. x0=descriptor:
// {trie,codepoints,count,byte_ids,costs,paths,candidates,candidate_capacity,output,output_capacity}
// costs[count+1] int64; paths[count] {cost,length,id} (24 bytes).
// Candidates capacity includes one fallback; output uint32 IDs.
// Returns count or UINT64_MAX (scratch/output can be partially written).
// No boundary preprocessing here. All codepoints must be valid Unicode scalars.
.globl _plamo_encode_segment
_plamo_encode_segment:
    stp x19, x30, [sp, #-96]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    stp x24, x25, [sp, #48]
    stp x26, x27, [sp, #64]
    mov x19, x0
    ldr x20, [x19, #16]
    ldr x21, [x19, #32]
    str xzr, [x21, x20, lsl #3]
    ldr x22, [x19, #40]
    ldr x23, [x19, #48]
    ldr x24, [x19, #56]
    cbz x24, Lencbad
    mov x25, x20
Lencback:
    cbz x25, Lencforwardstart
    sub x25, x25, #1
    ldr x0, [x19]
    ldr x1, [x19, #8]
    add x1, x1, x25, lsl #2
    sub x2, x20, x25
    mov x3, x23
    sub x4, x24, #1
    bl _plamo_trie_match
    cmn x0, #1
    b.eq Lencbad
    lsl x9, x0, #4
    add x9, x23, x9
    mov w10, #1
    mov w11, #-1
    stp w10, w11, [x9]
    mov w10, #0x6980
    movk w10, #0xff67, lsl #16 // -10000000
    stp w10, wzr, [x9, #8]
    add x2, x0, #1
    add x0, x21, x25, lsl #3
    mov x1, x23
    sub x3, x20, x25
    mov x9, #24
    madd x4, x25, x9, x22
    bl _plamo_unigram_step
    cbz x0, Lencbad
    mov x9, #24
    madd x9, x25, x9, x22
    ldr x10, [x9]
    str x10, [x21, x25, lsl #3]
    b Lencback
Lencforwardstart:
    mov x25, #0
    mov x26, #0
    ldr x27, [x19, #72]
Lencforward:
    cmp x25, x20
    b.hs Lencdone
    mov x9, #24
    madd x9, x25, x9, x22
    ldp x10, x11, [x9, #8]
    cmp x11, #0
    b.lt Lencfallback
    cmp x26, x27
    b.hs Lencbad
    ldr x12, [x19, #64]
    str w11, [x12, x26, lsl #2]
    add x26, x26, #1
    add x25, x25, x10
    b Lencforward
Lencfallback:
    ldr x9, [x19, #8]
    ldr w0, [x9, x25, lsl #2]
    ldr x1, [x19, #24]
    add x2, sp, #80
    bl _plamo_byte_fallback
    cbz x0, Lencbad
    sub x9, x27, x26
    cmp x0, x9
    b.hi Lencbad
    mov x9, #0
    ldr x10, [x19, #64]
    add x10, x10, x26, lsl #2
    add x11, sp, #80
Lenccopy:
    ldr w12, [x11, x9, lsl #2]
    str w12, [x10, x9, lsl #2]
    add x9, x9, #1
    cmp x9, x0
    b.lo Lenccopy
    add x26, x26, x0
    add x25, x25, #1
    b Lencforward
Lencdone:
    mov x0, x26
    b Lencreturn
Lencbad:
    mov x0, #-1
Lencreturn:
    ldp x26, x27, [sp, #64]
    ldp x24, x25, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #96
    ret
