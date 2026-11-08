.text
.p2align 2
// Trie context {nodes,capacity,used,root_index}; used initially 1.
// Optional zeroed uint32 root_index[0x110000] must be supplied before building.
// Direct Unicode-root lookup avoids a long sibling scan; NULL keeps compact mode.
// Nodes: {uint32 cp,child,sibling; int32 id,score; uint32 terminal} (24 bytes).
// x0=context,x1=codepoints,x2=length,x3=id,x4=integer score.
// Returns 1=inserted,0=capacity/empty. Reserves worst-case length before mutation.
.globl _plamo_trie_insert
_plamo_trie_insert:
    cbz x2, Ltribad
    ldp x5, x6, [x0]
    ldr x7, [x0, #16]
    cmp x7, x6
    b.hi Ltribad
    sub x6, x6, x7
    cmp x2, x6
    b.hi Ltribad
    ldr x15, [x0, #24]
    mov w16, #0x110000
    mov x8, #0 // parent node offset
    mov x9, #24
Ltrichar:
    ldr w10, [x1], #4
    add x11, x5, x8
    mov x17, #0
    cbnz x8, Ltrilinked
    cbz x15, Ltrilinked
    cmp w10, w16
    b.hs Ltrilinked
    add x17, x15, x10, lsl #2
    ldr w12, [x17]
    cbz w12, Ltrinew
    b Ltrifound
Ltrilinked:
    ldr w12, [x11, #4]
Ltrifind:
    cbz w12, Ltrinew
    umull x13, w12, w9
    ldr w14, [x5, x13]
    cmp w14, w10
    b.eq Ltrifound
    add x13, x5, x13
    ldr w12, [x13, #8]
    b Ltrifind
Ltrinew:
    mul x13, x7, x9
    add x13, x5, x13
    ldr w14, [x11, #4]
    stp w10, wzr, [x13]
    stp w14, wzr, [x13, #8]
    stp wzr, wzr, [x13, #16]
    str w7, [x11, #4]
    cbz x17, Ltrinoindex
    str w7, [x17]
Ltrinoindex:
    mov x12, x7
    add x7, x7, #1
Ltrifound:
    mul x8, x12, x9
    subs x2, x2, #1
    b.ne Ltrichar
    add x11, x5, x8
    stp w3, w4, [x11, #12]
    mov w10, #1
    str w10, [x11, #20]
    str x7, [x0, #16]
    mov x0, #1
    ret
Ltribad:
    mov x0, #0
    ret

// x0=context,x1=codepoints,x2=available,x3=candidates,x4=capacity.
// Enumerate matching prefixes into {length,id,score,reserved} (16 bytes).
// Return count, or UINT64_MAX if output capacity exhausted (partial output).
.globl _plamo_trie_match
_plamo_trie_match:
    ldr x15, [x0, #24]
    mov w16, #0x110000
    ldr x5, [x0]
    mov x6, #0 // current offset
    mov x7, #0 // consumed
    mov x8, #0 // output count
    mov x9, #24
Ltrmatchchar:
    cmp x7, x2
    b.hs Ltrmatchdone
    ldr w10, [x1, x7, lsl #2]
    cbnz x7, Ltrmatchlinked
    cbz x15, Ltrmatchlinked
    cmp w10, w16
    b.hs Ltrmatchlinked
    ldr w12, [x15, x10, lsl #2]
    cbz w12, Ltrmatchdone
    umull x6, w12, w9
    add x11, x5, x6
    b Ltrmatchfound
Ltrmatchlinked:
    add x11, x5, x6
    ldr w12, [x11, #4]
Ltrmatchfind:
    cbz w12, Ltrmatchdone
    umull x6, w12, w9
    add x11, x5, x6
    ldr w13, [x11]
    cmp w13, w10
    b.eq Ltrmatchfound
    ldr w12, [x11, #8]
    b Ltrmatchfind
Ltrmatchfound:
    add x7, x7, #1
    ldr w13, [x11, #20]
    cbz w13, Ltrmatchchar
    cmp x8, x4
    b.hs Ltrmatchbad
    ldp w13, w14, [x11, #12]
    stp w7, w13, [x3]
    stp w14, wzr, [x3, #8]
    add x3, x3, #16
    add x8, x8, #1
    b Ltrmatchchar
Ltrmatchdone:
    mov x0, x8
    ret
Ltrmatchbad:
    mov x0, #-1
    ret
