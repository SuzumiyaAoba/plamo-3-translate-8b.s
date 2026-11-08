.text
.p2align 2
// One backward-DP step. x0=suffix costs relative to current position,
// x1=matching candidates {uint32 length,int32 id,int32 score,uint32 reserved},
// x2=candidate count,x3=max available codepoints,x4=result {int64 cost,length,id}.
// Scores already rounded to int32 at vocabulary load time. Candidates may be
// unsorted: equal costs prefer longest piece, then earliest candidate.
// Includes byte fallback as a candidate {1,-1,-10000000,0} in caller.
// 1=path,0=no valid candidate. Bounds and signed subtraction overflow checked.
.globl _plamo_unigram_step
_plamo_unigram_step:
    mov x5, #0x1000000000000000
    mov x6, #0
    mov x7, #-1
    mov w8, #0xd300
    movk w8, #0xfece, lsl #16 // -20000000
Luni:
    cbz x2, Lunidone
    ldp w9, w10, [x1]
    ldrsw x11, [x1, #8]
    add x1, x1, #16
    sub x2, x2, #1
    cbz w9, Luni
    cmp x9, x3
    b.hi Luni
    cmp w11, w8
    b.le Luni
    ldr x12, [x0, x9, lsl #3]
    subs x12, x12, x11
    b.vs Luni
    cmp x12, x5
    b.lt Luniupdate
    b.gt Luni
    cmp x9, x6
    b.ls Luni
Luniupdate:
    mov x5, x12
    mov x6, x9
    sxtw x7, w10
    b Luni
Lunidone:
    stp x5, x6, [x4]
    str x7, [x4, #16]
    cmp x6, #0
    cset x0, ne
    ret
