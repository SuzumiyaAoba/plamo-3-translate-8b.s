.text
.p2align 2
// x0=source language,x1=output,x2=capacity. Exact input prefix of the same
// translation template, ending at its special message boundary.
.globl _plamo_translation_prefix
_plamo_translation_prefix:
    mov x4, x2
    mov x3, x1
    mov x1, x0
    mov x0, #0
    mov x2, #0
    b _plamo_translation_prompt

// x0=source text,x1=source language,x2=target language (NUL strings),
// x3=output,x4=byte capacity including NUL. Return bytes or UINT64_MAX.
// Basic single-request PFML3 prompt, exactly matching official chat_template.
// Caller validates language codes and excludes reserved tokens from user text.
.globl _plamo_translation_prompt
_plamo_translation_prompt:
    sub sp, sp, #80
    str x0, [sp, #24]
    str x1, [sp, #8]
    str x2, [sp, #40]
    adrp x9, Lpromptprefix@PAGE
    add x9, x9, Lpromptprefix@PAGEOFF
    str x9, [sp]
    adrp x9, Lpromptmsg@PAGE
    add x9, x9, Lpromptmsg@PAGEOFF
    str x9, [sp, #16]
    str x9, [sp, #48]
    adrp x9, Lpromptoutput@PAGE
    add x9, x9, Lpromptoutput@PAGEOFF
    str x9, [sp, #32]
    mov x5, #0
    mov x6, #0
Lpromptpart:
    ldr x7, [sp, x5, lsl #3]
    cbz x7, Lpromptfinished
Lpromptcopy:
    ldrb w8, [x7], #1
    cbz w8, Lpromptnext
    add x9, x6, #1
    cmp x9, x4
    b.hs Lpromptbad
    strb w8, [x3, x6]
    mov x6, x9
    b Lpromptcopy
Lpromptnext:
    add x5, x5, #1
    cmp x5, #7
    b.lo Lpromptpart
Lpromptfinished:
    cmp x6, x4
    b.hs Lpromptbad
    strb wzr, [x3, x6]
    mov x0, x6
    add sp, sp, #80
    ret
Lpromptbad:
    mov x0, #-1
    add sp, sp, #80
    ret
.section __TEXT,__const
Lpromptprefix: .asciz "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>"
Lpromptmsg: .asciz "<|plamo:msg|>"
Lpromptoutput: .asciz "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>"
