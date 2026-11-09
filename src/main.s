// Tokenizer and raw-prompt greedy generation CLI.
.text
.p2align 2
.macro address reg, symbol
    adrp \reg, \symbol@PAGE
    add \reg, \reg, \symbol@PAGEOFF
.endm
.globl _start
_start:
    // LC_MAIN entry receives argc/argv in x0/x1 from dyld.
    mov x28, #0
    mov x25, #0
    cmp x0, #2
    b.eq Ltokenmode
    cmp x0, #3
    b.ne Lusage
    ldr x9, [x1, #8]
    ldrb w10, [x9]
    cmp w10, #45
    b.ne Lusage
    ldrb w10, [x9, #1]
    cmp w10, #103
    b.eq Lgenerationoption
    cmp w10, #106
    b.eq Ljaoption
    cmp w10, #101
    b.ne Lusage
    mov x25, #2
    b Lgenerationoption
Ljaoption:
    mov x25, #1
Lgenerationoption:
    ldrb w10, [x9, #2]
    cbnz w10, Lusage
    mov x28, #1
    ldr x19, [x1, #16]
    b Linputready
Ltokenmode:
    ldr x19, [x1, #8]
Linputready:
    cbz x25, Lrawinput
    mov x0, x19
    address x1, lang_en
    address x2, lang_ja
    cmp x25, #1
    b.eq Lbuildprompt
    address x1, lang_ja
    address x2, lang_en
Lbuildprompt:
    address x3, prompt_bytes
    mov x4, #16384
    bl _plamo_translation_prompt
    cmn x0, #1
    b.eq Lfailure
    address x19, prompt_bytes
Lrawinput:
    mov x20, x19
Largend:
    ldrb w9, [x20], #1
    cbnz w9, Largend
    sub x20, x20, #1
    address x0, _plamo_vocab_path
    bl _plamo_map_file
    cbz x0, Lfailure
    mov x21, x0
    mov x22, x1
    mov x26, x0
    add x27, x0, x1
    add x1, x0, x1
    address x2, trie_context
    address x3, vocab_scratch
    mov x4, #4096
    address x5, byte_ids
    bl _plamo_vocab_load
    cmn x0, #1
    b.eq Lfailure
    // Keep original vocabulary mapped for generated-token decoding.
    mov x21, #0
Ldecode:
    cmp x19, x20
    b.hs Lpreprocess
    cmp x21, #4096
    b.hs Lfailure
    mov x0, x19
    mov x1, x20
    address x2, scalar_result
    bl _plamo_utf8_next
    cbz x0, Lfailure
    address x9, scalar_result
    ldp x19, x10, [x9]
    address x9, input_cp
    str w10, [x9, x21, lsl #2]
    add x21, x21, #1
    b Ldecode
Lpreprocess:
    address x0, prep_descriptor
    str x21, [x0, #8]
    bl _plamo_preprocess
    cmn x0, #1
    b.eq Lfailure
    mov x9, x0
    address x0, encode_descriptor
    str x9, [x0, #16]
    bl _plamo_encode_marked
    cmn x0, #1
    b.eq Lfailure
    mov x20, x0
    mov x19, #0
    cbnz x28, Lgenerate
Lprint:
    cmp x19, x20
    b.hs Lnewline
    address x9, output_ids
    ldr w0, [x9, x19, lsl #2]
    bl Lprint_integer
    add x19, x19, #1
    b Lprint
Lgenerate:
    cbz x20, Lfailure
    bl _plamo_model_open
    cbz x0, Lmodelfailure
    address x9, _plamo_model
    ldr x9, [x9, #16]
    mul x24, x20, x9
    lsl x24, x24, #2
    mov x0, x24
    bl _plamo_allocate
    cbz x0, Lmodelfailure
    mov x23, x0
    address x0, _plamo_model
    address x1, output_ids
    mov x2, x20
    mov x3, x23
    bl _plamo_prefill
    mov x25, x0
    mov x0, x23
    mov x1, x24
    bl _plamo_unmap_file
    cbz x25, Lmodelfailure
    mov x21, #0
Lnexttoken:
    address x9, _plamo_model
    ldr x0, [x9, #80]
    ldr x1, [x9, #8]
    bl _plamo_argmax
    cmn x0, #1
    b.eq Lmodelfailure
    mov x22, x0
    cmp x22, #1
    b.eq Lnewline
    cmp x22, #16
    b.eq Lnewline
    mov x0, x26
    mov x1, x27
    mov x2, x22
    address x3, decoded_bytes
    mov x4, #16384
    address x5, vocab_scratch
    mov x6, #4096
    bl _plamo_decode_token
    cmn x0, #1
    b.eq Lmodelfailure
    mov x2, x0
    mov x0, #1
    address x1, decoded_bytes
    bl _plamo_write_all
    add x21, x21, #1
    cmp x21, #256
    b.hs Lnewline
    address x0, _plamo_model
    mov x1, x22
    bl _plamo_forward
    cbz x0, Lmodelfailure
    b Lnexttoken
Lmodelfailure:
    mov x0, #2
    address x1, model_failure
    address x2, model_failure_end
    sub x2, x2, x1
    bl _plamo_write_all
    mov x0, #1
    b Lexit
Lnewline:
    mov x0, #1
    address x1, newline
    mov x2, #1
    bl _plamo_write_all
    mov x0, #0
    b Lexit
Lusage:
    mov x0, #2
    address x1, usage
    address x2, usage_end
    sub x2, x2, x1
    bl _plamo_write_all
    mov x0, #2
    b Lexit
Lfailure:
    mov x0, #2
    address x1, failure
    address x2, failure_end
    sub x2, x2, x1
    bl _plamo_write_all
    mov x0, #1
Lexit:
    mov x16, #1
    svc #0x80
Lprint_integer:
    sub sp, sp, #32
    str x30, [sp]
    add x1, sp, #31
    mov w9, #32
    strb w9, [x1]
    mov x2, #1
    mov x10, #10
Ldigit:
    udiv x11, x0, x10
    msub x12, x11, x10, x0
    add w12, w12, #48
    sub x1, x1, #1
    strb w12, [x1]
    add x2, x2, #1
    mov x0, x11
    cbnz x0, Ldigit
    mov x0, #1
    bl _plamo_write_all
    ldr x30, [sp]
    add sp, sp, #32
    ret
.section __TEXT,__const
newline: .ascii "\n"
usage: .ascii "usage: build/plamo-asm [-g|-j|-e] 'text' (-g raw, -j en-to-ja, -e ja-to-en)\n"
usage_end:
failure: .ascii "plamo: tokenizer failed (check vocabulary, UTF-8, or input capacity)\n"
failure_end:
model_failure: .ascii "plamo: model initialization or forward failed\n"
model_failure_end:
.data
.p2align 3
trie_context: .quad trie_nodes,1000000,1,trie_roots
prep_descriptor: .quad input_cp,0,temp_cp,16384,marked_cp,49152,boundary_flags
encode_descriptor: .quad trie_context,marked_cp,0,byte_ids,costs,paths,candidates,4097,output_ids,16384
.zerofill __DATA,__bss,trie_nodes,24000000,4
.zerofill __DATA,__bss,vocab_scratch,16384,4
.zerofill __DATA,__bss,byte_ids,1024,4
.zerofill __DATA,__bss,scalar_result,16,3
.zerofill __DATA,__bss,input_cp,16384,4
.zerofill __DATA,__bss,temp_cp,65536,4
.zerofill __DATA,__bss,marked_cp,196608,4
.zerofill __DATA,__bss,boundary_flags,16385,4
.zerofill __DATA,__bss,costs,393224,4
.zerofill __DATA,__bss,paths,1179648,4
.zerofill __DATA,__bss,candidates,65552,4
.zerofill __DATA,__bss,output_ids,65536,4

.zerofill __DATA,__bss,decoded_bytes,16384,4

.section __TEXT,__const
lang_en: .asciz "en"
lang_ja: .asciz "ja"
.zerofill __DATA,__bss,prompt_bytes,16384,4

.zerofill __DATA,__bss,trie_roots,4456448,4
