.text
.p2align 2
// Best-effort Darwin read-ahead for large original mapped weight matrices.
// x0=address,x1=bytes. Advice failures are harmless; ordinary loads remain valid.
// Apple Silicon pages are 16 KiB (also aligned on smaller page sizes).
.globl _plamo_prefetch_read
_plamo_prefetch_read:
    cmp x1, #0x100000
    b.lo Lprefetchret
    adrp x9, _plamo_io_policy@PAGE
    ldr x9, [x9, _plamo_io_policy@PAGEOFF]
    cbz x9, Lprefetchret
    adrp x10, _plamo_io_stats@PAGE
    add x10, x10, _plamo_io_stats@PAGEOFF
    ldr x9, [x10, #16]
    add x9, x9, #1
    str x9, [x10, #16]
    and x9, x0, #0x3fff
    sub x0, x0, x9
    add x1, x1, x9
    mov x2, #3 // MADV_WILLNEED, from the macOS SDK sys/mman.h
    mov x16, #75 // madvise
    svc #0x80
Lprefetchret:
    ret
.data
.p2align 3
.globl _plamo_io_policy
_plamo_io_policy: .quad 2

.globl _plamo_weight_mappings
_plamo_weight_mappings: .space 1024 // capacity 64 {base,size} records
.globl _plamo_weight_fds
_plamo_weight_fds:
    .rept 64
    .quad -1
    .endr
.globl _plamo_weight_mapping_count
_plamo_weight_mapping_count: .quad 0
Lweight_staging: .quad 0
// {bytes_staged,pread_calls,advice_calls,read_failures}, for benchmark diagnostics.
.globl _plamo_io_stats
_plamo_io_stats: .space 32

.text
.p2align 2
// x0=original BF16 matrix,x1=bytes -> readable pointer with identical bytes.
// Policies: 0=mapped loads, 1=read-ahead, 2=bounded bulk-read staging.
// The single-threaded inference engine reuses one <=1 GiB anonymous buffer.
// Non-model buffers and any failed advisory/read fall back to original data.
.globl _plamo_weight_acquire
_plamo_weight_acquire:
    cmp x1, #0x100000
    b.lo Lweight_ret
    stp x19, x30, [sp, #-80]!
    stp x20, x21, [sp, #16]
    stp x22, x23, [sp, #32]
    str x24, [sp, #48]
    mov x19, x0
    mov x20, x1
    adrp x9, _plamo_io_policy@PAGE
    ldr x9, [x9, _plamo_io_policy@PAGEOFF]
    cmp x9, #2
    b.ne Lweight_advice
    // A tail panel may already point into the active staging buffer.
    adrp x9, Lweight_staging@PAGE
    ldr x9, [x9, Lweight_staging@PAGEOFF]
    cbz x9, Lweight_not_staged
    cmp x19, x9
    b.lo Lweight_not_staged
    sub x10, x19, x9
    mov x11, #0x40000000
    cmp x10, x11
    b.hs Lweight_not_staged
    mov x0, x19
    b Lweight_done
Lweight_not_staged:
    mov x9, #0x40000000
    cmp x20, x9
    b.hi Lweight_advice
    adrp x9, _plamo_weight_mappings@PAGE
    add x9, x9, _plamo_weight_mappings@PAGEOFF
    mov x10, #0
    adrp x13, _plamo_weight_mapping_count@PAGE
    ldr x13, [x13, _plamo_weight_mapping_count@PAGEOFF]
    cbz x13, Lweight_advice
Lweight_find:
    ldp x11, x12, [x9], #16
    cmp x19, x11
    b.lo Lweight_next
    sub x22, x19, x11
    cmp x22, x12
    b.hs Lweight_next
    sub x12, x12, x22
    cmp x20, x12
    b.hi Lweight_advice
    adrp x9, _plamo_weight_fds@PAGE
    add x9, x9, _plamo_weight_fds@PAGEOFF
    ldr x21, [x9, x10, lsl #3]
    cmp x21, #0
    b.lt Lweight_advice
    b Lweight_buffer
Lweight_next:
    add x10, x10, #1
    cmp x10, x13
    b.lo Lweight_find
    b Lweight_advice
Lweight_buffer:
    adrp x9, Lweight_staging@PAGE
    ldr x23, [x9, Lweight_staging@PAGEOFF]
    cbnz x23, Lweight_readstart
    mov x0, #0
    mov x1, #0x40000000
    mov x2, #3
    mov x3, #0x1002
    mov x4, #-1
    mov x5, #0
    mov x16, #197
    svc #0x80
    b.cs Lweight_advice
    mov x23, x0
    adrp x9, Lweight_staging@PAGE
    str x23, [x9, Lweight_staging@PAGEOFF]
Lweight_readstart:
    mov x24, #0
Lweight_read:
    mov x0, x21
    add x1, x23, x24
    sub x2, x20, x24
    mov x9, #0x800000 // bounded 8 MiB reads
    cmp x2, x9
    csel x2, x2, x9, lo
    add x3, x22, x24
    adrp x9, _plamo_io_stats@PAGE
    add x9, x9, _plamo_io_stats@PAGEOFF
    ldr x10, [x9, #8]
    add x10, x10, #1
    str x10, [x9, #8]
    mov x16, #153 // pread
    svc #0x80
    b.cs Lweight_readerror
    cbz x0, Lweight_advice
    add x24, x24, x0
    cmp x24, x20
    b.lo Lweight_read
    adrp x9, _plamo_io_stats@PAGE
    add x9, x9, _plamo_io_stats@PAGEOFF
    ldr x10, [x9]
    add x10, x10, x20
    str x10, [x9]
    mov x0, x23
    b Lweight_done
Lweight_readerror:
    cmp x0, #4 // EINTR
    b.eq Lweight_read
    adrp x9, _plamo_io_stats@PAGE
    add x9, x9, _plamo_io_stats@PAGEOFF
    ldr x10, [x9, #24]
    add x10, x10, #1
    str x10, [x9, #24]
Lweight_advice:
    mov x0, x19
    mov x1, x20
    bl _plamo_prefetch_read
    mov x0, x19
Lweight_done:
    ldr x24, [sp, #48]
    ldp x22, x23, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #80
Lweight_ret:
    ret
