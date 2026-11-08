// Darwin syscalls. map_file(path) -> x0=address, x1=size.
// Failure is always (NULL,0); empty files are rejected. Caller owns mapping.
.text
.p2align 2
.globl _plamo_map_file
_plamo_map_file:
    mov x3, #0
    b Lmapcommon
// Same mapping plus an owned descriptor in x2; used for bulk weight reads.
// Caller closes x2. Failure returns (NULL,0,-1).
.globl _plamo_map_file_keep
_plamo_map_file_keep:
    mov x3, #1
Lmapcommon:
    stp x29, x30, [sp, #-48]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov x22, x3
    mov x29, sp
    mov x1, #0
    mov x2, #0
    mov x16, #5 // open
    svc #0x80
    b.cs 3f
    mov x19, x0
    mov x1, #0
    mov x2, #2 // SEEK_END
    mov x16, #199 // lseek
    svc #0x80
    b.cs 2f
    cmp x0, #0
    b.le 2f
    mov x20, x0
    mov x0, #0
    mov x1, x20
    mov x2, #1 // PROT_READ
    mov x3, #2 // MAP_PRIVATE
    mov x4, x19
    mov x5, #0
    mov x16, #197 // mmap
    svc #0x80
    b.cs 2f
    mov x21, x0
    cbnz x22, Lmapkept
    mov x0, x19
    mov x16, #6
    svc #0x80
    mov x2, #-1
    b Lmapsuccess
Lmapkept:
    mov x2, x19
Lmapsuccess:
    mov x0, x21
    mov x1, x20
    b 4f
2:  mov x0, x19
    mov x16, #6 // close, including errors after open
    svc #0x80
3:  mov x0, #0
    mov x1, #0
    mov x2, #-1
4:  ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// x0=mapping, x1=size; NULL is accepted as an already released mapping.
.globl _plamo_unmap_file
_plamo_unmap_file:
    cbz x0, 1f
    mov x16, #73
    svc #0x80
1:  ret

// Write full byte range, retrying interrupted and partial Darwin writes.
// x0=fd,x1=buffer,x2=bytes. Return 1=complete,0=error/no progress.
// No-op zero-length output succeeds without inspecting buffer.
.globl _plamo_write_all
_plamo_write_all:
    mov x3, x0
    mov x4, x1
    mov x5, x2
Lwriteagain:
    cbz x5, Lwriteok
    mov x0, x3
    mov x1, x4
    mov x2, x5
    mov x16, #4
    svc #0x80
    b.cs Lwriteerror
    cbz x0, Lwritebad
    cmp x0, x5
    b.hi Lwritebad
    add x4, x4, x0
    sub x5, x5, x0
    b Lwriteagain
Lwriteerror:
    cmp x0, #4 // EINTR
    b.eq Lwriteagain
Lwritebad:
    mov x0, #0
    ret
Lwriteok:
    mov x0, #1
    ret
