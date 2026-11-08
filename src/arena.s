.text
.p2align 2
// x0=bytes -> zero-filled anonymous private RW mapping or NULL.
.globl _plamo_allocate
_plamo_allocate:
    mov x1, x0
    cbz x1, Lallocbad
    mov x0, #0
    mov x2, #3
    mov x3, #0x1002 // MAP_ANON | MAP_PRIVATE
    mov x4, #-1
    mov x5, #0
    mov x16, #197
    svc #0x80
    b.cs Lallocbad
    ret
Lallocbad:
    mov x0, #0
    ret

// x0=arena base,x1=table {destination,offset}[], NULL terminated.
.globl _plamo_arena_patch
_plamo_arena_patch:
Larenaloop:
    ldp x2, x3, [x1], #16
    cbz x2, Larenadone
    add x3, x0, x3
    str x3, [x2]
    b Larenaloop
Larenadone:
    ret
