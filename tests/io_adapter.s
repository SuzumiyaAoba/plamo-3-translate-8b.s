.text
.p2align 2
// C-test adapter: x0=path, x1=address of size_t.
.globl _plamo_map_file_test
_plamo_map_file_test:
    stp x19, x30, [sp, #-16]!
    mov x19, x1
    bl _plamo_map_file
    str x1, [x19]
    ldp x19, x30, [sp], #16
    ret
