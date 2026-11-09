.text
.p2align 2
// Initialize the compiled model once. Paths are generated from its profile.
// Returns 1=ready,0=failure; a successful mapping lives for process lifetime.
// Repeated successful calls are idempotent. Not thread safe.
.globl _plamo_model_open
_plamo_model_open:
    stp x19, x30, [sp, #-48]!
    stp x20, x21, [sp, #16]
    str x22, [sp, #32]
    adrp x19, _plamo_weight_mappings@PAGE
    add x19, x19, _plamo_weight_mappings@PAGEOFF
    adrp x9, Lmodel_ready@PAGE
    ldr x10, [x9, Lmodel_ready@PAGEOFF]
    cbnz x10, Lopensuccess
    adrp x20, _plamo_weight_paths@PAGE
    add x20, x20, _plamo_weight_paths@PAGEOFF
    adrp x9, _plamo_weight_file_count@PAGE
    ldr x22, [x9, _plamo_weight_file_count@PAGEOFF]
    mov x21, #0
Lopenmap:
    ldr x0, [x20, x21, lsl #3]
    bl _plamo_map_file_keep
    cbz x0, Lopenfail
    lsl x9, x21, #4
    add x9, x19, x9
    stp x0, x1, [x9]
    adrp x9, _plamo_weight_fds@PAGE
    add x9, x9, _plamo_weight_fds@PAGEOFF
    str x2, [x9, x21, lsl #3]
    add x21, x21, #1
    cmp x21, x22
    b.lo Lopenmap
    adrp x9, _plamo_weight_mapping_count@PAGE
    str x22, [x9, _plamo_weight_mapping_count@PAGEOFF]
    mov x0, x19
    mov x1, x22
    adrp x2, _plamo_weight_schema@PAGE
    add x2, x2, _plamo_weight_schema@PAGEOFF
    bl _plamo_bind_weights
    cbz x0, Lopenfail
    adrp x9, _plamo_arena_size@PAGE
    ldr x0, [x9, _plamo_arena_size@PAGEOFF]
    bl _plamo_allocate
    cbz x0, Lopenfail
    adrp x9, Lmodel_arena@PAGE
    str x0, [x9, Lmodel_arena@PAGEOFF]
    adrp x1, _plamo_arena_patches@PAGE
    add x1, x1, _plamo_arena_patches@PAGEOFF
    bl _plamo_arena_patch
    adrp x0, _plamo_model_patches@PAGE
    add x0, x0, _plamo_model_patches@PAGEOFF
    bl _plamo_model_patch
    cbz x0, Lopenfail
    adrp x9, Lmodel_ready@PAGE
    mov x10, #1
    str x10, [x9, Lmodel_ready@PAGEOFF]
Lopensuccess:
    mov x0, #1
    b Lopenreturn
Lopenfail:
    adrp x9, Lmodel_arena@PAGE
    ldr x0, [x9, Lmodel_arena@PAGEOFF]
    str xzr, [x9, Lmodel_arena@PAGEOFF]
    adrp x9, _plamo_arena_size@PAGE
    ldr x1, [x9, _plamo_arena_size@PAGEOFF]
    bl _plamo_unmap_file
    mov x21, #0
Lopencleanup:
    lsl x9, x21, #4
    add x9, x19, x9
    ldp x0, x1, [x9]
    stp xzr, xzr, [x9]
    bl _plamo_unmap_file
    adrp x9, _plamo_weight_fds@PAGE
    add x9, x9, _plamo_weight_fds@PAGEOFF
    ldr x0, [x9, x21, lsl #3]
    mov x10, #-1
    str x10, [x9, x21, lsl #3]
    cmp x0, #0
    b.lt Lopenfdclosed
    mov x16, #6
    svc #0x80
Lopenfdclosed:
    add x21, x21, #1
    cmp x21, x22
    b.lo Lopencleanup
    adrp x9, _plamo_weight_mapping_count@PAGE
    str xzr, [x9, _plamo_weight_mapping_count@PAGEOFF]
    mov x0, #0
Lopenreturn:
    ldr x22, [sp, #32]
    ldp x20, x21, [sp, #16]
    ldp x19, x30, [sp], #48
    ret
.data
.p2align 3
Lmodel_ready: .quad 0

Lmodel_arena: .quad 0
