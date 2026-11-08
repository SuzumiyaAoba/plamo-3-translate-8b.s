.text
.p2align 2
// x0=descriptor {state,pre_mixer_weight,post_mixer_weight,normalized,
//               mixer_output,mixer_descriptor,mlp_branch_descriptor,hidden},
// s0=RoPE concentration. All scratch buffers disjoint from state and weights.
// Sets QKV input/mixer output/MLP state pointers; returns 1/0 (cache full).
// On cache-full failure state is unchanged; scratch may change.
.globl _plamo_layer
_plamo_layer:
    stp x19, x30, [sp, #-32]!
    str s0, [sp, #16]
    mov x19, x0
    ldp x0, x1, [x19]
    ldr x2, [x19, #24]
    ldr x3, [x19, #56]
    adrp x9, Llayerconst@PAGE
    ldr s0, [x9, Llayerconst@PAGEOFF]
    fmov s1, #1.0
    bl _plamo_rmsnorm
    ldr x0, [x19, #40]
    ldr x9, [x0]
    ldr x10, [x19, #24]
    str x10, [x9]
    ldr x10, [x19, #32]
    str x10, [x0, #40]
    ldr s0, [sp, #16]
    bl _plamo_mixer
    cbz x0, Llayerreturn
    ldr x0, [x19, #32]
    ldr x1, [x19, #16]
    mov x2, x0
    ldr x3, [x19, #56]
    adrp x9, Llayerconst@PAGE
    add x9, x9, Llayerconst@PAGEOFF
    ldp s0, s1, [x9]
    bl _plamo_rmsnorm
    ldr x0, [x19]
    ldr x1, [x19, #32]
    ldr x2, [x19, #56]
    bl _plamo_residual
    ldr x0, [x19, #48]
    ldr x9, [x19]
    str x9, [x0]
    bl _plamo_mlp_branch
    mov x0, #1
Llayerreturn:
    ldp x19, x30, [sp], #32
    ret
.section __TEXT,__const
.p2align 2
Llayerconst: .float 0.000001,0.2
