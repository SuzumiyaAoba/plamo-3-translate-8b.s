.text
.p2align 2
// Populate generated model descriptor weight fields after successful binding.
// x0=patch table {destination,weight_slot}[], NULL terminated. 1=success,0=missing.
.globl _plamo_model_patch
_plamo_model_patch:
    mov x1, x0
Lpatchcheck:
    ldp x2, x3, [x1], #16
    cbz x2, Lpatchstart
    ldr x3, [x3]
    cbz x3, Lpatchbad
    b Lpatchcheck
Lpatchstart:
    ldp x2, x3, [x0], #16
    cbz x2, Lpatchdone
    ldr x3, [x3]
    str x3, [x2]
    b Lpatchstart
Lpatchdone:
    mov x0, #1
    ret
Lpatchbad:
    mov x0, #0
    ret
