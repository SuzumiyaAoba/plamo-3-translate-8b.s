.text
.p2align 2
// d0=angle, x0=output {float sin,cos}. Valid finite |angle| <= 2^22.
// Double range reduction and Taylor polynomials; no libm calls.
.globl _plamo_sincos
_plamo_sincos:
    adrp x9, Ltrigconstants@PAGE
    add x9, x9, Ltrigconstants@PAGEOFF
    ldr d1, [x9]
    fdiv d2, d0, d1
    frintn d2, d2
    fmsub d0, d2, d1, d0
    fmul d3, d0, d0
    add x10, x9, #8
    ldr d4, [x10], #8
    mov x11, #7
Lsinpoly:
    ldr d5, [x10], #8
    fmadd d4, d4, d3, d5
    subs x11, x11, #1
    b.ne Lsinpoly
    fmov d5, #1.0
    fmadd d4, d4, d3, d5
    fmul d4, d4, d0
    fcvt s4, d4
    str s4, [x0]
    ldr d4, [x10], #8
    mov x11, #7
Lcospoly:
    ldr d5, [x10], #8
    fmadd d4, d4, d3, d5
    subs x11, x11, #1
    b.ne Lcospoly
    fmov d5, #1.0
    fmadd d4, d4, d3, d5
    fcvt s4, d4
    str s4, [x0, #4]
    ret
.section __TEXT,__const
.p2align 3
Ltrigconstants:
.double 6.283185307179586
.double 2.8114572543455206e-15,-7.647163731819816e-13,1.6059043836821613e-10,-2.505210838544172e-08,2.7557319223985893e-06,-0.0001984126984126984,0.008333333333333333,-0.16666666666666666
.double 4.779477332387385e-14,-1.1470745597729725e-11,2.08767569878681e-09,-2.755731922398589e-07,2.48015873015873e-05,-0.001388888888888889,0.041666666666666664,-0.5
