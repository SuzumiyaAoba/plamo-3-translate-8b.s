#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void plamo_bf16_matvec(const uint16_t*,const float*,float*,size_t,size_t);
extern void plamo_bf16_matvec_reference(const uint16_t*,const float*,float*,size_t,size_t);
extern void plamo_bf16_matvec_4row(const uint16_t*,const float*,float*,size_t,size_t);
typedef void (*kernel)(const uint16_t*,const float*,float*,size_t,size_t);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double bench(kernel f,const uint16_t*w,const float*x,float*y,size_t r,size_t c){double start=now();for(int i=0;i<20;i++)f(w,x,y,r,c);return (now()-start)/20;}
int main(void){size_t r=16384,c=4096;uint16_t *w=malloc(r*c*2);float *x=malloc(c*4),*a=malloc(r*4),*b=malloc(r*4);if(!w||!x||!a||!b)return 1;
 uint32_t seed=13;for(size_t i=0;i<r*c;i++){seed=seed*1664525+1013904223;w[i]=(uint16_t)(0x3c00+(seed%1024));}for(size_t i=0;i<c;i++)x[i]=(int)(i%31)-15;
 for(size_t rows=0;rows<=9;rows++)for(size_t cols=0;cols<=33;cols++){plamo_bf16_matvec(w,x,a,rows,cols);plamo_bf16_matvec_reference(w,x,b,rows,cols);if(memcmp(a,b,rows*4))return 2;}
 plamo_bf16_matvec(w,x,a,r,c);plamo_bf16_matvec_reference(w,x,b,r,c);if(memcmp(a,b,r*4))return 3;
 double four=bench(plamo_bf16_matvec_4row,w,x,b,r,c);
 double old=bench(plamo_bf16_matvec_reference,w,x,b,r,c),fast=bench(plamo_bf16_matvec,w,x,a,r,c);
 printf("matvec bitwise equivalence: PASS\n128 MiB weights: old %.2f ms, new %.2f ms, %.2fx (4-row %.2f ms)\n",old*1000,fast*1000,old/fast,four*1000);free(w);free(x);free(a);free(b);
}
