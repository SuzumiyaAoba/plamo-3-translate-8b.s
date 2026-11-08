#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void plamo_softmax(float*,uint64_t),plamo_softmax_reference(float*,uint64_t);
extern void plamo_swiglu(const float*,float*,uint64_t),plamo_swiglu_reference(const float*,float*,uint64_t);
extern void plamo_rmsnorm(const float*,const uint16_t*,float*,uint64_t,float,float),plamo_rmsnorm_reference(const float*,const uint16_t*,float*,uint64_t,float,float);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
 size_t counts[71];for(size_t i=0;i<68;i++)counts[i]=i;counts[68]=128;counts[69]=4096;counts[70]=16384;
 for(size_t ci=0;ci<71;ci++){
  size_t n=counts[ci];float *x=malloc((2*n+1)*4),*a=malloc((2*n+1)*4),*b=malloc((2*n+1)*4);uint16_t *w=malloc((n+1)*2);
  for(size_t i=0;i<2*n+1;i++)x[i]=(float)((int)(i%67)-33)*3.25f;
  plamo_swiglu(x,a,n);plamo_swiglu_reference(x,b,n);if(memcmp(a,b,n*4)){fprintf(stderr,"SwiGLU mismatch n=%zu\n",n);return 1;}
  memcpy(a,x,2*n*4);memcpy(b,x,2*n*4);plamo_swiglu(a,a,n);plamo_swiglu_reference(b,b,n);if(memcmp(a,b,n*4))return 2;
  memcpy(a,x,n*4);memcpy(b,x,n*4);plamo_softmax(a,n);plamo_softmax_reference(b,n);if(memcmp(a,b,n*4)){fprintf(stderr,"softmax mismatch n=%zu\n",n);return 3;}
  for(size_t i=0;i<n;i++)w[i]=(uint16_t)(0x3b80|(i%128)|((i%3)?0:0x8000));
  plamo_rmsnorm(x,w,a,n,1e-6f,0.2f);plamo_rmsnorm_reference(x,w,b,n,1e-6f,0.2f);if(memcmp(a,b,n*4))return 4;
  free(x);free(a);free(b);free(w);
 }
 size_t n=16384;float *x=malloc(n*8),*y=malloc(n*4);for(size_t i=0;i<2*n;i++)x[i]=(float)((int)(i%31)-15)*0.1f;
 double start=now();for(int i=0;i<200;i++)plamo_swiglu_reference(x,y,n);double old=(now()-start)/200;
 start=now();for(int i=0;i<200;i++)plamo_swiglu(x,y,n);double fast=(now()-start)/200;
 printf("SIMD activations: PASS (71 shapes, bitwise incl. in-place/tails); SwiGLU %.3f -> %.3f ms (%.2fx)\n",old*1e3,fast*1e3,old/fast);free(x);free(y);
}
