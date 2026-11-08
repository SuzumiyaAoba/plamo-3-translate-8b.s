#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void plamo_attention_head(const float*,const float*,const float*,float*,float*,uint64_t,uint64_t,uint64_t);
extern void plamo_attention_head_reference(const float*,const float*,const float*,float*,float*,uint64_t,uint64_t,uint64_t);
extern void plamo_rope_apply(float*,const float*,const float*,uint64_t,uint64_t);
extern void plamo_rope_apply_reference(float*,const float*,const float*,uint64_t,uint64_t);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
 size_t ns[]={1,3,4,5,17,128,1025},ds[]={1,3,4,5,7,8,128};
 for(size_t ni=0;ni<7;ni++)for(size_t di=0;di<7;di++){
  size_t n=ns[ni],d=ds[di],stride=d*4+3;float *q=malloc(d*4),*k=malloc(n*stride*4),*v=malloc(n*stride*4),*a=malloc(d*4),*b=malloc(d*4),*as=malloc(n*4),*bs=malloc(n*4);
  for(size_t j=0;j<d;j++)q[j]=(float)((int)(j%13)-6)*0.13f;
  for(size_t j=0;j<n*stride;j++){k[j]=(float)((int)(j%31)-15)*0.17f;v[j]=(float)((int)(j%17)-8)*0.23f;}
  plamo_attention_head(q,k,v,a,as,n,d,stride);plamo_attention_head_reference(q,k,v,b,bs,n,d,stride);
  if(memcmp(a,b,d*4)||memcmp(as,bs,n*4)){fprintf(stderr,"attention bit mismatch n=%zu d=%zu\n",n,d);return 1;}
  if(d==128&&n==1025){double t=now();for(int i=0;i<20;i++)plamo_attention_head_reference(q,k,v,b,bs,n,d,stride);double old=(now()-t)/20;t=now();for(int i=0;i<20;i++)plamo_attention_head(q,k,v,a,as,n,d,stride);double fast=(now()-t)/20;printf("attention 1025x128: %.3f -> %.3f ms (%.2fx)\n",old*1e3,fast*1e3,old/fast);}
  free(q);free(k);free(v);free(a);free(b);free(as);free(bs);
 }
 for(size_t half=1;half<=67;half++){
  float a[4*134],b[4*134],c[67],s[67];for(size_t j=0;j<half;j++){c[j]=(float)(j+1)/67;s[j]=(float)(67-j)/67;}for(size_t j=0;j<8*half;j++)a[j]=b[j]=(float)((int)(j%23)-11)*0.1f;
  plamo_rope_apply(a,c,s,4,half);plamo_rope_apply_reference(b,c,s,4,half);if(memcmp(a,b,8*half*4))return 2;
 }
 puts("SIMD Attention / RoPE: PASS (bitwise, tails, signed inputs, stride)");
}
