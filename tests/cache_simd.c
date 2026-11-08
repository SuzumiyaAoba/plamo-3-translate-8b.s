#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern int plamo_cache_append(struct cache*,const float*,const float*),plamo_cache_append_reference(struct cache*,const float*,const float*);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
 for(size_t width=1;width<35;width++)for(size_t cap=1;cap<7;cap++){
  float a[256]={0},b[256]={0},c[256]={0},d[256]={0},row[40];struct cache x={a,b,cap,0,width,1},y={c,d,cap,0,width,1};
  for(size_t t=0;t<2*cap+3;t++){for(size_t i=0;i<width;i++)row[i]=(float)(t*79+i*31);if(!plamo_cache_append(&x,row,row)||!plamo_cache_append_reference(&y,row,row))return 1;if(x.count!=y.count||memcmp(a,c,sizeof a)||memcmp(b,d,sizeof b))return 2;}
 }
 size_t cap=2049,w=512;float *a=calloc(cap*w,4),*b=calloc(cap*w,4),*row=calloc(w,4);struct cache c={a,b,cap,cap,w,1};
 double start=now();for(int i=0;i<20;i++)plamo_cache_append_reference(&c,row,row);double old=(now()-start)/20;
 start=now();for(int i=0;i<20;i++)plamo_cache_append(&c,row,row);double fast=(now()-start)/20;
 printf("SIMD cache moves: PASS (bitwise overlap/tails); sliding 2049x512 %.3f -> %.3f ms (%.2fx)\n",old*1e3,fast*1e3,old/fast);free(a);free(b);free(row);
}
