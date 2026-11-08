#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
struct matmul {const uint16_t *w;const float *x;float *y;size_t rows,cols,tokens,xstride,ystride;};
extern void plamo_bf16_matmul(const struct matmul*);
extern void plamo_bf16_matmul4(const struct matmul*);
extern void plamo_bf16_matmul8x3(const struct matmul*);
extern void plamo_bf16_matmul8(const struct matmul*);
extern void plamo_bf16_matvec(const uint16_t*,const float*,float*,size_t,size_t);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void separate(const struct matmul *m){for(size_t t=0;t<m->tokens;t++)plamo_bf16_matvec(m->w,m->x+t*m->xstride,m->y+t*m->ystride,m->rows,m->cols);}
int main(void){
 size_t r=16384,c=4096;uint16_t *w=malloc(r*c*2);float *x=malloc(32*c*4),*a=malloc(32*r*4),*b=malloc(32*r*4);if(!w||!x||!a||!b)return 1;
 for(size_t i=0;i<r*c;i++)w[i]=0x3f00+(i%256);for(size_t i=0;i<32*c;i++)x[i]=(int)(i%31)-15;
 size_t counts[]={1,4,8,16,32};
 for(size_t j=0;j<5;j++){size_t t=counts[j];struct matmul m={w,x,a,r,c,t,c,r};plamo_bf16_matmul(&m);m.y=b;separate(&m);if(memcmp(a,b,t*r*4))return 2;
  double old=1e9,fast=1e9,four=1e9,three=1e9,two=1e9;
  for(int i=0;i<5;i++){
   double start;
   if(i%2){start=now();plamo_bf16_matmul(&m);double v=now()-start;if(v<fast)fast=v;start=now();separate(&m);v=now()-start;if(v<old)old=v;}
   else{start=now();separate(&m);double v=now()-start;if(v<old)old=v;start=now();plamo_bf16_matmul(&m);v=now()-start;if(v<fast)fast=v;}
   start=now();plamo_bf16_matmul8(&m);double v2=now()-start;if(v2<two)two=v2;
   start=now();plamo_bf16_matmul8x3(&m);double v3=now()-start;if(v3<three)three=v3;
   start=now();plamo_bf16_matmul4(&m);double v4=now()-start;if(v4<four)four=v4;
  }
  printf("%zu tokens,128 MiB: matvec %.3f ms, matmul %.3f ms, %.2fx (4-token tile %.3f ms; 8x3 tile %.3f ms; 8x2 tile %.3f ms)\n",t,old*1e3,fast*1e3,old/fast,four*1e3,three*1e3,two*1e3);
 }
 free(w);free(x);free(a);free(b);
}
