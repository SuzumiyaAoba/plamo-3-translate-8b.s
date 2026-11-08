#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct matmul {const uint16_t *w; const float *x; float *y; size_t rows,cols,tokens,xstride,ystride;};
extern void plamo_bf16_matmul(const struct matmul*);
extern void plamo_bf16_matmul8x3(const struct matmul*);
extern void plamo_bf16_matvec(const uint16_t*,const float*,float*,size_t,size_t);
static uint32_t rng=19;
static uint32_t next(void){rng=rng*1664525u+1013904223u;return rng;}
int main(void){
 size_t shapes[][3]={{0,0,0},{0,16,4},{4,0,4},{1,1,1},{3,7,5},{4,4,4},{3,4,8},{5,12,17},{8,12,8},{16,128,9},{12,64,17},{128,4096,7},{64,16384,5}};
 for(size_t k=0;k<sizeof shapes/sizeof *shapes;k++){
  size_t r=shapes[k][0],c=shapes[k][1],t=shapes[k][2],xs=c+7,ys=r+9;
  uint16_t *w=malloc((r*c+1)*2);float *x=malloc((t*xs+1)*4),*a=malloc((t*ys+1)*4),*b=malloc((t*ys+1)*4);
  for(size_t i=0;i<r*c;i++)w[i]=(uint16_t)((next()&0x8000)|0x3800|(next()&0x7ff));
  for(size_t i=0;i<t*xs;i++)x[i]=((int)(next()%8193)-4096)*0.0009765625f;
  for(size_t i=0;i<t*ys+1;i++)a[i]=b[i]=-123456.f;
  struct matmul m={w,x,a,r,c,t,xs,ys};plamo_bf16_matmul(&m);
  for(size_t i=0;i<t;i++)plamo_bf16_matvec(w,x+i*xs,b+i*ys,r,c);
  if(memcmp(a,b,(t*ys+1)*4)){fprintf(stderr,"matmul mismatch shape %zu,%zu,%zu\n",r,c,t);return 1;}
  for(size_t i=0;i<t*ys+1;i++)a[i]=-123456.f;
  plamo_bf16_matmul8x3(&m);if(memcmp(a,b,(t*ys+1)*4)){fprintf(stderr,"8x3 mismatch shape %zu,%zu,%zu\n",r,c,t);return 2;}
  free(w);free(x);free(a);free(b);
 }
 puts("batched BF16 matmul: PASS (bitwise, signed data, strides, tails, empty shapes)");
}
