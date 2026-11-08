#include <stdint.h>
#include <stdio.h>
#include <math.h>
extern void plamo_rmsnorm(const float*,const uint16_t*,float*,size_t,float,float);
extern void plamo_bf16_matvec(const uint16_t*,const float*,float*,size_t,size_t);
int main(void) {
  uint16_t w[]={0x3f80,0x4000,0x4040,0x4080,0x40a0,0xbf80,0xc000,0xc040,0xc080,0xc0a0};
  float x[]={1,2,3,4,5}, y[2]={0};
  plamo_bf16_matvec(w,x,y,2,5);
  if(y[0]!=55 || y[1]!=-55) return 1;
  plamo_bf16_matvec(w,x,y,0,5);
  puts("BF16 matvec: PASS (NEON, scalar tail, multiple rows, zero rows)");
  uint16_t nw[]={0,0x3f80,0xbf00};
  float nx[]={1,2,3}, ny[3];
  plamo_rmsnorm(nx,nw,ny,3,1e-6f,0.2f);
  for(int i=0;i<3;i++) {
    float weights[]={0,1,-0.5f};
    float expected=nx[i]/sqrtf(14.0f/3+1e-6f)*(weights[i]+0.2f);
    if(fabsf(ny[i]-expected)>1e-6f) return 2;
  }
  plamo_rmsnorm(nx,nw,nx,3,1e-6f,0.2f);
  for(int i=0;i<3;i++) if(nx[i]!=ny[i]) return 3;
  plamo_rmsnorm(0,0,0,0,1e-6f,1);
  puts("RMSNorm: PASS (offset, in-place, zero count)");
  return 0;
}
