#include <math.h>
#include <stdio.h>
#include <stdint.h>
extern void plamo_rope_apply(float*,const float*,const float*,uint64_t,uint64_t);
int main(void){
 enum {H=3,D=128};float x[H*D],ref[H*D],c[D/2],s[D/2];
 for(int i=0;i<D/2;i++){double a=1234*pow(10000.,-2.*i/D);c[i]=cos(a);s[i]=sin(a);}
 for(int i=0;i<H*D;i++)x[i]=(i%19-9)*0.2f;
 for(int h=0;h<H;h++)for(int i=0;i<D/2;i++){
  float a=x[h*D+i],b=x[h*D+i+D/2];
  ref[h*D+i]=a*c[i]-b*s[i];ref[h*D+i+D/2]=b*c[i]+a*s[i];
 }
 plamo_rope_apply(x,c,s,H,D/2);
 for(int i=0;i<H*D;i++)if(fabsf(x[i]-ref[i])>1e-6f)return 1;
 // Identity at position zero, including multiple heads.
 for(int i=0;i<D/2;i++){c[i]=1;s[i]=0;}
 plamo_rope_apply(x,c,s,H,D/2);
 for(int i=0;i<H*D;i++)if(fabsf(x[i]-ref[i])>1e-6f)return 2;
 plamo_rope_apply(0,0,0,0,D/2);plamo_rope_apply(0,0,0,H,0);
 puts("RoPE apply: PASS (split-half, 128 dimensions, multiple heads, in-place)");
}
