#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
extern void plamo_mlp(const uint16_t*,const uint16_t*,const float*,float*,float*,uint64_t,uint64_t);
extern void plamo_residual(float*,const float*,uint64_t);
struct branch {float *state;const uint16_t *pre,*post,*gu,*down;float *normalized,*branch,*scratch;uint64_t hidden,intermediate;};
extern void plamo_mlp_branch(const struct branch*);
static float decode(uint16_t v){uint32_t bits=(uint32_t)v<<16;float f;memcpy(&f,&bits,4);return f;}
int main(void) {
 // Small dense network covers NEON rows and non-multiple-of-four tails.
 enum {H=5,I=7};uint16_t gu[2*I*H],down[H*I];float input[H],out[H],scratch[2*I],ref[H];
 const uint16_t vals[]={0xbf80,0xbf00,0,0x3f00,0x3f80};
 for(int i=0;i<2*I*H;i++)gu[i]=vals[(i*7+i/5)%5];
 for(int i=0;i<H*I;i++)down[i]=vals[(i*3+i/7)%5];
 for(int i=0;i<H;i++)input[i]=(i-2)*0.125f;
 double mid[I];
 for(int i=0;i<I;i++){double g=0,u=0;for(int j=0;j<H;j++){g+=decode(gu[i*H+j])*input[j];u+=decode(gu[(i+I)*H+j])*input[j];}mid[i]=g/(1+exp(-g))*u;}
 for(int i=0;i<H;i++){double v=0;for(int j=0;j<I;j++)v+=decode(down[i*I+j])*mid[j];ref[i]=v;}
 plamo_mlp(gu,down,input,out,scratch,H,I);
 for(int i=0;i<H;i++)if(fabsf(out[i]-ref[i])>1e-5)return 1;
 plamo_mlp(gu,down,input,input,scratch,H,I);
 for(int i=0;i<H;i++)if(input[i]!=out[i])return 2;
 plamo_residual(out,input,H);
 for(int i=0;i<H;i++)if(out[i]!=2*input[i])return 3;
 plamo_residual(0,0,0);
 // End-to-end branch with nonzero norm weights and independent double reference.
 uint16_t pre[H],post[H];float state[H],norm[H],branch[H];double variance=0;
 for(int i=0;i<H;i++){state[i]=(i-2)*0.3f;pre[i]=vals[i];post[i]=vals[H-1-i];variance+=(double)state[i]*state[i];}
 double normalized[H];for(int i=0;i<H;i++)normalized[i]=state[i]/sqrt(variance/H+1e-6)*(1+decode(pre[i]));
 for(int i=0;i<I;i++){double g=0,u=0;for(int j=0;j<H;j++){g+=decode(gu[i*H+j])*normalized[j];u+=decode(gu[(i+I)*H+j])*normalized[j];}mid[i]=g/(1+exp(-g))*u;}
 double br[H],brvar=0;for(int i=0;i<H;i++){br[i]=0;for(int j=0;j<I;j++)br[i]+=decode(down[i*I+j])*mid[j];brvar+=br[i]*br[i];}
 for(int i=0;i<H;i++)ref[i]=state[i]+br[i]/sqrt(brvar/H+1e-6)*(decode(post[i])+1/pow(5,1.5));
 struct branch b={state,pre,post,gu,down,norm,branch,scratch,H,I};plamo_mlp_branch(&b);
 for(int i=0;i<H;i++)if(fabsf(state[i]-ref[i])>2e-5f){fprintf(stderr,"branch %d: %g != %g\n",i,state[i],ref[i]);return 4;}
 puts("MLP forward / residual: PASS (dense BF16 weights, reference, in-place, normalized residual branch)");
}
