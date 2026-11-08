#include <math.h>
#include <stdio.h>
extern float plamo_exp(float);
extern void plamo_softmax(float*,unsigned long);
extern void plamo_swiglu(const float*,float*,unsigned long);
int main(void) {
 for(int i=-8600;i<=8700;i++) {float x=i/100.f,e=expf(x),a=plamo_exp(x);if(fabsf(a/e-1)>5e-6f){printf("exp %g %g %g\n",x,a,e);return 1;}}
 float x[]={1000,1001,999,-1000};plamo_softmax(x,4);
 double sum=exp(0)+exp(1)+exp(-1);
 if(fabs(x[0]-1/sum)>1e-6||fabs(x[1]-exp(1)/sum)>1e-6||fabs(x[2]-exp(-1)/sum)>1e-6||x[3]!=0)return 2;
 float h[]={-100,-2,0,2,100,1,2,3,4,5},out[5];plamo_swiglu(h,out,5);
 for(int i=0;i<5;i++){double e=h[i]/(1+exp(-h[i]))*h[i+5];if(fabs(out[i]-e)>1e-5)return 3;}
 plamo_softmax(0,0);plamo_swiglu(0,0,0);
 puts("exp / softmax / SwiGLU: PASS");
}
