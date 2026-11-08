#include <math.h>
#include <stdint.h>
#include <stdio.h>
extern void plamo_sincos(double,float*);
int main(void){
 float out[2];double maxerr=0;uint64_t rng=7;
 for(int i=0;i<100000;i++){rng=rng*6364136223846793005ULL+1;double a=((int64_t)(rng>>20)-((int64_t)1<<43))/2097152.0;plamo_sincos(a,out);double e=fmax(fabs(out[0]-sin(a)),fabs(out[1]-cos(a)));if(e>maxerr)maxerr=e;if(e>2e-7)return 1;}
 plamo_sincos(0,out);if(out[0]!=0||out[1]!=1)return 2;
 printf("sincos: PASS (100000 long-context angles, max abs error %.3g)\n",maxerr);
}
