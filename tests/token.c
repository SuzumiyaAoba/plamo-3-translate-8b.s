#include <stdint.h>
#include <stdio.h>
#include <math.h>
extern int plamo_embedding(const uint16_t*,uint64_t,uint64_t,uint64_t,float*);
extern uint64_t plamo_argmax(const float*,uint64_t);
int main(void){
 uint16_t w[]={0,0,0,0,0,0x3f80,0xc000,0x4040,0x3f00,0xbf00};float y[5];
 if(!plamo_embedding(w,1,2,5,y))return 1;
 float expected[]={1,-2,3,0.5f,-0.5f};
 for(int i=0;i<5;i++)if(y[i]!=expected[i])return 2;
 if(plamo_embedding(w,2,2,5,y)||y[0]!=1)return 3;
 float logits[]={-5,2,2,-1};if(plamo_argmax(logits,4)!=1)return 4;
 float infinities[]={-INFINITY,-INFINITY};if(plamo_argmax(infinities,2)!=0)return 5;
 logits[3]=NAN;if(plamo_argmax(logits,4)!=UINT64_MAX)return 6;
 if(plamo_argmax(0,0)!=UINT64_MAX)return 7;
 puts("embedding / greedy selection: PASS (BF16 conversion, bounds, ties, NaN)");
}
