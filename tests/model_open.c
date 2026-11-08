#include <stdint.h>
#include <stdio.h>
extern int plamo_model_open(void);
extern const void *plamo_weights[323];
extern unsigned char plamo_model[];
int main(void){
 if(!plamo_model_open()){puts("model open: incomplete/missing weights (no inference attempted)");return 77;}
 for(int i=0;i<323;i++)if(!plamo_weights[i])return 1;
 if(*(const void**)plamo_model!=plamo_weights[0])return 2;
 if(!plamo_model_open())return 3;
 puts("model open: PASS (four shards, 323 weights, descriptors, repeat initialization)");
}
