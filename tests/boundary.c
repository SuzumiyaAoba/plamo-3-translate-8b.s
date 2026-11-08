#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void plamo_repeat_boundaries(const uint32_t*,uint64_t,uint8_t*);
int main(void){
 uint32_t cp[]={65,32,32,66,66,66,66,10,10,10,10,0x65e5,0x65e5,0x65e5,0x65e5,32};uint8_t flags[17]={0};
 plamo_repeat_boundaries(cp,16,flags);
 for(int i=0;i<17;i++){int expected=i==1||i==3||i==7||i==11||i==15;if(flags[i]!=expected)return 1;}
 plamo_repeat_boundaries(0,0,flags);
 puts("repeat boundaries: PASS (spaces, Unicode runs, LF exclusion)");
}
