#include <stdint.h>
#include <stdio.h>
struct descriptor{const uint32_t *in;uint64_t n;uint32_t *temp;uint64_t tcap;uint32_t *out;uint64_t ocap;uint8_t *flags;};
extern uint64_t plamo_preprocess(const struct descriptor*);
int main(void){
 const char *s="a  b<|plamo:x|>cccc";uint32_t in[100],temp[300],out[900];uint8_t flags[301];unsigned n=0;for(;s[n];n++)in[n]=s[n];
 struct descriptor d={in,n,temp,300,out,900,flags};uint64_t k=plamo_preprocess(&d);
 const char *parts[]={"a","  ","b","<|plamo:x|>","cccc"};uint32_t expected[100];unsigned e=0;
 for(unsigned p=0;p<5;p++){if(p==1||p==3||p==4)expected[e++]=0xee00;for(unsigned j=0;parts[p][j];j++)expected[e++]=parts[p][j];if(p==1||p==3||p==4)expected[e++]=0xee00;}
 if(k!=e)return 1;for(unsigned i=0;i<e;i++)if(out[i]!=expected[i])return 2;
 d.tcap=1;if(plamo_preprocess(&d)!=UINT64_MAX)return 3;
 puts("tokenizer preprocessing: PASS (ordered special/repeat stages, capacity)");
}
