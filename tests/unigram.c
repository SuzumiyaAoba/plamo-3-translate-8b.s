#include <stdint.h>
#include <stdio.h>
struct candidate {uint32_t length;int32_t id,score;uint32_t reserved;};
struct path {int64_t cost,length,id;};
extern int plamo_unigram_step(const int64_t*,const struct candidate*,uint64_t,uint64_t,struct path*);
int main(void){
 // "abc": a+bc costs 5+3, ab+c costs 6+2. Longest prefix wins tie.
 int64_t costs[]={0,3,2,0};struct candidate c[]={{1,10,-5,0},{2,20,-6,0},{3,30,-9,0},{1,-1,-10000000,0}};struct path p;
 if(!plamo_unigram_step(costs,c,4,3,&p)||p.cost!=8||p.length!=2||p.id!=20)return 1;
 c[2].score=-7;if(!plamo_unigram_step(costs,c,4,3,&p)||p.id!=30||p.cost!=7)return 2;
 c[0].score=-20000000;c[1].length=0;c[2].length=4;
 if(!plamo_unigram_step(costs,c,4,3,&p)||p.id!=-1)return 3;
 if(plamo_unigram_step(costs,c,3,3,&p)||p.length||p.id!=-1)return 4;
 // INT64 subtraction overflow must not select a corrupt path.
 costs[1]=INT64_MAX;c[0].score=-1;
 if(plamo_unigram_step(costs,c,1,3,&p))return 5;
 puts("Unigram DP step: PASS (suffix costs, longest tie, fallback, invalid candidates)");
}
