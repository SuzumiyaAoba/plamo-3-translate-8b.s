#include <stdint.h>
#include <stdio.h>
struct node{uint32_t cp,child,sibling;int32_t id,score;uint32_t terminal;};
struct trie{struct node *nodes;uint64_t capacity,used;uint32_t *roots;};
struct descriptor{struct trie *trie;const uint32_t *cp;uint64_t n;uint32_t *bytes;int64_t *costs,*paths;void *candidates;uint64_t capacity;uint32_t *out;uint64_t outcap;};
extern int plamo_trie_insert(struct trie*,const uint32_t*,uint64_t,int32_t,int32_t);
extern uint64_t plamo_encode_segment(const struct descriptor*);
extern uint64_t plamo_encode_marked(const struct descriptor*);
int main(void){
 struct node nodes[64]={0};struct trie t={nodes,64,1,0};uint32_t abc[]={97,98,99},bc[]={98,99};
 plamo_trie_insert(&t,abc,1,10,-5);plamo_trie_insert(&t,abc,2,20,-6);plamo_trie_insert(&t,bc,2,30,-3);plamo_trie_insert(&t,abc+2,1,40,-2);
 uint32_t cp[]={97,98,99,0x1f600},bytes[256],out[16];int64_t costs[5],paths[12];uint32_t candidates[32];
 for(int i=0;i<256;i++)bytes[i]=1000+i;
 struct descriptor d={&t,cp,4,bytes,costs,paths,candidates,8,out,16};
 if(plamo_encode_segment(&d)!=6||out[0]!=20||out[1]!=40||out[2]!=1240||out[3]!=1159||out[4]!=1152||out[5]!=1128)return 1;
 d.outcap=1;if(plamo_encode_segment(&d)!=UINT64_MAX)return 2;
 d.n=0;if(plamo_encode_segment(&d)!=0)return 3;
 uint32_t marked[]={0xee00,97,0xee00,0xee00,98,99,0xee00};
 d.cp=marked;d.n=7;d.outcap=16;
 if(plamo_encode_marked(&d)!=2||out[0]!=10||out[1]!=30)return 4;
 if(d.cp!=marked||d.n!=7||d.outcap!=16)return 5;
 d.outcap=1;if(plamo_encode_marked(&d)!=UINT64_MAX)return 6;
 puts("Unigram segment encoder: PASS (global DP, longest tie, emoji fallback, boundaries, capacity)");
}
