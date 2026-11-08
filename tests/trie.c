#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
struct node{uint32_t cp,child,sibling;int32_t id,score;uint32_t terminal;};
struct trie{struct node *nodes;uint64_t capacity,used;uint32_t *roots;};
struct candidate{uint32_t length;int32_t id,score;uint32_t reserved;};
extern int plamo_trie_insert(struct trie*,const uint32_t*,uint64_t,int32_t,int32_t);
extern uint64_t plamo_trie_match(const struct trie*,const uint32_t*,uint64_t,struct candidate*,uint64_t);
int main(void){
 struct node nodes[32]={0};struct trie t={nodes,32,1,0};uint32_t a[]={0x65e5,0x672c,0x8a9e},b[]={0x65e5,0x672c,65};struct candidate c[4];
 if(!plamo_trie_insert(&t,a,1,10,-100)||!plamo_trie_insert(&t,a,3,30,-300)||!plamo_trie_insert(&t,b,3,40,-400))return 1;
 if(t.used!=5)return 2;
 if(plamo_trie_match(&t,a,3,c,4)!=2||c[0].length!=1||c[0].id!=10||c[1].length!=3||c[1].id!=30)return 3;
 if(plamo_trie_match(&t,b,3,c,4)!=2||c[1].id!=40)return 4;
 if(plamo_trie_match(&t,a,3,c,1)!=UINT64_MAX)return 5;
 uint64_t used=t.used;t.capacity=t.used;if(plamo_trie_insert(&t,a,3,5,-5)||t.used!=used)return 6;
 struct node indexed[32]={0};uint32_t *roots=calloc(0x110001,4);roots[0x110000]=0xdeadbeef;struct trie fast={indexed,32,1,roots};
 if(!plamo_trie_insert(&fast,a,1,10,-100)||!plamo_trie_insert(&fast,a,3,30,-300)||!plamo_trie_insert(&fast,b,3,40,-400))return 7;
 if(plamo_trie_match(&fast,a,3,c,4)!=2||c[1].id!=30)return 8;
 uint32_t edges[]={0,0x10ffff,0x110000,0xffffffff};
 for(unsigned i=0;i<4;i++){if(!plamo_trie_insert(&fast,edges+i,1,100+i,-100))return 9;if(plamo_trie_match(&fast,edges+i,1,c,4)!=1||c[0].id!=(int)(100+i))return 10;}
 if(!roots[0]||!roots[0x10ffff]||roots[0x110000]!=0xdeadbeef)return 11;
 free(roots);
 puts("Unicode trie: PASS (shared prefixes, branches, candidate IDs, capacity, direct Unicode root index)");
}
