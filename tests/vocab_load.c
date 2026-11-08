#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
struct node{uint32_t cp,child,sibling;int32_t id,score;uint32_t terminal;};
struct trie{struct node *nodes;uint64_t capacity,used;uint32_t *roots;};
struct candidate{uint32_t length;int32_t id,score;uint32_t reserved;};
extern uint64_t plamo_vocab_load(const char*,const char*,struct trie*,uint32_t*,uint64_t,uint32_t*);
extern uint64_t plamo_trie_match(const struct trie*,const uint32_t*,uint64_t,struct candidate*,uint64_t);
int main(void){
 FILE *f=fopen("models/plamo/tokenizer.jsonl","rb");if(!f){puts("official vocabulary load: SKIP (download files first)");return 0;}
 fseek(f,0,SEEK_END);long size=ftell(f);rewind(f);char *data=malloc(size);if(fread(data,1,size,f)!=(size_t)size)return 1;fclose(f);
 struct trie t={calloc(2000000,sizeof(struct node)),2000000,1,calloc(0x110000,4)};uint32_t cp[4096],bytes[256]={0};
 uint64_t n=plamo_vocab_load(data,data+size,&t,cp,4096,bytes);
 if(n!=107520){fprintf(stderr,"load failed: %llu nodes=%llu\n",(unsigned long long)n,(unsigned long long)t.used);return 2;}
 for(int i=0;i<256;i++)if(!bytes[i])return 3;
 uint32_t text[]={0x65e5,0x672c,0x8a9e};struct candidate c[32];
 if(!plamo_trie_match(&t,text,3,c,32))return 4;
 printf("official vocabulary load: PASS (%llu rows, %llu trie nodes)\n",(unsigned long long)n,(unsigned long long)t.used);
 free(t.roots);free(t.nodes);free(data);
}
