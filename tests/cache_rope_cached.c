#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern int plamo_cache_append(struct cache*,const float*,const float*);
extern void plamo_cache_rope(struct rotation*,float);
extern void plamo_cache_rope_cached(struct rotation*,float);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
 enum {H=4,D=128,W=H*D,C=1025};float *k=calloc(C*W,4),*v=calloc(C*W,4),*a=calloc(C*W,4),*b=calloc(C*W,4);float freq[D/2],ca[D/2],sa[D/2],cb[D/2],sb[D/2],row[W];
 for(int j=0;j<D/2;j++)freq[j]=1.0f/(1+j*j);
 struct cache cache={k,v,C,0,W,1};struct rotation full={&cache,a,freq,ca,sa,H,D,0,0},cached={&cache,b,freq,cb,sb,H,D,0,0};
 // Append, reset, sliding eviction, and a concentration change.
 for(int t=0;t<1040;t++){
  if(t==5)cache.count=0;
  for(int j=0;j<W;j++)row[j]=(float)((t+j*17)%127-63)/64;
  if(!plamo_cache_append(&cache,row,row))return 1;
  float scale=t<9?1:1.5545177f;
  plamo_cache_rope(&full,scale);plamo_cache_rope_cached(&cached,scale);
  size_t bytes=cache.count*W*4;
  if(memcmp(a,b,bytes)||memcmp(ca,cb,sizeof ca)||memcmp(sa,sb,sizeof sa)){fprintf(stderr,"cached RoPE mismatch at append %d\n",t);return 2;}
 }
 // Cost per append at long context; establish previous prefix each iteration.
 cache.count=C-1;plamo_cache_rope(&full,1);plamo_cache_rope_cached(&cached,1);cache.count=C;
 double old=now();for(int i=0;i<20;i++)plamo_cache_rope(&full,1);old=(now()-old)/20;
 double fast=now();for(int i=0;i<20;i++){cached.processed_count=C-1;plamo_cache_rope_cached(&cached,1);}fast=(now()-fast)/20;
 printf("cached RoPE: PASS (bitwise, reset/scale change/sliding); %d keys %.3f -> %.3f ms (%.1fx)\n",C,old*1e3,fast*1e3,old/fast);
 free(k);free(v);free(a);free(b);
}
