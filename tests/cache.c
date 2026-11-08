#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct cache{float *k,*v;uint64_t capacity,count,width,sliding;};
extern int plamo_cache_append(struct cache*,const float*,const float*);
int main(void){
 float k[15]={0},v[15]={0};struct cache c={k,v,3,0,5,0};float nk[5],nv[5];
 for(int t=0;t<3;t++){for(int j=0;j<5;j++){nk[j]=t*10+j;nv[j]=-nk[j];}if(!plamo_cache_append(&c,nk,nv)||c.count!=(unsigned)t+1)return 1;}
 float saved[15];memcpy(saved,k,sizeof k);if(plamo_cache_append(&c,nk,nv)||memcmp(saved,k,sizeof k)||c.count!=3)return 2;
 c.sliding=1;
 for(int t=3;t<8;t++){for(int j=0;j<5;j++){nk[j]=t*10+j;nv[j]=-nk[j];}if(!plamo_cache_append(&c,nk,nv)||c.count!=3)return 3;for(int p=0;p<3;p++)for(int j=0;j<5;j++)if(k[p*5+j]!=(t-2+p)*10+j||v[p*5+j]!=-k[p*5+j])return 4;}
 c.capacity=1;c.count=1;if(!plamo_cache_append(&c,nk,nv)||memcmp(k,nk,sizeof nk))return 5;
 c.capacity=0;if(plamo_cache_append(&c,nk,nv))return 6;
 puts("KV cache: PASS (append, global overflow, sliding eviction, capacity one)");
}
