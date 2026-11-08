#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
struct cache{float *k,*v;uint64_t capacity,count,width,sliding;};
struct descriptor{struct cache *cache;float *out;const float *freq;float *cos,*sin;uint64_t heads,dim;};
extern void plamo_cache_rope(const struct descriptor*,float);
int main(void){
 enum{N=3,H=2,D=4};float k[N*H*D],saved[N*H*D],out[N*H*D],freq[]={1,0.1f},c[2],s[2];
 for(int i=0;i<N*H*D;i++)k[i]=(i-10)*0.2f;memcpy(saved,k,sizeof k);
 struct cache cache={k,0,N,N,H*D,1};struct descriptor d={&cache,out,freq,c,s,H,D};plamo_cache_rope(&d,1);
 if(memcmp(k,saved,sizeof k))return 1;
 for(int t=0;t<N;t++)for(int h=0;h<H;h++)for(int j=0;j<D/2;j++){
 int a=(t*H+h)*D+j,b=a+D/2;float angle=t*freq[j];double co=cos(angle),si=sin(angle);
 if(fabs(out[a]-(k[a]*co-k[b]*si))>1e-6||fabs(out[b]-(k[b]*co+k[a]*si))>1e-6)return 2;
 }
 puts("cached-key RoPE: PASS (local coordinates, multiple heads, source preserved)");
}
