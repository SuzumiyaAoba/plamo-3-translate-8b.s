#include <math.h>
#include <stdio.h>
#include <stdint.h>
extern void plamo_attention_head(const float*,const float*,const float*,float*,float*,uint64_t,uint64_t,uint64_t);
struct gqa {const float *q,*k,*v;float *out,*scores;uint64_t tokens,dim,qheads,kvheads;};
extern void plamo_gqa(const struct gqa*);
int main(void){
 enum {D=5,N=4,S=15};float q[D],k[N*S],v[N*S],out[D],scores[N];double probs[N],total=0;
 for(int j=0;j<D;j++)q[j]=(j-2)*0.3f;
 for(int i=0;i<N*S;i++){k[i]=((i*7)%19-9)*0.1f;v[i]=((i*3)%17-8)*0.2f;}
 // Middle GQA KV head, strided token storage.
 for(int i=0;i<N;i++){double dot=0;for(int j=0;j<D;j++)dot+=q[j]*k[i*S+D+j];probs[i]=exp(dot/sqrt(D));total+=probs[i];}
 plamo_attention_head(q,k+D,v+D,out,scores,N,D,S);
 for(int j=0;j<D;j++){double ref=0;for(int i=0;i<N;i++)ref+=probs[i]/total*v[i*S+D+j];if(fabs(out[j]-ref)>2e-6)return 1;}
 // Selected one-token range must return exactly its value.
 plamo_attention_head(q,k+S+D,v+S+D,out,scores,1,D,S);
 for(int j=0;j<D;j++)if(out[j]!=v[S+D+j])return 2;
 enum {QH=6,KH=2,GD=4,GN=3};
 float gq[QH*GD],gk[GN*KH*GD],gv[GN*KH*GD],go[QH*GD],gs[GN];
 for(int i=0;i<QH*GD;i++)gq[i]=(i%7-3)*0.2f;
 for(int i=0;i<GN*KH*GD;i++){gk[i]=(i%11-5)*0.3f;gv[i]=(i%13-6)*0.1f;}
 struct gqa a={gq,gk,gv,go,gs,GN,GD,QH,KH};plamo_gqa(&a);
 for(int h=0;h<QH;h++){
  double p[GN],z=0;int kh=h/(QH/KH);
  for(int t=0;t<GN;t++){double dot=0;for(int j=0;j<GD;j++)dot+=gq[h*GD+j]*gk[(t*KH+kh)*GD+j];p[t]=exp(dot/sqrt(GD));z+=p[t];}
  for(int j=0;j<GD;j++){double r=0;for(int t=0;t<GN;t++)r+=p[t]/z*gv[(t*KH+kh)*GD+j];if(fabs(go[h*GD+j]-r)>2e-6)return 3;}
 }
 puts("attention / GQA: PASS (scaled scores, strided KV, selected range, grouped heads)");
}
