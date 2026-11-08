#include <stdint.h>
#include <math.h>
#include <stdio.h>
struct qkv{const float *in;const uint16_t *w,*qn,*kn;float *out;uint64_t hidden,qh,kh,dim;};
struct cache{float *k,*v;uint64_t cap,n,width,sliding;};
struct rotation{struct cache *cache;float *out;const float *freq;float *cos,*sin;uint64_t heads,dim,processed_count,scale_bits;};
struct gqa{float *q,*k,*v,*out,*scores;uint64_t tokens,dim,qh,kh;};
struct mixer{struct qkv *qkv;struct cache *cache;struct rotation *rotation;struct gqa *gqa;const uint16_t *ow;float *out;uint64_t hidden;};
extern int plamo_mixer(const struct mixer*,float);
struct mlp{float *state;const uint16_t *pre,*post,*gu,*down;float *normalized,*branch,*scratch;uint64_t hidden,intermediate;};
struct layer{float *state;const uint16_t *pre,*post;float *normalized,*output;struct mixer *mixer;struct mlp *mlp;uint64_t hidden;};
extern int plamo_layer(const struct layer*,float);
struct model{const uint16_t *embedding;uint64_t vocab,hidden;float *state;struct layer **layers;uint64_t count;float *scales;const uint16_t *norm;float *normalized;const uint16_t *head;float *logits;};
extern int plamo_forward(const struct model*,uint64_t);
extern int plamo_forward_hidden(const struct model*,uint64_t);
extern int plamo_prefill(const struct model*,const uint32_t*,uint64_t,float*);
int main(void){
 // H=2,QH=2,KH=1,D=2. V projection identity; O selects duplicated V.
 float input[]={2,3},qkvout[8],k[4],v[4],rk[4],freq[]={1},cosbuf[1],sinbuf[1],a[4],scores[2],out[2];uint16_t w[16]={0},norm[2]={0},ow[]={0x3f80,0,0x3f80,0,0,0x3f80,0,0x3f80};
 for(int r=0;r<8;r++)w[r*2+r%2]=0x3f80;
 struct qkv q={input,w,norm,norm,qkvout,2,2,1,2};struct cache c={k,v,2,0,2,0};struct rotation r={&c,rk,freq,cosbuf,sinbuf,1,2,0,0};struct gqa g={0,0,0,a,scores,0,2,2,1};struct mixer m={&q,&c,&r,&g,ow,out,2};
 if(!plamo_mixer(&m,1)||c.n!=1||out[0]!=4||out[1]!=6)return 1;
 // Identical values at two positions keep weighted sum unchanged.
 if(!plamo_mixer(&m,1)||c.n!=2||fabs(out[0]-4)>1e-6||fabs(out[1]-6)>1e-6)return 2;
 if(plamo_mixer(&m,1)||c.n!=2)return 3;
 // Complete layer, nonzero Attention and MLP branches; both invariant under
 // equal positive state components. Zero norm weights give constant offsets.
 float state[]={2,2},normalized[2],mout[2],mn[2],mb[2],ms[4];
 uint16_t gu[]={0x3f80,0,0,0x3f80,0x3f80,0,0,0x3f80},down[]={0x3f80,0,0,0x3f80};
 struct mlp mlp={state,norm,norm,gu,down,mn,mb,ms,2,2};struct layer layer={state,norm,norm,normalized,mout,&m,&mlp,2};c.n=0;
 if(!plamo_layer(&layer,1)||c.n!=1)return 4;
 // Independent scalar reference for each stage (symmetry yields equal lanes).
 double pre=2/sqrt(4+1e-6),mix=2*pre;
 double after=2+mix/sqrt(mix*mix+1e-6)*0.2;
 double mp=after/sqrt(after*after+1e-6),ml=mp/(1+exp(-mp))*mp;
 double expected=after+ml/sqrt(ml*ml+1e-6)/pow(5,1.5);
 if(fabs(state[0]-expected)>2e-6||fabs(state[1]-expected)>2e-6)return 5;
 c.n=c.cap;float old=state[0];if(plamo_layer(&layer,1)||state[0]!=old)return 6;
 uint16_t embedding[]={0x4000,0x4000,0x3f80,0x3f80},head[]={0x3f80,0,0,0x3f80};float final[2],logits[2],scales[]={1};struct layer *layers[]={&layer};
 struct model model={embedding,2,2,state,layers,1,scales,norm,final,head,logits};c.n=0;
 if(!plamo_forward(&model,0)||c.n!=1)return 7;
 double expected_logit=expected/sqrt(expected*expected+1e-6);
 if(fabs(logits[0]-expected_logit)>2e-6||fabs(logits[1]-expected_logit)>2e-6)return 8;
 float previous=state[0];if(plamo_forward(&model,2)||state[0]!=previous||c.n!=1)return 9;
 c.n=0;logits[0]=123;final[0]=456;
 if(!plamo_forward_hidden(&model,0)||c.n!=1||logits[0]!=123||final[0]!=456)return 10;
 c.n=0;uint32_t ids[]={0,0};float states[4];
 if(!plamo_prefill(&model,ids,2,states)||c.n!=2)return 11;
 if(fabs(logits[0]-expected_logit)>2e-6||fabs(logits[1]-expected_logit)>2e-6)return 12;
 // Layer-major prefill must equal token-major forward on distinct inputs.
 embedding[2]=0x3f80;embedding[3]=0x4040;c.n=0;
 if(!plamo_forward_hidden(&model,0)||!plamo_forward(&model,1))return 13;
 float sequential_logits[2]={logits[0],logits[1]},sequential_state[2]={state[0],state[1]};
 c.n=0;ids[1]=1;
 if(!plamo_prefill(&model,ids,2,states))return 14;
 for(int j=0;j<2;j++)if(fabs(logits[j]-sequential_logits[j])>1e-6||fabs(state[j]-sequential_state[j])>1e-6)return 15;
 puts("Attention mixer: PASS (projection, cache, RoPE, GQA, output projection, complete Decoder layer, model forward)");
}
