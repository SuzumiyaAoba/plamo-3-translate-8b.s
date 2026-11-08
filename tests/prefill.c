#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *allocations[2048];static size_t allocated;
static void *mem(size_t n,size_t bytes){void *p=calloc(n?n:1,bytes);if(!p||allocated==2048)abort();allocations[allocated++]=p;return p;}
static uint16_t *weights(size_t n,unsigned seed){uint16_t *p=mem(n,2);for(size_t i=0;i<n;i++){seed=seed*1664525u+1013904223u;p[i]=(uint16_t)(0x3c00|(seed&0x1ff)|((seed>>6)&0x8000));}return p;}
static float *buffer(size_t n){return mem(n,4);}
static struct model *model(size_t h,size_t inter){
 size_t d=4,qh=2,kh=1,a=qh*d,q=(qh+2*kh)*d,vocab=13;
 struct model *m=mem(1,sizeof *m);m->embedding=weights(vocab*h,13);m->vocab=vocab;m->hidden=h;m->state=buffer(h);m->count=2;m->layers=mem(2,sizeof *m->layers);m->scales=buffer(2);m->norm=weights(h,71);m->normalized=buffer(h);m->head=weights(vocab*h,113);m->logits=buffer(vocab);
 for(size_t i=0;i<2;i++){
  struct layer *l=mem(1,sizeof *l);m->layers[i]=l;l->state=m->state;l->hidden=h;l->pre=weights(h,19+i);l->post=weights(h,20+i);l->normalized=buffer(h);l->branch=buffer(h);
  struct mixer *mix=mem(1,sizeof *mix);l->mixer=mix;mix->hidden=h;mix->weight=weights(h*a,22+i);mix->out=l->branch;
  struct qkv *qq=mem(1,sizeof *qq);mix->qkv=qq;qq->input=l->normalized;qq->weight=weights(q*h,27+i);qq->qn=weights(d,32+i);qq->kn=weights(d,35+i);qq->out=buffer(q);qq->hidden=h;qq->qh=qh;qq->kh=kh;qq->dim=d;
  struct cache *c=mem(1,sizeof *c);mix->cache=c;c->capacity=i?64:3;c->width=kh*d;c->k=buffer(c->capacity*c->width);c->v=buffer(c->capacity*c->width);c->sliding=!i;
  struct rotation *r=mem(1,sizeof *r);mix->rotation=r;r->cache=c;r->out=buffer(c->capacity*c->width);float *f=buffer(d/2);f[0]=1;f[1]=0.01f;r->freq=f;r->cos=buffer(d/2);r->sin=buffer(d/2);r->heads=kh;r->dim=d;
  struct gqa *g=mem(1,sizeof *g);mix->gqa=g;g->out=buffer(a);g->scores=buffer(c->capacity);g->dim=d;g->qh=qh;g->kh=kh;
  struct mlp *p=mem(1,sizeof *p);l->mlp=p;p->state=m->state;p->pre=weights(h,38+i);p->post=weights(h,40+i);p->gu=weights(2*inter*h,51+i);p->down=weights(h*inter,61+i);p->normalized=buffer(h);p->branch=buffer(h);p->scratch=buffer(2*inter);p->hidden=h;p->intermediate=inter;
  m->scales[i]=i?1.5545177f:1;
 }
 return m;
}
int main(void){
 size_t counts[]={0,1,2,3,4,5,7,8,9,17};
 for(size_t shape=0;shape<2;shape++)for(size_t prefix=0;prefix<=2;prefix+=2)for(size_t k=0;k<10;k++){
  size_t h=shape?7:8,inter=shape?11:12,n=counts[k];struct model *a=model(h,inter),*b=model(h,inter);
  for(size_t i=0;i<prefix;i++)if(!plamo_forward_hidden(a,i)||!plamo_forward_hidden(b,i))return 1;
  uint32_t ids[20];for(size_t i=0;i<n;i++)ids[i]=(i*7+3)%13;
  float *as=buffer((n+1)*h),*bs=buffer((n+1)*h);
  int ok=plamo_prefill(a,ids,n,as),ref=plamo_prefill_reference(b,ids,n,bs);
  if(ok!=ref||memcmp(as,bs,n*h*4)||memcmp(a->state,b->state,h*4)||memcmp(a->logits,b->logits,13*4)){fprintf(stderr,"prefill mismatch h=%zu n=%zu prefix=%zu\n",h,n,prefix);return 2;}
  for(size_t i=0;i<2;i++){struct cache *ac=a->layers[i]->mixer->cache,*bc=b->layers[i]->mixer->cache;if(ac->count!=bc->count||memcmp(ac->k,bc->k,ac->count*ac->width*4)||memcmp(ac->v,bc->v,ac->count*ac->width*4)){fprintf(stderr,"cache mismatch\n");return 3;}}
  // Subsequent generation must work after the temporary batch workspace is freed.
  if(n&&(!plamo_forward(a,4)||!plamo_forward(b,4)||memcmp(a->logits,b->logits,13*4))){fprintf(stderr,"post-prefill forward mismatch\n");return 4;}
  for(size_t i=0;i<allocated;i++)free(allocations[i]);allocated=0;
 }
 puts("batched prefill: PASS (40 cases, bitwise states/logits/KV, sliding eviction, existing prefixes, subsequent decode)");
}
