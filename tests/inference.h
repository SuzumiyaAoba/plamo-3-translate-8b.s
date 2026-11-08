#ifndef PLAMO_TEST_INFERENCE_H
#define PLAMO_TEST_INFERENCE_H
#include <stddef.h>
#include <stdint.h>
struct qkv {const float *input; const uint16_t *weight,*qn,*kn; float *out; uint64_t hidden,qh,kh,dim;};
struct cache {float *k,*v; uint64_t capacity,count,width,sliding;};
struct rotation {struct cache *cache; float *out; const float *freq; float *cos,*sin; uint64_t heads,dim,processed_count,scale_bits;};
struct gqa {float *q,*k,*v,*out,*scores; uint64_t tokens,dim,qh,kh;};
struct mixer {struct qkv *qkv; struct cache *cache; struct rotation *rotation; struct gqa *gqa; const uint16_t *weight; float *out; uint64_t hidden;};
struct mlp {float *state; const uint16_t *pre,*post,*gu,*down; float *normalized,*branch,*scratch; uint64_t hidden,intermediate;};
struct layer {float *state; const uint16_t *pre,*post; float *normalized,*branch; struct mixer *mixer; struct mlp *mlp; uint64_t hidden;};
struct model {const uint16_t *embedding; uint64_t vocab,hidden; float *state; struct layer **layers; uint64_t count; float *scales; const uint16_t *norm; float *normalized; const uint16_t *head; float *logits;};
extern int plamo_prefill(struct model*,const uint32_t*,uint64_t,float*);
extern int plamo_prefill_reference(struct model*,const uint32_t*,uint64_t,float*);
extern int plamo_forward_hidden(struct model*,uint64_t);
extern int plamo_forward(struct model*,uint64_t);
#endif
