#ifndef PLAMO_RUNTIME_MODEL_H
#define PLAMO_RUNTIME_MODEL_H
#include <stddef.h>
#include <stdint.h>

/* Layouts shared with the existing assembly ABI. Inference stays in assembly. */
struct qkv { const float *input; const uint16_t *weight, *qn, *kn; float *out; uint64_t hidden, qh, kh, dim; };
struct cache { float *k, *v; uint64_t capacity, count, width, sliding; };
struct rotation { struct cache *cache; float *out; const float *freq; float *cos, *sin; uint64_t heads, dim, processed_count, scale_bits; };
struct gqa { float *q, *k, *v, *out, *scores; uint64_t tokens, dim, qh, kh; };
struct mixer { struct qkv *qkv; struct cache *cache; struct rotation *rotation; struct gqa *gqa; const uint16_t *weight; float *out; uint64_t hidden; };
struct mlp { float *state; const uint16_t *pre, *post, *gu, *down; float *normalized, *branch, *scratch; uint64_t hidden, intermediate; };
struct layer { float *state; const uint16_t *pre, *post; float *normalized, *branch; struct mixer *mixer; struct mlp *mlp; uint64_t hidden; };
struct model { const uint16_t *embedding; uint64_t vocab, hidden; float *state; struct layer **layers; uint64_t count; float *scales; const uint16_t *norm; float *normalized; const uint16_t *head; float *logits; };
struct plamo_mapping { void *base; uint64_t bytes; };
struct plamo_weight_spec { const char *name; uint64_t rank; const uint64_t *shape; void **slot; };

_Static_assert(sizeof(struct model) == 88, "assembly model ABI");
_Static_assert(offsetof(struct layer, mixer) == 40, "assembly layer ABI");
_Static_assert(offsetof(struct cache, count) == 24, "assembly cache ABI");
_Static_assert(offsetof(struct rotation, processed_count) == 56, "assembly rotation ABI");

extern struct model plamo_model;
enum { PLAMO_MAX_WEIGHT_FILES = 64 };
extern struct plamo_mapping plamo_weight_mappings[PLAMO_MAX_WEIGHT_FILES];
extern int64_t plamo_weight_fds[PLAMO_MAX_WEIGHT_FILES];
extern uint64_t plamo_weight_mapping_count, plamo_weight_file_count;
extern const char plamo_vocab_path[], plamo_model_name[], plamo_model_repository[], plamo_model_revision[];
extern struct plamo_weight_spec plamo_weight_schema[];
extern uint64_t plamo_model_patches[], plamo_io_policy, plamo_io_stats[4];
extern int plamo_model_open(void);
extern int plamo_model_patch(void *);
extern int plamo_prefill(struct model *, const uint32_t *, uint64_t, float *);
extern int plamo_forward(struct model *, uint64_t);
extern uint64_t plamo_argmax(const float *, uint64_t);

/* Reset logical request state without touching weights or clearing a huge arena. */
void plamo_reset_request(struct model *);
#endif
