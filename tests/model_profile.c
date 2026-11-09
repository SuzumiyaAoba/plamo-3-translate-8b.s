/* Exercise the real loader with sparse BF16 fixtures; no inference or weights. */
#include "../runtime/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern const char *plamo_weight_paths[];

int main(int argc, char **argv) {
    if (argc != 7) return 64;
    /* Start with the last file withheld, then repair it and retry in-process. */
    if (plamo_model_open()) return 1;
    if (plamo_weight_mapping_count) return 2;
    for (size_t i = 0; i < PLAMO_MAX_WEIGHT_FILES; ++i)
        if (plamo_weight_mappings[i].base || plamo_weight_mappings[i].bytes || plamo_weight_fds[i] != -1) return 3;
    const char *last = plamo_weight_paths[plamo_weight_file_count - 1];
    char held[512];
    snprintf(held, sizeof held, "%s.withheld", last);
    if (rename(held, last) || !plamo_model_open()) return 4;
    if (plamo_weight_mapping_count != plamo_weight_file_count) return 5;
    struct model *m = &plamo_model;
    if (m->hidden != strtoull(argv[1], NULL, 10) || m->count != strtoull(argv[2], NULL, 10) || m->vocab != 107520) return 6;
    int tied = atoi(argv[5]);
    if ((m->head == m->embedding) != tied) return 7;
    for (uint64_t i = 0; i < m->count; ++i) {
        struct layer *l = m->layers[i];
        struct qkv *q = l->mixer->qkv;
        if (!l->pre || !l->post || !l->mlp->gu || !l->mlp->down || !l->mixer->weight) return 8;
        if (q->hidden != m->hidden || q->qh != strtoull(argv[3], NULL, 10) || q->kh != strtoull(argv[4], NULL, 10) || q->dim != 128) return 9;
        if (!l->mixer->cache->k || !l->mixer->cache->v || l->mixer->cache->count) return 10;
    }
    uint64_t count = 0;
    for (struct plamo_weight_spec *s = plamo_weight_schema; s->name; ++s) {
        if (!*s->slot) return 11;
        ++count;
    }
    if (count != strtoull(argv[6], NULL, 10)) return 12;
    void *first = plamo_weight_mappings[0].base;
    if (!plamo_model_open() || plamo_weight_mappings[0].base != first) return 13;
    printf("%s: PASS (%llu files, %llu tensors, retry, idempotency, tied=%d)\n",
           plamo_model_name, plamo_weight_file_count, count, tied);
    return 0;
}
