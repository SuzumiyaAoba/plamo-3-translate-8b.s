#include "../runtime/prompt_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "prefix check failed at %d: %s\n", __LINE__, #x); exit(1); } } while (0)
int main(void) {
    enum { L = 2, N = 3, W = 5, CAP = 7 };
    float data[L][3][CAP * W], original[L][3][CAP * W];
    struct cache caches[L] = {0}; struct rotation rotations[L] = {0};
    struct mixer mixers[L] = {0}; struct layer layers[L] = {0}; struct layer *pointers[L];
    struct model model = {.count = L, .layers = pointers};
    uint32_t ids[] = {1, 19, 7, 5, 9, 11}, wrong[] = {1, 18, 7, 5, 9, 11};
    for (unsigned l = 0; l < L; ++l) {
        for (unsigned b = 0; b < 3; ++b)
            for (unsigned i = 0; i < CAP * W; ++i) {
                uint32_t bits = (l * 17 + b * 33 + i) * UINT32_C(0x1234567);
                memcpy(&data[l][b][i], &bits, 4);
            }
        caches[l] = (struct cache){.k = data[l][0], .v = data[l][1], .count = N, .capacity = CAP, .width = W};
        rotations[l] = (struct rotation){.cache = &caches[l], .out = data[l][2], .processed_count = N, .scale_bits = 123 + l};
        mixers[l] = (struct mixer){.cache = &caches[l], .rotation = &rotations[l]};
        layers[l].mixer = &mixers[l]; pointers[l] = &layers[l];
    }
    memcpy(original, data, sizeof data);
    struct plamo_prompt_cache *a = plamo_prompt_cache_capture(&model, ids, N); CHECK(a);
    CHECK(plamo_prompt_cache_tokens(a) == N && plamo_prompt_cache_bytes(a) >= L * 3 * N * W * 4);
    CHECK(!plamo_prompt_cache_capture(&model, ids, 0));
    CHECK(!plamo_prompt_cache_capture(&model, ids, PLAMO_PROMPT_PREFIX_LIMIT + 1));
    caches[1].count = N + 1; CHECK(!plamo_prompt_cache_capture(&model, ids, N)); caches[1].count = N;
    caches[1].width = UINT64_MAX; CHECK(!plamo_prompt_cache_capture(&model, ids, N)); caches[1].width = W;
    memset(data, 0, sizeof data);
    struct plamo_prompt_cache *b = plamo_prompt_cache_capture(&model, wrong, N); CHECK(b);
    for (unsigned l = 0; l < L; ++l) caches[l].count = rotations[l].processed_count = 0;
    CHECK(!plamo_prompt_cache_restore(a, &model, wrong, 6));
    CHECK(!plamo_prompt_cache_restore(a, &model, ids, N - 1));
    CHECK(!plamo_prompt_cache_restore(a, &model, ids, N));
    caches[1].capacity = N - 1;
    CHECK(!plamo_prompt_cache_restore(a, &model, ids, 6));
    CHECK(caches[0].count == 0 && data[0][0][0] == 0); // Failed last destination cannot modify the first.
    caches[1].capacity = CAP;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CHECK(plamo_prompt_cache_restore(a, &model, ids, 6) == N);
        for (unsigned l = 0; l < L; ++l) {
            CHECK(caches[l].count == N && rotations[l].processed_count == N && rotations[l].scale_bits == 123 + l);
            for (unsigned k = 0; k < 3; ++k) {
                CHECK(!memcmp(data[l][k], original[l][k], N * W * 4));
                for (unsigned i = N * W; i < CAP * W; ++i) CHECK(data[l][k][i] == 0);
            }
        }
        CHECK(plamo_prompt_cache_restore(b, &model, wrong, 6) == N);
        for (unsigned l = 0; l < L; ++l)
            for (unsigned k = 0; k < 3; ++k)
                for (unsigned i = 0; i < CAP * W; ++i) CHECK(data[l][k][i] == 0);
    }
    CHECK(!plamo_prompt_cache_restore(NULL, &model, ids, 6));
    CHECK(!plamo_prompt_cache_tokens(NULL) && !plamo_prompt_cache_bytes(NULL));
    plamo_prompt_cache_free(a); plamo_prompt_cache_free(b); plamo_prompt_cache_free(NULL);
    puts("template prefix snapshots: PASS (bitwise KV, exact IDs, immutable independent snapshots, atomic misses, bounded memory, repeated restoration)");
    return 0;
}
