#include "../runtime/engine.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

extern void plamo_bf16_matvec(const uint16_t *, const float *, float *, uint64_t, uint64_t);

static int copy_tests(void) {
    enum { SIZE = 1024 * 1024 + 117 };
    unsigned char *expected = malloc(SIZE);
    if (!expected) return 1;
    for (size_t i = 0; i < SIZE; ++i) expected[i] = (unsigned char)(i * 13 + i / 11);
    FILE *files[2] = {tmpfile(), tmpfile()};
    struct plamo_mapping source[2], out[2];
    int64_t fds[2];
    for (size_t i = 0; i < 2; ++i) {
        if (!files[i] || fwrite(expected, 1, SIZE, files[i]) != SIZE || fflush(files[i])) return 2;
        fds[i] = fileno(files[i]);
        void *base = mmap(NULL, SIZE, PROT_READ, MAP_PRIVATE, (int)fds[i], 0);
        if (base == MAP_FAILED) return 3;
        source[i] = (struct plamo_mapping){base, SIZE};
    }
    char error[256];
    if (!plamo_resident_copy(source, fds, 2, 0, out, error, sizeof error)) return 4;
    for (size_t i = 0; i < 2; ++i)
        if (out[i].base == source[i].base || out[i].bytes != SIZE || memcmp(out[i].base, expected, SIZE)) return 5;
    /* Same original BF16 bytes, including an offset, give identical kernel bits. */
    float input[7], a[8], b[8];
    for (size_t i = 0; i < 7; ++i) input[i] = (float)((int)i - 4) / 7;
    plamo_bf16_matvec((uint16_t *)((char *)source[0].base + 32), input, a, 8, 7);
    plamo_bf16_matvec((uint16_t *)((char *)out[0].base + 32), input, b, 8, 7);
    if (memcmp(a, b, sizeof a)) return 6;
    void *slots[] = {source[0].base, (char *)source[0].base + 37, (char *)source[1].base + SIZE - 1};
    struct plamo_weight_spec specs[] = {{"a", 0, NULL, &slots[0]}, {"b", 0, NULL, &slots[1]}, {"c", 0, NULL, &slots[2]}, {0}};
    if (!plamo_rebase_weights(specs, source, out, 2) || slots[0] != out[0].base ||
        slots[1] != (char *)out[0].base + 37 || slots[2] != (char *)out[1].base + SIZE - 1) return 13;
    if (!plamo_rebase_weights(specs, out, source, 2)) return 14;
    slots[2] = expected; /* An invalid last slot must leave earlier slots untouched. */
    if (plamo_rebase_weights(specs, source, out, 2) || slots[0] != source[0].base || slots[1] != (char *)source[0].base + 37) return 15;
    plamo_resident_release(out, 2);
    if (out[0].base || out[1].base) return 7;
    if (plamo_resident_copy(source, fds, 2, UINT64_MAX, out, error, sizeof error) || out[0].base || out[1].base) return 8;
    if (plamo_resident_copy(source, fds, 2, UINT64_C(1) << 62, out, error, sizeof error) || out[0].base || out[1].base) return 16;
    int64_t bad_fds[] = {fds[0], -1};
    if (plamo_resident_copy(source, bad_fds, 2, 0, out, error, sizeof error)) return 9;
    /* Failure after the first completed shard must roll back every new mapping. */
    if (ftruncate((int)fds[1], 0)) return 10;
    if (plamo_resident_copy(source, fds, 2, 0, out, error, sizeof error) || out[0].base || out[1].base) return 11;
    if (memcmp(source[0].base, expected, SIZE) || fcntl((int)fds[0], F_GETFD) < 0) return 12;
    for (size_t i = 0; i < 2; ++i) { munmap(source[i].base, SIZE); fclose(files[i]); }
    free(expected);
    return 0;
}

static int reset_tests(void) {
    float state[2] = {3, 7}, keys[2] = {17, 19}, values[2] = {23, 29};
    struct cache cache = {keys, values, 2, 2, 1, 1};
    struct rotation rotation = {.cache = &cache, .processed_count = 2, .scale_bits = 123};
    struct gqa gqa = {.tokens = 2};
    struct mixer mixer = {.cache = &cache, .rotation = &rotation, .gqa = &gqa};
    struct layer layer = {.state = keys, .mixer = &mixer};
    struct layer *layers[] = {&layer};
    struct model model = {.state = state, .layers = layers, .count = 1};
    for (unsigned i = 0; i < 2; ++i) {
        plamo_reset_request(&model);
        if (layer.state != state || cache.count || rotation.processed_count || rotation.scale_bits || gqa.tokens) return 1;
        if (keys[0] != 17 || values[1] != 29 || state[1] != 7 || cache.capacity != 2) return 2;
    }
    return 0;
}

int main(void) {
    int result = copy_tests();
    if (result) { fprintf(stderr, "resident copy test failed: %d\n", result); return result; }
    result = reset_tests();
    if (result) return 20 + result;
    puts("persistent runtime: PASS (exact BF16 copies, offsets, rollback, capacity guard, request reset)");
    return 0;
}
