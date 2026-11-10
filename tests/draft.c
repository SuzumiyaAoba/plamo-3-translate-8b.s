#include "../runtime/draft.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

static void append(struct cache *c, unsigned id) {
    if (c->count == c->capacity) {
        memmove(c->k, c->k + c->width, (size_t)((c->count - 1) * c->width * 4));
        memmove(c->v, c->v + c->width, (size_t)((c->count - 1) * c->width * 4));
        --c->count;
    }
    for (unsigned j = 0; j < c->width; ++j) {
        c->k[c->count * c->width + j] = (float)(id * 100 + j);
        c->v[c->count * c->width + j] = -(float)(id * 100 + j);
    }
    ++c->count;
}

static int fake_peer(void) {
    uint32_t ready[] = {PLAMO_DRAFT_MAGIC, 1, 107520, PLAMO_DRAFT_MAX};
    if (getenv("_PLAMO_TEST_BAD_READY")) ready[0] = 0;
    assert(fwrite(ready, sizeof ready, 1, stdout) == 1 && !fflush(stdout));
    uint32_t header[4];
    unsigned proposals = 0;
    while (fread(header, sizeof header, 1, stdin) == 1) {
        assert(header[3] <= 16384);
        char text[16384];
        assert(fread(text, 1, header[3], stdin) == header[3]);
        struct { uint32_t ok, count, ids[PLAMO_DRAFT_MAX]; } reply = {.ok = 1};
        if (header[0] == 2) {
            if (++proposals == 2 && getenv("_PLAMO_TEST_FAIL_SECOND")) return 0;
            if (header[1] == 107519) return 0; // Deliberate pipe closure.
            assert(header[2] <= PLAMO_DRAFT_MAX);
            reply.count = header[2];
            for (unsigned i = 0; i < reply.count; ++i) reply.ids[i] = header[1] + i;
            if (header[1] == 42) ++reply.ids[0]; // Invalid known first token.
        }
        assert(fwrite(&reply, sizeof reply, 1, stdout) == 1 && !fflush(stdout));
    }
    return 0;
}

static void client(const char *executable) {
    char path[PATH_MAX];
    assert(realpath(executable, path));
    assert(!setenv("PLAMO_DRAFT_WORKER", path, 1) && !setenv("PLAMO_DRAFT_DATA_DIR", "/tmp", 1));
    assert(!setenv("PLAMO_SPECULATIVE", "0", 1));
    assert(!plamo_draft_open());
    assert(!setenv("PLAMO_SPECULATIVE", "1", 1));
    struct plamo_draft *d = plamo_draft_open();
    assert(d && plamo_draft_reset(d, "test 日本語", 1));
    uint32_t ids[PLAMO_DRAFT_MAX];
    assert(!plamo_draft_propose(d, 10, 0, ids));
    assert(plamo_draft_propose(d, 10, 4, ids) == 4 && ids[0] == 10 && ids[3] == 13);
    assert(plamo_draft_accept(d, 2));
    assert(!plamo_draft_propose(d, 42, 4, ids));
    plamo_draft_close(d);
    d = plamo_draft_open(); assert(d);
    assert(!plamo_draft_propose(d, 107519, 4, ids));
    plamo_draft_close(d);
    assert(!setenv("_PLAMO_TEST_BAD_READY", "1", 1));
    assert(!plamo_draft_open());
    plamo_draft_close(NULL);
    puts("Draft client: PASS (process isolation, UTF-8 frames, proposals/acceptance, opt-out, invalid reply, EOF, bad handshake)");
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--draft-peer")) return fake_peer();
    // Every truncation boundary, including multiple evictions and a prefix
    // already at capacity. Compare with independently appended accepted rows.
    for (unsigned initial = 0; initial <= 12; ++initial)
        for (unsigned processed = 0; processed <= 8; ++processed)
            for (unsigned accepted = 0; accepted <= processed; ++accepted) {
                float k[36] = {0}, v[36] = {0}, ek[36] = {0}, ev[36] = {0};
                struct cache c = {k, v, 12, 0, 3, 1}, expected = {ek, ev, 12, 0, 3, 1};
                struct rotation rotation = {.cache = &c, .processed_count = 99, .scale_bits = 123};
                struct gqa gqa = {0};
                struct mixer mixer = {.cache = &c, .rotation = &rotation, .gqa = &gqa};
                struct layer layer = {.mixer = &mixer}, *layers[] = {&layer};
                struct model model = {.layers = layers, .count = 1};
                for (unsigned i = 0; i < initial; ++i) { append(&c, i); append(&expected, i); }
                struct plamo_draft_checkpoint *s = plamo_draft_checkpoint(&model, 8);
                assert(s);
                for (unsigned i = 0; i < processed; ++i) append(&c, initial + i);
                for (unsigned i = 0; i < accepted; ++i) append(&expected, initial + i);
                assert(!plamo_draft_restore(s, processed, processed + 1));
                assert(plamo_draft_restore(s, processed, accepted));
                assert(c.count == expected.count);
                assert(!memcmp(k, ek, (size_t)(c.count * c.width * 4)));
                assert(!memcmp(v, ev, (size_t)(c.count * c.width * 4)));
                assert(!rotation.processed_count && !rotation.scale_bits && gqa.tokens == c.count);
                plamo_draft_checkpoint_free(s);
            }
    puts("Draft rollback: PASS (all prefixes, partially/full sliding caches, repeated evictions, rejected arguments)");
    client(argv[0]);
    return 0;
}
