#include "engine.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct node { uint32_t cp, child, sibling; int32_t id, score; uint32_t terminal; };
struct trie { struct node *nodes; uint64_t capacity, used; uint32_t *roots; };
struct preprocess { const uint32_t *in; uint64_t count; uint32_t *temp; uint64_t temp_capacity; uint32_t *out; uint64_t capacity; uint8_t *flags; };
struct encoder { struct trie *trie; const uint32_t *cp; uint64_t count; uint32_t *bytes; int64_t *costs; void *paths, *candidates; uint64_t capacity; uint32_t *out; uint64_t out_capacity; };
struct scalar { const char *next; uint64_t cp; };
extern uint64_t plamo_vocab_load(const char *, const char *, struct trie *, uint32_t *, uint64_t, uint32_t *);
extern int plamo_utf8_next(const char *, const char *, struct scalar *);
extern uint64_t plamo_preprocess(const struct preprocess *);
extern uint64_t plamo_encode_marked(const struct encoder *);
extern uint64_t plamo_translation_prompt(const char *, const char *, const char *, char *, uint64_t);
extern uint64_t plamo_decode_token(const char *, const char *, uint64_t, char *, uint64_t, uint32_t *, uint64_t);

struct plamo_engine {
    struct trie trie;
    const char *vocab, *rows[107521];
    size_t vocab_size;
    int tokenizer_only;
    uint32_t scratch[4096], bytes[256], input[4096], temp[16384], marked[49152], ids[16384];
    uint8_t flags[16385];
    int64_t costs[49153];
    unsigned char paths[1179648], candidates[65552];
    char prompt[16384], decoded[16384];
};

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t hash(const void *data, size_t count) {
    const unsigned char *p = data;
    uint64_t value = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; ++i) { value ^= p[i]; value *= UINT64_C(1099511628211); }
    return value;
}

void plamo_reset_request(struct model *model) {
    for (uint64_t i = 0; i < model->count; ++i) {
        struct layer *layer = model->layers[i];
        layer->state = model->state;
        layer->mixer->cache->count = 0;
        layer->mixer->rotation->processed_count = 0;
        layer->mixer->rotation->scale_bits = 0;
        layer->mixer->gqa->tokens = 0;
    }
}

static int preload(struct plamo_engine_info *info, char *error, size_t cap) {
    struct plamo_mapping copies[4], original[4];
    memcpy(original, plamo_weight_mappings, sizeof original);
    if (!plamo_resident_copy(original, plamo_weight_fds, 4, PLAMO_MEMORY_RESERVE, copies, error, cap)) return 0;
    if (!plamo_rebase_weights(plamo_weight_schema, original, copies, 4)) {
        snprintf(error, cap, "a bound tensor is outside the original files");
        plamo_resident_release(copies, 4);
        return 0;
    }
    if (!plamo_model_patch(plamo_model_patches)) {
        (void)plamo_rebase_weights(plamo_weight_schema, copies, original, 4);
        (void)plamo_model_patch(plamo_model_patches);
        plamo_resident_release(copies, 4);
        snprintf(error, cap, "cannot update model descriptors for preloaded weights");
        return 0;
    }
    memcpy(plamo_weight_mappings, copies, sizeof copies);
    for (size_t i = 0; i < 4; ++i) {
        munmap(original[i].base, (size_t)original[i].bytes);
        close((int)plamo_weight_fds[i]);
        plamo_weight_fds[i] = -1;
    }
    plamo_io_policy = 0;
    info->preloaded_weight_bytes = info->weight_bytes;
    info->memory_mode = "resident";
    return 1;
}

struct plamo_engine *plamo_engine_open(enum plamo_memory_mode mode, int tokenizer_only,
                                     struct plamo_engine_info *info, char *error, size_t cap) {
    double start = now();
    memset(info, 0, sizeof *info);
    struct plamo_engine *e = calloc(1, sizeof *e);
    if (!e) { snprintf(error, cap, "cannot allocate tokenizer state"); return NULL; }
    int fd = open("models/plamo/tokenizer.jsonl", O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size <= 0) {
        if (fd >= 0) close(fd);
        snprintf(error, cap, "tokenizer is missing; run plamo-download first");
        goto fail;
    }
    e->vocab_size = (size_t)st.st_size;
    void *vocab = mmap(NULL, e->vocab_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (vocab == MAP_FAILED) { snprintf(error, cap, "cannot map tokenizer"); goto fail; }
    e->vocab = vocab;
    e->trie = (struct trie){calloc(1000000, sizeof(struct node)), 1000000, 1, calloc(0x110000, sizeof(uint32_t))};
    if (!e->trie.nodes || !e->trie.roots) { snprintf(error, cap, "cannot allocate vocabulary trie"); goto fail; }
    uint64_t rows = plamo_vocab_load(e->vocab, e->vocab + e->vocab_size, &e->trie, e->scratch, 4096, e->bytes);
    if (rows != 107520) { snprintf(error, cap, "vocabulary must contain 107520 valid rows"); goto fail; }
    /* Index original rows once; decoding no longer scans the full JSONL per token. */
    size_t row = 0;
    e->rows[0] = e->vocab;
    for (size_t i = 0; i < e->vocab_size; ++i)
        if (e->vocab[i] == '\n' && row < 107520) e->rows[++row] = e->vocab + i + 1;
    if (row == 107519 && e->vocab[e->vocab_size - 1] != '\n') e->rows[++row] = e->vocab + e->vocab_size;
    if (row != 107520) { snprintf(error, cap, "invalid vocabulary row boundaries"); goto fail; }
    e->tokenizer_only = tokenizer_only;
    info->memory_mode = tokenizer_only ? "tokenizer" : "stream";
    info->physical_bytes = plamo_physical_memory();
    info->reclaimable_bytes = plamo_reclaimable_memory();
    if (!tokenizer_only) {
        if (!plamo_model_open()) { snprintf(error, cap, "missing or invalid weight shards; run plamo-download --weights"); goto fail; }
        for (size_t i = 0; i < 4; ++i) info->weight_bytes += plamo_weight_mappings[i].bytes;
        if (mode != PLAMO_MEMORY_STREAM && !preload(info, error, cap)) {
            if (mode == PLAMO_MEMORY_RESIDENT) goto fail;
            snprintf(info->memory_note, sizeof info->memory_note, "%s", error);
            plamo_io_policy = 2;
        }
    }
    info->load_seconds = now() - start;
    return e;
fail:
    if (e->vocab) munmap((void *)e->vocab, e->vocab_size);
    free(e->trie.nodes);
    free(e->trie.roots);
    free(e);
    return NULL;
}

uint64_t plamo_engine_tokenize(struct plamo_engine *e, const char *text, unsigned direction,
                              const uint32_t **ids, char *error, size_t cap) {
    if (direction > 2) { snprintf(error, cap, "unsupported translation direction"); return UINT64_MAX; }
    const char *input = text;
    if (direction) {
        const char *source = direction == 1 ? "en" : "ja", *target = direction == 1 ? "ja" : "en";
        if (plamo_translation_prompt(text, source, target, e->prompt, sizeof e->prompt) == UINT64_MAX) {
            snprintf(error, cap, "translation prompt exceeds the byte capacity"); return UINT64_MAX;
        }
        input = e->prompt;
    }
    const char *end = input + strlen(input);
    uint64_t count = 0;
    while (input < end) {
        struct scalar scalar;
        if (count == 4096 || !plamo_utf8_next(input, end, &scalar)) {
            snprintf(error, cap, "prompt exceeds 4096 Unicode scalars or is not valid UTF-8"); return UINT64_MAX;
        }
        e->input[count++] = (uint32_t)scalar.cp;
        input = scalar.next;
    }
    struct preprocess p = {e->input, count, e->temp, 16384, e->marked, 49152, e->flags};
    uint64_t marked = plamo_preprocess(&p);
    if (marked == UINT64_MAX) { snprintf(error, cap, "tokenizer preprocessing failed"); return UINT64_MAX; }
    struct encoder encoder = {&e->trie, e->marked, marked, e->bytes, e->costs, e->paths, e->candidates, 4097, e->ids, 16384};
    uint64_t tokens = plamo_encode_marked(&encoder);
    if (tokens == UINT64_MAX) { snprintf(error, cap, "tokenizer encoding failed"); return UINT64_MAX; }
    *ids = e->ids;
    return tokens;
}

int plamo_engine_translate(struct plamo_engine *e, const char *text, unsigned direction,
                          unsigned max_tokens, const struct plamo_callbacks *cb,
                          struct plamo_result *result, char *error, size_t cap) {
    memset(result, 0, sizeof *result);
    if (e->tokenizer_only || !max_tokens || max_tokens > 4096) {
        snprintf(error, cap, "generation requires a loaded model and max_tokens in 1..4096"); return 0;
    }
    double start = now();
    const uint32_t *ids;
    uint64_t count = plamo_engine_tokenize(e, text, direction, &ids, error, cap);
    if (count == UINT64_MAX) return 0;
    if (!count) { snprintf(error, cap, "generation requires a nonempty prompt"); return 0; }
    if (count > 4096) { snprintf(error, cap, "API generation accepts at most 4096 prompt tokens"); return 0; }
    result->prompt_tokens = count;
    result->tokenize_seconds = now() - start;
    plamo_reset_request(&plamo_model);
    if (cb && cb->prefill) cb->prefill(count, cb->context);
    uint64_t before_bytes = plamo_io_stats[0], before_reads = plamo_io_stats[1];
    size_t bytes = (size_t)count * (size_t)plamo_model.hidden * sizeof(float);
    float *states = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (states == MAP_FAILED) { snprintf(error, cap, "cannot allocate prompt states"); return 0; }
    double prefill_start = now();
    int ok = plamo_prefill(&plamo_model, ids, count, states);
    result->prefill_seconds = now() - prefill_start;
    munmap(states, bytes);
    if (!ok) { snprintf(error, cap, "prefill failed"); return 0; }
    result->prefill_logits_hash = hash(plamo_model.logits, plamo_model.vocab * 4);
    result->prefill_state_hash = hash(plamo_model.state, plamo_model.hidden * 4);
    result->finish_reason = "length";
    double decode_start = now();
    for (unsigned i = 0; i < max_tokens; ++i) {
        uint64_t id = plamo_argmax(plamo_model.logits, plamo_model.vocab);
        if (id == 1 || id == 16) { result->finish_reason = "stop"; break; }
        if (id >= plamo_model.vocab) { snprintf(error, cap, "invalid generated token"); return 0; }
        uint64_t n = plamo_decode_token(e->rows[id], e->rows[id + 1], 0, e->decoded, sizeof e->decoded, e->scratch, 4096);
        if (n == UINT64_MAX) { snprintf(error, cap, "token decoding failed"); return 0; }
        double token_ready = now() - start;
        if (cb && cb->token && !cb->token((uint32_t)id, e->decoded, (size_t)n, cb->context)) {
            result->finish_reason = "cancelled"; break;
        }
        if (!result->completion_tokens) result->first_token_seconds = token_ready;
        ++result->completion_tokens;
        if (i + 1 == max_tokens) break;
        if (!plamo_forward(&plamo_model, id)) { snprintf(error, cap, "decode failed"); return 0; }
        ++result->decode_steps;
    }
    result->decode_seconds = now() - decode_start;
    result->total_seconds = now() - start;
    result->staged_bytes = plamo_io_stats[0] - before_bytes;
    result->read_calls = plamo_io_stats[1] - before_reads;
    result->logits_hash = hash(plamo_model.logits, plamo_model.vocab * 4);
    result->state_hash = hash(plamo_model.state, plamo_model.hidden * 4);
    return 1;
}
