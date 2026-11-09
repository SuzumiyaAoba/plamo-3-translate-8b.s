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
    struct plamo_engine_info info;
    struct plamo_weight_cache *weight_cache;
    struct plamo_parallel *parallel;
    struct plamo_metal *metal;
    struct trie trie;
    const char *vocab, *rows[107521];
    size_t vocab_size;
    int tokenizer_only, prepared;
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

static void model_files_error(char *error, size_t cap, const char *message) {
    /* Generated names always end in the selectable size, e.g. "-2b". */
    snprintf(error, cap, "%s for %s; run plamo-download --weights --model %s",
             message, plamo_model_name, strrchr(plamo_model_name, '-') + 1);
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
    struct plamo_mapping copies[PLAMO_MAX_WEIGHT_FILES], original[PLAMO_MAX_WEIGHT_FILES];
    size_t count = (size_t)plamo_weight_mapping_count;
    memcpy(original, plamo_weight_mappings, count * sizeof *original);
    if (!plamo_resident_copy(original, plamo_weight_fds, count, PLAMO_MEMORY_RESERVE, copies, error, cap)) return 0;
    if (!plamo_rebase_weights(plamo_weight_schema, original, copies, count)) {
        snprintf(error, cap, "a bound tensor is outside the original files");
        plamo_resident_release(copies, count);
        return 0;
    }
    if (!plamo_model_patch(plamo_model_patches)) {
        (void)plamo_rebase_weights(plamo_weight_schema, copies, original, count);
        (void)plamo_model_patch(plamo_model_patches);
        plamo_resident_release(copies, count);
        snprintf(error, cap, "cannot update model descriptors for preloaded weights");
        return 0;
    }
    memcpy(plamo_weight_mappings, copies, count * sizeof *copies);
    for (size_t i = 0; i < count; ++i) {
        munmap(original[i].base, (size_t)original[i].bytes);
        close((int)plamo_weight_fds[i]);
        plamo_weight_fds[i] = -1;
    }
    plamo_io_policy = 0;
    info->preloaded_weight_bytes = info->weight_bytes;
    info->memory_mode = "resident";
    return 1;
}

static uint64_t working_weight_bytes(const struct model *model) {
    uint64_t bytes = model->vocab * model->hidden * 2;
    for (uint64_t i = 0; i < model->count; ++i) {
        const struct layer *layer = model->layers[i];
        const struct qkv *q = layer->mixer->qkv;
        const struct mlp *m = layer->mlp;
        bytes += (q->qh + 2 * q->kh) * q->dim * q->hidden * 2;
        bytes += layer->mixer->hidden * q->qh * q->dim * 2;
        bytes += m->hidden * m->intermediate * 6;
    }
    return bytes;
}

struct plamo_engine *plamo_engine_open(enum plamo_memory_mode mode, int tokenizer_only, unsigned threads, enum plamo_backend backend,
                                     struct plamo_engine_info *info, char *error, size_t cap) {
    double start = now();
    memset(info, 0, sizeof *info);
    info->threads = 1;
    info->backend = "cpu";
    struct plamo_engine *e = calloc(1, sizeof *e);
    if (!e) { snprintf(error, cap, "cannot allocate tokenizer state"); return NULL; }
    int fd = open(plamo_vocab_path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size <= 0) {
        if (fd >= 0) close(fd);
        model_files_error(error, cap, "tokenizer is missing");
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
        if (!plamo_model_open()) { model_files_error(error, cap, "missing or invalid weight files"); goto fail; }
        for (size_t i = 0; i < plamo_weight_mapping_count; ++i) info->weight_bytes += plamo_weight_mappings[i].bytes;
        if (backend != PLAMO_BACKEND_CPU) {
            enum plamo_metal_packing packing = PLAMO_PACK_AUTO;
            const char *storage = getenv("PLAMO_WEIGHT_STORAGE");
            if (storage && strcmp(storage, "auto")) {
                if (!strcmp(storage, "raw")) packing = PLAMO_PACK_RAW;
                else if (!strcmp(storage, "lossless")) packing = PLAMO_PACK_LOSSLESS;
                else { snprintf(error, cap, "PLAMO_WEIGHT_STORAGE must be auto, raw or lossless"); goto fail; }
            }
            struct plamo_metal_config config = {
                .memory = (enum plamo_metal_memory)mode,
                .precision = PLAMO_METAL_FP32,
                .packing = packing, .working_weight_bytes = working_weight_bytes(&plamo_model),
                .maps = plamo_weight_mappings, .fds = plamo_weight_fds,
                .count = (size_t)plamo_weight_mapping_count, .schema = plamo_weight_schema,
                .reserve = PLAMO_MEMORY_RESERVE, .available = plamo_reclaimable_memory
            };
            e->metal = plamo_metal_open(&config, error, cap);
            if (!e->metal && backend != PLAMO_BACKEND_AUTO) goto fail;
            if (!e->metal) snprintf(info->gpu.note, sizeof info->gpu.note, "%s", error);
        }
        if (e->metal) {
            plamo_metal_get_info(e->metal, &info->gpu);
            info->cache = info->gpu.cache;
            info->backend = "metal";
            info->memory_mode = mode == PLAMO_MEMORY_AUTO ? "adaptive" : mode == PLAMO_MEMORY_RESIDENT ? "resident" : "stream";
            if (mode == PLAMO_MEMORY_RESIDENT) info->preloaded_weight_bytes = info->cache.cached_bytes;
        } else if (mode == PLAMO_MEMORY_RESIDENT) {
            if (!preload(info, error, cap)) goto fail;
        } else if (mode == PLAMO_MEMORY_AUTO) {
            e->weight_cache = plamo_weight_cache_open(plamo_weight_mappings, plamo_weight_fds,
                                                     (size_t)plamo_weight_mapping_count, PLAMO_MEMORY_RESERVE,
                                                     plamo_reclaimable_memory, error, cap);
            if (e->weight_cache) {
                plamo_weight_cache_get_info(e->weight_cache, &info->cache);
                plamo_weight_context = e->weight_cache;
                plamo_weight_provider = plamo_weight_cache_acquire;
                plamo_weight_chunk_bytes = info->cache.staging_bytes;
                info->memory_mode = "adaptive";
            } else {
                snprintf(info->memory_note, sizeof info->memory_note, "%s", error);
                /* Even an arena allocation failure keeps automatic reads small. */
                plamo_io_policy = 0;
                plamo_weight_chunk_bytes = 1024 * 1024;
            }
        }
    }
    if (!tokenizer_only && !e->metal) {
        e->parallel = plamo_parallel_open(threads);
        info->threads = plamo_parallel_threads(e->parallel);
    }
    info->load_seconds = now() - start;
    e->info = *info;
    return e;
fail:
    if (e->vocab) munmap((void *)e->vocab, e->vocab_size);
    free(e->trie.nodes);
    free(e->trie.roots);
    free(e);
    return NULL;
}

void plamo_engine_get_info(struct plamo_engine *e, struct plamo_engine_info *out) {
    *out = e->info;
    if (e->weight_cache) plamo_weight_cache_get_info(e->weight_cache, &out->cache);
    if (e->metal) { plamo_metal_get_info(e->metal, &out->gpu); out->cache = out->gpu.cache; }
}

void plamo_engine_maintain_memory(struct plamo_engine *e) {
    if (e->weight_cache) plamo_weight_cache_rebalance(e->weight_cache);
    if (e->metal) plamo_metal_maintain(e->metal);
}

static int prepare_matrix(struct plamo_engine *e, const uint16_t *weights, uint64_t rows, uint64_t columns) {
    if (!rows || !columns) return 1;
    uint64_t row_bytes = columns * 2, window = e->info.cache.staging_bytes / row_bytes;
    if (!window) return 1; // A wider-than-window row keeps its ordinary mapped path.
    if (rows <= window) window = rows;
    else if (window >= 8) window = window / 8 * 8;
    // Match the projections' prefill/decode row boundaries exactly. Loading
    // only cacheable tiles avoids reading the unretainable part of large models.
    for (uint64_t first = 0; first < rows; first += window) {
        uint64_t count = rows - first;
        if (count > window) count = window;
        const void *source = (const char *)weights + first * row_bytes;
        int ok = e->metal ? plamo_metal_preload(e->metal, source, count * row_bytes) :
                           plamo_weight_cache_preload(e->weight_cache, source, count * row_bytes);
        if (!ok) return 0;
    }
    return 1;
}

static int prepare_matrices(struct plamo_engine *e) {
    // Follow execution order, rather than safetensors order. Embedding rows
    // are selected by the future prompt; loading the whole embedding would
    // displace frequently used projections. A tied output head is still loaded.
    for (uint64_t i = 0; i < plamo_model.count; ++i) {
        const struct layer *layer = plamo_model.layers[i];
        const struct mixer *mixer = layer->mixer;
        const struct qkv *qkv = mixer->qkv;
        const struct mlp *mlp = layer->mlp;
        uint64_t attention = qkv->qh * qkv->dim;
        if (!prepare_matrix(e, qkv->weight, (qkv->qh + 2 * qkv->kh) * qkv->dim, qkv->hidden) ||
            !prepare_matrix(e, mixer->weight, mixer->hidden, attention) ||
            !prepare_matrix(e, mlp->gu, mlp->intermediate * 2, mlp->hidden) ||
            !prepare_matrix(e, mlp->down, mlp->hidden, mlp->intermediate)) return 0;
    }
    return prepare_matrix(e, plamo_model.head, plamo_model.vocab, plamo_model.hidden);
}

int plamo_engine_prepare(struct plamo_engine *e, char *error, size_t cap) {
    if (e->prepared) return 1;
    int adaptive = !strcmp(e->info.memory_mode, "adaptive");
    if (!adaptive && !(e->metal && !strcmp(e->info.memory_mode, "resident"))) {
        e->prepared = 1; return 1;
    }
    double start = now();
    if (adaptive && !prepare_matrices(e)) goto fail;
    if (e->metal && !plamo_metal_prepare_model(e->metal, &plamo_model)) goto fail;
    if (e->metal && !plamo_metal_warm_cache(e->metal)) goto fail;
    struct plamo_engine_info info;
    plamo_engine_get_info(e, &info);
    e->info.preloaded_weight_bytes = info.cache.cached_bytes;
    e->info.preload_seconds = now() - start;
    e->info.load_seconds += e->info.preload_seconds;
    e->prepared = 1;
    return 1;
fail:
    snprintf(error, cap, "cannot prepare original BF16 weights before API readiness");
    return 0;
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
    plamo_engine_maintain_memory(e);
    if (cb && cb->prefill) cb->prefill(count, cb->context);
    uint64_t before_bytes = plamo_io_stats[0], before_reads = plamo_io_stats[1];
    struct plamo_weight_cache_info before_cache = {0}, after_cache;
    struct plamo_engine_info sample;
    plamo_engine_get_info(e, &sample);
    before_cache = sample.cache;
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
    uint64_t decode_bytes = plamo_io_stats[0], decode_reads = plamo_io_stats[1];
    struct plamo_metal_info gpu_before = {0}, gpu_after;
    if (e->metal) plamo_metal_get_info(e->metal, &gpu_before);
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
    result->decode_staged_bytes = plamo_io_stats[0] - decode_bytes;
    result->decode_read_calls = plamo_io_stats[1] - decode_reads;
    if (e->metal) {
        plamo_metal_get_info(e->metal, &gpu_after);
        result->decode_gpu_seconds = gpu_after.gpu_seconds - gpu_before.gpu_seconds;
        result->decode_command_buffers = gpu_after.command_buffers - gpu_before.command_buffers;
        result->decode_gpu_dispatches = gpu_after.dispatches - gpu_before.dispatches;
    }
    result->total_seconds = now() - start;
    result->staged_bytes = plamo_io_stats[0] - before_bytes;
    result->read_calls = plamo_io_stats[1] - before_reads;
    if (e->weight_cache || e->metal) {
        plamo_engine_get_info(e, &sample);
        after_cache = sample.cache;
        result->cache_hits = after_cache.hits - before_cache.hits;
        result->cache_hit_bytes = after_cache.hit_bytes - before_cache.hit_bytes;
        result->cache_misses = after_cache.misses - before_cache.misses;
    }
    result->logits_hash = hash(plamo_model.logits, plamo_model.vocab * 4);
    result->state_hash = hash(plamo_model.state, plamo_model.hidden * 4);
    return 1;
}
