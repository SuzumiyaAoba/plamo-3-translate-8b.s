/* A single persistent model process. Binary requests, JSONL events on stdout.
   Request: four little-endian u32 values {op, direction, max_tokens, text_bytes},
   followed by exact UTF-8 bytes. op=1 translate, 2 tokenize, 3 cancel (no body).
   The HTTP adapter owns this private pipe; it never starts a process per request. */
#include "engine.h"
#include "draft.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/qos.h>

static int channel_failed;

static int read_exact(void *data, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = read(STDIN_FILENO, (char *)data + done, bytes - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return !done && !n ? 0 : -1;
        done += (size_t)n;
    }
    return 1;
}

static void json_string(const char *text) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}

static void report_error(const char *code, const char *message) {
    printf("{\"event\":\"error\",\"code\":"); json_string(code);
    printf(",\"message\":"); json_string(message);
    printf("}\n"); fflush(stdout);
}

static void report_memory(struct plamo_engine *engine) {
    struct plamo_engine_info info;
    plamo_engine_get_info(engine, &info);
    printf("{\"mode\":\"%s\",\"weight_bytes\":%" PRIu64 ",\"preloaded_weight_bytes\":%" PRIu64 ","
           "\"estimated_reclaimable_bytes_at_start\":%" PRIu64 ",\"physical_bytes\":%" PRIu64 ",\"reserve_bytes\":%" PRIu64 ","
           "\"budget_bytes\":%" PRIu64 ",\"max_budget_bytes\":%" PRIu64 ",\"cached_weight_bytes\":%" PRIu64 ",\"cached_storage_bytes\":%" PRIu64 ",\"staging_bytes\":%" PRIu64 ","
           "\"cache_hits\":%" PRIu64 ",\"cache_hit_bytes\":%" PRIu64 ",\"cache_misses\":%" PRIu64 ",\"resizes\":%" PRIu64 ",\"note\":",
           info.memory_mode, info.weight_bytes, info.preloaded_weight_bytes,
           info.reclaimable_bytes, info.physical_bytes, PLAMO_MEMORY_RESERVE,
           info.cache.budget_bytes, info.cache.max_budget_bytes, info.cache.cached_bytes, info.cache.storage_bytes, info.cache.staging_bytes,
           info.cache.hits, info.cache.hit_bytes, info.cache.misses, info.cache.resizes);
    json_string(info.memory_note);
    putchar('}');
}

static void prefill_event(uint64_t count, void *context) {
    printf("{\"event\":\"prefill\",\"prompt_tokens\":%" PRIu64 ",\"memory\":", count);
    report_memory(context);
    puts("}");
    fflush(stdout);
}

static void report_compute(struct plamo_engine *engine) {
    struct plamo_engine_info info;
    plamo_engine_get_info(engine, &info);
    printf("{\"backend\":"); json_string(info.backend);
    printf(",\"device\":"); json_string(info.gpu.device);
    printf(",\"precision\":"); json_string(info.gpu.precision[0] ? info.gpu.precision : "fp32");
    printf(",\"weight_storage\":"); json_string(info.gpu.weight_storage[0] ? info.gpu.weight_storage : "bf16");
    printf(",\"packed_weight_bytes\":%" PRIu64, info.gpu.packed_weight_bytes);
    printf(",\"disk_cache_hits\":%" PRIu64 ",\"disk_cache_weight_bytes\":%" PRIu64 ",\"disk_cache_storage_bytes\":%" PRIu64,
           info.gpu.disk_cache_hits, info.gpu.disk_cache_weight_bytes, info.gpu.disk_cache_storage_bytes);
    printf(",\"disk_cache_mapped_bytes\":%" PRIu64, info.gpu.disk_cache_mapped_bytes);
    printf(",\"disk_cache_mapping_bytes\":%" PRIu64, info.gpu.disk_cache_mapping_bytes);
    printf(",\"residency_sets\":%s", info.gpu.residency_sets ? "true" : "false");
    printf(",\"verification_batches\":%" PRIu64 ",\"verified_tokens\":%" PRIu64 ",\"committed_tokens\":%" PRIu64,
           info.gpu.verification_batches, info.gpu.verified_tokens, info.gpu.committed_tokens);
    printf(",\"gpu_dispatches\":%" PRIu64 ",\"command_buffers\":%" PRIu64 ",\"fused_mlps\":%" PRIu64 ",\"fused_decodes\":%" PRIu64 ",\"fused_prefills\":%" PRIu64 ",\"cpu_fallbacks\":%" PRIu64 ",\"errors\":%" PRIu64 ",\"gpu_seconds\":%.9f,"
           "\"allocated_bytes\":%" PRIu64 ",\"recommended_working_set_bytes\":%" PRIu64 ",\"note\":",
           info.gpu.dispatches, info.gpu.command_buffers, info.gpu.fused_mlps, info.gpu.fused_decodes, info.gpu.fused_prefills, info.gpu.cpu_fallbacks, info.gpu.errors, info.gpu.gpu_seconds,
           info.gpu.allocated_bytes, info.gpu.recommended_working_set_bytes);
    json_string(info.gpu.note); putchar('}');
}

static int token_event(uint32_t id, const char *bytes, size_t count, void *context) {
    (void)context;
    struct pollfd fd = {STDIN_FILENO, POLLIN, 0};
    if (poll(&fd, 1, 0) > 0) {
        if (fd.revents & POLLHUP) return 0;
        if (fd.revents & POLLIN) {
            uint32_t header[4];
            if (read_exact(header, sizeof header) != 1 || header[0] != 3 || header[1] || header[2] || header[3]) channel_failed = 1;
            return 0;
        }
    }
    printf("{\"event\":\"token\",\"token_id\":%u,\"hex\":\"", id);
    for (size_t i = 0; i < count; ++i) printf("%02x", (unsigned char)bytes[i]);
    printf("\"}\n");
    return fflush(stdout) == 0;
}

static void report_result(const struct plamo_result *r, struct plamo_engine *engine) {
    struct rusage usage;
    struct plamo_engine_info info;
    plamo_engine_get_info(engine, &info);
    getrusage(RUSAGE_SELF, &usage);
    printf("{\"event\":\"done\",\"finish_reason\":\"%s\","
           "\"usage\":{\"prompt_tokens\":%" PRIu64 ",\"completion_tokens\":%" PRIu64 ",\"total_tokens\":%" PRIu64 ",\"cached_prompt_tokens\":%" PRIu64 "},"
           "\"timings\":{\"tokenize_seconds\":%.9f,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,"
           "\"first_token_seconds\":%.9f,\"total_seconds\":%.9f,\"decode_steps\":%" PRIu64 ","
           "\"prefill_gpu_seconds\":%.9f,\"prefill_command_buffers\":%" PRIu64 ",\"prefill_gpu_dispatches\":%" PRIu64 ","
           "\"decode_gpu_seconds\":%.9f,\"decode_command_buffers\":%" PRIu64 ",\"decode_gpu_dispatches\":%" PRIu64 "},"
           "\"weight_io\":{\"staged_bytes\":%" PRIu64 ",\"read_calls\":%" PRIu64 ","
           "\"cache_hits\":%" PRIu64 ",\"cache_hit_bytes\":%" PRIu64 ",\"cache_misses\":%" PRIu64 ","
           "\"decode_staged_bytes\":%" PRIu64 ",\"decode_read_calls\":%" PRIu64 ",\"decode_mapped_bytes\":%" PRIu64 "},"
           "\"process_peak_rss_bytes\":%ld,"
           "\"verification\":{\"prefill_logits_hash\":\"%016" PRIx64 "\",\"prefill_state_hash\":\"%016" PRIx64 "\","
           "\"logits_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\"},\"memory\":",
           r->finish_reason, r->prompt_tokens, r->completion_tokens, r->prompt_tokens + r->completion_tokens, r->cached_prompt_tokens,
           r->tokenize_seconds, r->prefill_seconds, r->decode_seconds, r->first_token_seconds, r->total_seconds,
           r->decode_steps, r->prefill_gpu_seconds, r->prefill_command_buffers, r->prefill_gpu_dispatches,
           r->decode_gpu_seconds, r->decode_command_buffers, r->decode_gpu_dispatches,
           r->staged_bytes, r->read_calls, r->cache_hits, r->cache_hit_bytes, r->cache_misses,
           r->decode_staged_bytes, r->decode_read_calls, r->decode_mapped_bytes, usage.ru_maxrss,
           r->prefill_logits_hash, r->prefill_state_hash, r->logits_hash, r->state_hash);
    report_memory(engine);
    printf(",\"speculative\":{\"enabled\":%s,\"block_tokens\":%u,\"batches\":%" PRIu64 ",\"consumed_tokens\":%" PRIu64
           ",\"proposed_tokens\":%" PRIu64 ",\"accepted_tokens\":%" PRIu64 ",\"draft_seconds\":%.9f}",
           info.speculative_tokens ? "true" : "false", info.speculative_tokens,
           r->speculative_batches, r->speculative_tokens, r->proposed_tokens, r->accepted_tokens, r->draft_seconds);
    printf(",\"compute\":"); report_compute(engine);
    puts("}");
    fflush(stdout);
}

static void usage(void) {
    puts("usage: plamo-worker [--backend auto|cpu|metal|metal-fp32] [--memory auto|resident|stream] [--threads auto|1..64] [--tokenizer-only]\n"
         "Private persistent worker for plamo-server. auto prefers an available Metal GPU.\n"
         "resident: preload BF16 weights; requires weights + 2 GiB spare.\n"
         "auto: preload a bounded BF16 cache before readiness; stream the remaining row tiles.\n"
         "threads: CPU compute threads; auto uses performance cores; 1 disables parallelism.\n"
         "OS paging is possible; preloading is not mlock or a hard residency guarantee.");
}

/* This process owns only an approximate proposer. Its private binary channel
   never emits public tokens; the parent verifies every proposal with 8B. */
static int draft_peer(struct plamo_engine *engine) {
    const char *setting = getenv("PLAMO_DRAFT_MIN_MARGIN");
    char *end;
    float minimum_margin = setting ? strtof(setting, &end) : 0.0f;
    if (setting && (!*setting || *end || !isfinite(minimum_margin) || minimum_margin < 0)) return 64;
    setting = getenv("PLAMO_DRAFT_MIN_TOKENS");
    unsigned long minimum_tokens = setting ? strtoul(setting, &end, 10) : 2;
    if (setting && (!*setting || *end || minimum_tokens < 1 || minimum_tokens > PLAMO_DRAFT_MAX)) return 64;
    setting = getenv("PLAMO_DRAFT_TRACE");
    int trace = setting && !strcmp(setting, "1");
    uint32_t ready[] = {PLAMO_DRAFT_MAGIC, 1, 107520, PLAMO_DRAFT_MAX};
    if (fwrite(ready, sizeof ready, 1, stdout) != 1 || fflush(stdout)) return 1;
    struct plamo_draft_checkpoint *saved = NULL;
    unsigned processed = 0, proposed = 0;
    uint32_t ids[PLAMO_DRAFT_MAX];
    for (;;) {
        uint32_t frame[4];
        int read = read_exact(frame, sizeof frame);
        if (!read) break;
        if (read != 1) return 1;
        struct { uint32_t ok, count, ids[PLAMO_DRAFT_MAX]; } reply = {0};
        char error[512];
        if (frame[0] == 1 && frame[1] <= 2 && !frame[2] && frame[3] <= 16384) {
            char text[16385];
            if (read_exact(text, frame[3]) != 1) return 1;
            text[frame[3]] = 0;
            plamo_draft_checkpoint_free(saved); saved = NULL;
            struct plamo_result result;
            reply.ok = !memchr(text, 0, frame[3]) &&
                plamo_engine_translate(engine, text, frame[1], 1, NULL, &result, error, sizeof error);
            if (trace) fprintf(stderr, "draft reset direction=%u\n", frame[1]);
        } else if (frame[0] == 2 && frame[1] < plamo_model.vocab && frame[2] && frame[2] <= PLAMO_DRAFT_MAX && !frame[3] && !saved) {
            saved = plamo_draft_checkpoint(&plamo_model, frame[2]);
            if (saved) {
                ids[0] = frame[1]; processed = 0; proposed = 1;
                float margins[PLAMO_DRAFT_MAX] = {0};
                uint32_t alternatives[PLAMO_DRAFT_MAX] = {0};
                float confidence = INFINITY;
                int failed = 0;
                while (proposed < frame[2] && ids[proposed - 1] != 1 && ids[proposed - 1] != 16) {
                    if (minimum_margin && proposed >= minimum_tokens && confidence < minimum_margin) break;
                    if (!plamo_forward(&plamo_model, ids[proposed - 1])) { failed = 1; break; }
                    ++processed;
                    uint32_t id = (uint32_t)plamo_argmax(plamo_model.logits, plamo_model.vocab);
                    if (id >= plamo_model.vocab) { failed = 1; break; }
                    if (minimum_margin || trace) {
                        float maximum = plamo_model.logits[id];
                        plamo_model.logits[id] = -INFINITY;
                        uint32_t alternative = (uint32_t)plamo_argmax(plamo_model.logits, plamo_model.vocab);
                        plamo_model.logits[id] = maximum;
                        alternatives[proposed] = alternative;
                        margins[proposed] = maximum - plamo_model.logits[alternative];
                        if (margins[proposed] < confidence) confidence = margins[proposed];
                        if (minimum_margin && proposed >= minimum_tokens && confidence < minimum_margin) break;
                    }
                    ids[proposed++] = id;
                }
                if (trace) {
                    fprintf(stderr, "draft proposal count=%u processed=%u", proposed, processed);
                    for (unsigned i = 0; i < proposed; ++i) fprintf(stderr, " %u:%.4f/%u", ids[i], margins[i], alternatives[i]);
                    fputc('\n', stderr);
                }
                reply.ok = !failed; reply.count = proposed;
                memcpy(reply.ids, ids, proposed * sizeof *ids);
            }
        } else if (frame[0] == 3 && saved && frame[1] <= proposed && !frame[2] && !frame[3]) {
            unsigned accepted = frame[1];
            if (trace) fprintf(stderr, "draft accept count=%u\n", accepted);
            // No rollback is needed when every forwarded token was consumed.
            // Preserve the already valid rotated-key cache in that case.
            if (accepted == processed) reply.ok = 1;
            else if (accepted < processed) reply.ok = plamo_draft_restore(saved, processed, accepted);
            else if (accepted == processed + 1) reply.ok = plamo_forward(&plamo_model, ids[processed]);
            plamo_draft_checkpoint_free(saved); saved = NULL;
        }
        if (fwrite(&reply, sizeof reply, 1, stdout) != 1 || fflush(stdout) || !reply.ok) return 1;
    }
    plamo_draft_checkpoint_free(saved);
    return 0;
}

int main(int argc, char **argv) {
    enum plamo_memory_mode mode = PLAMO_MEMORY_AUTO;
    int tokenizer_only = 0, peer = 0;
    unsigned threads = 0;
    enum plamo_backend backend = PLAMO_BACKEND_AUTO;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--draft-peer")) { peer = 1; continue; }
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(); return 0; }
        if (!strcmp(argv[i], "--tokenizer-only")) { tokenizer_only = 1; continue; }
        if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            const char *value = argv[++i];
            if (!strcmp(value, "auto")) backend = PLAMO_BACKEND_AUTO;
            else if (!strcmp(value, "cpu")) backend = PLAMO_BACKEND_CPU;
            else if (!strcmp(value, "metal")) backend = PLAMO_BACKEND_METAL;
            else if (!strcmp(value, "metal-fp32")) backend = PLAMO_BACKEND_METAL_FP32;
            else { usage(); return 64; }
            continue;
        }
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            const char *value = argv[++i];
            if (!strcmp(value, "auto")) { threads = 0; continue; }
            char *end;
            unsigned long count = strtoul(value, &end, 10);
            if (value[0] < '1' || value[0] > '9' || *end || count > 64) { usage(); return 64; }
            threads = (unsigned)count;
            continue;
        }
        if (!strcmp(argv[i], "--memory") && i + 1 < argc) {
            const char *value = argv[++i];
            if (!strcmp(value, "auto")) mode = PLAMO_MEMORY_AUTO;
            else if (!strcmp(value, "resident")) mode = PLAMO_MEMORY_RESIDENT;
            else if (!strcmp(value, "stream")) mode = PLAMO_MEMORY_STREAM;
            else { usage(); return 64; }
        } else { usage(); return 64; }
    }
    char error[512];
    if (peer) {
        if (strcmp(plamo_model_name, "plamo-3-translate-0.8b") && strcmp(plamo_model_name, "plamo-3-translate-2b")) return 64;
        setenv("_PLAMO_DRAFT_PEER", "1", 1);
    }
    struct plamo_engine_info info;
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    struct plamo_engine *engine = plamo_engine_open(mode, tokenizer_only, threads, backend, &info, error, sizeof error);
    if (!engine) { report_error("startup_failed", error); return 1; }
    if (!plamo_engine_prepare(engine, error, sizeof error)) { report_error("startup_failed", error); return 1; }
    if (peer) return draft_peer(engine);
    plamo_engine_get_info(engine, &info);
    printf("{\"event\":\"ready\",\"protocol\":1,\"model\":"); json_string(plamo_model_name);
    printf(",\"repository\":"); json_string(plamo_model_repository);
    printf(",\"revision\":"); json_string(plamo_model_revision);
    printf(",\"worker_pid\":%d,\"load_seconds\":%.9f,\"preload_seconds\":%.9f,\"threads\":%u,\"memory\":", getpid(), info.load_seconds, info.preload_seconds, info.threads);
    report_memory(engine);
    printf(",\"prompt_cache\":{\"en_ja_tokens\":%" PRIu64 ",\"ja_en_tokens\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"prepare_seconds\":%.9f}",
           info.prefix_tokens[0], info.prefix_tokens[1], info.prefix_bytes, info.prefix_seconds);
    printf(",\"speculative\":{\"enabled\":%s,\"block_tokens\":%u}", info.speculative_tokens ? "true" : "false", info.speculative_tokens);
    printf(",\"compute\":"); report_compute(engine);
    puts("}"); fflush(stdout);
    for (;;) {
        struct pollfd idle = {STDIN_FILENO, POLLIN, 0};
        int ready = poll(&idle, 1, 1000);
        if (!ready) { plamo_engine_maintain_memory(engine); continue; }
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) { report_error("protocol_error", "cannot poll request pipe"); return 1; }
        uint32_t header[4];
        int read = read_exact(header, sizeof header);
        if (!read) return 0;
        if (read != 1) { report_error("protocol_error", "incomplete request header"); return 1; }
        unsigned op = header[0], direction = header[1], max_tokens = header[2], bytes = header[3];
        if (op == 3 && !direction && !max_tokens && !bytes) continue;
        if ((op != 1 && op != 2) || bytes > 16384) { report_error("protocol_error", "invalid request frame"); return 1; }
        char text[16385];
        if (read_exact(text, bytes) != 1) { report_error("protocol_error", "incomplete text"); return 1; }
        text[bytes] = 0;
        if (memchr(text, 0, bytes)) { report_error("invalid_input", "NUL characters are not supported"); continue; }
        if (op == 2) {
            const uint32_t *ids;
            uint64_t count = plamo_engine_tokenize(engine, text, direction, &ids, error, sizeof error);
            if (count == UINT64_MAX) { report_error("invalid_input", error); continue; }
            printf("{\"event\":\"tokenized\",\"prompt_tokens\":%" PRIu64 ",\"token_ids\":[", count);
            for (uint64_t i = 0; i < count; ++i) printf("%s%u", i ? "," : "", ids[i]);
            printf("]}\n"); fflush(stdout);
        } else {
            struct plamo_result result;
            struct plamo_callbacks callbacks = {prefill_event, token_event, engine};
            if (!plamo_engine_translate(engine, text, direction, max_tokens, &callbacks, &result, error, sizeof error))
                report_error(result.prompt_tokens ? "inference_failed" : "invalid_input", error);
            else report_result(&result, engine);
            if (channel_failed) return 1;
        }
    }
}
