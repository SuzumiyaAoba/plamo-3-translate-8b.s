/* A single persistent model process. Binary requests, JSONL events on stdout.
   Request: four little-endian u32 values {op, direction, max_tokens, text_bytes},
   followed by exact UTF-8 bytes. op=1 translate, 2 tokenize, 3 cancel (no body).
   The HTTP adapter owns this private pipe; it never starts a process per request. */
#include "engine.h"
#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

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

static void prefill_event(uint64_t count, void *context) {
    (void)context;
    printf("{\"event\":\"prefill\",\"prompt_tokens\":%" PRIu64 "}\n", count);
    fflush(stdout);
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

static void report_result(const struct plamo_result *r) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("{\"event\":\"done\",\"finish_reason\":\"%s\","
           "\"usage\":{\"prompt_tokens\":%" PRIu64 ",\"completion_tokens\":%" PRIu64 ",\"total_tokens\":%" PRIu64 "},"
           "\"timings\":{\"tokenize_seconds\":%.9f,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,"
           "\"first_token_seconds\":%.9f,\"total_seconds\":%.9f,\"decode_steps\":%" PRIu64 "},"
           "\"weight_io\":{\"staged_bytes\":%" PRIu64 ",\"read_calls\":%" PRIu64 "},"
           "\"process_peak_rss_bytes\":%ld,"
           "\"verification\":{\"prefill_logits_hash\":\"%016" PRIx64 "\",\"prefill_state_hash\":\"%016" PRIx64 "\","
           "\"logits_hash\":\"%016" PRIx64 "\",\"state_hash\":\"%016" PRIx64 "\"}}\n",
           r->finish_reason, r->prompt_tokens, r->completion_tokens, r->prompt_tokens + r->completion_tokens,
           r->tokenize_seconds, r->prefill_seconds, r->decode_seconds, r->first_token_seconds, r->total_seconds,
           r->decode_steps, r->staged_bytes, r->read_calls, usage.ru_maxrss,
           r->prefill_logits_hash, r->prefill_state_hash, r->logits_hash, r->state_hash);
    fflush(stdout);
}

static void usage(void) {
    puts("usage: plamo-worker [--memory auto|resident|stream] [--tokenizer-only]\n"
         "Private persistent worker for plamo-server. Inference kernels remain in assembly.\n"
         "resident: original BF16 file bytes in private memory; requires weights + 2 GiB spare.\n"
         "auto: preload when capacity permits, otherwise retain the low-memory I/O policy.\n"
         "OS paging is possible; preloading is not mlock or a hard residency guarantee.");
}

int main(int argc, char **argv) {
    enum plamo_memory_mode mode = PLAMO_MEMORY_AUTO;
    int tokenizer_only = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(); return 0; }
        if (!strcmp(argv[i], "--tokenizer-only")) { tokenizer_only = 1; continue; }
        if (!strcmp(argv[i], "--memory") && i + 1 < argc) {
            const char *value = argv[++i];
            if (!strcmp(value, "auto")) mode = PLAMO_MEMORY_AUTO;
            else if (!strcmp(value, "resident")) mode = PLAMO_MEMORY_RESIDENT;
            else if (!strcmp(value, "stream")) mode = PLAMO_MEMORY_STREAM;
            else { usage(); return 64; }
        } else { usage(); return 64; }
    }
    char error[512];
    struct plamo_engine_info info;
    struct plamo_engine *engine = plamo_engine_open(mode, tokenizer_only, &info, error, sizeof error);
    if (!engine) { report_error("startup_failed", error); return 1; }
    printf("{\"event\":\"ready\",\"protocol\":1,\"worker_pid\":%d,\"load_seconds\":%.9f,\"memory\":{"
           "\"mode\":\"%s\",\"weight_bytes\":%" PRIu64 ",\"preloaded_weight_bytes\":%" PRIu64 ","
           "\"estimated_reclaimable_bytes_at_start\":%" PRIu64 ",\"physical_bytes\":%" PRIu64 ",\"reserve_bytes\":%" PRIu64 ",\"note\":",
           getpid(), info.load_seconds, info.memory_mode, info.weight_bytes, info.preloaded_weight_bytes,
           info.reclaimable_bytes, info.physical_bytes, PLAMO_MEMORY_RESERVE);
    json_string(info.memory_note); printf("}}\n"); fflush(stdout);
    for (;;) {
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
            struct plamo_callbacks callbacks = {prefill_event, token_event, NULL};
            if (!plamo_engine_translate(engine, text, direction, max_tokens, &callbacks, &result, error, sizeof error))
                report_error(result.prompt_tokens ? "inference_failed" : "invalid_input", error);
            else report_result(&result);
            if (channel_failed) return 1;
        }
    }
}
