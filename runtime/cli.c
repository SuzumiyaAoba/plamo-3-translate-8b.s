/* The CLI and persistent worker share the same native inference engine. */
#include "engine.h"
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/qos.h>

static void usage(FILE *stream) {
    fprintf(stream,
        "usage: plamo [--backend auto|cpu|metal|metal-fp32] [--memory auto|resident|stream]\n"
        "             [--threads auto|1..64] [--max-tokens 1..4096] [-g|-j|-e] TEXT\n"
        "  TEXT       print token IDs without loading model weights\n"
        "  -g TEXT    generate from a raw prompt\n"
        "  -j TEXT    translate English to Japanese\n"
        "  -e TEXT    translate Japanese to English\n"
        "  auto       use an available Metal GPU and adapt the weight cache\n"
        "  FP32 activations and arithmetic; original BF16 weights; greedy generation\n");
}

static int count(const char *value, unsigned limit, unsigned *out) {
    char *end;
    errno = 0;
    unsigned long number = strtoul(value, &end, 10);
    if (errno || value[0] < '1' || value[0] > '9' || *end || number > limit) return 0;
    *out = (unsigned)number;
    return 1;
}

static int token(uint32_t id, const char *bytes, size_t size, void *context) {
    (void)id; (void)context;
    return fwrite(bytes, 1, size, stdout) == size && !fflush(stdout);
}

int main(int argc, char **argv) {
    enum plamo_backend backend = PLAMO_BACKEND_AUTO;
    enum plamo_memory_mode memory = PLAMO_MEMORY_AUTO;
    unsigned threads = 0, max_tokens = 256, direction = 0;
    int generation = 0, separator = 0;
    const char *text = NULL;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!separator && (!strcmp(arg, "--help") || !strcmp(arg, "-h"))) { usage(stdout); return 0; }
        if (!separator && !strcmp(arg, "--")) { separator = 1; continue; }
        if (!separator && (!strcmp(arg, "-g") || !strcmp(arg, "-j") || !strcmp(arg, "-e"))) {
            if (generation || text) goto invalid;
            generation = 1;
            direction = arg[1] == 'j' ? 1 : arg[1] == 'e' ? 2 : 0;
            // The next argument is text even when it begins with a dash.
            if (++i == argc) goto invalid;
            text = argv[i];
            continue;
        }
        if (!separator && (!strcmp(arg, "--backend") || !strcmp(arg, "--memory") ||
                           !strcmp(arg, "--threads") || !strcmp(arg, "--max-tokens"))) {
            if (++i == argc) goto invalid;
            const char *value = argv[i];
            if (!strcmp(arg, "--backend")) {
                if (!strcmp(value, "auto")) backend = PLAMO_BACKEND_AUTO;
                else if (!strcmp(value, "cpu")) backend = PLAMO_BACKEND_CPU;
                else if (!strcmp(value, "metal")) backend = PLAMO_BACKEND_METAL;
                else if (!strcmp(value, "metal-fp32")) backend = PLAMO_BACKEND_METAL_FP32;
                else goto invalid;
            } else if (!strcmp(arg, "--memory")) {
                if (!strcmp(value, "auto")) memory = PLAMO_MEMORY_AUTO;
                else if (!strcmp(value, "resident")) memory = PLAMO_MEMORY_RESIDENT;
                else if (!strcmp(value, "stream")) memory = PLAMO_MEMORY_STREAM;
                else goto invalid;
            } else if (!strcmp(arg, "--threads")) {
                if (!strcmp(value, "auto")) threads = 0;
                else if (!count(value, 64, &threads)) goto invalid;
            } else if (!count(value, 4096, &max_tokens)) goto invalid;
            continue;
        }
        if (text || (!separator && arg[0] == '-' && arg[1])) goto invalid;
        text = arg;
    }
    if (!text) { usage(argc == 1 ? stdout : stderr); return argc == 1 ? 0 : 2; }
    char error[512];
    struct plamo_engine_info info;
    (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    struct plamo_engine *engine = plamo_engine_open(memory, !generation, threads, backend, &info, error, sizeof error);
    if (!engine) { fprintf(stderr, "plamo: %s\n", error); return 1; }
    if (!generation) {
        const uint32_t *ids;
        uint64_t length = plamo_engine_tokenize(engine, text, 0, &ids, error, sizeof error);
        if (length == UINT64_MAX) { fprintf(stderr, "plamo: %s\n", error); return 1; }
        for (uint64_t i = 0; i < length; ++i) printf("%" PRIu32 " ", ids[i]);
    } else {
        struct plamo_result result;
        const struct plamo_callbacks callbacks = {.token = token};
        if (!plamo_engine_translate(engine, text, direction, max_tokens, &callbacks, &result, error, sizeof error)) {
            fprintf(stderr, "plamo: %s\n", error); return 1;
        }
        if (ferror(stdout) || !strcmp(result.finish_reason, "cancelled")) return 1;
    }
    return putchar('\n') == EOF || fflush(stdout) ? 1 : 0;
invalid:
    usage(stderr);
    return 2;
}
