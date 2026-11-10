/* Weight-dependent differential probe; intentionally outside model-free tests.
   Dump every complete state/logit vector before each emitted token, then the
   final vectors, so acceptance, EOS, length and cancellation can be compared. */
#include "../runtime/engine.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct capture { unsigned count, cancel; };
static int vectors(void) {
    return fwrite(plamo_model.state, 4, plamo_model.hidden, stdout) == plamo_model.hidden &&
        fwrite(plamo_model.logits, 4, plamo_model.vocab, stdout) == plamo_model.vocab;
}
static int token(uint32_t id, const char *bytes, size_t count, void *context) {
    (void)bytes; (void)count;
    struct capture *capture = context;
    if (capture->count == capture->cancel) return 0;
    if (fwrite(&id, 4, 1, stdout) != 1 || !vectors()) return 0;
    ++capture->count;
    return 1;
}
int main(void) {
    char error[1024]; struct plamo_engine_info info;
    struct plamo_engine *e = plamo_engine_open(PLAMO_MEMORY_AUTO, 0, 1, PLAMO_BACKEND_METAL, &info, error, sizeof error);
    if (!e || !plamo_engine_prepare(e, error, sizeof error)) { fprintf(stderr, "%s\n", error); return 1; }
    uint32_t header[] = {0x504c5632, (uint32_t)plamo_model.hidden, (uint32_t)plamo_model.vocab};
    if (fwrite(header, sizeof header, 1, stdout) != 1 || fflush(stdout)) return 1;
    uint32_t request[4];
    while (fread(request, sizeof request, 1, stdin) == 1) {
        if (request[2] > 16384) return 1;
        char text[16385];
        if (fread(text, 1, request[2], stdin) != request[2]) return 1;
        text[request[2]] = 0;
        struct capture capture = {0, request[3]};
        struct plamo_callbacks cb = {.token = token, .context = &capture};
        struct plamo_result result;
        int ok = plamo_engine_translate(e, text, request[0], request[1], &cb, &result, error, sizeof error);
        if (!ok) { plamo_engine_get_info(e, &info); fprintf(stderr, "%s (%s)\n", error, info.gpu.note); return 1; }
        uint32_t marker = UINT32_MAX;
        uint64_t summary[] = {result.completion_tokens, result.decode_steps,
            !strcmp(result.finish_reason, "length") ? 0 : !strcmp(result.finish_reason, "stop") ? 1 : 2,
            result.speculative_batches, result.speculative_tokens, result.proposed_tokens, result.accepted_tokens, result.prompt_tokens};
        if (fwrite(&marker, 4, 1, stdout) != 1 || fwrite(summary, sizeof summary, 1, stdout) != 1 || !vectors() || fflush(stdout)) return 1;
    }
    return ferror(stdin) ? 1 : 0;
}
