#include "../runtime/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* A complete synthetic vocabulary tests ownership and API boundaries without
   model downloads. Every byte has a fallback; repeated normal rows are valid. */
int main(void) {
    char path[] = "/tmp/plamo-tokenizer-XXXXXX", error[256];
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    FILE *file = fdopen(fd, "w");
    if (!file) { close(fd); unlink(path); return 2; }
    fputs("[\"unused\", 0, \"NORMAL\"]\n", file);
    for (unsigned i = 0; i < 256; ++i) fprintf(file, "[\"<0x%02X>\", 0, \"BYTE\"]\n", i);
    for (unsigned i = 257; i < 107520; ++i) fputs("[\"x\", -1, \"NORMAL\"]\n", file);
    if (fclose(file)) { unlink(path); return 3; }
    struct plamo_tokenizer *a = plamo_tokenizer_open(path, error, sizeof error);
    struct plamo_tokenizer *b = plamo_tokenizer_open(path, error, sizeof error);
    unlink(path);  // Both live mappings must remain valid after unlink.
    if (!a || !b) return 4;
    const char *text = "日本語😀", *decoded;
    const uint32_t *ids;
    uint32_t saved[32];
    uint64_t count = plamo_tokenizer_encode(a, text, 0, &ids, error, sizeof error);
    if (count != strlen(text)) return 5;
    memcpy(saved, ids, (size_t)count * sizeof *ids);
    if (plamo_tokenizer_encode(b, "other", 0, &ids, error, sizeof error) == UINT64_MAX) return 6;
    for (uint64_t i = 0; i < count; ++i) {
        if (plamo_tokenizer_decode(a, saved[i], &decoded) != 1 || (unsigned char)*decoded != (unsigned char)text[i]) return 7;
    }
    if (plamo_tokenizer_decode(a, 107520, &decoded) != UINT64_MAX) return 8;
    if (plamo_tokenizer_encode(a, "\xff", 0, &ids, error, sizeof error) != UINT64_MAX) return 9;
    if (plamo_tokenizer_encode(a, text, 3, &ids, error, sizeof error) != UINT64_MAX) return 10;
    char oversized[4098]; memset(oversized, 'a', sizeof oversized); oversized[sizeof oversized - 1] = 0;
    if (plamo_tokenizer_encode(a, oversized, 0, &ids, error, sizeof error) != UINT64_MAX) return 11;
    if (plamo_tokenizer_encode(a, "", 0, &ids, error, sizeof error)) return 12;
    if (plamo_tokenizer_prefix(a, 0, &ids, error, sizeof error) == UINT64_MAX ||
        plamo_tokenizer_prefix(a, 1, &ids, error, sizeof error) == UINT64_MAX) return 13;
    plamo_tokenizer_close(a);
    if (plamo_tokenizer_encode(b, text, 0, &ids, error, sizeof error) != count ||
        memcmp(ids, saved, (size_t)count * sizeof *ids)) return 14;
    plamo_tokenizer_close(b);
    plamo_tokenizer_close(NULL);
    if (plamo_tokenizer_open(path, error, sizeof error)) return 15;
    char malformed[] = "/tmp/plamo-tokenizer-XXXXXX";
    fd = mkstemp(malformed);
    if (fd < 0) return 16;
    if (write(fd, "invalid\n", 8) != 8) { close(fd); unlink(malformed); return 17; }
    close(fd);
    a = plamo_tokenizer_open(malformed, error, sizeof error);
    unlink(malformed);
    if (a) return 18;
    puts("tokenizer ownership: PASS (independent mappings/workspace, byte round trips, invalid input, limits, prefixes, close)");
    return 0;
}
