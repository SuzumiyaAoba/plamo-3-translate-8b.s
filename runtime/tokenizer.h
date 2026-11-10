#ifndef PLAMO_TOKENIZER_H
#define PLAMO_TOKENIZER_H
#include <stddef.h>
#include <stdint.h>

/* Owns the vocabulary mapping, row index, trie and reusable encoding workspace.
   Returned IDs/bytes are borrowed until the next call on this tokenizer.
   direction: 0 raw, 1 English to Japanese, 2 Japanese to English. */
struct plamo_tokenizer;
struct plamo_tokenizer *plamo_tokenizer_open(const char *path, char *error, size_t cap);
void plamo_tokenizer_close(struct plamo_tokenizer *);
uint64_t plamo_tokenizer_encode(struct plamo_tokenizer *, const char *text, unsigned direction,
                                const uint32_t **ids, char *error, size_t cap);
/* Fixed input prefix: direction 0 selects English, 1 selects Japanese. */
uint64_t plamo_tokenizer_prefix(struct plamo_tokenizer *, unsigned direction,
                                const uint32_t **ids, char *error, size_t cap);
uint64_t plamo_tokenizer_decode(struct plamo_tokenizer *, uint32_t id, const char **bytes);
#endif
