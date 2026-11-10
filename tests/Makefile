# Model-free native checks. Each rule specifies only its link inputs; the
# common recipe filters out headers and applies frameworks only where needed.
TEST_NAMES = math io load json tensor find activation model \
    attention rope token utf8 unigram score string vocab \
    trie encode vocab-load boundary preprocess qkv cache trig \
    rope-table cache-rope mixer arena decode matmul prefill cache-rope-cached \
    simd-attention simd-activation weight-io cache-simd runtime prompt-cache disk-cache draft \
    weight-cache parallel metal packing metal-decode tokenizer
TEST_BINS = $(addprefix build/test-,$(TEST_NAMES))

.PHONY: test test-metal test-draft
test: $(TEST_BINS)
	@set -e; for test in $(TEST_BINS); do ./$$test; done

test-metal: build/test-metal
	./build/test-metal
test-draft: build/test-draft
	./build/test-draft

$(TEST_BINS) build/test-model-open build/test-model-open-retry:
	mkdir -p build
	$(CC) $(CFLAGS) $(filter %.s %.c %.m,$^) $(TEST_LIBS) -o $@

build/test-runtime build/test-metal build/test-metal-decode build/test-packing: TEST_LIBS = $(METAL_FLAGS)

build/test-math: src/math.s src/prefetch.s tests/math.c

build/bench-metal-layout: tests/bench_metal_layout.m runtime/packing.h build/metal_layout_kernels.h
	mkdir -p build
	$(CC) $(CFLAGS) tests/bench_metal_layout.m $(METAL_FLAGS) -o $@

build/test-draft: runtime/draft.c runtime/draft.h runtime/model.h tests/draft.c

build/test-disk-cache: runtime/disk_cache.c runtime/disk_cache.h runtime/packing.h tests/disk_cache.c

build/test-io: src/io.s tests/io_adapter.s tests/io.c

build/test-load: src/load.s tests/load.c

build/test-json: src/json.s src/load.s tests/json.c

build/test-tensor: src/tensor.s src/json.s src/load.s tests/tensor.c

build/test-find: src/find.s src/tensor.s src/json.s src/load.s tests/find.c

build/test-activation: src/activation.s tests/activation.c

build/test-model: src/model.s src/math.s src/prefetch.s src/activation.s tests/model.c

build/test-attention: src/attention.s src/activation.s tests/attention.c

build/test-rope: src/rope.s tests/rope.c

build/test-token: src/token.s tests/token.c

build/test-utf8: src/utf8.s tests/utf8.c

build/test-unigram: src/unigram.s tests/unigram.c

build/test-score: src/score.s tests/score.c

build/test-string: src/string.s src/utf8.s tests/string.c

build/test-vocab: src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s tests/vocab.c

build/test-trie: src/trie.s tests/trie.c

build/test-encode: src/marked.s src/encode.s src/trie.s src/unigram.s src/utf8.s tests/encode.c

build/test-tokenizer: src/vocab_load.s src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s src/trie.s src/preprocess.s src/boundary.s src/marked.s src/encode.s src/unigram.s src/decode.s src/prompt.s runtime/tokenizer.c runtime/tokenizer.h tests/tokenizer.c

build/test-vocab-load: src/vocab_load.s src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s src/trie.s tests/vocab_load.c

build/test-boundary: src/boundary.s tests/boundary.c

build/test-preprocess: src/preprocess.s src/boundary.s tests/preprocess.c

build/test-qkv: src/qkv.s src/math.s src/prefetch.s tests/qkv.c

build/test-cache: src/cache.s tests/cache.c

build/test-trig: src/trig.s tests/trig.c

build/test-rope-table: src/rope_table.s src/rope_constants.s src/trig.s tests/rope_table.c

build/test-cache-rope: src/cache_rope.s src/rope.s src/rope_table.s src/trig.s tests/cache_rope.c

build/test-mixer: src/prefill.s src/batch.s src/matmul.s src/matmul8.s src/matmul8x3.s src/forward.s src/token.s src/layer.s src/model.s src/mixer.s src/qkv.s src/math.s src/prefetch.s src/cache.s src/cache_rope.s src/rope.s src/rope_table.s src/trig.s src/attention.s src/activation.s src/arena.s src/io.s tests/mixer.c

build/test-model-open: src/model_profile.s src/arena.s src/model_open.s src/prefetch.s src/io.s src/bind.s src/resolve.s src/find.s src/tensor.s src/json.s src/load.s src/model_init.s src/weight_schema.s src/model_layout.s src/rope_constants.s tests/model_open.c

build/test-arena: src/arena.s src/io.s tests/arena.c

build/trace-model: src/model_profile.s src/math.s src/prefetch.s src/io.s src/load.s src/json.s src/tensor.s src/find.s src/activation.s src/model.s src/attention.s src/rope.s src/token.s src/utf8.s src/unigram.s src/score.s src/string.s src/vocab.s src/trie.s src/encode.s src/vocab_load.s src/boundary.s src/preprocess.s src/marked.s src/qkv.s src/cache.s src/trig.s src/rope_table.s src/rope_constants.s src/cache_rope.s src/mixer.s src/layer.s src/forward.s src/resolve.s src/bind.s src/weight_schema.s src/model_init.s src/model_layout.s src/model_open.s src/arena.s tests/trace_model.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-decode: src/decode.s src/prompt.s src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s tests/decode.c

build/bench-matvec: src/math.s src/prefetch.s tests/matvec_reference.s tests/matvec4_reference.s tests/bench_matvec.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-matmul: src/matmul.s src/matmul8.s src/matmul8x3.s src/math.s src/prefetch.s tests/matmul.c

build/bench-matmul: src/matmul.s src/matmul8.s src/matmul8x3.s src/math.s src/prefetch.s tests/bench_matmul.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-prefill: src/prefill.s src/batch.s src/matmul.s src/matmul8.s src/matmul8x3.s src/forward.s src/token.s src/layer.s src/model.s src/mixer.s src/qkv.s src/math.s src/prefetch.s src/cache.s src/cache_rope.s src/rope.s src/rope_table.s src/trig.s src/attention.s src/activation.s src/arena.s src/io.s tests/prefill_reference.s tests/prefill.c tests/inference.h

build/test-cache-rope-cached: src/cache_rope.s src/cache.s src/rope.s src/rope_table.s src/trig.s tests/cache_rope_cached.c

build/bench-model: $(filter-out src/main.s,$(wildcard src/*.s)) tests/bench_model.c tests/inference.h
	mkdir -p build
	$(CC) $(CFLAGS) -DPLAMO_IO_POLICY $(filter %.s %.c,$^) -o $@

build/test-simd-attention: src/attention.s src/rope.s src/activation.s tests/attention_reference.s tests/rope_reference.s tests/bench_attention.c

build/test-simd-activation: src/activation.s src/math.s src/prefetch.s tests/activation_reference.s tests/rmsnorm_reference.s tests/simd_activation.c

build/test-weight-io: src/prefetch.s tests/weight_io.c

build/test-cache-simd: src/cache.s tests/cache_reference.s tests/cache_simd.c tests/inference.h

build/test-model-open-retry: src/model_profile.s src/arena.s src/model_open.s src/prefetch.s src/io.s src/bind.s src/resolve.s src/find.s src/tensor.s src/json.s src/load.s src/model_init.s src/weight_schema.s src/model_layout.s src/rope_constants.s tests/model_open_retry.c

build/test-prompt-cache: runtime/prompt_cache.c runtime/prompt_cache.h runtime/model.h tests/prompt_cache.c

build/test-packing: runtime/packing.h build/metal_kernels.h tests/packing.m

build/test-runtime: $(filter-out src/main.s,$(wildcard src/*.s)) $(RUNTIME) $(RUNTIME_HEADERS) tests/runtime.c

build/test-weight-cache: src/math.s src/matmul.s src/matmul8.s src/matmul8x3.s src/prefetch.s runtime/weight_cache.c runtime/weight_cache.h runtime/model.h tests/weight_cache.c

build/test-parallel: src/math.s src/matmul.s src/matmul8.s src/matmul8x3.s src/prefetch.s runtime/parallel.c runtime/compute.h runtime/parallel.h runtime/model.h tests/parallel.c

build/test-metal: src/token.s src/math.s src/activation.s src/rope_table.s src/trig.s src/matmul.s src/matmul8.s src/matmul8x3.s src/prefetch.s $(METAL_RUNTIME) $(METAL_HEADERS) tests/metal.c

build/test-metal-decode: $(filter-out src/main.s,$(wildcard src/*.s)) $(METAL_RUNTIME) $(METAL_HEADERS) tests/metal_decode.c
