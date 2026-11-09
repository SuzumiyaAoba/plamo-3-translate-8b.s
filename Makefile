.DEFAULT_GOAL := all
CC = clang
CFLAGS = -O2 -arch arm64 -Wall -Wextra
PYTHON ?= python3
.PHONY: test-tools profile-probes profile-probe
test-tools: profile-probes
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
profile-probes:
	$(MAKE) MODEL=0.8b profile-probe
	$(MAKE) MODEL=2b profile-probe
	$(MAKE) MODEL=8b profile-probe
	$(MAKE) MODEL=31b profile-probe

.PHONY: test clean
test: build/test-cache-rope-cached build/test-simd-attention build/test-simd-activation build/test-weight-io build/test-cache-simd build/test-math build/test-io build/test-load build/test-json build/test-tensor build/test-find build/test-activation build/test-model build/test-attention build/test-rope build/test-token build/test-utf8 build/test-unigram build/test-score build/test-string build/test-vocab build/test-trie build/test-encode build/test-vocab-load build/test-boundary build/test-preprocess build/test-qkv build/test-cache build/test-trig build/test-rope-table build/test-cache-rope build/test-mixer build/test-arena build/test-decode build/test-matmul build/test-prefill
	./build/test-math
	./build/test-io
	./build/test-load
	./build/test-json
	./build/test-tensor
	./build/test-find
	./build/test-activation
	./build/test-model
	./build/test-attention
	./build/test-rope
	./build/test-token
	./build/test-utf8
	./build/test-unigram
	./build/test-score
	./build/test-string
	./build/test-vocab
	./build/test-trie
	./build/test-encode
	./build/test-vocab-load
	./build/test-boundary
	./build/test-preprocess
	./build/test-qkv
	./build/test-cache
	./build/test-trig
	./build/test-rope-table
	./build/test-cache-rope
	./build/test-mixer
	./build/test-arena
	./build/test-decode
	./build/test-matmul
	./build/test-prefill
	./build/test-cache-rope-cached
	./build/test-simd-attention
	./build/test-simd-activation
	./build/test-weight-io
	./build/test-cache-simd
	./build/test-runtime
build/test-math: src/math.s src/prefetch.s tests/math.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@
clean:
	rm -rf build

MODEL ?= 8b
BUILD_DIR ?= $(if $(filter 8b,$(MODEL)),build,build/$(MODEL))
GENERATED_NAMES = weight_schema.s model_layout.s rope_constants.s model_profile.s
GENERATED = $(addprefix $(BUILD_DIR)/generated/,$(GENERATED_NAMES))
CORE = $(filter-out src/main.s $(addprefix src/,$(GENERATED_NAMES)),$(wildcard src/*.s))
PROFILE_INPUTS = $(wildcard tools/profiles/$(MODEL)/*.json) tools/generate_model.py tools/model_catalog.py tools/weight_schema.py tools/model_layout.py tools/rope_constants.py

.PHONY: all models
all: $(BUILD_DIR)/plamo $(BUILD_DIR)/plamo-worker
models:
	$(MAKE) MODEL=0.8b all
	$(MAKE) MODEL=2b all
	$(MAKE) MODEL=8b all
	$(MAKE) MODEL=31b all

$(BUILD_DIR)/generated/.stamp: $(PROFILE_INPUTS)
	mkdir -p $(BUILD_DIR)/generated
	$(PYTHON) tools/generate_model.py --model $(MODEL) --output $(BUILD_DIR)/generated
	touch $@
$(GENERATED): $(BUILD_DIR)/generated/.stamp

$(BUILD_DIR)/plamo: src/main.s $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp
	mkdir -p $(BUILD_DIR)
	$(CC) -arch arm64 -nostdlib -Wl,-e,_start $(filter %.s,$^) -lSystem -o $@

$(BUILD_DIR)/plamo-worker: $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp runtime/worker.c runtime/engine.c runtime/resident.c runtime/engine.h runtime/model.h runtime/resident.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(filter %.s %.c,$^) -o $@

profile-probe: $(BUILD_DIR)/test-profile
$(BUILD_DIR)/test-profile: src/arena.s src/model_open.s src/prefetch.s src/io.s src/bind.s src/resolve.s src/find.s src/tensor.s src/json.s src/load.s src/model_init.s $(GENERATED) $(BUILD_DIR)/generated/.stamp tests/model_profile.c runtime/model.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(filter %.s %.c,$^) -o $@

build/test-io: src/io.s tests/io_adapter.s tests/io.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-load: src/load.s tests/load.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-json: src/json.s src/load.s tests/json.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-tensor: src/tensor.s src/json.s src/load.s tests/tensor.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-find: src/find.s src/tensor.s src/json.s src/load.s tests/find.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-activation: src/activation.s tests/activation.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-model: src/model.s src/math.s src/prefetch.s src/activation.s tests/model.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-attention: src/attention.s src/activation.s tests/attention.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-rope: src/rope.s tests/rope.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-token: src/token.s tests/token.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-utf8: src/utf8.s tests/utf8.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-unigram: src/unigram.s tests/unigram.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-score: src/score.s tests/score.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-string: src/string.s src/utf8.s tests/string.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-vocab: src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s tests/vocab.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-trie: src/trie.s tests/trie.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-encode: src/marked.s src/encode.s src/trie.s src/unigram.s src/utf8.s tests/encode.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-vocab-load: src/vocab_load.s src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s src/trie.s tests/vocab_load.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-boundary: src/boundary.s tests/boundary.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-preprocess: src/preprocess.s src/boundary.s tests/preprocess.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-qkv: src/qkv.s src/math.s src/prefetch.s tests/qkv.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-cache: src/cache.s tests/cache.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-trig: src/trig.s tests/trig.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-rope-table: src/rope_table.s src/rope_constants.s src/trig.s tests/rope_table.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-cache-rope: src/cache_rope.s src/rope.s src/rope_table.s src/trig.s tests/cache_rope.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-mixer: src/prefill.s src/batch.s src/matmul.s src/matmul8.s src/matmul8x3.s src/forward.s src/token.s src/layer.s src/model.s src/mixer.s src/qkv.s src/math.s src/prefetch.s src/cache.s src/cache_rope.s src/rope.s src/rope_table.s src/trig.s src/attention.s src/activation.s src/arena.s src/io.s tests/mixer.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-model-open: src/model_profile.s src/arena.s src/model_open.s src/prefetch.s src/io.s src/bind.s src/resolve.s src/find.s src/tensor.s src/json.s src/load.s src/model_init.s src/weight_schema.s src/model_layout.s src/rope_constants.s tests/model_open.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-arena: src/arena.s src/io.s tests/arena.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/trace-model: src/model_profile.s src/math.s src/prefetch.s src/io.s src/load.s src/json.s src/tensor.s src/find.s src/activation.s src/model.s src/attention.s src/rope.s src/token.s src/utf8.s src/unigram.s src/score.s src/string.s src/vocab.s src/trie.s src/encode.s src/vocab_load.s src/boundary.s src/preprocess.s src/marked.s src/qkv.s src/cache.s src/trig.s src/rope_table.s src/rope_constants.s src/cache_rope.s src/mixer.s src/layer.s src/forward.s src/resolve.s src/bind.s src/weight_schema.s src/model_init.s src/model_layout.s src/model_open.s src/arena.s tests/trace_model.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-decode: src/decode.s src/prompt.s src/vocab.s src/string.s src/utf8.s src/score.s src/json.s src/load.s tests/decode.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/bench-matvec: src/math.s src/prefetch.s tests/matvec_reference.s tests/matvec4_reference.s tests/bench_matvec.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-matmul: src/matmul.s src/matmul8.s src/matmul8x3.s src/math.s src/prefetch.s tests/matmul.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@
build/bench-matmul: src/matmul.s src/matmul8.s src/matmul8x3.s src/math.s src/prefetch.s tests/bench_matmul.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-prefill: src/prefill.s src/batch.s src/matmul.s src/matmul8.s src/matmul8x3.s src/forward.s src/token.s src/layer.s src/model.s src/mixer.s src/qkv.s src/math.s src/prefetch.s src/cache.s src/cache_rope.s src/rope.s src/rope_table.s src/trig.s src/attention.s src/activation.s src/arena.s src/io.s tests/prefill_reference.s tests/prefill.c tests/inference.h
	mkdir -p build
	$(CC) $(CFLAGS) $(filter %.s %.c,$^) -o $@

build/test-cache-rope-cached: src/cache_rope.s src/cache.s src/rope.s src/rope_table.s src/trig.s tests/cache_rope_cached.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/bench-model: $(filter-out src/main.s,$(wildcard src/*.s)) tests/bench_model.c tests/inference.h
	mkdir -p build
	$(CC) $(CFLAGS) -DPLAMO_IO_POLICY $(filter %.s %.c,$^) -o $@

build/test-simd-attention: src/attention.s src/rope.s src/activation.s tests/attention_reference.s tests/rope_reference.s tests/bench_attention.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-simd-activation: src/activation.s src/math.s src/prefetch.s tests/activation_reference.s tests/rmsnorm_reference.s tests/simd_activation.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-weight-io: src/prefetch.s tests/weight_io.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

build/test-cache-simd: src/cache.s tests/cache_reference.s tests/cache_simd.c tests/inference.h
	mkdir -p build
	$(CC) $(CFLAGS) $(filter %.s %.c,$^) -o $@

build/test-model-open-retry: src/model_profile.s src/arena.s src/model_open.s src/prefetch.s src/io.s src/bind.s src/resolve.s src/find.s src/tensor.s src/json.s src/load.s src/model_init.s src/weight_schema.s src/model_layout.s src/rope_constants.s tests/model_open_retry.c
	mkdir -p build
	$(CC) $(CFLAGS) $^ -o $@

test: build/test-runtime
build/test-runtime: $(filter-out src/main.s,$(wildcard src/*.s)) runtime/engine.c runtime/resident.c runtime/engine.h runtime/model.h runtime/resident.h tests/runtime.c
	mkdir -p build
	$(CC) $(CFLAGS) $(filter %.s %.c,$^) -o $@
