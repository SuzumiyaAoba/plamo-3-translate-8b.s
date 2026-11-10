.DEFAULT_GOAL := all
CC = clang
CFLAGS = -O2 -arch arm64 -Wall -Wextra
PYTHON ?= python3
.PHONY: test-tools profile-probes profile-probe
test-tools: profile-probes
	PLAMO_TEST_CLI="$(abspath $(BUILD_DIR)/plamo)" $(PYTHON) -m unittest discover -s tests -p 'test_*.py'
profile-probes:
	$(MAKE) MODEL=0.8b profile-probe
	$(MAKE) MODEL=2b profile-probe
	$(MAKE) MODEL=8b profile-probe
	$(MAKE) MODEL=31b profile-probe

.PHONY: clean
clean:
	rm -rf build

MODEL ?= 8b
BUILD_DIR ?= $(if $(filter 8b,$(MODEL)),build,build/$(MODEL))
GENERATED_NAMES = weight_schema.s model_layout.s rope_constants.s model_profile.s
GENERATED = $(addprefix $(BUILD_DIR)/generated/,$(GENERATED_NAMES))
CORE = $(filter-out src/main.s $(addprefix src/,$(GENERATED_NAMES)),$(wildcard src/*.s))
METAL_RUNTIME = runtime/metal.m runtime/metal_weights.m runtime/metal_model.m runtime/disk_cache.c
METAL_HEADERS = runtime/metal.h runtime/metal_internal.h runtime/model.h runtime/compute.h runtime/parallel.h runtime/weight_cache.h runtime/packing.h runtime/disk_cache.h build/metal_kernels.h
RESIDENT_SOURCE ?= runtime/resident.c
RUNTIME = runtime/engine.c runtime/tokenizer.c $(RESIDENT_SOURCE) runtime/weight_cache.c runtime/parallel.c runtime/prompt_cache.c $(METAL_RUNTIME) runtime/draft.c
RUNTIME_HEADERS = runtime/engine.h runtime/tokenizer.h runtime/resident.h runtime/prompt_cache.h runtime/draft.h $(METAL_HEADERS)
METAL_FLAGS = -Ibuild -framework Foundation -framework Metal
PROFILE_INPUTS = $(wildcard tools/profiles/$(MODEL)/*.json) tools/generate_model.py tools/model_catalog.py tools/weight_schema.py tools/model_layout.py tools/rope_constants.py

.PHONY: all models
all: $(BUILD_DIR)/plamo $(BUILD_DIR)/plamo-asm $(BUILD_DIR)/plamo-worker build/pack-metal-weights
test-tools: $(BUILD_DIR)/plamo
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

build/metal_kernels.h: runtime/kernels.metal tools/embed_metal.py
	mkdir -p build
	$(PYTHON) tools/embed_metal.py $< $@

build/metal_layout_kernels.h: runtime/kernels.metal tests/metal_layout.metal tools/embed_metal_layout.py
	mkdir -p build
	$(PYTHON) tools/embed_metal_layout.py runtime/kernels.metal tests/metal_layout.metal $@

build/pack-metal-weights: tools/pack_weights.m runtime/disk_cache.h runtime/model.h runtime/packing.h build/metal_kernels.h
	mkdir -p build
	$(CC) $(CFLAGS) tools/pack_weights.m $(METAL_FLAGS) -o $@

$(BUILD_DIR)/plamo-asm: src/main.s $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp
	mkdir -p $(BUILD_DIR)
	$(CC) -arch arm64 -nostdlib -Wl,-e,_start $(filter %.s,$^) -lSystem -o $@

$(BUILD_DIR)/plamo: $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp runtime/cli.c $(RUNTIME) $(RUNTIME_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(filter %.s %.c %.m,$^) $(METAL_FLAGS) -o $@

$(BUILD_DIR)/plamo-worker: $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp runtime/worker.c $(RUNTIME) $(RUNTIME_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(filter %.s %.c %.m,$^) $(METAL_FLAGS) -o $@

$(BUILD_DIR)/generation-probe: $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp tests/generation_probe.c $(RUNTIME) $(RUNTIME_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(filter %.s %.c %.m,$^) $(METAL_FLAGS) -o $@

profile-probe: $(BUILD_DIR)/test-profile
$(BUILD_DIR)/test-profile: src/arena.s src/model_open.s src/prefetch.s src/io.s src/bind.s src/resolve.s src/find.s src/tensor.s src/json.s src/load.s src/model_init.s $(GENERATED) $(BUILD_DIR)/generated/.stamp tests/model_profile.c runtime/model.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(filter %.s %.c,$^) -o $@

$(BUILD_DIR)/bench-backend: $(CORE) $(GENERATED) $(BUILD_DIR)/generated/.stamp $(RUNTIME) $(RUNTIME_HEADERS) tests/bench_model.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -DPLAMO_RUNTIME_BACKEND -DPLAMO_IO_POLICY $(filter %.s %.c %.m,$^) $(METAL_FLAGS) -o $@

include tests/Makefile
