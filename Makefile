# C port of parakeet-redux inference. `make` builds $(BUILD)/parakeet and
# $(BUILD)/libparakeet.a.
#
#   make                                    own kernels (NEON / AVX2 / generic C), no dependencies
#   make KERNEL=generic BUILD=build-gen     force the plain-C micro-kernel (any CPU)
#   make BLAS=accelerate BUILD=build-blas   CBLAS reference build (Accelerate; BLAS=openblas on Linux)
#   make ARCH=x86_64 BUILD=build-x86        cross-build on macOS (SIMD=none for CPUs without AVX2)
#   make MARCH=-march=armv8-a               Linux aarch64 without the dot-product extension
#   make SANITIZE=thread BUILD=build-tsan   ThreadSanitizer build
#   make check                              bundled clip vs its stored reference (no Python packages needed)
#   make test AUDIO=file.wav                compare against transcribe.py (needs onnxruntime in .venv)
#   make concurrent                         shared-model concurrency test (tests/concurrent.c)
#
# The weights are not in this repository: clone https://huggingface.co/eschmidbauer/parakeet-redux-c
# into models/parakeet-redux-c (or set MODEL). `make test` and `make stages` also need the ONNX
# export with its transcribe.py (https://huggingface.co/eschmidbauer/parakeet-redux-onnx) in $(ONNX_DIR).

CC        ?= cc
BLAS      ?= none
KERNEL    ?= native
BUILD     ?= build
MODEL     ?= models/parakeet-redux-c/parakeet-redux.bin
ONNX_DIR  ?= models/parakeet-redux-onnx
AUDIO     ?= tests/data/tts-16k.wav
UNAME_S   := $(shell uname -s)
HOST_ARCH := $(shell uname -m)
ARCH      ?= $(HOST_ARCH)

CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -fno-math-errno -fPIC -D_DEFAULT_SOURCE -D_DARWIN_C_SOURCE
LDLIBS  := -lm

ifeq ($(BLAS),accelerate)
  CFLAGS += -DPK_BLAS_ACCELERATE
  LDLIBS += -framework Accelerate
else ifeq ($(BLAS),openblas)
  CFLAGS += -DPK_BLAS_OPENBLAS
  LDLIBS += -lopenblas
endif
ifeq ($(KERNEL),generic)
  CFLAGS += -DPK_GENERIC_KERNELS
endif
ifeq ($(UNAME_S),Darwin)
  ifneq ($(ARCH),$(HOST_ARCH))
    CFLAGS  += -arch $(ARCH)
    LDFLAGS += -arch $(ARCH)
  endif
endif
ifeq ($(ARCH),x86_64)
  SIMD ?= avx2
  ifeq ($(SIMD),avx2)
    CFLAGS += -mavx2 -mfma -mf16c
  endif
endif
ifeq ($(ARCH),aarch64)
  MARCH ?= -march=armv8.2-a+dotprod+fp16
  CFLAGS += $(MARCH)
endif
ifeq ($(SANITIZE),thread)
  CFLAGS  += -fsanitize=thread -g -O1
  LDFLAGS += -fsanitize=thread
endif
LDLIBS += -lpthread

SRC := $(wildcard src/*.c)
OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(SRC))
HDR := $(wildcard src/*.h)

all: $(BUILD)/parakeet $(BUILD)/libparakeet.a $(BUILD)/concurrent_test

$(BUILD)/parakeet: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/libparakeet.a: $(filter-out $(BUILD)/main.o,$(OBJ))
	rm -f $@
	ar rcs $@ $^

$(BUILD)/concurrent_test: tests/concurrent.c $(BUILD)/libparakeet.a $(HDR)
	$(CC) $(CFLAGS) $(LDFLAGS) -Isrc -o $@ tests/concurrent.c $(BUILD)/libparakeet.a $(LDLIBS)

$(BUILD)/%.o: src/%.c $(HDR) | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

check: $(BUILD)/parakeet $(BUILD)/concurrent_test
	$(PYTHON) tests/compare.py tests/data/tts-16k.wav --ref tests/data/tts-16k.expected.json --bin $(BUILD)/parakeet --model $(MODEL)
	$(PYTHON) tests/compare.py tests/data/tts-16k.wav --ref tests/data/tts-16k.expected.json --bin $(BUILD)/parakeet --model $(MODEL) --fast
	$(BUILD)/concurrent_test $(MODEL) tests/data/tts-16k.wav 3 2

test: $(BUILD)/parakeet
	$(PYTHON) tests/compare.py $(AUDIO) --bin $(BUILD)/parakeet --model $(MODEL) --onnx-dir $(ONNX_DIR)
	$(PYTHON) tests/compare.py $(AUDIO) --bin $(BUILD)/parakeet --model $(MODEL) --onnx-dir $(ONNX_DIR) --fast

stages: $(BUILD)/parakeet
	$(PYTHON) tests/stages.py $(AUDIO) --bin $(BUILD)/parakeet --model $(MODEL) --onnx-dir $(ONNX_DIR) --seconds 40

concurrent: $(BUILD)/concurrent_test
	$(BUILD)/concurrent_test $(MODEL) $(AUDIO) 4 1 --lock

clean:
	rm -rf $(BUILD)

.PHONY: all check test stages concurrent clean
