# parakeet-redux-c

A dependency-free C implementation of
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux),
the 1.58-bit version of NVIDIA's parakeet-tdt-0.6b-v3 speech recognizer: no
Python, ONNX Runtime or BLAS at run time. It keeps the ternary encoder
weights packed at 2 bits per weight and brings its own thread pool,
NEON/AVX2 matrix kernels, FFT and resampler.

The weights are one file, `parakeet-redux.bin` (254 MB), hosted on Hugging
Face at [eschmidbauer/parakeet-redux-c](https://huggingface.co/eschmidbauer/parakeet-redux-c)
together with the script that converts it from the ONNX export at
[eschmidbauer/parakeet-redux-onnx](https://huggingface.co/eschmidbauer/parakeet-redux-onnx).

```bash
git clone https://huggingface.co/eschmidbauer/parakeet-redux-c models/parakeet-redux-c
make                                       # build/parakeet and build/libparakeet.a
./build/parakeet speech.wav                # WAV at any rate, or anything ffmpeg can decode
./build/parakeet speech.wav --timestamps word
./build/parakeet speech.wav --json --threads 8
./build/parakeet speech.wav --fast         # int8 kernels: faster, may change a few words
./build/parakeet talk.wav --stream         # print each window as soon as it is decoded
arecord -f S16_LE -r 16000 -c 1 | ./build/parakeet --stream -
./build/parakeet --jobs 4 *.wav            # four files at a time on one model
./build/parakeet --packed weights.pkw x.wav   # write once, then mmap: 0.03 s start-up
```

The binary looks for `parakeet-redux.bin` in `.`, then in
`models/parakeet-redux-c`, or wherever `--model` / `PARAKEET_MODEL` says (a
file or a directory holding it).

## Results

On a 256 s phone call the default (exact) mode produces the same text, words,
timestamps and sentence segments as the ONNX Runtime reference pipeline.
Apple M2 Max, 12 threads:

| Mode | Time | Real time | Peak RSS | Words vs reference |
| --- | --- | --- | --- | --- |
| ONNX Runtime (`transcribe.py`, 8 threads) | 8.5 s | 30x | | reference |
| exact (default) | 7.3 s | 35x | 1.2 GB | identical (604 words) |
| `--fast` | 4.5 s | 56x | 1.2 GB | 6 of 604 differ |

`--fast` quantizes activations to int8 per row and 128-block and stores the
decoder weights as float16, the equivalent of ONNX Runtime's
`accuracy_level 4`. `--stream` cuts the audio at pauses as it arrives, so its
windows (and, on noisy audio, some words near the cuts) can differ from the
offline run, which sees the whole recording first.

## Library

`build/libparakeet.a` with the header `src/pk.h` is built for one model shared
by many threads: `pk_model_load` returns a read-only model, optionally locked
in memory or mapped from a packed file, and every thread transcribes through
its own `pk_context`, which holds a thread pool of any size (1 for pure
request-level concurrency) and the scratch memory. Nothing in the library
exits or prints: failures come back as NULL or -1 with `pk_error_message()`,
warnings go to `pk_set_log()`.

```c
pk_options opt = {.lock_in_memory = 1, .packed_path = "/var/cache/parakeet.pkw"};
pk_model *model = pk_model_load("models/parakeet-redux-c/parakeet-redux.bin", &opt);   /* once */
/* per thread: */
pk_context *ctx = pk_context_new(1);
pk_result *r = pk_transcribe(ctx, model, pcm, n_samples, PK_TS_WORD);
```

`pk_stream_*` feeds audio incrementally and hands back results one piece at a
time with absolute timestamps. `pk_stream_set_pause` adds end-of-utterance
cuts (decode as soon as speech is followed by that much silence), and
`pk_stream_set_manual` with `pk_stream_process` lets a worker pool run the
decoding on threads of its own choosing, which is how
[mod_parakeet_redux](https://github.com/eschmidbauer/mod_parakeet_redux)
embeds it in FreeSWITCH. `pk_read_audio` decodes WAV files at any rate (and
other formats through ffmpeg, when present) to 16 kHz mono.

The library also builds with CMake (`add_subdirectory` or `FetchContent` give
the `parakeet::parakeet` target), and the static archive is position
independent so it can go into a shared object.

## Building

| Command | Build |
| --- | --- |
| `make` | own kernels: NEON on arm64, AVX2 on x86_64, plain C elsewhere |
| `make KERNEL=generic` | force the plain-C kernels (any CPU) |
| `make BLAS=accelerate` / `BLAS=openblas` | CBLAS reference build |
| `make ARCH=x86_64` | cross-build on macOS (`SIMD=none` for CPUs without AVX2) |
| `make MARCH=-march=armv8-a` | Linux aarch64 without the dot-product extension |
| `make SANITIZE=thread` | ThreadSanitizer build |

Linux and macOS, gcc or clang, C11 plus POSIX (`pthread`, `mmap`, `mlock`).

## Tests

- `make check` runs the bundled 12 s clip in `tests/data/` against its stored
  ONNX Runtime reference in exact and fast mode plus the concurrent test, and
  needs only a C compiler and Python.
- `make test AUDIO=file.wav` compares any file with `transcribe.py` from a
  checkout of the ONNX export in `models/parakeet-redux-onnx` (or `ONNX_DIR`);
  it needs `uv venv .venv && uv pip install onnxruntime numpy onnx`, or the
  equivalent with pip.
- `make stages AUDIO=file.wav` compares features, subsampler, encoder, VAD and
  greedy tokens against ONNX Runtime stage by stage, from the same checkout.
- `make concurrent` transcribes several clips at once on one model and checks
  each result against a sequential run.

## License and attribution

The model is moondream's [parakeet-redux](https://huggingface.co/moondream/parakeet-redux),
derived from NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
both under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Neither
moondream nor NVIDIA endorse this port.
