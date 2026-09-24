#!/usr/bin/env python3
"""Stage-by-stage comparison of the C binary with ONNX Runtime.

    python tests/stages.py speech.wav [--seconds 20] [--bin build/parakeet] [--model models/parakeet-redux-c/parakeet-redux.bin]
                           [--onnx-dir models/parakeet-redux-onnx]

Cuts the first N seconds of the file to a temporary clip, runs the binary with
--dump, then recomputes every stage with ONNX Runtime and prints the largest
deviation of features, subsampler output, encoder output, VAD probabilities
and the greedy token/duration sequence.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = HERE / "models" / "parakeet-redux-c" / "parakeet-redux.bin"
DEFAULT_ONNX_DIR = HERE / "models" / "parakeet-redux-onnx"
ONNX_DIR = Path(sys.argv[sys.argv.index("--onnx-dir") + 1]) if "--onnx-dir" in sys.argv else DEFAULT_ONNX_DIR
if not (ONNX_DIR / "transcribe.py").exists():
    sys.exit(f"error: {ONNX_DIR / 'transcribe.py'} not found; clone https://huggingface.co/eschmidbauer/parakeet-redux-onnx into {ONNX_DIR}")
sys.path.insert(0, str(ONNX_DIR))
import transcribe  # noqa: E402


def report(name: str, ref: np.ndarray, got: np.ndarray) -> bool:
    if ref.shape != got.shape:
        print(f"  {name:<15} SHAPE MISMATCH ref {ref.shape} c {got.shape}")
        return False
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    scale = np.abs(ref).max() or 1.0
    print(f"  {name:<15} shape {str(ref.shape):<14} max abs diff {diff.max():.3e}  (rel {diff.max() / scale:.3e}, mean {diff.mean():.3e})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--bin", type=Path, default=HERE / "build" / "parakeet")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    pcm = transcribe.read_audio(args.audio)[: int(args.seconds * transcribe.SAMPLE_RATE)]
    tmp = Path(tempfile.mkdtemp(prefix="pk-stages-"))
    clip = tmp / "clip.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(transcribe.SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32768).astype("<i2").tobytes())
    pcm = transcribe.read_audio(clip)  # exactly what both sides read
    print(f"clip: {pcm.size / 16000:.2f}s -> {clip}")

    subprocess.run([str(args.bin), "--model", str(args.model), "--dump", str(tmp), "--json", "--timestamps", "word",
                    "--threads", str(args.threads), str(clip)], check=True, capture_output=True)

    model = transcribe.OnnxParakeet(args.onnx_dir, threads=args.threads)
    ok = True

    # VAD probabilities of the first 120 s block (the binary dumps them only when the clip is cut)
    vad_graph = onnx.load(str(args.onnx_dir / "vad-model.onnx"))
    vad_graph.graph.output.append(onnx.helper.make_tensor_value_info("linear", onnx.TensorProto.FLOAT, None))
    session = ort.InferenceSession(vad_graph.SerializeToString(), providers=["CPUExecutionProvider"])
    if (tmp / "vad0_probs.npy").exists():
        block = pcm[: round(transcribe.BLOCK_SECONDS * transcribe.SAMPLE_RATE)]
        features, lengths = model.features(block)
        probabilities, valid, _ = session.run(None, {"audio_signal": features, "length": lengths})
        ok &= report("vad", probabilities[0, : int(valid[0])], np.load(tmp / "vad0_probs.npy"))
    else:
        print("  vad          (not run: clip is under 30 s; use --seconds 31 or more to test the VAD)")

    pieces = model.cut_segments(pcm)
    print(f"  {len(pieces)} segment(s): " + ", ".join(f"{a / 16000:.2f}-{b / 16000:.2f}s" for a, b in pieces))
    for index, (start, end) in enumerate(pieces):
        clip_pcm = pcm[start:end]
        tag = f"seg{index}"
        features, lengths = model.features(clip_pcm)
        ok &= report(f"{tag} features", features[0].T, np.load(tmp / f"{tag}_features.npy"))
        _, valid, linear = session.run(None, {"audio_signal": features, "length": lengths})
        n = int(valid[0])
        ok &= report(f"{tag} subsample", linear[0, :n], np.load(tmp / f"{tag}_subsample.npy"))
        encoded = model.encode(clip_pcm)
        ok &= report(f"{tag} encoder", encoded, np.load(tmp / f"{tag}_encoder.npy"))
        tokens, durations = model.greedy(encoded)
        steps = np.load(tmp / f"{tag}_steps.npy")
        ref_steps = np.stack([tokens, durations], 1)
        if ref_steps.shape == steps.shape and np.array_equal(ref_steps, steps):
            print(f"  {tag} greedy    identical: {len(tokens)} steps, {sum(t != model.blank_id for t in tokens)} tokens")
        else:
            ok = False
            print(f"  {tag} greedy    DIFFERS: reference {len(tokens)} steps, c {len(steps)} steps")
            for i, (a, b) in enumerate(zip(ref_steps, steps)):
                if not np.array_equal(a, b):
                    print(f"    first difference at step {i}: ref token {a[0]} ({model.vocab.pieces.get(int(a[0]))!r}) dur {a[1]}, "
                          f"c token {b[0]} ({model.vocab.pieces.get(int(b[0]))!r}) dur {b[1]}")
                    break
            print("    ref text:", model.vocab.decode(tokens))
            print("    c   text:", model.vocab.decode([int(t) for t in steps[:, 0]]))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
