#!/usr/bin/env python3
"""Compare the C binary with transcribe.py (ONNX Runtime) on one or more files.

    python tests/compare.py speech.wav [--bin build/parakeet] [--model models/parakeet-redux-c/parakeet-redux.bin]
                            [--onnx-dir models/parakeet-redux-onnx] [--ref ref.json]

The reference is transcribe.py from a checkout of https://huggingface.co/eschmidbauer/parakeet-redux-onnx
(needs onnxruntime and numpy). With --ref a stored reference JSON is used
instead and no Python packages are needed.
Runs both with word timestamps and reports differences in the text, the word
sequence and the timestamps. Exit status 1 on any difference.
"""
from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = HERE / "models" / "parakeet-redux-c" / "parakeet-redux.bin"
DEFAULT_ONNX_DIR = HERE / "models" / "parakeet-redux-onnx"


def run_reference(onnx_dir: Path, audio: Path, threads: int) -> dict:
    script = onnx_dir / "transcribe.py"
    if not script.exists():
        sys.exit(f"error: {script} not found; clone https://huggingface.co/eschmidbauer/parakeet-redux-onnx into {onnx_dir}")
    cmd = [sys.executable, str(script), str(audio), "--json", "--timestamps", "word", "--threads", str(threads), "--model-dir", str(onnx_dir)]
    started = time.perf_counter()
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"  reference: {time.perf_counter() - started:.2f}s  ({out.stderr.strip().splitlines()[-1]})")
    return json.loads(out.stdout)


def run_binary(binary: Path, model: Path, audio: Path, threads: int, extra: list[str]) -> dict:
    cmd = [str(binary), "--model", str(model), "--json", "--timestamps", "word", "--threads", str(threads), *extra, str(audio)]
    started = time.perf_counter()
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"  c binary:  {time.perf_counter() - started:.2f}s  ({out.stderr.strip().splitlines()[-1]})")
    return json.loads(out.stdout)


def words_of(result: dict) -> list[dict]:
    return [w for s in result["segments"] for w in s.get("words", [])]


def compare(ref: dict, got: dict, tol: float) -> bool:
    ok = True
    if ref["text"] != got["text"]:
        ok = False
        print("  TEXT DIFFERS")
        for line in difflib.unified_diff(ref["text"].split(), got["text"].split(), "reference", "c", lineterm="", n=2):
            print("   ", line)
    else:
        print(f"  text: identical ({len(ref['text'].split())} words)")
    rw, gw = words_of(ref), words_of(got)
    if [w["word"] for w in rw] != [w["word"] for w in gw]:
        ok = False
        print(f"  WORD LIST DIFFERS ({len(rw)} vs {len(gw)} words)")
        for line in difflib.unified_diff([w["word"] for w in rw], [w["word"] for w in gw], "reference", "c", lineterm="", n=2):
            print("   ", line)
    else:
        worst = max((max(abs(a["start"] - b["start"]), abs(a["end"] - b["end"])) for a, b in zip(rw, gw)), default=0.0)
        if worst > tol:
            ok = False
            print(f"  TIMESTAMPS DIFFER: max deviation {worst:.4f}s")
            for a, b in zip(rw, gw):
                if abs(a["start"] - b["start"]) > tol or abs(a["end"] - b["end"]) > tol:
                    print(f"    {a['word']!r}: ref {a['start']:.3f}-{a['end']:.3f}  c {b['start']:.3f}-{b['end']:.3f}")
        else:
            print(f"  words: identical ({len(rw)} words, max timestamp deviation {worst:.2e}s)")
    rs = [(s["text"], s["start"], s["end"]) for s in ref["segments"]]
    gs = [(s["text"], s["start"], s["end"]) for s in got["segments"]]
    if len(rs) != len(gs) or any(a[0] != b[0] or abs(a[1] - b[1]) > tol or abs(a[2] - b[2]) > tol for a, b in zip(rs, gs)):
        ok = False
        print(f"  SEGMENTS DIFFER ({len(rs)} vs {len(gs)})")
    else:
        print(f"  segments: identical ({len(rs)})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--bin", type=Path, default=HERE / "build" / "parakeet")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="parakeet-redux.bin for the binary")
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR, help="ONNX export checkout with transcribe.py")
    parser.add_argument("--ref", type=Path, default=None, help="cached reference JSON (skips running transcribe.py)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--fast", action="store_true", help="run the binary with --fast and only report how many words change")
    parser.add_argument("--args", default="", help="extra arguments for the binary, e.g. --args '--threads 4'")
    args = parser.parse_args()
    extra = args.args.split() + (["--fast"] if args.fast else [])
    all_ok = True
    for audio in args.audio:
        print(f"== {audio}")
        ref = json.loads(args.ref.read_text()) if args.ref else run_reference(args.onnx_dir, audio, args.threads)
        got = run_binary(args.bin, args.model, audio, args.threads, extra)
        if args.fast:
            rw, gw = ref["text"].split(), got["text"].split()
            ops = difflib.SequenceMatcher(a=rw, b=gw).get_opcodes()
            changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
            print(f"  fast mode: {changed} of {len(rw)} words differ from the exact reference")
            for tag, i1, i2, j1, j2 in ops:
                if tag != "equal":
                    print(f"    {tag}: {' '.join(rw[i1:i2])!r} -> {' '.join(gw[j1:j2])!r}")
        else:
            all_ok &= compare(ref, got, args.tol)
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
