#!/usr/bin/env python3
"""Compare native-engine output against the frozen FP32 ONNX oracle.

Both files are raw float32 (N, steps, C) post-softmax probabilities in the same
crop order, so this is a positional comparison with no re-preprocessing.

Usage: python3 compare_native.py <oracle.f32> <native.f32> <n> <steps> [dict]

`dict` is either a PaddleOCR inference.yml (PostProcess.character_dict, blank at
index 0 and a trailing space -- app.py's convention) or a plain one-per-line
dictionary. Without it the decode comparison runs on raw class indices, which is
what the correctness gate actually needs; with it the text is also readable.
"""

from __future__ import annotations

import sys

import numpy as np


def load_chars(path):
    if path is None:
        return None
    if path.endswith((".yml", ".yaml")):
        import yaml
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return [""] + list(cfg["PostProcess"]["character_dict"]) + [" "]
    with open(path, encoding="utf-8") as fh:
        return [""] + [line.rstrip("\n") for line in fh]


def decode(probs, chars):
    ids = probs.argmax(axis=1)
    out, prev = [], -1
    for i in ids.tolist():
        if i != 0 and i != prev:
            out.append(chars[i] if chars and i < len(chars) else f"<{i}>")
        prev = i
    return "".join(out)


def main():
    oracle = np.fromfile(sys.argv[1], np.float32)
    native = np.fromfile(sys.argv[2], np.float32)
    n, steps = int(sys.argv[3]), int(sys.argv[4])
    chars = load_chars(sys.argv[5] if len(sys.argv) > 5 else None)
    classes = oracle.size // (n * steps)
    oracle = oracle.reshape(n, steps, classes)
    native = native.reshape(n, steps, classes)

    diff = np.abs(oracle - native)
    agree = oracle.argmax(2) == native.argmax(2)
    exact, mismatched = 0, []
    for i in range(n):
        a, b = decode(oracle[i], chars), decode(native[i], chars)
        if a == b:
            exact += 1
        elif len(mismatched) < 5:
            mismatched.append((i, a, b))

    print(f"crops={n} steps={steps} classes={classes}")
    print(f"max abs prob diff  : {diff.max():.3e}")
    print(f"mean abs prob diff : {diff.mean():.3e}")
    print(f"timestep argmax agreement : {agree.mean()*100:.4f}%  ({int(agree.sum())}/{agree.size})")
    print(f"full-string exact decode match : {exact}/{n} ({exact/n*100:.2f}%)")
    for i, a, b in mismatched:
        print(f"  crop {i}: oracle={a!r} native={b!r}")
    worst = np.unravel_index(diff.argmax(), diff.shape)
    print(f"worst element: crop={worst[0]} t={worst[1]} class={worst[2]} "
          f"oracle={oracle[worst]:.6f} native={native[worst]:.6f}")
    return 0 if diff.max() < 1e-3 and agree.all() else 1


if __name__ == "__main__":
    sys.exit(main())
