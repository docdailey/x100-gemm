#!/usr/bin/env python3
"""Freeze real-crop inputs and FP32 ONNX oracle outputs for the native engine.

For each requested width this writes a raw float32 (N,3,32,W) input file and the
matching (N,steps,37) post-softmax oracle file produced by ONNX Runtime CPU on
rosetta-r34-dyn.onnx. The native engine on the board consumes the input file and
emits an output file in the same layout, so comparison is byte-for-byte
positional with no re-preprocessing on either side.

Usage: python3 prep_validation.py <cropdir> <oracle.onnx> <outdir> <height> W [W ...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main():
    cropdir, oracle_path, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    height = int(sys.argv[4])
    widths = [int(a) for a in sys.argv[5:]]
    outdir.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(oracle_path, providers=["CPUExecutionProvider"])
    files = sorted(glob.glob(f"{cropdir}/crop_*.png"))
    meta = {"crops": [Path(f).name for f in files], "height": height, "widths": {}}

    for W in widths:
        batch = np.empty((len(files), 3, height, W), np.float32)
        for i, f in enumerate(files):
            crop = cv2.imread(f, cv2.IMREAD_COLOR)
            resized = cv2.resize(crop, (W, height)).astype(np.float32) / 127.5 - 1.0
            batch[i] = np.transpose(resized, (2, 0, 1))
        batch = np.ascontiguousarray(batch)
        batch.tofile(outdir / f"in_W{W}.f32")

        outs = []
        for i in range(len(files)):
            outs.append(sess.run(None, {"x": batch[i:i + 1]})[0][0])
        oracle = np.ascontiguousarray(np.stack(outs).astype(np.float32))
        oracle.tofile(outdir / f"oracle_W{W}.f32")
        meta["widths"][str(W)] = {"n": len(files), "steps": int(oracle.shape[1])}
        print(f"W={W}: {len(files)} crops, steps={oracle.shape[1]}, "
              f"in={batch.nbytes/1e6:.1f}MB oracle={oracle.nbytes/1e6:.1f}MB")

    with open(outdir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
