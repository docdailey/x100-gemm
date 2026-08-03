#!/usr/bin/env python3
"""Produce real perspective-corrected text crops for Rosetta validation.

Runs the LIVE detector path (`OCRBackend._det_input` / `_boxes` / `_crop`,
imported unmodified from app.py) over a real document image, then resizes each
crop to Rosetta's contract -- height 32, aspect-preserved width -- and applies
Paddle's recognizer normalization (pixel/127.5 - 1).

Emits one .npy per crop, shape (3, 32, W), float32, plus crops.json describing
them. These are the inputs shared by the ONNX oracle and the native engine.

Usage: python3 make_rec_crops.py <image> <outdir> [max_crops] [max_width]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import OCRBackend, _order_box  # noqa: E402,F401  (detector path, unmodified)


def rosetta_input(crop, target_h=32, align=1, max_w=1600):
    h, w = crop.shape[:2]
    width = max(8, min(max_w, int(np.ceil(target_h * w / h))))
    if align > 1:
        width = int(np.ceil(width / align) * align)
    resized = cv2.resize(crop, (width, target_h)).astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(np.transpose(resized, (2, 0, 1)))


def main():
    image_path, outdir = sys.argv[1], Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    max_w = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
    outdir.mkdir(parents=True, exist_ok=True)

    det_dir = Path(os.environ["VISION_MODEL_DIR"]) / "PP-OCRv6_tiny_det_onnx"
    det = ort.InferenceSession(str(det_dir / "inference.onnx"),
                               providers=["CPUExecutionProvider"])

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    det_input, ratio_h, ratio_w = OCRBackend._det_input(image)
    prob = det.run(None, {det.get_inputs()[0].name: det_input})[0][0, 0]
    boxes = OCRBackend._boxes(prob, ratio_h, ratio_w)
    crops = [OCRBackend._crop(image, box) for box in boxes][:limit]

    entries = []
    for i, crop in enumerate(crops):
        x = rosetta_input(crop, max_w=max_w)
        name = f"crop_{i:04d}.npy"
        np.save(outdir / name, x)
        cv2.imwrite(str(outdir / f"crop_{i:04d}.png"), crop)
        entries.append({"file": name, "shape": list(x.shape),
                        "src_hw": [int(crop.shape[0]), int(crop.shape[1])]})

    with open(outdir / "crops.json", "w", encoding="utf-8") as fh:
        json.dump({"image": image_path, "count": len(entries), "crops": entries}, fh, indent=1)
    widths = [e["shape"][2] for e in entries]
    print(f"{len(entries)} crops -> {outdir}")
    if widths:
        print(f"width min={min(widths)} max={max(widths)} mean={sum(widths)/len(widths):.1f}")


if __name__ == "__main__":
    main()
