"""Create SpaceMIT-oriented static QDQ OCR models from calibration images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class DetectorReader(CalibrationDataReader):
    def __init__(self, images: list[Path]) -> None:
        batches = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot read {path}")
            resized = cv2.resize(image, (736, 736)).astype(np.float32) / 255.0
            resized = (resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            batches.append({"x": np.transpose(resized, (2, 0, 1))[None].astype(np.float32)})
        self._iterator = iter(batches)

    def get_next(self):
        return next(self._iterator, None)


def strip_redundant_kernel_shapes(path: Path) -> int:
    model = onnx.load(path)
    removed = 0
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose", "QLinearConv"}:
            continue
        for attribute in list(node.attribute):
            if attribute.name == "kernel_shape":
                node.attribute.remove(attribute)
                removed += 1
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return removed


parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("images", nargs="+", type=Path)
args = parser.parse_args()

quantize_static(
    args.model,
    args.output,
    DetectorReader(args.images),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
print(f"removed kernel_shape attributes: {strip_redundant_kernel_shapes(args.output)}")
