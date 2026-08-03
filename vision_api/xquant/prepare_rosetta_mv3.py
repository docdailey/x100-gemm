"""Prepare fixed-batch MobileNetV3 Rosetta for an XQuant compatibility probe."""

from pathlib import Path

import numpy as np
import onnx


work = Path("/work")
model = onnx.load(work / "rosetta-mv3.onnx")
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = ""
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = ""
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
onnx.checker.check_model(model)
onnx.save(model, work / "rosetta-mv3-b1.onnx")

rng = np.random.default_rng(20260730)
paths = []
for index in range(8):
    path = work / f"mv3-calib-{index:02d}.npy"
    np.save(path, rng.uniform(-1.0, 1.0, (1, 3, 32, 100)).astype(np.float32))
    paths.append(path)
(work / "calib_mv3.txt").write_text("\n".join(map(str, paths)) + "\n")
