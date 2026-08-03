"""Fix Rosetta's batch shape and create a deterministic architecture-probe corpus."""

from pathlib import Path

import numpy as np
import onnx


work = Path("/work")
source = work / "rosetta-r34.onnx"
target = work / "rosetta-r34-b16.onnx"

def fixed_batch(batch: int, destination: Path) -> None:
    model = onnx.load(source)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = ""
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = ""
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = batch
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def fixed_shape(batch: int, width: int, destination: Path) -> None:
    model = onnx.load(source)
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    input_shape[0].dim_param = ""
    input_shape[0].dim_value = batch
    input_shape[3].dim_value = width
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    output_shape[0].dim_param = ""
    output_shape[0].dim_value = batch
    output_shape[1].dim_value = width // 4
    for node in model.graph.node:
        if "helper.constant.0" not in node.output:
            continue
        value = next(attribute for attribute in node.attribute if attribute.name == "value")
        value.t.CopyFrom(onnx.numpy_helper.from_array(np.array([-1, width // 4, 37], dtype=np.int64)))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


fixed_batch(16, target)
fixed_batch(1, work / "rosetta-r34-b1.onnx")
fixed_shape(1, 128, work / "rosetta-r34-b1-w128.onnx")

rng = np.random.default_rng(20260730)
paths = []
for index in range(8):
    # Paddle's classic recognizer normalization maps image bytes to [-1, 1].
    batch = rng.uniform(-1.0, 1.0, (16, 3, 32, 100)).astype(np.float32)
    path = work / f"rosetta-calib-{index:02d}.npy"
    np.save(path, batch)
    paths.append(path)
(work / "calib_rosetta.txt").write_text("\n".join(map(str, paths)) + "\n")
single_paths = []
for index in range(8):
    sample = rng.uniform(-1.0, 1.0, (1, 3, 32, 100)).astype(np.float32)
    path = work / f"rosetta-calib-b1-{index:02d}.npy"
    np.save(path, sample)
    single_paths.append(path)
(work / "calib_rosetta_b1.txt").write_text("\n".join(map(str, single_paths)) + "\n")
wide_paths = []
for index in range(8):
    sample = rng.uniform(-1.0, 1.0, (1, 3, 32, 128)).astype(np.float32)
    path = work / f"rosetta-calib-w128-{index:02d}.npy"
    np.save(path, sample)
    wide_paths.append(path)
(work / "calib_rosetta_w128.txt").write_text("\n".join(map(str, wide_paths)) + "\n")
print(f"wrote fixed batch-16 and batch-1 models plus calibration batches")
