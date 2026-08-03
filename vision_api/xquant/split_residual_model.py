"""Split a residual ONNX model into accelerator-safe segments around Add nodes."""

import argparse
from pathlib import Path

import onnx


parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("output_dir", type=Path)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
model = onnx.shape_inference.infer_shapes(onnx.load(args.model))
shaped = args.output_dir / "shaped.onnx"
onnx.save(model, shaped)

adds = [node for node in model.graph.node if node.op_type == "Add"]
entry = [item.name for item in model.graph.input]
manifest = []
for index, add in enumerate(adds):
    outputs = list(add.input)
    destination = args.output_dir / f"segment-{index:02d}.onnx"
    onnx.utils.extract_model(str(shaped), str(destination), entry, outputs, check_model=False)
    onnx.checker.check_model(onnx.load(destination))
    manifest.append(f"{destination.name}\t{','.join(entry)}\t{','.join(outputs)}\t{add.output[0]}")
    entry = [add.output[0]]

destination = args.output_dir / f"segment-{len(adds):02d}-tail.onnx"
outputs = [item.name for item in model.graph.output]
onnx.utils.extract_model(str(shaped), str(destination), entry, outputs, check_model=False)
onnx.checker.check_model(onnx.load(destination))
manifest.append(f"{destination.name}\t{','.join(entry)}\t{','.join(outputs)}\t-")
(args.output_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n")
shaped.unlink()
print(f"wrote {len(manifest)} segments around {len(adds)} residual merges")
