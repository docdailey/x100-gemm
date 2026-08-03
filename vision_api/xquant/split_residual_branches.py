"""Split every residual branch into a single-path accelerator model."""

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
    fields = []
    for branch, output in zip(("a", "b"), add.input):
        if output in entry:
            fields.append("IDENTITY")
            continue
        destination = args.output_dir / f"stage-{index:02d}-{branch}.onnx"
        onnx.utils.extract_model(str(shaped), str(destination), entry, [output], check_model=False)
        onnx.checker.check_model(onnx.load(destination))
        fields.append(destination.name)
    manifest.append(f"{index}\t{','.join(entry)}\t{fields[0]}\t{add.input[0]}\t{fields[1]}\t{add.input[1]}\t{add.output[0]}")
    entry = [add.output[0]]

# The post-residual pool/Gemm/Softmax tail is deliberately CPU-only.
destination = args.output_dir / "tail.onnx"
outputs = [item.name for item in model.graph.output]
onnx.utils.extract_model(str(shaped), str(destination), entry, outputs, check_model=False)
onnx.checker.check_model(onnx.load(destination))
manifest.append(f"tail\t{','.join(entry)}\t{destination.name}\t{','.join(outputs)}")
(args.output_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n")
shaped.unlink()
print(f"wrote single-path models for {len(adds)} residual stages plus a CPU tail")
