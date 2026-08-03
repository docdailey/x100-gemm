"""Replace binary residual Add nodes with equivalent ONNX Sum nodes."""

import argparse
from pathlib import Path

import onnx


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

model = onnx.load(args.input)
replaced = 0
for node in model.graph.node:
    if node.op_type == "Add":
        node.op_type = "Sum"
        node.name = f"{node.name}.CPU-Sum"
        replaced += 1
onnx.checker.check_model(model)
onnx.save(model, args.output)
print(f"replaced {replaced} residual Adds with equivalent Sums")
