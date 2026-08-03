"""Express residual Add as variadic Mean followed by multiplication by two."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
model = onnx.load(args.input)
nodes = []
replaced = 0
for node in model.graph.node:
    if node.op_type != "Add":
        nodes.append(node)
        continue
    stem = node.name or f"Residual{replaced}"
    mean_output = f"{node.output[0]}.mean"
    scale_name = f"{stem}.two"
    model.graph.initializer.append(numpy_helper.from_array(np.array(2.0, dtype=np.float32), scale_name))
    nodes.append(helper.make_node("Mean", list(node.input), [mean_output], name=f"{stem}.CPU-Mean"))
    nodes.append(helper.make_node("Mul", [mean_output, scale_name], list(node.output), name=f"{stem}.Scale2"))
    replaced += 1
model.graph.ClearField("node")
model.graph.node.extend(nodes)
onnx.checker.check_model(model)
onnx.save(model, args.output)
print(f"replaced {replaced} residual Adds with Mean-times-two")
