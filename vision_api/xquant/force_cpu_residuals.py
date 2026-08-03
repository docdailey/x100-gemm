"""Force residual additions outside SpaceMIT EP while preserving float outputs.

SpaceMIT EP 2.0.5 segfaults while compiling the first two-input residual Add.
FP64 is unsupported by the accelerator, so casts around only those Adds create a
clean CPU-provider boundary without moving the convolutional blocks off IME.
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


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
    stem = node.name or f"ResidualAdd_{replaced}"
    lhs64 = f"{stem}.lhs.fp64"
    rhs64 = f"{stem}.rhs.fp64"
    sum64 = f"{stem}.sum.fp64"
    nodes.extend(
        [
            helper.make_node("Cast", [node.input[0]], [lhs64], name=f"{stem}.CastL", to=TensorProto.DOUBLE),
            helper.make_node("Cast", [node.input[1]], [rhs64], name=f"{stem}.CastR", to=TensorProto.DOUBLE),
            helper.make_node("Add", [lhs64, rhs64], [sum64], name=f"{stem}.CPU"),
            helper.make_node("Cast", [sum64], [node.output[0]], name=f"{stem}.CastOut", to=TensorProto.FLOAT),
        ]
    )
    replaced += 1

model.graph.ClearField("node")
model.graph.node.extend(nodes)
onnx.checker.check_model(model)
onnx.save(model, args.output)
print(f"forced {replaced} residual Adds through FP64 CPU boundaries")
