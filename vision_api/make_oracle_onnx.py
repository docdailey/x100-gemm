#!/usr/bin/env python3
"""Build a dynamic-width ONNX Runtime oracle from the fixed-shape Rosetta export.

The shipped export pins the input to [N,3,32,100] and ends in a Reshape to a
hardcoded [-1,25,37] -- both of which only hold for width 100. For validating a
width-agnostic native engine the oracle must be width-agnostic too, so this
script marks the width dimension dynamic and cuts the graph at the Softmax
output (dropping the export's Mul-by-1.0 and the width-specific Reshape, which
are exactly what extract_rosetta.py drops as well).

Usage: python3 make_oracle_onnx.py rosetta-r34.onnx rosetta-r34-dyn.onnx
"""

from __future__ import annotations

import sys

import onnx


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = onnx.load(src)
    graph = model.graph

    width_dim = graph.input[0].type.tensor_type.shape.dim[3]
    width_dim.ClearField("dim_value")
    width_dim.dim_param = "W"

    keep = []
    for node in graph.node:
        if node.op_type in ("Mul", "Reshape") and node.output[0] in (
                "Mul.1", graph.output[0].name):
            continue
        keep.append(node)
    del graph.node[:]
    graph.node.extend(keep)

    softmax_out = "p2o.pd_op.softmax.0.0"
    assert any(softmax_out in n.output for n in graph.node), "softmax output missing"
    del graph.output[:]
    graph.output.extend([onnx.helper.make_tensor_value_info(
        softmax_out, onnx.TensorProto.FLOAT, ["N", "W_out", 37])])

    for vi in list(graph.value_info):
        graph.value_info.remove(vi)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"wrote {dst}: input W dynamic, output {softmax_out} [N, W_out, 37]")


if __name__ == "__main__":
    main()
