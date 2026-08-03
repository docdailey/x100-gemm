"""Extract a model prefix or test it with the SpaceMIT execution provider."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def extract(model_path: Path, node_index: int, output_path: Path) -> None:
    model = onnx.load(model_path)
    node = model.graph.node[node_index]
    output_name = next(name for name in node.output if name)
    onnx.utils.extract_model(
        str(model_path),
        str(output_path),
        [item.name for item in model.graph.input],
        [output_name],
        check_model=False,
    )
    extracted = onnx.load(output_path)
    onnx.checker.check_model(extracted)
    print(
        f"node={node_index} op={node.op_type} name={node.name!r} "
        f"extracted_nodes={len(extracted.graph.node)} output={output_name!r}",
        flush=True,
    )


def load(model_path: Path) -> None:
    import spacemit_ort  # noqa: F401
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"loaded providers={session.get_providers()}", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("extract", "load"))
parser.add_argument("model", type=Path)
parser.add_argument("--node", type=int)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

if args.action == "extract":
    if args.node is None or args.output is None:
        parser.error("extract requires --node and --output")
    extract(args.model, args.node, args.output)
else:
    load(args.model)
