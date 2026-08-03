"""Run XQuant Rosetta as single-path SpaceMIT segments with CPU residual sums."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Stage:
    lhs: object
    rhs: object
    output_name: str


class SegmentedRosetta:
    def __init__(self, directory: Path, profile: bool = False) -> None:
        import spacemit_ort  # noqa: F401
        import onnxruntime as ort

        self.directory = directory
        providers = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
        self.stages: list[Stage] = []
        lines = (directory / "manifest.tsv").read_text().splitlines()
        for line in lines[:-1]:
            fields = line.split("\t")
            lhs_options = ort.SessionOptions()
            rhs_options = ort.SessionOptions()
            if profile:
                lhs_options.enable_profiling = True
                rhs_options.enable_profiling = True
            lhs = ort.InferenceSession(str(directory / fields[2]), sess_options=lhs_options, providers=providers)
            rhs = ort.InferenceSession(str(directory / fields[4]), sess_options=rhs_options, providers=providers)
            self.stages.append(Stage(lhs, rhs, fields[6]))
        tail_fields = lines[-1].split("\t")
        self.tail = ort.InferenceSession(
            str(directory / tail_fields[2]), providers=["CPUExecutionProvider"]
        )

    def run(self, image: np.ndarray) -> np.ndarray:
        value = image
        for stage in self.stages:
            lhs = stage.lhs.run(None, {stage.lhs.get_inputs()[0].name: value})[0]
            rhs = stage.rhs.run(None, {stage.rhs.get_inputs()[0].name: value})[0]
            value = np.add(lhs, rhs, dtype=np.float32)
        return self.tail.run(None, {self.tail.get_inputs()[0].name: value})[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    model = SegmentedRosetta(args.directory)
    sample = np.zeros((1, 3, 32, 100), dtype=np.float32)
    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        output = model.run(sample)
        timings.append((time.perf_counter() - start) * 1000)
    print(f"output={output.shape} timings_ms={timings}")


if __name__ == "__main__":
    main()
