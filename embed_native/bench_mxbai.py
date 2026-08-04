#!/usr/bin/env python3
"""Runs on the K3 board. Benchmarks the native engine (embeddings/sec, per-text latency) across
hart counts (1/2/4/8) and text lengths, vs the live ryzen server. Reports STEADY-STATE numbers
(discards the first few reps): a single warm process was observed to slow down substantially
after the first 1-2 calls under sustained 8-hart load (see PROGRESS.md) -- likely a package-
level DVFS/power-budget response the board doesn't expose via standard Linux cpufreq (checked:
no scaling_cur_freq node on the AI harts). A real server sees sustained load, not one-off calls,
so steady-state is the honest number to report, not the faster cold-start figure."""
import json
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bert_wordpiece import BertWordPieceTokenizer
from mxbai_native import MxbaiEngine

HERE = os.path.dirname(os.path.abspath(__file__))
RYZEN_URL = "http://192.168.68.6:8100/embed_batch/mxbai"

LENGTHS = {
    "short (~5 tok)": "The quick brown fox jumps.",
    "medium (~30 tok)": (
        "Embeddings map discrete tokens into a continuous vector space where semantic "
        "similarity corresponds to geometric proximity, enabling nearest-neighbor search."
    ),
    "long (~130 tok)": (
        "The SpaceMIT K3 board exposes eight A100 RISC-V harts, each with a 1024-bit vector "
        "register file (VLEN=1024) and an IME-2 matrix/dot-product extension. Four of the "
        "eight harts (8, 10, 12, 14) are never contended for IME-2 work; the other four "
        "(9, 11, 13, 15) share IME-2 hardware pairwise with a primary partner, though plain "
        "RVV vector FMA work has been measured to scale cleanly across all eight harts with "
        "no such contention. This engine uses only plain RVV FMA, never IME-2, so it is "
        "expected to scale cleanly across every hart count tested here, unlike engines that "
        "lean on the contended int8 dot-product path for their heavy compute."
    ),
    "near-max (~260 tok)": (
        "In mathematics, a matrix is a rectangular array of numbers, symbols, or expressions, "
        "arranged in rows and columns, which is used to represent a mathematical object or a "
        "property of such an object. For example, the dimensions of the matrix below are 2 x 3 "
        "(read 'two by three'), because there are two rows and three columns. The individual "
        "items, or entries, of a matrix are often denoted by a variable with two subscripts. "
        "Matrices are commonly related to linear algebra. Notable exceptions include incidence "
        "matrices and adjacency matrices in graph theory. This article focuses on matrices "
        "related to linear algebra, and, unless otherwise specified, all matrices represent "
        "linear maps or may be viewed as such. Square matrices, matrices with the same number "
        "of rows and columns, play a major role in matrix theory. Square matrices of a given "
        "dimension form a noncommutative ring, which is one of the most common examples of a "
        "noncommutative ring. The determinant of a square matrix is a number associated with "
        "the matrix, which is fundamental for the study of a square matrix; for example, a "
        "square matrix is invertible if and only if it has a nonzero determinant, and the "
        "eigenvalues of a square matrix are the roots of a polynomial determinant."
    ),
}

WARMUP = 2
REPS = 6


def bench_native(eng, tok, text, nthreads):
    ids = tok.encode(text, max_length=512)
    times = []
    for r in range(WARMUP + REPS):
        _, dt = eng.embed(ids, nthreads=nthreads)
        if r >= WARMUP:
            times.append(dt * 1000.0)
    return len(ids), times


def bench_ryzen(text):
    times = []
    for r in range(WARMUP + REPS):
        t0 = time.time()
        resp = requests.post(RYZEN_URL, json={"texts": [text]}, timeout=60)
        resp.raise_for_status()
        dt = (time.time() - t0) * 1000.0
        if r >= WARMUP:
            times.append(dt)
    return times


def main():
    tok = BertWordPieceTokenizer(os.path.join(HERE, "mxbai_vocab.txt"))
    eng = MxbaiEngine(os.path.join(HERE, "mxbai.bin"))

    results = {"native": {}, "ryzen": {}}

    print("=== ryzen server (steady-state, single-request latency) ===")
    for label, text in LENGTHS.items():
        times = bench_ryzen(text)
        med = statistics.median(times)
        results["ryzen"][label] = {"median_ms": med, "times": times}
        print(f"{label:22s} median={med:8.1f}ms  min={min(times):8.1f}ms  max={max(times):8.1f}ms")

    print("\n=== native K3 engine, hart scaling (steady-state) ===")
    for label, text in LENGTHS.items():
        results["native"][label] = {}
        for nt in (1, 2, 4, 8):
            n_tok, times = bench_native(eng, tok, text, nt)
            med = statistics.median(times)
            results["native"][label][nt] = {"median_ms": med, "n_tok": n_tok, "times": times}
            print(f"{label:22s} nt={nt}  S={n_tok:4d}  median={med:9.1f}ms  "
                  f"emb/s={1000.0/med:6.2f}  min={min(times):8.1f}  max={max(times):8.1f}")

    print("\n=== summary: native (nt=8, steady-state) vs ryzen ===")
    for label in LENGTHS:
        nat = results["native"][label][8]["median_ms"]
        ryz = results["ryzen"][label]["median_ms"]
        speedup = ryz / nat
        print(f"{label:22s} native={nat:8.1f}ms  ryzen={ryz:8.1f}ms  "
              f"ryzen/native={speedup:.2f}x {'(native faster)' if speedup>1 else '(ryzen faster)'}")

    print("\n=== hart scaling factor (nt=8 vs nt=1, per length) ===")
    for label in LENGTHS:
        t1 = results["native"][label][1]["median_ms"]
        t8 = results["native"][label][8]["median_ms"]
        print(f"{label:22s} nt=1: {t1:8.1f}ms  nt=8: {t8:8.1f}ms  speedup={t1/t8:.2f}x")

    with open(os.path.join(HERE, "bench_mxbai_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'bench_mxbai_results.json')}")


if __name__ == "__main__":
    main()
