#!/usr/bin/env python3
"""Runs on the K3 board. Validates libmxbai_rvv.so against the oracle (live ryzen server +
local PyTorch, saved by gen_oracle_mxbai.py). Mirrors validate_native.py for nomic. Note: this
model's own oracle-vs-oracle noise floor (local vs remote) is ~1e-3 to 6e-3 max_abs (much larger
than nomic's ~1e-5) -- traced to the model's weights being stored/shipped in fp16
(torch_dtype: float16 in config.json) and accumulating more rounding divergence across 24 layers
than nomic's 12; NOT a defect in this engine, which computes in FP32 throughout from FP32-cast
weights. Expect this engine to land in that same ~1e-3 noise band, not nomic's ~1e-5 one."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bert_wordpiece import BertWordPieceTokenizer
from mxbai_native import MxbaiEngine
from test_texts import TEST_TEXTS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "oracle_mxbai.json")) as f:
        oracle = json.load(f)

    tok = BertWordPieceTokenizer(os.path.join(HERE, "mxbai_vocab.txt"))
    eng = MxbaiEngine(os.path.join(HERE, "mxbai.bin"))

    worst_local = worst_remote = 0.0
    for i, text in enumerate(TEST_TEXTS):
        safe = text if text.strip() else " "
        ids = tok.encode(safe, max_length=512)
        emb, dt = eng.embed(ids, nthreads=8)
        emb = np.array(emb)
        local = np.array(oracle["local"][i])
        remote = np.array(oracle["remote"][i])

        def cmp(a, b):
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            return cos, float(np.max(np.abs(a - b))), float(np.mean(np.abs(a - b)))

        cos_l, max_l, mean_l = cmp(emb, local)
        cos_r, max_r, mean_r = cmp(emb, remote)
        worst_local = max(worst_local, max_l)
        worst_remote = max(worst_remote, max_r)
        print(f"[{i}] S={len(ids):4d} t={dt*1000:7.2f}ms "
              f"vs_local  cos={cos_l:.8f} max={max_l:.3e} mean={mean_l:.3e} | "
              f"vs_remote cos={cos_r:.8f} max={max_r:.3e} mean={mean_r:.3e} | {text[:40]!r}")

    print(f"\nworst max_abs vs local-pytorch oracle:  {worst_local:.3e}")
    print(f"worst max_abs vs live ryzen server:     {worst_remote:.3e}")
    eng.close()


if __name__ == "__main__":
    main()
