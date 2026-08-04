#!/usr/bin/env python3
"""Runs on the K3 board. Validates libnomic_rvv.so against the oracle (both the live ryzen
server's output and the local-PyTorch run, saved earlier by gen_oracle_nomic.py as
oracle_nomic.json) over the shared TEST_TEXTS set. Reports cosine similarity + max/mean abs
diff per text, at nthreads=8 (correctness must hold at every hart count, but 8 is what
production would run; hart-count-invariance is checked separately in bench_native.py since
linear/attention op partitioning by output-channel/head is inherently order-independent per
output element -- no reduction-order dependency on hart count, unlike qwen's GQA attention
fusion, so this is expected to be exactly reproducible across nthreads, not just close)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bert_wordpiece import BertWordPieceTokenizer
from nomic_native import NomicEngine
from test_texts import TEST_TEXTS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "oracle_nomic.json")) as f:
        oracle = json.load(f)

    tok = BertWordPieceTokenizer(os.path.join(HERE, "nomic_vocab.txt"))
    eng = NomicEngine(os.path.join(HERE, "nomic.bin"))

    worst_local = worst_remote = 0.0
    for i, text in enumerate(TEST_TEXTS):
        safe = text if text.strip() else " "
        ids = tok.encode(safe)
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
