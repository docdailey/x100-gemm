#!/usr/bin/env python3
"""Same purpose as gen_oracle_nomic.py, for mxbai-embed-large-v1 via the /embed_batch/mxbai
endpoint."""
import json
import os

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from test_texts import TEST_TEXTS

OUT = os.path.join(os.path.dirname(__file__), "oracle_mxbai.json")
RYZEN_URL = "http://192.168.68.6:8100/embed_batch/mxbai"


def main():
    m = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    local = m.encode(TEST_TEXTS, batch_size=64, show_progress_bar=False)
    local = [e.tolist() for e in local]

    r = requests.post(RYZEN_URL, json={"texts": TEST_TEXTS}, timeout=60)
    r.raise_for_status()
    remote = r.json()["embeddings"]

    diffs = []
    for i, (a, b) in enumerate(zip(local, remote)):
        a = np.array(a); b = np.array(b)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        maxabs = float(np.max(np.abs(a - b)))
        meanabs = float(np.mean(np.abs(a - b)))
        diffs.append({"idx": i, "text": TEST_TEXTS[i][:60], "cos": cos, "max_abs": maxabs, "mean_abs": meanabs})
        print(f"[{i}] cos={cos:.8f} max_abs={maxabs:.3e} mean_abs={meanabs:.3e} text={TEST_TEXTS[i][:50]!r}")

    with open(OUT, "w") as f:
        json.dump({"texts": TEST_TEXTS, "local": local, "remote": remote, "diffs": diffs}, f)
    print(f"\nwrote {OUT}")
    worst = max(diffs, key=lambda d: d["max_abs"])
    print(f"worst max_abs={worst['max_abs']:.3e} at idx {worst['idx']}")


if __name__ == "__main__":
    main()
