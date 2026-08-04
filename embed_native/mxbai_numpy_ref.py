#!/usr/bin/env python3
"""Pure NumPy reimplementation of the algorithm mxbai_rvv.c will implement, reading mxbai.bin.
Development oracle only, same purpose as nomic_numpy_ref.py: catch logic bugs in a fast Python
loop before touching the K3 board."""
import struct
import sys

import numpy as np
from scipy.special import erf


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:8] == b"EMBDMXB1", raw[:8]
    ints = struct.unpack("<16i", raw[8:72])
    flts = struct.unpack("<2f", raw[72:80])
    (hidden, n_layer, n_head, head_dim, inter, vocab, type_vocab, pad_id,
     pooling_mode, normalize, use_rope, max_pos, activation,
     has_qkv_bias, has_out_bias, has_mlp_bias) = ints
    eps, _rope_base = flts
    off = 128
    buf = np.frombuffer(raw, dtype=np.float32)

    def take(n):
        nonlocal off
        start = off // 4
        arr = buf[start:start + n]
        off += n * 4
        return arr

    word_emb = take(vocab * hidden).reshape(vocab, hidden)
    pos_emb = take(max_pos * hidden).reshape(max_pos, hidden)
    type_emb = take(type_vocab * hidden).reshape(type_vocab, hidden)
    emb_ln_w = take(hidden); emb_ln_b = take(hidden)
    layers = []
    for _ in range(n_layer):
        qw = take(hidden * hidden).reshape(hidden, hidden); qb = take(hidden)
        kw = take(hidden * hidden).reshape(hidden, hidden); kb = take(hidden)
        vw = take(hidden * hidden).reshape(hidden, hidden); vb = take(hidden)
        ow = take(hidden * hidden).reshape(hidden, hidden); ob = take(hidden)
        n1w = take(hidden); n1b = take(hidden)
        fc1w = take(inter * hidden).reshape(inter, hidden); fc1b = take(inter)
        fc2w = take(hidden * inter).reshape(hidden, inter); fc2b = take(hidden)
        n2w = take(hidden); n2b = take(hidden)
        layers.append(dict(qw=qw, qb=qb, kw=kw, kb=kb, vw=vw, vb=vb, ow=ow, ob=ob,
                            n1w=n1w, n1b=n1b, fc1w=fc1w, fc1b=fc1b, fc2w=fc2w, fc2b=fc2b,
                            n2w=n2w, n2b=n2b))
    return dict(hidden=hidden, n_layer=n_layer, n_head=n_head, head_dim=head_dim, inter=inter,
                eps=eps, word_emb=word_emb, pos_emb=pos_emb, type_emb=type_emb,
                emb_ln_w=emb_ln_w, emb_ln_b=emb_ln_b, layers=layers, normalize=normalize)


def layernorm(x, w, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu_exact(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def forward(m, ids):
    S = len(ids)
    H, nh, hd = m["hidden"], m["n_head"], m["head_dim"]
    x = m["word_emb"][ids] + m["pos_emb"][:S] + m["type_emb"][0]
    x = layernorm(x, m["emb_ln_w"], m["emb_ln_b"], m["eps"])

    for ly in m["layers"]:
        q = (x @ ly["qw"].T + ly["qb"]).reshape(S, nh, hd)
        k = (x @ ly["kw"].T + ly["kb"]).reshape(S, nh, hd)
        v = (x @ ly["vw"].T + ly["vb"]).reshape(S, nh, hd)
        scale = 1.0 / np.sqrt(hd)
        scores = np.einsum("ihd,jhd->hij", q, k) * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores); w = w / w.sum(axis=-1, keepdims=True)
        attn = np.einsum("hij,jhd->ihd", w, v).reshape(S, H)
        proj = attn @ ly["ow"].T + ly["ob"]
        x = layernorm(proj + x, ly["n1w"], ly["n1b"], m["eps"])

        h = gelu_exact(x @ ly["fc1w"].T + ly["fc1b"])
        out = h @ ly["fc2w"].T + ly["fc2b"]
        x = layernorm(out + x, ly["n2w"], ly["n2b"], m["eps"])

    pooled = x[0]  # CLS token
    if m["normalize"]:
        pooled = pooled / np.linalg.norm(pooled)
    return pooled


if __name__ == "__main__":
    import json
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from bert_wordpiece import BertWordPieceTokenizer
    from test_texts import TEST_TEXTS

    m = load(os.path.join(os.path.dirname(__file__), "mxbai.bin"))
    tok = BertWordPieceTokenizer(os.path.join(os.path.dirname(__file__), "mxbai_vocab.txt"))

    with open(os.path.join(os.path.dirname(__file__), "oracle_mxbai.json")) as f:
        oracle = json.load(f)

    worst = 0.0
    for i, text in enumerate(TEST_TEXTS):
        safe = text if text.strip() else " "
        ids = np.array(tok.encode(safe, max_length=512), dtype=np.int64)
        emb = forward(m, ids)
        ref = np.array(oracle["local"][i])
        cos = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-12))
        maxabs = float(np.max(np.abs(emb - ref)))
        worst = max(worst, maxabs)
        print(f"[{i}] S={len(ids)} cos={cos:.8f} max_abs={maxabs:.3e} text={text[:50]!r}")
    print(f"\nworst max_abs={worst:.3e}")
