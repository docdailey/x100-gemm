#!/usr/bin/env python3
"""Pure NumPy reimplementation of exactly the algorithm nomic_rvv.c implements, reading the same
nomic.bin the C engine reads. Development oracle only: proves the extraction + algorithm design
are right independently of the C/RVV code, in a fast Python iteration loop before touching the
K3 board (round-tripping scp+ssh+build is much slower than fixing logic bugs here first).
"""
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    magic = raw[:8]
    assert magic == b"EMBDNOM1", magic
    ints = struct.unpack("<16i", raw[8:8 + 64])
    flts = struct.unpack("<2f", raw[72:80])
    (hidden, n_layer, n_head, head_dim, inter, vocab, type_vocab, pad_id,
     pooling_mode, normalize, use_rope, max_pos, activation,
     has_qkv_bias, has_out_bias, has_mlp_bias) = ints
    eps, rope_base = flts
    off = 128
    buf = np.frombuffer(raw, dtype=np.float32)

    def take(n):
        nonlocal off
        start = off // 4
        arr = buf[start:start + n]
        off += n * 4
        return arr

    word_emb = take(vocab * hidden).reshape(vocab, hidden)
    type_emb = take(type_vocab * hidden).reshape(type_vocab, hidden)
    emb_ln_w = take(hidden)
    emb_ln_b = take(hidden)
    layers = []
    for _ in range(n_layer):
        q = take(hidden * hidden).reshape(hidden, hidden)
        k = take(hidden * hidden).reshape(hidden, hidden)
        v = take(hidden * hidden).reshape(hidden, hidden)
        o = take(hidden * hidden).reshape(hidden, hidden)
        fc11 = take(inter * hidden).reshape(inter, hidden)
        fc12 = take(inter * hidden).reshape(inter, hidden)
        fc2 = take(hidden * inter).reshape(hidden, inter)
        n1w = take(hidden); n1b = take(hidden)
        n2w = take(hidden); n2b = take(hidden)
        layers.append(dict(q=q, k=k, v=v, o=o, fc11=fc11, fc12=fc12, fc2=fc2,
                            n1w=n1w, n1b=n1b, n2w=n2w, n2b=n2b))
    return dict(hidden=hidden, n_layer=n_layer, n_head=n_head, head_dim=head_dim,
                inter=inter, eps=eps, rope_base=rope_base, word_emb=word_emb,
                type_emb=type_emb, emb_ln_w=emb_ln_w, emb_ln_b=emb_ln_b, layers=layers,
                normalize=normalize)


def layernorm(x, w, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def rope(x, head_dim, base):
    # x: [S, n_head, head_dim]
    S = x.shape[0]
    i = np.arange(head_dim // 2)
    inv_freq = base ** (-2.0 * i / head_dim)
    pos = np.arange(S)[:, None]
    ang = pos * inv_freq[None, :]  # [S, hd/2]
    cos = np.cos(ang)[:, None, :]  # [S,1,hd/2]
    sin = np.sin(ang)[:, None, :]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    out = np.empty_like(x)
    out[..., :head_dim // 2] = x1 * cos - x2 * sin
    out[..., head_dim // 2:] = x1 * sin + x2 * cos
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def forward(m, ids):
    S = len(ids)
    H, nh, hd = m["hidden"], m["n_head"], m["head_dim"]
    x = m["word_emb"][ids] + m["type_emb"][0]
    x = layernorm(x, m["emb_ln_w"], m["emb_ln_b"], m["eps"])

    for ly in m["layers"]:
        q = x @ ly["q"].T
        k = x @ ly["k"].T
        v = x @ ly["v"].T
        q = q.reshape(S, nh, hd); k = k.reshape(S, nh, hd); v = v.reshape(S, nh, hd)
        q = rope(q, hd, m["rope_base"]); k = rope(k, hd, m["rope_base"])
        # non-causal attention per head
        scale = 1.0 / np.sqrt(hd)
        scores = np.einsum("ihd,jhd->hij", q, k) * scale  # [nh,S,S]
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        attn = np.einsum("hij,jhd->ihd", w, v).reshape(S, H)
        proj = attn @ ly["o"].T
        x = layernorm(proj + x, ly["n1w"], ly["n1b"], m["eps"])

        val = x @ ly["fc11"].T
        gate = x @ ly["fc12"].T
        h = val * silu(gate)
        mlp_out = h @ ly["fc2"].T
        x = layernorm(mlp_out + x, ly["n2w"], ly["n2b"], m["eps"])

    pooled = x.mean(axis=0)
    if m["normalize"]:
        pooled = pooled / np.linalg.norm(pooled)
    return pooled


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from bert_wordpiece import BertWordPieceTokenizer
    from test_texts import TEST_TEXTS

    m = load(os.path.join(os.path.dirname(__file__), "nomic.bin"))
    tok = BertWordPieceTokenizer(os.path.join(os.path.dirname(__file__), "nomic_vocab.txt"))

    import json
    with open(os.path.join(os.path.dirname(__file__), "oracle_nomic.json")) as f:
        oracle = json.load(f)

    worst = 0.0
    for i, text in enumerate(TEST_TEXTS):
        safe = text if text.strip() else " "
        ids = np.array(tok.encode(safe), dtype=np.int64)
        emb = forward(m, ids)
        ref = np.array(oracle["local"][i])
        cos = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-12))
        maxabs = float(np.max(np.abs(emb - ref)))
        worst = max(worst, maxabs)
        print(f"[{i}] S={len(ids)} cos={cos:.8f} max_abs={maxabs:.3e} text={text[:50]!r}")
    print(f"\nworst max_abs={worst:.3e}")
