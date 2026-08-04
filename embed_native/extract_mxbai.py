#!/usr/bin/env python3
"""Deterministic, re-runnable extraction of mxbai-embed-large-v1 (standard HF BertModel, no
custom remote code) to a flat FP32 weight blob + fixed binary header, mirroring extract_nomic.py
but for a genuinely different architecture family: absolute learned position embeddings (not
RoPE), exact GELU (not SwiGLU), biases on every linear projection, CLS-token pooling (not mean).

Verified facts (not assumed): pooling_mode_cls_token=true / mean=false in 1_Pooling/config.json,
modules.json has only Transformer+Pooling (no Normalize module), graphify-embedder/server.py's
own encode() call passes no normalize_embeddings=True for either model -- same "no
normalization" conclusion as nomic, confirmed the same way. config_sentence_transformers.json's
"prompts.query" prefix is a CALLER convention (asymmetric retrieval usage pattern), not part of
the model's forward pass -- the reference server never applies it (plain .encode(text)), so the
native engine matches that by tokenizing exactly what it's given, no prefix injection.
hidden_act="gelu" in HF is exact erf-based GELU (torch.nn.functional.gelu default), not the
tanh-approximate "gelu_new"/"gelu_pytorch_tanh" variant -- verified via config, not assumed from
the name alone (BERT configs have shipped either historically).
"""
import json
import os
import struct

HF_HOME = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ["HF_HOME"] = HF_HOME

import torch
from sentence_transformers import SentenceTransformer

OUT_DIR = os.path.dirname(__file__)
BIN_PATH = os.path.join(OUT_DIR, "mxbai.bin")
MANIFEST_PATH = os.path.join(OUT_DIR, "mxbai_manifest.json")
VOCAB_PATH = os.path.join(OUT_DIR, "mxbai_vocab.txt")

HEADER_FMT = "<8s16i2f48x"  # identical layout to extract_nomic.py's header


def main():
    m = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    auto = m[0].auto_model
    cfg = auto.config
    sd = auto.state_dict()

    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    hidden = cfg.hidden_size
    head_dim = hidden // n_head
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    type_vocab = cfg.type_vocab_size
    max_pos = cfg.max_position_embeddings

    assert cfg.hidden_act == "gelu"
    assert cfg.position_embedding_type == "absolute"
    assert cfg.model_type == "bert"

    tensors = []

    def add(name, t):
        arr = t.detach().to(torch.float32).contiguous().numpy()
        tensors.append((name, arr))

    add("word_embeddings", sd["embeddings.word_embeddings.weight"])
    add("position_embeddings", sd["embeddings.position_embeddings.weight"])  # [max_pos, hidden]
    add("token_type_embeddings", sd["embeddings.token_type_embeddings.weight"])
    add("emb_ln.weight", sd["embeddings.LayerNorm.weight"])
    add("emb_ln.bias", sd["embeddings.LayerNorm.bias"])

    for L in range(n_layer):
        p = f"encoder.layer.{L}."
        add(f"L{L}.attn.q.weight", sd[p + "attention.self.query.weight"])
        add(f"L{L}.attn.q.bias", sd[p + "attention.self.query.bias"])
        add(f"L{L}.attn.k.weight", sd[p + "attention.self.key.weight"])
        add(f"L{L}.attn.k.bias", sd[p + "attention.self.key.bias"])
        add(f"L{L}.attn.v.weight", sd[p + "attention.self.value.weight"])
        add(f"L{L}.attn.v.bias", sd[p + "attention.self.value.bias"])
        add(f"L{L}.attn.o.weight", sd[p + "attention.output.dense.weight"])
        add(f"L{L}.attn.o.bias", sd[p + "attention.output.dense.bias"])
        add(f"L{L}.norm1.weight", sd[p + "attention.output.LayerNorm.weight"])
        add(f"L{L}.norm1.bias", sd[p + "attention.output.LayerNorm.bias"])
        add(f"L{L}.fc1.weight", sd[p + "intermediate.dense.weight"])   # [I,H]
        add(f"L{L}.fc1.bias", sd[p + "intermediate.dense.bias"])
        add(f"L{L}.fc2.weight", sd[p + "output.dense.weight"])          # [H,I]
        add(f"L{L}.fc2.bias", sd[p + "output.dense.bias"])
        add(f"L{L}.norm2.weight", sd[p + "output.LayerNorm.weight"])
        add(f"L{L}.norm2.bias", sd[p + "output.LayerNorm.bias"])

    manifest = {
        "model": "mxbai-embed-large-v1", "hidden": hidden, "n_layer": n_layer, "n_head": n_head,
        "head_dim": head_dim, "intermediate": inter, "vocab": vocab, "type_vocab": type_vocab,
        "pad_token_id": cfg.pad_token_id, "layer_norm_eps": cfg.layer_norm_eps,
        "max_position_embeddings": max_pos, "pooling": "cls", "normalize": False,
        "activation": "gelu", "tensors": [],
    }

    header = struct.pack(
        HEADER_FMT,
        b"EMBDMXB1",
        hidden, n_layer, n_head, head_dim, inter, vocab, type_vocab, cfg.pad_token_id,
        1,   # pooling_mode: 1=cls
        0,   # normalize: 0 (verified: no Normalize module, server doesn't set it)
        0,   # use_rope: 0 (absolute position embeddings)
        max_pos,
        1,   # activation: 1=gelu (exact, erf-based)
        1, 1, 1,  # has_qkv_bias, has_out_bias, has_mlp_bias: all 1 (verified: bias on every Linear)
        float(cfg.layer_norm_eps), 0.0,  # rope_base unused
    )
    assert len(header) == 128, len(header)

    offset = 0
    with open(BIN_PATH, "wb") as f:
        f.write(header)
        for name, arr in tensors:
            nbytes = arr.nbytes
            f.write(arr.tobytes())
            manifest["tensors"].append({"name": name, "shape": list(arr.shape), "offset": 128 + offset, "nbytes": nbytes})
            offset += nbytes

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    vocab_dict = m.tokenizer.get_vocab()
    with open(VOCAB_PATH, "w") as f:
        for tok, idx in sorted(vocab_dict.items(), key=lambda kv: kv[1]):
            f.write(tok + "\n")

    print(f"wrote {BIN_PATH} ({(128+offset)/1e6:.1f} MB), {MANIFEST_PATH}, {VOCAB_PATH}")
    print(f"vocab size from tokenizer: {len(vocab_dict)} (config vocab_size={vocab})")
    print(f"do_lower_case: {getattr(m.tokenizer, 'do_lower_case', 'unknown')}")
    print(f"special tokens: cls={m.tokenizer.cls_token!r}:{m.tokenizer.cls_token_id} "
          f"sep={m.tokenizer.sep_token!r}:{m.tokenizer.sep_token_id} "
          f"pad={m.tokenizer.pad_token!r}:{m.tokenizer.pad_token_id} "
          f"unk={m.tokenizer.unk_token!r}:{m.tokenizer.unk_token_id}")
    print(f"max_seq_length: {m.max_seq_length}")


if __name__ == "__main__":
    main()
