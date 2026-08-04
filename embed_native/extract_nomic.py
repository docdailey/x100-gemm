#!/usr/bin/env python3
"""Deterministic, re-runnable extraction of nomic-embed-text-v1.5 (NomicBertModel) to a flat
FP32 weight blob + JSON manifest consumable by the native RVV engine (nomic_rvv.c on the K3).

Architecture verified against the actual forward-pass code (not just config.json):
/home/willy/.cache/huggingface/modules/transformers_modules/nomic_hyphen_ai/
  nomic_hyphen_bert_hyphen_2048/7710840340a098cfb869c4f65e87cf2b1b70caca/modeling_hf_nomic_bert.py
(read on ryzen, read-only, not the graphify-embedder venv). Key facts pulled from that file,
not assumed: prenorm=False (post-norm: x=LN(x+Attn(x)); x=LN(x+MLP(x))), emb_ln applied once
right after word+token_type embeddings, RoPE is GPT-NeoX-style (rotate first/second half,
non-interleaved) over the full head_dim, MLP is fc2(fc11(x) * silu(fc12(x))) with NO norm_layer
(norm_layer=False default, so NomciBertGatedMLP.norm is nn.Identity()), no biases anywhere
except LayerNorm weight/bias. Pooling is mean-over-tokens (attention-mask-weighted), NOT the
built-in NomicBertPooler (that's unused by sentence-transformers). No final L2 normalize:
confirmed both by the absence of a "2_Normalize" module in modules.json and by
graphify-embedder/server.py calling .encode() without normalize_embeddings=True.
"""
import json
import os
import struct

HF_HOME = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ["HF_HOME"] = HF_HOME

import torch
from sentence_transformers import SentenceTransformer

OUT_DIR = os.path.dirname(__file__)
BIN_PATH = os.path.join(OUT_DIR, "nomic.bin")
MANIFEST_PATH = os.path.join(OUT_DIR, "nomic_manifest.json")
VOCAB_PATH = os.path.join(OUT_DIR, "nomic_vocab.txt")

# Binary layout consumed directly by nomic_rvv.c (embed_native/engine.c), no JSON parsing on
# the K3 board. Fixed 128-byte header (see engine.c's EmbedHeader / read_header for the mirror
# of this exact field order), then tensor data immediately after, in the fixed order `add()`
# below appends them -- both sides just need to agree on order + shapes, which are fully
# determined by the header's integer fields (hidden/n_layer/n_head/intermediate/vocab/...).
HEADER_FMT = "<8s16i2f48x"  # magic, 16x int32, 2x float32, 48 bytes reserved/zero = 128 bytes total



def main():
    m = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu")
    auto = m[0].auto_model
    cfg = auto.config
    sd = auto.state_dict()

    n_layer = cfg.n_layer
    n_head = cfg.n_head
    hidden = cfg.n_embd
    head_dim = hidden // n_head
    inter = cfg.n_inner
    vocab = cfg.vocab_size
    type_vocab = cfg.type_vocab_size

    assert cfg.prenorm is False
    assert cfg.qkv_proj_bias is False
    assert cfg.mlp_fc1_bias is False and cfg.mlp_fc2_bias is False
    assert cfg.rotary_emb_interleaved is False
    assert abs(cfg.rotary_emb_fraction - 1.0) < 1e-9
    assert cfg.causal is False

    tensors = []  # (name, np.float32 flat array)

    def add(name, t):
        arr = t.detach().to(torch.float32).contiguous().numpy()
        tensors.append((name, arr))

    add("word_embeddings", sd["embeddings.word_embeddings.weight"])       # [vocab, hidden]
    add("token_type_embeddings", sd["embeddings.token_type_embeddings.weight"])  # [type_vocab, hidden]
    add("emb_ln.weight", sd["emb_ln.weight"])
    add("emb_ln.bias", sd["emb_ln.bias"])

    for L in range(n_layer):
        p = f"encoder.layers.{L}."
        wqkv = sd[p + "attn.Wqkv.weight"]  # [3*hidden, hidden] = concat(Q,K,V) rows
        q_w, k_w, v_w = wqkv.split(hidden, dim=0)
        add(f"L{L}.attn.q.weight", q_w)
        add(f"L{L}.attn.k.weight", k_w)
        add(f"L{L}.attn.v.weight", v_w)
        add(f"L{L}.attn.o.weight", sd[p + "attn.out_proj.weight"])
        add(f"L{L}.mlp.fc11.weight", sd[p + "mlp.fc11.weight"])  # value/up
        add(f"L{L}.mlp.fc12.weight", sd[p + "mlp.fc12.weight"])  # gate
        add(f"L{L}.mlp.fc2.weight", sd[p + "mlp.fc2.weight"])
        add(f"L{L}.norm1.weight", sd[p + "norm1.weight"])
        add(f"L{L}.norm1.bias", sd[p + "norm1.bias"])
        add(f"L{L}.norm2.weight", sd[p + "norm2.weight"])
        add(f"L{L}.norm2.bias", sd[p + "norm2.bias"])

    manifest = {
        "model": "nomic-embed-text-v1.5",
        "hidden": hidden,
        "n_layer": n_layer,
        "n_head": n_head,
        "head_dim": head_dim,
        "intermediate": inter,
        "vocab": vocab,
        "type_vocab": type_vocab,
        "pad_token_id": cfg.pad_token_id,
        "layer_norm_eps": cfg.layer_norm_epsilon,
        "rope_base": cfg.rotary_emb_base,
        "pooling": "mean",
        "normalize": False,
        "tensors": [],
    }

    header = struct.pack(
        HEADER_FMT,
        b"EMBDNOM1",
        hidden, n_layer, n_head, head_dim, inter, vocab, type_vocab, cfg.pad_token_id,
        0,   # pooling_mode: 0=mean
        0,   # normalize: 0=no final L2 norm (confirmed via server.py, no Normalize ST module)
        1,   # use_rope: 1 (nomic is RoPE-only, no absolute position embeddings)
        0,   # max_position_embeddings: unused when use_rope=1
        0,   # activation: 0=swiglu/silu
        0, 0, 0,  # has_qkv_bias, has_out_bias, has_mlp_bias: all 0 (verified no biases)
        float(cfg.layer_norm_epsilon), float(cfg.rotary_emb_base),
    )
    assert len(header) == 128, len(header)

    offset = 0
    with open(BIN_PATH, "wb") as f:
        f.write(header)
        for name, arr in tensors:
            nbytes = arr.nbytes
            f.write(arr.tobytes())
            manifest["tensors"].append({
                "name": name, "shape": list(arr.shape),
                "offset": 128 + offset, "nbytes": nbytes,
            })
            offset += nbytes

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # Vocab for the pure-Python WordPiece tokenizer used on-device (no `tokenizers`/`transformers`
    # package on the K3 board). tokenizer.json/vocab.txt come from the HF snapshot.
    snap_dir = os.path.dirname(m.tokenizer.vocab_file if hasattr(m.tokenizer, "vocab_file") else "")
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
