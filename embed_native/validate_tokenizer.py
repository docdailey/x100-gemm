#!/usr/bin/env python3
"""Validate bert_wordpiece.py's pure-Python tokenizer against the real HF tokenizer, byte-for-
byte on input_ids, over the shared test-text set. Must pass before the native engine is trusted
to tokenize its own inputs."""
import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")

from transformers import AutoTokenizer
from bert_wordpiece import BertWordPieceTokenizer
from test_texts import TEST_TEXTS

hf_tok = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
my_tok = BertWordPieceTokenizer(os.path.join(os.path.dirname(__file__), "nomic_vocab.txt"))

n_fail = 0
for t in TEST_TEXTS:
    safe = t if t.strip() else " "
    ref = hf_tok(safe)["input_ids"]
    mine = my_tok.encode(safe)
    ok = ref == mine
    if not ok:
        n_fail += 1
        print(f"MISMATCH text={t!r}\n  ref ={ref}\n  mine={mine}")
print(f"\n{len(TEST_TEXTS)-n_fail}/{len(TEST_TEXTS)} matched, {n_fail} mismatches")
