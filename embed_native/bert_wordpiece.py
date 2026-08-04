#!/usr/bin/env python3
"""Dependency-free reimplementation of HF's BertTokenizer (BasicTokenizer + WordpieceTokenizer),
for on-device tokenization on the K3 board (no `tokenizers`/`transformers` package there --
verified absent, see progress doc). Ported line-for-line from transformers'
models/bert/tokenization_bert.py behavior for do_lower_case=True, tokenize_chinese_chars=True,
strip_accents=None (follows do_lower_case). Validated byte-for-byte against the real HF
tokenizer on the project's test-text set (see validate_tokenizer.py) before being trusted.
"""
import unicodedata


def _is_whitespace(c):
    if c in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(c) == "Zs"


def _is_control(c):
    if c in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(c).startswith("C")


def _is_punctuation(c):
    cp = ord(c)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(c).startswith("P")


def _is_chinese_char(cp):
    return (
        (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
        (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
        (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
        (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)
    )


def _clean_text(text):
    out = []
    for c in text:
        cp = ord(c)
        if cp == 0 or cp == 0xFFFD or _is_control(c):
            continue
        out.append(" " if _is_whitespace(c) else c)
    return "".join(out)


def _tokenize_chinese_chars(text):
    out = []
    for c in text:
        cp = ord(c)
        if _is_chinese_char(cp):
            out.append(" "); out.append(c); out.append(" ")
        else:
            out.append(c)
    return "".join(out)


def _strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _whitespace_tokenize(text):
    return text.strip().split()


def _run_split_on_punc(text):
    chars = list(text)
    i = 0
    out = [[]]
    while i < len(chars):
        c = chars[i]
        if _is_punctuation(c):
            out.append([c]); out.append([])
        else:
            out[-1].append(c)
        i += 1
    return ["".join(x) for x in out if x]


def basic_tokenize(text, do_lower_case=True):
    text = _clean_text(text)
    text = _tokenize_chinese_chars(text)
    orig_tokens = _whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
        if do_lower_case:
            token = token.lower()
            token = _strip_accents(token)
        split_tokens.extend(_run_split_on_punc(token))
    return _whitespace_tokenize(" ".join(split_tokens))


class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab  # dict token -> id
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]
        out = []
        start = 0
        n = len(chars)
        while start < n:
            end = n
            cur = None
            while start < end:
                sub = "".join(chars[start:end])
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return [self.unk_token]
            out.append(cur)
            start = end
        return out


class BertWordPieceTokenizer:
    def __init__(self, vocab_path, do_lower_case=True, cls_token="[CLS]", sep_token="[SEP]",
                 pad_token="[PAD]", unk_token="[UNK]"):
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.wordpiece = WordpieceTokenizer(self.vocab, unk_token=unk_token)
        self.cls_token, self.sep_token = cls_token, sep_token
        self.pad_token, self.unk_token = pad_token, unk_token
        self.cls_id = self.vocab[cls_token]
        self.sep_id = self.vocab[sep_token]
        self.pad_id = self.vocab[pad_token]
        self.unk_id = self.vocab[unk_token]

    def tokenize(self, text):
        out = []
        for tok in basic_tokenize(text, self.do_lower_case):
            out.extend(self.wordpiece.tokenize(tok))
        return out

    def encode(self, text, max_length=None):
        """Returns input_ids including [CLS]/[SEP], BERT-style single-segment encoding."""
        pieces = self.tokenize(text)
        ids = [self.cls_id] + [self.vocab.get(p, self.unk_id) for p in pieces] + [self.sep_id]
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.sep_id]
        return ids


if __name__ == "__main__":
    import sys
    tok = BertWordPieceTokenizer(sys.argv[1] if len(sys.argv) > 1 else "nomic_vocab.txt")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        ids = tok.encode(line)
        print(ids)
