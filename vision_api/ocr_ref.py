#!/usr/bin/env python3
"""NumPy reference interpreter for an extracted OCR blob (v3).

Development/validation tool only -- never used by the deployed service. It
implements exactly the semantics ocr_rvv.c implements, so that when the C engine
disagrees with ONNX Runtime the blob is already excluded as a suspect.
"""

from __future__ import annotations

import struct

import numpy as np
from scipy.special import erf as _erf  # noqa: F401  (optional; falls back below)

OP_CONV, OP_MAXPOOL, OP_AVGPOOL, OP_FC, OP_SE = 1, 2, 3, 4, 5
ACT_NONE, ACT_RELU, ACT_GELU, ACT_HARDSWISH, ACT_HARDSIGMOID = 0, 1, 2, 3, 4
OP_FIELDS = 32
F = {k: i for i, k in enumerate(
    ["op", "in_buf", "out_buf", "add_buf", "act", "oc", "ic", "kh", "kw",
     "sh", "sw", "ph", "pw", "ceil_mode", "count_include_pad", "w_off",
     "b_off", "group", "se_off", "se_squeeze", "fc_layout", "fc_softmax"])}


class Blob:
    def __init__(self, path):
        raw = np.fromfile(path, dtype=np.uint8)
        assert raw[:4].tobytes() == b"OCR1", raw[:4].tobytes()
        (self.version, n_ops, self.n_bufs, self.final_buf, ops_off, data_off,
         n_floats, self.n_classes, self.mr, self.input_h, _, _) = struct.unpack(
            "<12i", raw[4:52].tobytes())
        self.ops = raw[ops_off:ops_off + n_ops * OP_FIELDS * 4].view(np.int32
                                                                    ).reshape(n_ops, OP_FIELDS)
        self.data = raw[data_off:data_off + n_floats * 4].view(np.float32)

    def g(self, op, key):
        return int(op[F[key]])

    def conv_weight(self, op):
        oc, ic, kh, kw = (self.g(op, k) for k in ("oc", "ic", "kh", "kw"))
        group = self.g(op, "group")
        off = self.g(op, "w_off")
        if group > 1:
            return self.data[off:off + oc * kh * kw].reshape(oc, 1, kh, kw)
        k = ic * kh * kw
        panels = (oc + self.mr - 1) // self.mr
        packed = self.data[off:off + panels * k * self.mr].reshape(panels, k, self.mr)
        flat = np.concatenate([packed[p].T for p in range(panels)], axis=0)[:oc]
        return flat.reshape(oc, ic, kh, kw)


def activate(x, act):
    if act == ACT_RELU:
        return np.maximum(x, 0.0)
    if act == ACT_GELU:
        return (x * 0.5 * (1.0 + _erf(x / np.float32(1.4142135381698608)))).astype(np.float32)
    if act == ACT_HARDSWISH:
        return (x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)).astype(np.float32)
    if act == ACT_HARDSIGMOID:
        return np.clip(x / 6.0 + 0.5, 0.0, 1.0).astype(np.float32)
    return x


def conv2d(x, w, b, sh, sw, ph, pw, group):
    ic, ih, iw = x.shape
    oc, _, kh, kw = w.shape
    oh = (ih + 2 * ph - kh) // sh + 1
    ow = (iw + 2 * pw - kw) // sw + 1
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    if group > 1:
        out = np.empty((oc, oh, ow), np.float32)
        for c in range(oc):
            acc = np.zeros((oh, ow), np.float32)
            for i in range(kh):
                for j in range(kw):
                    acc += w[c, 0, i, j] * xp[c, i:i + (oh - 1) * sh + 1:sh,
                                              j:j + (ow - 1) * sw + 1:sw]
            out[c] = acc + b[c]
        return out
    cols = np.empty((ic * kh * kw, oh * ow), np.float32)
    idx = 0
    for c in range(ic):
        for i in range(kh):
            for j in range(kw):
                cols[idx] = xp[c, i:i + (oh - 1) * sh + 1:sh,
                               j:j + (ow - 1) * sw + 1:sw].reshape(-1)
                idx += 1
    return (w.reshape(oc, -1) @ cols + b[:, None]).reshape(oc, oh, ow)


def pool_out(size, k, s, p, ceil_mode):
    num = size + 2 * p - k
    o = (-(-num // s) if ceil_mode else num // s) + 1
    if ceil_mode and (o - 1) * s >= size + p:
        o -= 1
    return o


def pool(x, kind, kh, kw, sh, sw, ph, pw, ceil_mode, cip):
    c, ih, iw = x.shape
    oh, ow = pool_out(ih, kh, sh, ph, ceil_mode), pool_out(iw, kw, sw, pw, ceil_mode)
    out = np.empty((c, oh, ow), np.float32)
    for i in range(oh):
        for j in range(ow):
            hs, ws = i * sh - ph, j * sw - pw
            h0, h1 = max(hs, 0), min(hs + kh, ih)
            w0, w1 = max(ws, 0), min(ws + kw, iw)
            win = x[:, h0:h1, w0:w1].reshape(c, -1)
            if kind == OP_MAXPOOL:
                out[:, i, j] = win.max(axis=1)
            else:
                out[:, i, j] = win.sum(axis=1) / (kh * kw if cip else win.shape[1])
    return out


def forward(blob, x):
    bufs = [None] * blob.n_bufs
    bufs[blob.g(blob.ops[0], "in_buf")] = x.astype(np.float32)
    for op in blob.ops:
        kind = blob.g(op, "op")
        src = bufs[blob.g(op, "in_buf")]
        if kind == OP_CONV:
            w = blob.conv_weight(op)
            bias = blob.data[blob.g(op, "b_off"):blob.g(op, "b_off") + blob.g(op, "oc")]
            y = conv2d(src, w, bias, *(blob.g(op, k) for k in ("sh", "sw", "ph", "pw", "group")))
            s = blob.g(op, "se_squeeze")
            if s:
                c = blob.g(op, "oc")
                off = blob.g(op, "se_off")
                w1 = blob.data[off:off + s * c].reshape(s, c); off += s * c
                b1 = blob.data[off:off + s]; off += s
                w2 = blob.data[off:off + c * s].reshape(c, s); off += c * s
                b2 = blob.data[off:off + c]
                gap = y.mean(axis=(1, 2))
                gate = np.clip((w2 @ np.maximum(w1 @ gap + b1, 0.0) + b2) / 6.0 + 0.5, 0.0, 1.0)
                y = y * gate[:, None, None]
            if blob.g(op, "add_buf") >= 0:
                y = y + bufs[blob.g(op, "add_buf")]
            y = activate(y, blob.g(op, "act"))
        elif kind in (OP_MAXPOOL, OP_AVGPOOL):
            y = pool(src, kind, *(blob.g(op, k) for k in
                                  ("kh", "kw", "sh", "sw", "ph", "pw",
                                   "ceil_mode", "count_include_pad")))
        elif kind == OP_FC:
            oc = blob.g(op, "oc")
            w = blob.conv_weight(op).reshape(oc, -1)      # (OUT, IN)
            bias = blob.data[blob.g(op, "b_off"):blob.g(op, "b_off") + oc]
            y = w @ src[:, 0, :] + bias[:, None]          # (OUT, T)
            if blob.g(op, "fc_layout"):
                y = y.T                                    # (T, OUT)
                if blob.g(op, "fc_softmax"):
                    y = y - y.max(axis=1, keepdims=True)
                    e = np.exp(y)
                    y = e / e.sum(axis=1, keepdims=True)
                y = y.astype(np.float32)[None, ...]
            else:
                y = y.astype(np.float32)[:, None, :]       # (OUT, 1, T)
        else:
            raise RuntimeError(f"unknown op {kind}")
        bufs[blob.g(op, "out_buf")] = y
    out = bufs[blob.final_buf]
    return out[0] if out.ndim == 3 else out
