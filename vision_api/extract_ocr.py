#!/usr/bin/env python3
"""Extract a PaddleOCR recognizer ONNX into a flat FP32 blob + linear op program.

Handles both recognizers used on this board from one code path:
  * PP-OCRv6 Tiny  (PP-LCNet-style: depthwise/pointwise MBConv, GELU, SE,
                    HardSwish 1D sequence mixer, two-layer FC head, 6906 classes)
  * Rosetta        (ResNet34-vd + CTC, 37 classes)

Everything the runtime should not have to think about is resolved here:
  - BatchNormalization (2d and 1d) is folded into the preceding conv.
  - A separate `Add` of a [1,C,1,1] constant is folded in as the conv bias.
  - Residual adds and activations are fused onto the conv that produces them.
  - Squeeze/Unsqueeze/Transpose that are no-ops in an NCHW buffer with H==1 are
    dropped; the runtime asserts H==1 where that assumption is load-bearing.
  - Squeeze-and-Excite collapses to a single op (its convs are 1x1 on a 1x1
    spatial map, i.e. plain GEMVs).

Output is one self-describing binary (header + op table + packed float32) that
ocr_rvv.c mmaps directly, plus a JSON manifest.

Usage: python3 extract_ocr.py <model.onnx> <out.bin>
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"OCR1"
VERSION = 3
OP_CONV, OP_MAXPOOL, OP_AVGPOOL, OP_FC, OP_SE = 1, 2, 3, 4, 5
ACT_NONE, ACT_RELU, ACT_GELU, ACT_HARDSWISH, ACT_HARDSIGMOID = 0, 1, 2, 3, 4
OP_FIELDS = 32
MR = 8

SQRT2 = 1.4142135381698608


def resolve_tensors(graph):
    values = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            (attr,) = [a for a in node.attribute if a.name == "value"]
            values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def attrs_of(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.TENSOR:
            out[a.name] = numpy_helper.to_array(a.t)
    return out


def strip_identity(nodes):
    """Drop Identity nodes and rewrite every reference through them."""
    alias = {}
    kept = []
    for node in nodes:
        if node.op_type == "Identity":
            alias[node.output[0]] = node.input[0]
            continue
        kept.append(node)

    def root(name):
        seen = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    for node in kept:
        for i, name in enumerate(node.input):
            node.input[i] = root(name)
    return kept, root


class Program:
    def __init__(self, values):
        self.values = values
        self.ops = []
        self.blobs = []
        self.blob_len = 0
        self.buf_of = {}
        self.users = {}
        self.free = []
        self.n_bufs = 0
        self.notes = []

    def add_data(self, array):
        array = np.ascontiguousarray(array, dtype=np.float32)
        offset = self.blob_len
        self.blobs.append(array.reshape(-1))
        self.blob_len += array.size
        return offset

    def new_buf(self):
        if self.free:
            return self.free.pop()
        buf = self.n_bufs
        self.n_bufs += 1
        return buf

    def release(self, name):
        buf = self.buf_of.get(name)
        if buf is None:
            return
        self.users[buf] -= 1
        if self.users[buf] == 0 and buf not in self.free:
            self.free.append(buf)

    def bind(self, name, buf, readers):
        self.buf_of[name] = buf
        self.users[buf] = readers

    def alias(self, dst, src, readers):
        buf = self.buf_of[src]
        self.buf_of[dst] = buf
        self.users[buf] += readers - 1

    def emit(self, **kw):
        record = dict(op=0, in_buf=0, out_buf=0, add_buf=-1, act=ACT_NONE,
                      oc=0, ic=0, kh=1, kw=1, sh=1, sw=1, ph=0, pw=0,
                      ceil_mode=0, count_include_pad=0, w_off=0, b_off=0,
                      group=1, se_off=0, se_squeeze=0, fc_layout=0, fc_softmax=0,
                      name="")
        record.update(kw)
        self.ops.append(record)


def count_readers(nodes, graph, const_names):
    readers = {}
    for node in nodes:
        for name in node.input:
            if name in const_names:
                continue
            readers[name] = readers.get(name, 0) + 1
    for out in graph.output:
        readers[out.name] = readers.get(out.name, 0) + 1
    return readers


def fold_bn(weight, scale, offset, mean, var, eps):
    factor = scale / np.sqrt(var + eps)
    return (weight * factor.reshape(-1, *([1] * (weight.ndim - 1)))).astype(np.float32), \
           (offset - mean * factor).astype(np.float32)


def pack_conv_weight(w, group):
    """Dense conv -> MR-row panels (k-major). Depthwise -> plain (C, KH*KW)."""
    oc = w.shape[0]
    if group > 1:
        assert group == oc and w.shape[1] == 1, f"unsupported grouped conv {w.shape} group={group}"
        return w.reshape(oc, -1).astype(np.float32)
    k = int(np.prod(w.shape[1:]))
    flat = w.reshape(oc, k)
    panels = (oc + MR - 1) // MR
    packed = np.zeros((panels, k, MR), dtype=np.float32)
    for p in range(panels):
        rows = min(MR, oc - p * MR)
        packed[p, :, :rows] = flat[p * MR:p * MR + rows].T
    return packed


class Extractor:
    def __init__(self, graph):
        self.graph = graph
        self.values = resolve_tensors(graph)
        nodes = [n for n in graph.node if n.op_type != "Constant"]
        self.nodes, _ = strip_identity(nodes)
        self.readers = count_readers(self.nodes, graph, set(self.values))
        self.p = Program(self.values)
        self.consumed = set()

    def is_const(self, name):
        return name in self.values

    def const(self, name):
        return self.values[name]

    # ---- pattern helpers -------------------------------------------------
    def match_gelu(self, i, src):
        """Div(src,sqrt2) Erf Add(1) Mul(src,.) Mul(0.5) -> end index, or None."""
        n = self.nodes
        if i + 4 >= len(n):
            return None
        d, e, a, m1, m2 = n[i:i + 5]
        if d.op_type != "Div" or d.input[0] != src or not self.is_const(d.input[1]):
            return None
        if not np.isclose(float(self.const(d.input[1])), SQRT2):
            return None
        if e.op_type != "Erf" or e.input[0] != d.output[0]:
            return None
        if a.op_type != "Add" or a.input[0] != e.output[0] or not self.is_const(a.input[1]):
            return None
        if not np.isclose(float(self.const(a.input[1])), 1.0):
            return None
        if m1.op_type != "Mul" or set(m1.input) != {src, a.output[0]}:
            return None
        if m2.op_type != "Mul" or m2.input[0] != m1.output[0] or not self.is_const(m2.input[1]):
            return None
        if not np.isclose(float(self.const(m2.input[1])), 0.5):
            return None
        return i + 5, m2.output[0]

    def match_hardswish(self, i, src):
        """HardSigmoid(src) then Mul(hs, src)."""
        n = self.nodes
        if i + 1 >= len(n):
            return None
        hs, mul = n[i], n[i + 1]
        if hs.op_type != "HardSigmoid" or hs.input[0] != src:
            return None
        at = attrs_of(hs)
        assert np.isclose(at.get("alpha", 0.2), 1.0 / 6, atol=1e-4) and \
               np.isclose(at.get("beta", 0.5), 0.5), f"unexpected HardSigmoid {at}"
        if mul.op_type != "Mul" or set(mul.input) != {hs.output[0], src}:
            return None
        return i + 2, mul.output[0]

    def match_se(self, i, src):
        """ReduceMean -> Conv1x1+bias -> Relu -> Conv1x1+bias -> HardSigmoid -> Mul."""
        n = self.nodes
        if i + 6 >= len(n):
            return None
        rm, c1, a1, rl, c2, a2, hs, mul = n[i:i + 8]
        if rm.op_type != "ReduceMean" or rm.input[0] != src:
            return None
        if attrs_of(rm).get("axes") != [2, 3]:
            return None
        if c1.op_type != "Conv" or a1.op_type != "Add" or rl.op_type != "Relu":
            return None
        if c2.op_type != "Conv" or a2.op_type != "Add" or hs.op_type != "HardSigmoid":
            return None
        if mul.op_type != "Mul" or set(mul.input) != {src, hs.output[0]}:
            return None
        w1 = self.const(c1.input[1])[:, :, 0, 0]          # (S, C)
        b1 = self.const(a1.input[1]).reshape(-1)
        w2 = self.const(c2.input[1])[:, :, 0, 0]          # (C, S)
        b2 = self.const(a2.input[1]).reshape(-1)
        return i + 8, mul.output[0], w1, b1, w2, b2

    def trailing_act(self, i, src):
        """Any activation applied directly to `src`. Returns (act, end, out)."""
        n = self.nodes
        if i < len(n) and n[i].op_type == "Relu" and n[i].input[0] == src:
            return ACT_RELU, i + 1, n[i].output[0]
        hit = self.match_gelu(i, src)
        if hit:
            return (ACT_GELU,) + hit
        hit = self.match_hardswish(i, src)
        if hit:
            return (ACT_HARDSWISH,) + hit
        return ACT_NONE, i, src

    # ---- main walk -------------------------------------------------------
    def run(self):
        p = self.p
        n = self.nodes
        input_name = self.graph.input[0].name
        p.bind(input_name, p.new_buf(), self.readers.get(input_name, 1))

        i = 0
        while i < len(n):
            node = n[i]
            op = node.op_type
            a = attrs_of(node)

            if op == "Conv":
                i = self.emit_conv(i, node, a)
                continue

            if op in ("MaxPool", "AveragePool"):
                src = node.input[0]
                pads = a.get("pads", [0, 0, 0, 0])
                assert pads[0] == pads[2] and pads[1] == pads[3], "asymmetric pool pad"
                k = a["kernel_shape"]
                s = a.get("strides", [1, 1])
                dst = p.new_buf()
                src_buf = p.buf_of[src]
                p.release(src)
                p.emit(op=OP_MAXPOOL if op == "MaxPool" else OP_AVGPOOL,
                       name=node.output[0], in_buf=src_buf, out_buf=dst,
                       kh=k[0], kw=k[1], sh=s[0], sw=s[1], ph=pads[0], pw=pads[1],
                       ceil_mode=a.get("ceil_mode", 0),
                       count_include_pad=a.get("count_include_pad", 0))
                p.bind(node.output[0], dst, self.readers.get(node.output[0], 1))
                i += 1
                continue

            if op in ("Squeeze", "Unsqueeze", "Transpose"):
                i = self.emit_reshape_like(i, node, a)
                continue

            if op == "MatMul":
                i = self.emit_fc(i, node)
                continue

            if op == "Mul":
                # A Mul by a scalar 1.0 is a paddle2onnx export artifact.
                other = [t for t in node.input if self.is_const(t)]
                assert other and np.allclose(self.const(other[0]), 1.0), \
                    f"unexpected standalone Mul at {i}: {node.input}"
                p.notes.append("dropped paddle2onnx Mul-by-1.0 artifact")
                p.alias(node.output[0], [t for t in node.input if t not in other][0],
                        self.readers.get(node.output[0], 1))
                i += 1
                continue

            if op == "Reshape":
                shape = self.const(node.input[1])
                p.notes.append(f"dropped trailing Reshape to {np.asarray(shape).tolist()} "
                               "(width-specific; runtime derives steps from stride arithmetic)")
                p.alias(node.output[0], node.input[0], self.readers.get(node.output[0], 1))
                i += 1
                continue

            raise RuntimeError(f"unhandled op {op} at cleaned-node index {i}")

        return p, p.buf_of[self.graph.output[0].name]

    def emit_reshape_like(self, i, node, a):
        """Squeeze/Unsqueeze on axis 2 and cancelling Transposes are no-ops here.

        The buffer model is NCHW with an explicit (c, h, w); an axis-2
        squeeze/unsqueeze is only valid when H == 1, which the runtime shape pass
        checks. A Transpose(0,2,1) is only a no-op if a second one undoes it --
        the final lone Transpose feeds the FC head and is absorbed there.
        """
        p = self.p
        if node.op_type in ("Squeeze", "Unsqueeze"):
            axes = a.get("axes")
            if axes is None and len(node.input) > 1:
                axes = np.asarray(self.const(node.input[1])).reshape(-1).tolist()
            assert list(axes) == [2], f"unsupported {node.op_type} axes {axes}"
            p.alias(node.output[0], node.input[0], self.readers.get(node.output[0], 1))
            return i + 1
        assert a["perm"] == [0, 2, 1]
        nxt = self.nodes[i + 1] if i + 1 < len(self.nodes) else None
        if nxt is not None and nxt.op_type == "Transpose" and \
                nxt.input[0] == node.output[0] and attrs_of(nxt)["perm"] == [0, 2, 1]:
            self.p.notes.append("cancelled a back-to-back Transpose(0,2,1) pair")
            p.alias(node.output[0], node.input[0], 1)
            p.alias(nxt.output[0], node.output[0], self.readers.get(nxt.output[0], 1))
            return i + 2
        # Lone transpose: channel-major (C,1,T) -> row-major (1,T,C). The FC that
        # consumes it absorbs the transpose via its fc_layout flag.
        p.alias(node.output[0], node.input[0], self.readers.get(node.output[0], 1))
        return i + 1

    def emit_conv(self, i, node, a):
        p, n = self.p, self.nodes
        group = a.get("group", 1)
        assert a.get("dilations", [1, 1]) == [1, 1], "dilated conv unsupported"
        w = self.const(node.input[1])
        pads = a.get("pads", [0, 0, 0, 0])
        assert pads[0] == pads[2] and pads[1] == pads[3], "asymmetric conv pad"
        strides = a.get("strides", [1, 1])
        oc = int(w.shape[0])

        j = i + 1
        bias = np.zeros(oc, np.float32)
        out_name = node.output[0]

        # BatchNormalization (2d or 1d) or an explicit [1,C,1,1] bias Add.
        if j < len(n) and n[j].op_type == "BatchNormalization" and n[j].input[0] == out_name:
            bn = n[j]
            eps = attrs_of(bn)["epsilon"]
            scale, offset, mean, var = (self.const(t) for t in bn.input[1:5])
            w, bias = fold_bn(w, scale, offset, mean, var, eps)
            out_name = bn.output[0]
            j += 1
        elif j < len(n) and n[j].op_type == "Add" and n[j].input[0] == out_name \
                and self.is_const(n[j].input[1]):
            bias = self.const(n[j].input[1]).reshape(-1).astype(np.float32)
            assert bias.size == oc, f"bias {bias.size} != oc {oc}"
            out_name = n[j].output[0]
            j += 1

        # Optional Squeeze/Unsqueeze around 1D convs -- no-ops in NCHW with H==1.
        while j < len(n) and n[j].op_type in ("Squeeze", "Unsqueeze") \
                and n[j].input[0] == out_name:
            axes = attrs_of(n[j]).get("axes")
            if axes is None and len(n[j].input) > 1:
                axes = np.asarray(self.const(n[j].input[1])).reshape(-1).tolist()
            assert list(axes) == [2]
            out_name = n[j].output[0]
            j += 1
            if j < len(n) and n[j].op_type == "BatchNormalization" and n[j].input[0] == out_name:
                bn = n[j]
                eps = attrs_of(bn)["epsilon"]
                scale, offset, mean, var = (self.const(t) for t in bn.input[1:5])
                # BN(conv(x) + b) == conv(x)*f + (b*f + extra), so an already
                # folded bias has to be rescaled by the same factor.
                factor = scale / np.sqrt(var + eps)
                w, extra = fold_bn(w, scale, offset, mean, var, eps)
                bias = (bias * factor + extra).astype(np.float32)
                out_name = bn.output[0]
                j += 1

        se = None
        hit = self.match_se(j, out_name)
        if hit:
            j, out_name, sw1, sb1, sw2, sb2 = hit
            se = (sw1, sb1, sw2, sb2)

        add_src = None
        if j < len(n) and n[j].op_type == "Add" and out_name in n[j].input \
                and not any(self.is_const(t) for t in n[j].input):
            other = [t for t in n[j].input if t != out_name]
            other = other[0] if other else out_name
            if other in p.buf_of:
                add_src = other
                out_name = n[j].output[0]
                j += 1

        act, j, out_name = self.trailing_act(j, out_name)

        src = node.input[0]
        src_buf = p.buf_of[src]
        # Allocate before releasing: the conv reads a neighbourhood of src (and
        # all of add_src) while writing dst, so they must not alias.
        dst = p.new_buf()
        p.release(src)
        if add_src is not None:
            p.release(add_src)

        w_off = p.add_data(pack_conv_weight(w, group))
        b_off = p.add_data(bias)
        se_off, se_squeeze = 0, 0
        if se is not None:
            sw1, sb1, sw2, sb2 = se
            se_squeeze = int(sw1.shape[0])
            se_off = p.add_data(np.concatenate(
                [sw1.reshape(-1), sb1.reshape(-1), sw2.reshape(-1), sb2.reshape(-1)]))

        p.emit(op=OP_CONV, name=node.input[1],
               in_buf=src_buf, out_buf=dst,
               add_buf=-1 if add_src is None else p.buf_of[add_src],
               act=act, oc=oc, ic=int(w.shape[1]) * group,
               kh=int(w.shape[2]), kw=int(w.shape[3]),
               sh=strides[0], sw=strides[1], ph=pads[0], pw=pads[1],
               w_off=w_off, b_off=b_off, group=group,
               se_off=se_off, se_squeeze=se_squeeze)
        p.bind(out_name, dst, self.readers.get(out_name, 1))
        return j

    def emit_fc(self, i, node):
        p, n = self.p, self.nodes
        w = self.const(node.input[1])              # (IN, OUT)
        add = n[i + 1]
        assert add.op_type == "Add" and self.is_const(add.input[1])
        bias = self.const(add.input[1]).reshape(-1)
        out_name = add.output[0]
        j = i + 2
        softmax = 0
        if j < len(n) and n[j].op_type == "Softmax" and n[j].input[0] == out_name:
            assert attrs_of(n[j])["axis"] == 2
            softmax = 1
            out_name = n[j].output[0]
            j += 1

        src = node.input[0]
        src_buf = p.buf_of[src]
        # An FC here is exactly a 1x1 conv over a (IN, 1, T) tensor, so it reuses
        # the conv GEMM path. Its source is therefore always channel-major, and
        # fc_layout describes the OUTPUT: 0 keeps (OUT, 1, T) so a following FC
        # can consume it directly, 1 emits (1, T, OUT) for the caller + softmax.
        dst = p.new_buf()
        p.release(src)
        w_off = p.add_data(pack_conv_weight(
            np.ascontiguousarray(w.T).reshape(w.shape[1], w.shape[0], 1, 1), 1))
        b_off = p.add_data(bias)
        p.emit(op=OP_FC, name=node.input[1], in_buf=src_buf, out_buf=dst,
               oc=int(w.shape[1]), ic=int(w.shape[0]),
               w_off=w_off, b_off=b_off, fc_layout=softmax, fc_softmax=softmax)
        p.bind(out_name, dst, self.readers.get(out_name, 1))
        return j


def write_blob(p, final_buf, out_path, model_path, input_height, n_classes):
    data = np.concatenate(p.blobs) if p.blobs else np.zeros(0, np.float32)
    header_ints = 12
    header_bytes = 4 + header_ints * 4
    ops_bytes = len(p.ops) * OP_FIELDS * 4
    data_offset = header_bytes + ops_bytes
    data_offset += (-data_offset) % 64

    keys = ["op", "in_buf", "out_buf", "add_buf", "act", "oc", "ic", "kh", "kw",
            "sh", "sw", "ph", "pw", "ceil_mode", "count_include_pad", "w_off",
            "b_off", "group", "se_off", "se_squeeze", "fc_layout", "fc_softmax"]
    with open(out_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<12i", VERSION, len(p.ops), p.n_bufs, final_buf,
                             header_bytes, data_offset, data.size, n_classes, MR,
                             input_height, 0, 0))
        for op in p.ops:
            row = [op[k] for k in keys] + [0] * (OP_FIELDS - len(keys))
            fh.write(struct.pack(f"<{OP_FIELDS}i", *row))
        fh.write(b"\0" * (data_offset - header_bytes - ops_bytes))
        fh.write(data.tobytes())

    manifest = {
        "source_onnx": model_path, "magic": MAGIC.decode(), "version": VERSION,
        "n_ops": len(p.ops), "n_bufs": p.n_bufs, "final_buf": final_buf,
        "n_classes": n_classes, "input_height": input_height, "gemm_mr": MR,
        "weight_floats": int(data.size), "weight_bytes": int(data.size) * 4,
        "notes": p.notes, "ops": p.ops,
    }
    with open(out_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return data


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    model = onnx.load(model_path)
    graph = model.graph
    dims = [d.dim_value if d.dim_value else d.dim_param
            for d in graph.input[0].type.tensor_type.shape.dim]
    input_height = int(dims[2])
    ex = Extractor(graph)
    p, final_buf = ex.run()
    n_classes = p.ops[-1]["oc"]
    data = write_blob(p, final_buf, out_path, model_path, input_height, n_classes)

    kinds = {}
    for op in p.ops:
        key = {OP_CONV: "conv", OP_MAXPOOL: "maxpool", OP_AVGPOOL: "avgpool",
               OP_FC: "fc", OP_SE: "se"}[op["op"]]
        if op["op"] == OP_CONV and op["group"] > 1:
            key = "conv-depthwise"
        kinds[key] = kinds.get(key, 0) + 1
    fused_se = sum(1 for o in p.ops if o["se_squeeze"] > 0)
    acts = {}
    for op in p.ops:
        acts[op["act"]] = acts.get(op["act"], 0) + 1
    print(f"ops={len(p.ops)} {kinds} fused_SE={fused_se} acts={acts}")
    print(f"bufs={p.n_bufs} final_buf={final_buf} classes={n_classes} H={input_height}")
    print(f"weights: {data.size} floats = {data.size*4/1e6:.2f} MB -> {out_path}")
    for note in sorted(set(p.notes)):
        print(f"note: {note}")


if __name__ == "__main__":
    main()
