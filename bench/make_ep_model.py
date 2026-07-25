#!/usr/bin/env python3
"""Build ONNX models that hit the SpaceMIT EP's int8 IME path (spine-kernel route).
MatMulInteger is NOT registered by the EP; DynamicQuantizeMatMul and Conv ARE.
Run on the board (needs python3-onnx): python3 make_ep_model.py dqm 2048
Then: onnxruntime_perf_test -e spacemit -t 8 /tmp/<kind>/model.onnx /tmp/res.txt
"""
import sys, os, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

kind = sys.argv[1] if len(sys.argv) > 1 else "dqm"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
outdir = f"/tmp/{kind}"; os.makedirs(f"{outdir}/test_data_set_0", exist_ok=True)

if kind == "dqm":
    # DynamicQuantizeMatMul (com.microsoft): A(f32) x B(int8, quantized weight) -> f32
    A  = helper.make_tensor_value_info("A", TensorProto.FLOAT, [N, N])
    Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, N])
    B  = helper.make_tensor("B", TensorProto.INT8, [N, N],
                            np.random.randint(-8, 8, N*N).astype(np.int8).tobytes(), raw=True)
    bs = helper.make_tensor("b_scale", TensorProto.FLOAT, [1], np.array([0.02], np.float32).tobytes(), raw=True)
    bz = helper.make_tensor("b_zp", TensorProto.INT8, [1], np.array([0], np.int8).tobytes(), raw=True)
    node = helper.make_node("DynamicQuantizeMatMul", ["A", "B", "b_scale", "b_zp"], ["Y"],
                            domain="com.microsoft")
    g = helper.make_graph([node], "dqm", [A], [Y], [B, bs, bz])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17),
                                            helper.make_opsetid("com.microsoft", 1)])
    inp = np.random.randn(N, N).astype(np.float32)
    open(f"{outdir}/test_data_set_0/input_0.pb", "wb").write(numpy_helper.from_array(inp, "A").SerializeToString())

elif kind == "conv":
    # 1x1 Conv as a GEMM: [1,C,H,W] * [C,C,1,1]. C=1024, H=W=64 -> C*H*W*C macs.
    C, H = 1024, 64
    X  = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C, H, H])
    Y  = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, C, H, H])
    W  = helper.make_tensor("W", TensorProto.FLOAT, [C, C, 1, 1],
                            (np.random.randn(C*C)*0.02).astype(np.float32).tobytes(), raw=True)
    node = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[1, 1])
    g = helper.make_graph([node], "conv", [X], [Y], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    inp = np.random.randn(1, C, H, H).astype(np.float32)
    open(f"{outdir}/test_data_set_0/input_0.pb", "wb").write(numpy_helper.from_array(inp, "X").SerializeToString())
    print(f"conv macs/inference = {C*C*H*H:.3e}  (TOPS = 2*that/avg_s/1e12)")
else:
    print("usage: make_ep_model.py [dqm|conv] N"); sys.exit(1)

onnx.save(m, f"{outdir}/model.onnx")
print(f"saved {outdir}/model.onnx  kind={kind} N={N}")
if kind == "dqm":
    print(f"dqm macs/inference = {N*N*N:.3e}  (TOPS = 2*{N}^3/avg_s/1e12)")
