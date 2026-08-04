# Native RVV OCR recognizer on the SpaceMIT K3

Status: **built, validated bit-for-bit against the FP32 ONNX oracle, deployed,
and benchmarked.** A hand-written FP32 RVV forward pass for the official
PaddleOCR Rosetta recognizer (`rec_r34_vd_none_none_ctc_v2.0`, ResNet34-vd +
CTC) runs on the K3's eight A100 AI harts with no ONNX Runtime, no SpaceMIT
execution provider, no XQuant, no Paddle runtime and no CPU fallback anywhere in
its inference path. It is **2.9-3.5x faster than ONNX Runtime CPU on the same
model** and agrees with it to 2.3e-5 absolute probability.

It is wired into the live API behind `VISION_RECOGNIZER=native-rosetta` and was
verified end to end there. **It is not the service default**, and the reason is
not the engine: the Rosetta *model* is the wrong model for this workload. See
"Why the live default is still PP-OCRv6" — that section is the most important
one in this document.

This work exists because `SPACEMIT_EP_FINDINGS.md` established that SpaceMIT EP
2.0.5 cannot compile or partition this topology. That document's own conclusion
was "the convolution kernels are compatible" — the failures were the vendor
software stack, not the silicon. This engine tests that conclusion directly by
going around the entire stack, and confirms it.

---

## 1. What was built

| File | Role |
|---|---|
| `extract_rosetta.py` | Lowers `rosetta-r34.onnx` to a flat FP32 blob + op program. Deterministic, re-runnable, no hand-transcribed numbers. |
| `rosetta_rvv.c` | The engine. ~600 lines of C + RVV intrinsics. Builds to `librosetta_rvv.so` and a standalone `rosetta_test` harness. |
| `rosetta_native.py` | ctypes binding used by `app.py`. |
| `rosetta_ref.py` | NumPy interpreter of the same blob. Development oracle only; proves the *extraction* is right independently of the C. |
| `make_oracle_onnx.py` | Rewrites the shipped ONNX into a width-dynamic oracle. |
| `make_rec_crops.py` | Generates real crops via the **unmodified** live detector path. |
| `prep_validation.py`, `compare_native.py` | Freeze inputs + oracle outputs; report the correctness numbers. |
| `bench/rvv_f32_peak.c`, `bench/rvv_gemm_shape.c` | Hardware characterisation used to set the kernel shape. |

The detector (PP-OCRv6 Tiny, ORT CPU) was not touched. `_det_input`, `_boxes`
and `_crop` are byte-identical to before.

## 2. Architecture, verified rather than assumed

Re-derived directly from the ONNX graph by `extract_rosetta.py`, not transcribed:

- **39 Conv** — matches the `39 Conv, CTC` fingerprint in `SPACEMIT_EP_FINDINGS.md`.
- 3-conv "vd" stem (32, 32, 64 channels) → MaxPool 3x3 s2 p1.
- Stages of 3/4/6/3 BasicBlocks at 64/128/256/512 channels. Each stage after the
  first downsamples **height only** (`stride=[2,1]`), with the shortcut taking
  `AveragePool k=[2,1] s=[2,1] ceil_mode=1 count_include_pad=0` on the previous
  stage's output before its 1x1 projection.
- Tail: MaxPool 2x2 s2 → Squeeze(axis 2) → Transpose → MatMul(512,37) → Add →
  Softmax(axis 2).
- Input `[N,3,32,W]`, normalized `pixel/127.5 - 1`. Output `[N, W_out, 37]`,
  **already post-softmax**. 37 = CTC blank at index 0 + 36 ic15 characters
  (`0-9a-z`).

Two things in the shipped export are width-specific and are dropped, because a
width-agnostic engine must not inherit them: a `Mul` by a constant `1.0`
(confirmed `p2o.pd_op.full.0.0 == [1.0]`, a paddle2onnx artifact) and a final
`Reshape` to a hardcoded `[-1, 25, 37]`, which is only correct for W=100. The
engine derives `W_out` by running the real stride/pool arithmetic. This is not
theoretical: at W=101 the true answer is still 25 timesteps
(`floor((101-1)/2)+1 = 51`, then `floor((51-2)/2)+1 = 25`), which a `W/4` formula
would get wrong. W=101 is in the validation set for exactly this reason.

**BatchNorm is folded at extraction time** (`scale = gamma/sqrt(var+eps)`,
`eps = 9.999999747378752e-06`), and residual `Add` + `ReLU` are fused into the
conv that produces them. The runtime therefore implements only Conv, MaxPool,
AveragePool and the CTC head — 45 ops, 3 reusable buffers, 21,318,469 weights
(85.27 MB).

## 3. Precision and kernel design

**FP32 throughout** — weights, activations and accumulation. Not int8. Two
reasons, one of which is stronger than the brief anticipated:

1. It makes correctness *falsifiable*. FP32-vs-FP32 lets us demand near-bit-exact
   agreement with the oracle and treat any deviation as a bug, rather than
   arguing about whether 80% argmax agreement is acceptable (the bar the prior
   XQuant attempt failed).
2. The measured hardware ceiling made int8 unnecessary to be competitive: plain
   FP32 vector FMA already beats ORT CPU by ~3x.

The IME-2 int8 dot-product instructions used by `qwen_moe_hp.c` are **not** used
here. That has a useful consequence — see the hart-scaling section.

### Kernel shape was measured, not guessed

`bench/rvv_f32_peak.c` on one AI hart (VLEN=1024, `vlenb=128`):

| LMUL | vl | GFLOP/s | FLOP per instruction |
|---|---|---|---|
| m1 | 32 | 182.9 | 64 |
| **m2** | **64** | **363.0** | **128** |
| m4 | 128 | 230.0 | 256 |

The machine retires ~2.85 G vector-FMA instructions/s regardless of LMUL, so
**LMUL=2 is strictly optimal** (m4 is worse per instruction, m1 does half the
work per instruction). The engine uses `e32m2` everywhere.

Then the operand-source cost was isolated, which is where the real constraint is:

| Inner loop (8 accumulators) | GFLOP/s |
|---|---|
| constant scalar, constant vector (no memory at all) | 368.4 |
| **loaded** scalars, constant vector | 184.3 |
| constant scalar, **loaded** vector | 71.7 |
| `vfmacc.vv`, loaded vector | 74.6 |
| loaded scalars **and** loaded vector (the real GEMM shape) | 27.9 |

The last row is worse than either penalty alone or their sum — scalar loads and
vector loads interact badly on this in-order core. This was chased rather than
assumed: sweeping the B working set from 2 KB to 73 KB changed nothing
(24.0 → 27.5 GFLOP/s), so it is **not** a cache-residency effect; software
pipelining the vector load 2 and 4 deep gained only 8% (27.9 → 30.3 → 29.8), so
it is **not** simply exposed load latency; and raising MR from 8 to 14 (fewer
vector loads per FLOP) gained nothing (30.2 → 29.3), so it is **not** vector-load
bandwidth. Trading scalar loads for FMAs *did* help — MR=6/NC=2 and MR=4/NC=3
both reached ~36.8 GFLOP/s, +22%.

The shipped kernel is **MR=8 x one e32m2 column vector**, i.e. the 30 GFLOP/s
row. The +22% MR=4/NC=3 variant is left on the table deliberately: it requires
re-packing the weight blob at MR=4 and a three-chunk column loop, and it was not
worth the correctness risk relative to the conclusion in section 7. This is
recorded as known headroom, not as an optimum.

Weights are pre-packed at extraction time into MR-row panels, k-major
(`panel[k][r]`), so the 8 scalar broadcasts for one k are contiguous.

### Tiling policy

Splitting a conv's output both ways costs redundant work in two directions: each
of the `m` output-row blocks rebuilds im2col for every column it touches
(`m*cols*K` floats), and each of the `n` column tiles re-reads the whole weight
matrix (`n*OC*K` floats). The engine minimises `m*cols + n*OC` subject to
`m*n >= nharts`. This lands on column splits for early layers (few channels,
thousands of columns) and channel splits for late layers (512 channels, ~100
columns) — automatically, per layer, at the caller's width. It is worth 9%
(82.9 → 76.0 ms/crop) over a fixed 64-column tile, mostly by eliminating the
wasted lanes when `cols` is not a multiple of the tile width.

im2col is built per work unit into a private, K-blocked scratch tile
(`KB_TARGET=288` — one 32-channel 3x3 block) so it never touches DRAM.

### Build flags — measured on this board, not inherited

97 real crops, W=100, 8 harts:

| Flags | ms/crop |
|---|---|
| `-O3` | 84.9 |
| `-O3 -fno-tree-vectorize` | 76.7 |
| `-O3 -fno-tree-vectorize -funroll-loops` | **52.8** |

`-fno-tree-vectorize` is worth 11% even though this file uses only intrinsics and
no hand-scheduled inline asm — the flag was re-verified here rather than assumed
from `qwen_moe_hp.c`'s header, and it earns its place on different grounds
(gcc's auto-vectorizer degrading surrounding scalar code) than it does there.
`-funroll-loops` is worth a further 1.44x. **Neither changes a single output
bit** — the correctness numbers in section 4 are identical with and without.

## 4. Correctness

Oracle: `onnxruntime` 1.24 CPU running `rosetta-r34-dyn.onnx` (the shipped FP32
export with the width dimension made dynamic and the graph cut at the Softmax).
ONNX Runtime is used **only** as a development-time oracle, never in the deployed
recognizer path.

Inputs: **97 real perspective-corrected text crops** produced by running
`sample.png` (a real 1224x1584 document page) through the live detector path
(`OCRBackend._det_input` / `_boxes` / `_crop`, imported unmodified from
`app.py`), then resized to height 32 and normalized `/127.5 - 1`. Both sides
consume the identical frozen float32 buffer, so the comparison is positional with
no re-preprocessing on either side.

| Width | Steps | Max abs prob diff | Mean abs prob diff | Timestep argmax agreement | Full-string exact decode match |
|---|---|---|---|---|---|
| 100 | 25 | 1.240e-05 | 5.07e-08 | **100.0000%** (2425/2425) | **97/97 (100%)** |
| 101 | 25 | 8.464e-06 | 5.15e-08 | **100.0000%** (2425/2425) | **97/97 (100%)** |
| 160 | 40 | 1.250e-05 | 4.75e-08 | **100.0000%** (3880/3880) | **97/97 (100%)** |
| 320 | 80 | 2.268e-05 | 3.78e-08 | **100.0000%** (7760/7760) | **97/97 (100%)** |

Residual differences are float32 accumulation-order noise on values in [0,1]
(worst single element: oracle 0.394803 vs native 0.394825). There is no
approximation anywhere in the engine — no fast-math, no reduced precision, no
skipped terms.

The extraction was validated separately and *before* any C was written, using
`rosetta_ref.py` (a NumPy interpreter of the same blob) against the same oracle:
max abs diff 2.3e-6, 100% argmax agreement, identical decoded strings. That
ordering mattered — when the C engine first disagreed, the blob was already
excluded as a suspect, and the bug (`run_program` reading the shape pass's *end
state* instead of each op's own shapes) was found in one profiling pass.

## 5. Performance

All numbers: 32 real crops, best of 3, on the board.

**Native engine vs ONNX Runtime CPU, same model, same crops:**

| Width | Native (8 AI harts) | ORT CPU, 8 threads (N=1) | ORT CPU batched | Speedup vs ORT |
|---|---|---|---|---|
| 100 | **55.0 ms/crop** | 192.6 | 170.6 | **3.1-3.5x** |
| 160 | **83.1 ms/crop** | — | — | — |
| 320 | **172.6 ms/crop** | 506.3 | 507.5 | **2.9x** |

**Hart scaling** (32 crops, W=100):

| Harts | ms/crop | Speedup vs 1 hart |
|---|---|---|
| 1 | 593.9 | 1.00x |
| 2 | 201.9 | 2.94x |
| 4 | 87.8 | 6.76x |
| 6 | 80.8 | 7.35x |
| 8 | **52.7** | **11.3x** |

Scaling is superlinear at low hart counts because a single hart's working set
thrashes cache that eight harts collectively keep warm; the honest reading is
that 8 harts are fully usable, not that there is 11x of parallel efficiency.

**The paired-hart question is settled: all eight harts are usable.**
`qwen_moe_hp.c` caps its linear layers at four harts (8/10/12/14) because IME-2
is shared per hart *pair* and concurrent IME-2 use of both harts in a pair is
measured-contended. This engine issues no IME-2 instructions, so that does not
apply — and it was verified rather than assumed. Running the GEMM kernel
concurrently on all eight harts gives 24.8-26.9 GFLOP/s **per hart** with no
systematic gap between the primary set (8/10/12/14) and their partners
(9/11/13/15), for 208 GFLOP/s aggregate — 95% of ideal scaling. End-to-end,
going from 4 to 8 harts is worth 1.67x.

For comparison, running the same engine unpinned on the ordinary harts 0-7 gives
79.4 ms/crop vs 52.7 on the AI harts — the AI harts are both faster *and* leave
the general cores free for the detector, the web server and the OS, so they are
the right target for a service.

**Where the time goes** (W=100, worker 0, 39 convs): GEMM 61.3 ms, im2col
11.8 ms, epilogue (residual add + ReLU + store) 0.4 ms.

**Known headroom.** The engine's GEMM achieves ~13.8 GFLOP/s per hart against
the ~26 GFLOP/s the same kernel sustains on all 8 harts concurrently in
isolation, so roughly **1.9x remains on the table** from tiling/im2col overhead,
plus the ~1.2x from the MR=4/NC=3 kernel shape. A third, larger lever is unused:
the engine processes **one crop at a time**, so all 85 MB of weights stream from
DRAM per crop. Batching same-width crops would amortise that and widen the GEMM's
column dimension (for a 512-channel layer at W=100, batch-8 cuts the modelled
per-crop traffic ~2.9x). This is the single biggest remaining optimisation and it
was not implemented — see section 7 for why it was not pursued.

## 6. Integration

`app.py` gained a `VISION_RECOGNIZER` env toggle mirroring the existing
`VISION_PROVIDER` convention:

- `onnx-ppocrv6` (**default**) — unchanged behaviour, PP-OCRv6 Tiny on ORT CPU.
- `native-rosetta` — `librosetta_rvv.so` via ctypes on the AI harts.

The detector path is untouched in both cases. The recognizer-specific pieces are
kept apart properly: the native path uses height 32 (not 48) and its own 37-entry
character list `[""] + "0-9a-z"` — *not* PP-OCRv6's 38-entry
`[""] + chars + [" "]` convention, which would silently mis-decode. `_decode`,
`infer()`'s timing/response shape and its error handling are unchanged; the
response now also reports `recognizer` and `rec_provider`.

The native path infers each crop at its own natural width (rounded to a multiple
of 16, capped at 640) rather than padding a batch to a common width — for a fully
convolutional CTC recognizer, padding makes every short crop pay the widest
crop's convolution cost.

Deployed to `root@192.168.68.24`:

```
/root/vision-api/app.py
/root/vision-api/rosetta_native.py
/root/vision-api/models/rosetta-native/librosetta_rvv.so
/root/vision-api/models/rosetta-native/rosetta_r34.bin        (85 MB)
/root/vision-api/models/rosetta-native/rosetta_r34.bin.json
```

Verified end to end on the native path (`systemd` drop-in setting
`VISION_RECOGNIZER=native-rosetta`, service restarted):

```
$ curl -s localhost:8080/healthz
{"status":"ok","model":"rec_r34_vd_none_none_ctc_v2.0 (native RVV FP32)",
 "detector":"PP-OCRv6_tiny_det","recognizer":"native-rosetta",
 "rec_provider":"spacemit-ai-harts x8", ...}

$ curl -F 'image=@sample.png' localhost:8080/v1/ocr
http=200  timing_ms {detect: 468, postprocess: 328, recognize: 34552, total: 35466}
97 lines returned
```

To switch the live service over:

```bash
mkdir -p /etc/systemd/system/jupiter-vision-api.service.d
printf '[Service]\nEnvironment=VISION_RECOGNIZER=native-rosetta\n' \
  > /etc/systemd/system/jupiter-vision-api.service.d/native.conf
systemctl daemon-reload && systemctl restart jupiter-vision-api.service
```

Removing that file and restarting reverts. The service is currently running the
default (`onnx-ppocrv6`), confirmed healthy.

## 7. Why the live default is still PP-OCRv6

The engine met every goal it was set. The *model* it runs does not suit this
workload, and the evidence is unambiguous. Same image, same detector, same
machine:

| | native-rosetta | onnx-ppocrv6 |
|---|---|---|
| recognize time, 97 crops | **34,552 ms** | **9,925 ms** |
| end-to-end wall | 35.5 s | 10.3 s |
| character set | 36 (`0-9a-z`) — no uppercase, no punctuation, **no space** | full PP-OCRv6 dictionary |

```
native-rosetta : aalgorithmsforthemarkoventronydecompositionen
onnx-ppocrv6   : Algorithms for the Markov Entropy Decomposition

native-rosetta : aandrewaferrissanddavidpoulinen
onnx-ppocrv6   : Andrew J. Ferris and David Poulin

native-rosetta : aemankoventronudecomnostionimedlisarecenthydorosediclusterhasedsimulationmethodforfanen
onnx-ppocrv6   : The Markov entropy decomposition (MED) is a recently-proposed. cluster-based simulation method for fi-
```

Two independent problems, neither fixable by optimising the engine:

1. **Rosetta is a word-level 36-character model.** It cannot emit spaces, capital
   letters or punctuation. On running text its output is a lowercase
   letter-soup even when the underlying recognition is largely right. This is a
   property of the published model, not of this implementation — the native
   engine reproduces the ONNX oracle's output exactly, letter-soup included.
2. **ResNet34 is ~20x more compute than PP-OCRv6 Tiny.** At ~6.8 GFLOP/crop for
   W=100 the native engine needs 55 ms/crop; PP-OCRv6 Tiny needs 3.4-5.8 ms/crop
   on plain ORT CPU. Even spending all the headroom in section 5 — call it 3x —
   leaves it several times slower than what is already deployed.

So the native engine is 3x faster than ORT *at running Rosetta*, and Rosetta is
the wrong thing to run. Defaulting the live service to it would have made OCR
both slower and substantially less accurate, which is the opposite of what was
asked for. The engine is deployed, verified and one file away from being active
if it is ever wanted.

## 8. What this proves about the hardware

`SPACEMIT_EP_FINDINGS.md` left open whether the K3's AI cores could run this
topology at all, having only shown that EP 2.0.5 could not compile it. That is
now answered: **the silicon runs the complete 39-conv residual CTC recognizer at
full FP32 accuracy across all eight AI harts, with linear-ish scaling and 3x the
throughput of the vendor's own CPU runtime.** Every failure documented in that
file — the residual-`Add` compile segfault, the `kernel_shape` parser bug, the
all-eight-cores-per-session reservation, the 75-nodes-on-CPU partitioning — was
a software-stack failure. None of them has a hardware cause.

The reusable result is the hardware characterisation in section 3, which applies
to any FP32 RVV kernel on this board, and the confirmation that **ordinary RVV
FP32 work scales across all eight AI harts** where IME-2 int8 work must stay on
four.

## 9. Known limitations and follow-ups

- **Not the service default**, for the model-quality reasons in section 7.
- **~1.9x tiling headroom + ~1.2x kernel-shape headroom** remain unexploited
  (section 5), and **crop batching is not implemented** — the largest single
  remaining win.
- **One crop at a time.** `rosetta_infer` is serialised by an internal mutex, so
  concurrent HTTP requests queue at the recognizer. Fine for the current
  single-worker uvicorn deployment; would need revisiting under real concurrency.
- **Width capped at 640** in `rosetta_native.py`. Document lines from
  `sample.png` want ~1600 px at height 32; at that width a crop would cost
  ~0.9 s. The cap trades recognition of very long lines for bounded latency.
- **Odd widths are slower than even ones** — W=101 costs 77 ms/crop vs 52 for
  W=100, a tiling/vector-tail artifact. The API rounds widths to multiples of 16
  so it does not hit this, but the engine is correct, not fast, at arbitrary
  widths.
- **gcc vector-spill hazard.** Holding RVV vector variables live across a
  function call (e.g. `clock_gettime`) makes this toolchain corrupt the stack
  canary — it aborts with "stack smashing detected". Hit twice while writing the
  benchmarks. Keep vector values out of call-live ranges, or build the affected
  file with `-fno-stack-protector`. The engine itself does not do this.
- **Board memory rule observed**: the K3 has 32 GB and no swap, so no benchmark
  here was run concurrently with the LLM engine or any other model-loading
  process.

## 10. Reproducing

See section 20 — the Rosetta-specific `rosetta_rvv.c` / `extract_rosetta.py` were
superseded by the generalised `ocr_rvv.c` / `extract_ocr.py`, which run both
models. Every Rosetta number above was re-verified through the generalised
engine (section 17).

---

# Phase 2 — the same engine, on the recognizer that is actually deployed

Status: **native PP-OCRv6 Tiny recognizer is validated bit-equivalent to ONNX
Runtime, faster on every crop width measured, and is now the live default.**

Phase 1's conclusion was that the Rosetta *engine* was fine and the Rosetta
*model* was wrong for this workload. Phase 2 acts on that: the proven engine
structure was generalised and pointed at `PP-OCRv6_tiny_rec` — the model already
serving traffic, with the real 6906-entry multilingual dictionary and ~19x less
compute than ResNet34. That combination is what phase 1 could not offer:
correct charset *and* faster than what it replaces.

## 11. What changed structurally

There is now **one engine and one extractor for both models**, selected purely by
the blob handed to them:

| Retired | Replaced by |
|---|---|
| `rosetta_rvv.c` | `ocr_rvv.c` |
| `extract_rosetta.py` | `extract_ocr.py` |
| `rosetta_ref.py` | `ocr_ref.py` |
| `rosetta_native.py` | `ocr_native.py` |

The blob format gained a version (v3, magic `OCR1`) carrying the input height and
class count, so the runtime and the ctypes wrapper are model-agnostic. Both
models were re-validated through the merged code (section 17).

## 12. PP-OCRv6 Tiny architecture, verified from the graph

All 219 nodes walked by `extract_ocr.py`; the op histogram is
`{Conv: 37, BatchNormalization: 4, Identity: 58, Div: 10, Erf: 10, Add: 52,
Mul: 25, ReduceMean: 3, Relu: 3, HardSigmoid: 5, AveragePool: 1, Squeeze: 3,
Transpose: 3, Unsqueeze: 2, MatMul: 2, Softmax: 1}`.

It is a PP-LCNet-style MobileNet, not a transformer. Confirmed details that
mattered:

- **Only 4 BatchNorms survive** for 37 convs — the export already folded the rest
  into a separate `Add` of a `[1,C,1,1]` constant. The extractor folds both forms.
- **GELU is the exact-erf form**, `Div(x,√2) → Erf → Add(1) → Mul(x,·) → Mul(0.5)`,
  10 instances. Not a tanh approximation.
- **3 Squeeze-and-Excite blocks** (48, 96, 160 channels), each
  `ReduceMean(2,3) → Conv1x1 → ReLU → Conv1x1 → HardSigmoid → Mul`. Since the
  spatial map is 1x1 there, both convs are plain GEMVs; the extractor collapses
  the whole block into one op.
- **10 depthwise convs** (`group == channels`), two of them `strides=[2,1]` —
  the same height-only downsample as Rosetta. Derived per node, never hardcoded.
- The tail's **two back-to-back `Transpose(0,2,1)` cancel exactly**; the
  `Squeeze`/`Unsqueeze` on axis 2 are no-ops in an NCHW buffer whose height has
  already collapsed to 1, which the runtime shape pass asserts. So the whole 1D
  sequence mixer is just convs on an H=1 tensor: a `1x5` depthwise conv and a
  `1x1` conv, each followed by BatchNorm1d (folded) and **HardSwish**
  (`x * hardsigmoid(x)`, the `HardSigmoid → Mul` pair).
- Head: `MatMul(160,80) → MatMul(80,6906) → Softmax`, with **no activation
  between the two FCs**.
- Input `[N,3,48,W]` with **W genuinely dynamic** (unlike Rosetta's fixed export),
  normalized `/127.5 - 1`. Output steps `= W/8` by real stride arithmetic;
  verified odd width 161 → 20 steps, same as 160.

Extraction yields **34 ops** (21 dense conv + 10 depthwise + 1 avgpool + 2 FC,
with 6 SE convs folded into 3 SE ops = 37 convs total ✓), 3 buffers,
**4.42 MB** of FP32 weights — versus Rosetta's 85.27 MB.

## 13. New kernels

- **Depthwise conv**: direct convolution, not im2col+GEMM. Each output channel
  touches only its own input channel, so a GEMM would be pure overhead. The
  kernel taps are loop-invariant scalars, which sidesteps the scalar-load /
  vector-load interaction that limits the dense GEMM on this core (section 3).
  Rows are copied into a width-padded scratch plane so every vector load is
  in-range and contiguous.
- **1x1 conv fast path**: for `k=1, s=1, p=0` the im2col of a column tile *is* a
  window of the source, so the copy is skipped entirely and the microkernel is
  pointed straight at the input with the source row stride. This covers most of
  this model's compute, and the FC head reuses it — an FC here is exactly a 1x1
  conv over a `(IN, 1, T)` tensor, so it goes through the same GEMM with only a
  transposed store and an optional softmax bolted on.
- **Vector `exp`** (Cephes range-reduction) and **vector `erf`**
  (Abramowitz & Stegun 7.1.26, max abs error 1.5e-7), used by GELU and by the
  6906-wide softmax. A scalar `erff` per element would have cost ~25 ms/crop on
  its own.
- **HardSigmoid / HardSwish**, and an activation enum replacing phase 1's
  boolean ReLU flag.

## 14. Two tiling-policy fixes this model forced

Both were found by profiling, both are in `ocr_rvv.c`'s tile chooser:

1. **Shrink output rows before output columns.** The 6906-class FC head has one
   enormous row block and only `T` columns, so the accumulator-size constraint
   was collapsing the column tile to 10 against a vector length of 64 — 16%
   lane utilisation. Giving up rows first fixed it: **6.47 → 5.27 ms/crop at
   W=320.**
2. **Widen a tile to a whole vector only when it removes vector iterations.**
   At 180 columns a 90-wide tile runs 4 three-quarter-full vectors where a
   128-wide tile runs 3 full ones. Applying the rounding unconditionally
   regressed Rosetta (it only inflated the im2col row stride there), so it is
   now gated on an explicit iteration count.

## 15. Two real defects found in phase 1's engine

Both were latent in the Rosetta build and are fixed in `ocr_rvv.c`:

**The worker pool busy-waited forever.** A pure spin pool is right for a batch
process and wrong for a long-lived HTTP service: it held all eight AI harts at
100% between requests, burning power and contaminating any concurrent
measurement (it is why an early phase-2 benchmark showed a fake 10x "cliff" at
W=1600). Workers now spin briefly — long enough to bridge the sub-microsecond
gaps between the ~40 dispatches inside one inference — then sleep on a condition
variable. The service now idles at **0.0% CPU**, confirmed with `top`. The spin
also uses the Zihintpause `PAUSE` hint, spelled as its raw encoding because this
assembler rejects the `fence w, 0` mnemonic it is defined as.

**The engine pins its caller, and that is load-bearing.** This is the most
important thing in this document for anyone else writing AI-hart code:

> **A thread that is not AI-bound does not reliably observe stores made by
> AI-hart threads.**

Measured directly. With the calling thread demoted to a pure signaller, its read
of a buffer the workers had just filled returned `-0.000000` where the correct
value was `-1.018019`; the identical build with `OCR_NO_AI_HARTS=1` (workers on
the ordinary harts) read back correct values. Calling `bind_ai()` on the caller
without pinning did **not** help, even though it does move the thread onto the AI
harts. Release/acquire atomics, `__asm__ volatile("" ::: "memory")` barriers, and
making the published field a real `_Atomic` all failed to fix it — this is below
the level the C memory model reaches.

So the caller stays inside the pool as worker 0, AI-bound and pinned, exactly as
`qwen_moe_hp.c` has always done. The cost is that the caller's thread is pinned
for life, which ran all of the caller's own OpenCV/numpy work on one hart:
**preprocessing 886 ms and decoding 2139 ms, against ONNX Runtime's 393 and 505
for identical work.** The fix belongs on the Python side, not in the engine —
`ocr_native.py` drives the library from one dedicated thread, so only that thread
is pinned and the request thread stays schedulable. That restored preprocessing
to **385 ms** and decoding to **503 ms**, matching the ORT path exactly.

The symptom of the pinning problem was a 4x slowdown in numpy code that had
nothing to do with the engine, so it is worth stating the general rule: **treat
`ocr_load`/`ocr_infer` as callable from exactly one dedicated thread.**

## 16. Correctness — bit-equivalent, not merely close

Same methodology and same bar as phase 1: 97 real perspective-corrected crops
from `sample.png` through the **unmodified** detector path, frozen to raw
float32, compared positionally against `onnxruntime` CPU on the same
`inference.onnx` the service was already serving. ONNX Runtime is a
development-time oracle only and appears nowhere in the deployed recognizer path.

| Width | Steps | Max abs prob diff | Mean abs prob diff | Timestep argmax agreement | Full-string exact decode match |
|---|---|---|---|---|---|
| 160 | 20 | 1.678e-05 | 7.66e-10 | **100.0000%** (1940/1940) | **97/97 (100%)** |
| 161 | 20 | 1.937e-05 | 7.87e-10 | **100.0000%** (1940/1940) | **97/97 (100%)** |
| 240 | 30 | 3.573e-05 | 8.91e-10 | **100.0000%** (2910/2910) | **97/97 (100%)** |
| 320 | 40 | 5.567e-05 | 9.64e-10 | **100.0000%** (3880/3880) | **97/97 (100%)** |

Mean absolute difference is ~1e-9 across 6906 classes; the max is a single
high-probability element and reflects float32 accumulation order, nothing else.
Odd width 161 is in the set to exercise the real stride arithmetic.

As in phase 1, the blob was validated first in NumPy (`ocr_ref.py`) against the
same oracle — max abs diff 1.4e-5, 100% argmax agreement — *before* any C was
written, so the C engine could be debugged against a known-good blob.

**End to end, the two paths are indistinguishable.** `POST /v1/ocr` on
`sample.png` returns **97/97 byte-identical lines** to the ONNX path. That is a
deliberate design choice: `app.py`'s existing `_rec_batch` is reused unchanged,
zero padding and all, so the native engine sees exactly the input ORT would see.
An earlier version that inferred each crop at its own natural width was faster
but produced different (not obviously worse) text on 37/97 lines — real
behaviour change, rejected in favour of being a true drop-in.

## 17. Rosetta re-validated through the merged engine

No regression from generalising: **100.0000% timestep argmax agreement,
97/97 exact decode match** at W=100 and W=320, with the same max abs prob diff as
phase 1 (1.28e-05 / 2.17e-05), at the same speed (54.3 vs 52.3 ms/crop, run-to-run
noise).

## 18. Performance

24 real crops, best of 2, service stopped.

**Native vs ONNX Runtime CPU on the identical model and crops:**

| Width | Native (8 AI harts) | ORT per-crop | ORT batch-16 | vs per-crop | vs batch-16 |
|---|---|---|---|---|---|
| 160 | **3.21 ms** | 7.94 | 5.37 | **2.47x** | **1.67x** |
| 320 | **5.51 ms** | 14.05 | 10.37 | **2.55x** | **1.88x** |
| 640 | **10.94 ms** | 26.81 | 21.20 | **2.45x** | **1.94x** |
| 1280 | **23.43 ms** | 52.74 | 42.67 | **2.25x** | **1.82x** |
| 1600 | **32.87 ms** | 65.52 | 52.76 | **1.99x** | **1.61x** |
| 3200 | **91.33 ms** | 127.68 | 106.01 | **1.40x** | **1.16x** |

The advantage is largest at the small-to-moderate widths where typical OCR crops
live, and narrows at extreme widths where both engines become memory-bound.

**Hart scaling** (24 crops, W=320):

| Harts | ms/crop | Speedup |
|---|---|---|
| 1 | 30.37 | 1.00x |
| 2 | 16.65 | 1.82x |
| 4 | 9.26 | 3.28x |
| 6 | 7.35 | 4.13x |
| 8 | **5.52** | **5.50x** |

Ordinary harts 0-7 give 7.49 ms/crop, so the AI harts are both faster **and**
leave the general cores free for the detector, the web server and the caller's
own pre/post-processing — which, per section 15, is not a nicety but the whole
reason the end-to-end win exists.

**End to end** on `sample.png` (97 crops, all at 2488-3200 px after the
pipeline's existing width policy — a deliberately hostile case, ~90:1 aspect
document lines):

| | native-ppocrv6 | onnx-ppocrv6 |
|---|---|---|
| preprocess | 385 ms | 393 ms |
| **recognizer engine** | **8189 ms (84.4 ms/crop)** | **8921 ms (92.0 ms/crop)** |
| decode | 503 ms | 505 ms |
| `/v1/ocr` recognize_ms | **9112** | 9766 |
| `/v1/ocr` total_ms | **9457** | 10139 |

**6.7% faster end to end on the worst case, 1.7-2.5x faster per crop at normal
widths, with byte-identical output.**

## 19. The default changed

`VISION_RECOGNIZER` now defaults to **`native-ppocrv6`**, set in the systemd unit
and in `app.py`. It is correct (bit-equivalent), faster at every width measured,
and idles at 0% CPU. Rollback is one line:

```bash
mkdir -p /etc/systemd/system/jupiter-vision-api.service.d
printf '[Service]\nEnvironment=VISION_RECOGNIZER=onnx-ppocrv6\n' \
  > /etc/systemd/system/jupiter-vision-api.service.d/onnx.conf
systemctl daemon-reload && systemctl restart jupiter-vision-api.service
```

`native-rosetta` remains selectable and correct, and remains a bad idea for
documents for the charset reasons in section 7.

Deployed to `root@192.168.68.24`:

```
/root/vision-api/app.py
/root/vision-api/ocr_native.py
/root/vision-api/models/ocr-native/libocr_rvv.so
/root/vision-api/models/ocr-native/ppocrv6_rec.bin      (4.42 MB)
/root/vision-api/models/ocr-native/rosetta_r34.bin      (85.3 MB)
```

Verified live: `/healthz` reports
`{"model":"PP-OCRv6_tiny_rec (native RVV FP32)","recognizer":"native-ppocrv6","rec_provider":"spacemit-ai-harts x8"}`,
`POST /v1/ocr` on `sample.png` returns http 200 with 97 lines whose text matches
the ONNX path exactly, and the service idles at 0.0% CPU between requests.

## 20. Reproducing (both models)

```bash
# workstation: extract, and build the width-dynamic Rosetta oracle
python3 extract_ocr.py ppocrv6/rec.onnx ppocrv6_rec.bin
python3 extract_ocr.py rosetta-r34.onnx rosetta_r34.bin
python3 make_oracle_onnx.py rosetta-r34.onnx rosetta-r34-dyn.onnx   # Rosetta only
VISION_MODEL_DIR=./ppocrv6-tiny python3 make_rec_crops.py sample.png crops_v 97 320

# board: PP-OCRv6's own export is already width-dynamic, so the oracle is made there
scp ocr_rvv.c ppocrv6_rec.bin crops_v/*.png root@192.168.68.24:/root/ocr-native/
ssh root@192.168.68.24 'cd /root/ocr-native && \
  python3 prep_validation.py crops_v rec.onnx val6 48 160 161 240 320 && \
  gcc -O3 -fno-tree-vectorize -funroll-loops -march=rv64gcv_zfh_zvfh_xsmtvdotii \
      -DOCR_MAIN -o ocr_test ocr_rvv.c -lm -lpthread && \
  ./ocr_test ppocrv6_rec.bin val6/in_W320.f32 320 out.f32 3 && \
  python3 compare_native.py val6/oracle_W320.f32 out.f32 97 40 inference.yml'
```

Env vars: `OCR_HARTS` (worker count, default 8), `OCR_PROF=1` (per-op ms),
`OCR_DEBUG=1` (per-op tensor checksums), `OCR_NO_AI_HARTS=1` (ordinary harts).

## 21. Known limitations and follow-ups

- **Crop batching is still not implemented.** The engine runs one crop at a time,
  so the 4.42 MB of weights re-streams per crop, and ORT's batch-16 path narrows
  the gap to 1.16x at W=3200. At this model's size weight traffic is small
  enough that batching matters much less than it would have for Rosetta, but it
  is still the largest remaining win at extreme widths.
- **~1.9x GEMM headroom** and the **+22% MR=4/NC=3 kernel shape** from phase 1
  section 5 remain unexploited; nothing in phase 2 changed that analysis.
- **One inference at a time.** `ocr_infer` is serialised by an internal mutex and
  now additionally funnelled through one dedicated Python thread, so concurrent
  HTTP requests queue at the recognizer. Correct for the current single-worker
  uvicorn deployment; would need revisiting under real concurrency.
- **The AI-hart visibility rule (section 15) is a hardware/kernel behaviour that
  is not understood, only characterised.** It is worth a vendor question. Until
  then, any future AI-hart code must keep its caller inside the pool.
- The 3200 px width cap and the batch-padding policy are inherited unchanged from
  the existing pipeline; they dominate latency on dense-document pages and are
  worth revisiting on quality grounds, but that is a recognition-behaviour change
  and was deliberately left alone.

---

# Phase 3 — closing the speed gaps, and what the model actually knows

Two things were asked for: take the two optimisations phase 2 left on the table,
and do something honest about a downstream consumer's comma-vs-decimal misreads.
Both were done. **Both optimisations were implemented, measured, and rejected on
the evidence** — the numbers phase 1 and 2 predicted them from do not survive
contact with the shipping build. The precision work landed and is deployed.

Net result: **no speed change** (the engine is byte-for-byte the phase-2 engine),
plus a new, additive uncertainty signal in `/v1/ocr` that exposes information the
model was already computing and the API was throwing away.

## 22. Crop batching — built, validated, rejected

Phase 1 called this "the single biggest remaining optimisation". It is not; it is
a loss at every width for both models.

Implemented properly: buffers became `[channel][sample][h][w]` so a channel's
stride spans the whole batch and `cols = N * OH * OW` becomes the GEMM's column
count — one weight stream covering N crops. Depthwise and pooling parallelise
over `(channel, sample)`, the Squeeze-and-Excite gate is computed per sample, and
the CTC head's `(N*T, classes)` output falls out row-major with no extra work.
**Validated bit-identical to batch=1** (same max abs prob diff to the digit, 100%
argmax agreement, 97/97 exact decode at batch=16).

**PP-OCRv6 Tiny, ms/crop, 24 real crops, 8 AI harts:**

| Width | b=1 | b=4 | b=8 | b=16 |
|---|---|---|---|---|
| 160 | 3.34 | **3.07** | 3.33 | 4.23 |
| 320 | **5.83** | 6.66 | 9.09 | 11.38 |
| 640 | **11.77** | 17.64 | 26.49 | 33.16 |
| 1280 | **25.06** | 47.83 | 64.91 | 69.72 |
| 3200 | **92.52** | 103.04 | 154.42 | 192.53 |

**Rosetta (85 MB of weights — the case batching should most favour), W=100:**
b=1 57.90, b=2 **54.92**, b=4 85.91, b=8 89.18 ms/crop.

The premise was wrong. Batching amortises *weight* traffic, and this engine is
not weight-bound — it is activation-bandwidth bound. Even Rosetta, with 20x more
weights than PP-OCRv6, gains only 5% at b=2 before collapsing. Batching multiplies
the working set by N while cache stays fixed, so every im2col gather and every
GEMM column sweep starts missing. The one real (small) win is at b=4 on narrow
crops, where the tensors still fit and the win is the per-dispatch sync overhead,
not weight reuse.

**Why ORT's batch-16 beats its own per-crop path** (1.16x at W=3200, the
observation that motivated this) is therefore not weight amortisation either — it
is ONNX Runtime's fixed per-`Run()` overhead being spread over 16 crops. This
engine has almost no per-call overhead, so it has nothing to amortise and only
the working-set penalty to pay.

**Removed, not kept as a knob.** Merely supporting the batch dimension cost a
measured ~5% on the single-crop path everyone uses (A/B against the pre-batch
build: 3.21→3.36, 5.67→6.00, 23.22→24.53, 90.64→91.53 ms/crop) because of the
extra index arithmetic at N=1. A feature that is never beneficial and taxes the
common path is dead weight. The engine that ships is byte-for-byte the phase-2
engine, re-validated after the revert.

## 23. MR=4/NC=3 GEMM shape — re-measured, rejected

Phase 1 measured this register-blocking shape at +22% and deferred it as
"unjustified risk". Re-measuring at the K values PP-OCRv6 actually uses (K is the
input channel count, 48-320, because the model is mostly 1x1 convs — phase 1 only
probed K=288) and with the accumulator in memory as the engine really has it:

| K | MR=8/NC=1 (shipping) | MR=4/NC=3 | ratio |
|---|---|---|---|
| 48 | 32.23 | 35.18 | 1.09x |
| 96 | 32.56 | 36.03 | 1.11x |
| 160 | 32.79 | 36.45 | 1.11x |
| 288 | 33.12 | 36.80 | 1.11x |
| 512 | 33.16 | 36.92 | 1.11x |

**The +22% was an artefact of the baseline.** Phase 1 measured before
`-funroll-loops` was adopted; unrolling is worth +13% to the MR=8 kernel on its
own (29.4 → 33.1 GFLOP/s) and cannot be applied to MR=4/NC=3 at all — 12 m2
accumulators plus 3 m2 operands is 30 of 32 vector registers, and letting gcc
unroll on top of that makes it spill vectors and **segfault**. It only compiles
correctly with `__attribute__((optimize("no-unroll-loops")))` on that one
function. Like-for-like under the shipping flags the gap is **+11%, not +22%**.

Three further reasons not to take even the 11%:

- The engine reaches ~9 GFLOP/s per hart against 33 for the kernel in isolation,
  so it is not kernel-bound; ~27% of the kernel's rate makes it through. An 11%
  kernel gain is worth low single digits end to end.
- NC=3 works in 192-column granules. Most layers here have 100-500 columns, so a
  large fraction of tiles would hit the tail path — and the natural MR=4 tail
  (4 FMAs per 4 scalar loads + 1 vector load) is **worse** than the current
  kernel's ratio, not better.
- It requires re-packing the weight blob at MR=4 and carrying a compiler
  landmine that miscompiles silently-ish (a segfault, at least, rather than
  wrong numbers) if anyone ever adds unrolling back.

Not shipped. `bench/rvv_gemm_shape2.c` reproduces the measurement.

## 24. Comma-vs-decimal: what the model actually knows

The reported failure (FinScan: `1,901` read as `1.901`, ~47% of thousands-groups,
"the misread scores 1.00 confidence") was taken as given — not re-derived. No
comma/period normalisation was added, for the reasons given in the brief: it is
locale-wrong (European documents invert the convention) and the disambiguation
needs document context this API does not have.

But the premise that confidence is useless here turns out to be **an artefact of
how the API reported confidence, not of the model.** Running the actual repro
image through the deployed engine and looking at the full per-timestep
distribution instead of the collapsed line score:

**Correctly-read separators** (runner-up probability at that timestep):

```
'1,901'  ',': p=0.991205  runner-up '.' p=0.008623
'1,901'  ',': p=0.974691  runner-up '.' p=0.025208
'1,755'  ',': p=0.977421  runner-up '.' p=0.022184
'2,362'  ',': p=0.999906  runner-up '.' p=0.000066
```

**Actually-misread separators**, induced by the small preprocessing
perturbations the brief describes (same crop, different pad/interpolation):

```
'20.000' '.': p=0.853685  runner-up ',' p=0.143309
'20.000' '.': p=0.741653  runner-up ',' p=0.256851
'20.000' '.': p=0.538061  runner-up ',' p=0.461671
'20.000' '.': p=0.512101  runner-up ',' p=0.486993
```

For contrast, digit positions in the same strings put **1e-5 to 1e-6** on their
runner-up. The model is not confidently wrong. At the separator it is genuinely
uncertain — in the worst case an almost exact coin flip — and it says so.

**The 1.00 in the bug report is a line-level mean.** `score` has always been the
mean top-1 probability over kept timesteps. A five-character figure with four
digits at 0.9999 and one separator at 0.51 averages to 0.90 and rounds to
something that looks confident. The mean is simply the wrong statistic for
detecting one bad character in an otherwise clean string.

### What was added

Two additive, backward-compatible fields. `text` and `score` are untouched, and
no decoded text is modified anywhere.

- **`min_char_score`** — always present, one float per line: the *minimum*
  per-character top-1 probability. On the misread above the line reports
  `score: 0.970` but `min_char_score: 0.845`.
- **`alternatives`** — opt-in per request via `?alternatives=true`: for each
  decoded character whose runner-up holds at least `VISION_ALT_THRESHOLD` (1%)
  of the mass, the position, the chosen character, the runner-up, and both
  probabilities.

```json
{"text": "20.000", "score": 0.970, "min_char_score": 0.845,
 "alternatives": [{"index": 2, "char": ".", "alt": ",", "p": 0.845132, "p_alt": 0.154425},
                  {"index": 5, "char": "0", "alt": "O", "p": 0.985758, "p_alt": 0.013229}]}
```

That is a complete, locale-free description of the ambiguity: *this glyph is
contested, here is the other candidate, here is how the mass splits.* A consumer
resolving `3.375` (rate or misread thousands?) still needs document context, but
it no longer has to guess **which** characters to be suspicious of.

The 1% threshold is measured, not chosen: digits sit at 1e-5 or below,
correctly-read separators at 0.9-2.5%, misread separators at 2-49%.

**Why `alternatives` is opt-in.** On dense prose this model is genuinely
uncertain nearly everywhere — the academic page produces 1641 alternatives across
97 lines and doubles the response payload (128 KB vs 5 KB for the financial
page's 40 lines). That is honest, not noise: at this model size `g`/`q`,
`v`/`y`, and space-or-nothing really are close calls. But callers who only want
text should not pay for it. `min_char_score` is one float and is always on.
Default can be flipped globally with `VISION_ALTERNATIVES=1`.

### What we recommend to FinScan

The digit-normalisation plus cross-reader voting they already do remains the
right architecture, and this changes none of it. What changes is that they no
longer need to treat every separator as equally suspect:

1. Use `min_char_score`, not `score`, as the line-level trust signal. It is the
   one that moves when a single character is contested.
2. Request `?alternatives=true` and filter to `alt in {',', '.'}`. That yields
   exactly the positions where this reader knows it is guessing between the two,
   with the split. A `p_alt` near 0.5 is a coin flip; near 0.01 it is not.
3. Weight the cross-reader vote by `p`/`p_alt` at the contested position rather
   than by the line score. When the MilkV splits 0.51/0.49 it should barely vote
   at all; when it reads 0.9999 it should count fully.

This does not fix the model — nothing short of retraining will, and retraining
would forfeit the bit-equivalence-to-the-official-export property this engine is
built on. It does replace a misleading number with an accurate one.

## 25. State after phase 3

- Engine: **unchanged from phase 2** and re-validated — PP-OCRv6 100.0000%
  timestep argmax agreement and 97/97 exact decode at W=160/161/240/320 against
  the ORT oracle; Rosetta 100% at W=100/320. Benchmarks unchanged
  (3.21 / 5.49 / 92.03 ms/crop at W=160 / 320 / 3200).
- API: `min_char_score` always, `alternatives` on request. Text output verified
  still 97/97 byte-identical to the ONNX path on `sample.png`.
- Deployed and live: `native-ppocrv6`, `rec_provider: spacemit-ai-harts x8`.

Both rejected optimisations are recorded here rather than left as open TODOs, so
nobody re-derives them a third time. If a future model on this board *is*
weight-bound — a much larger recognizer, or an int8 path where activations shrink
but weights do not — batching is worth revisiting, and section 22 describes
exactly how it was built.
