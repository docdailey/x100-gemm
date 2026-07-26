# research_feed_paths.md — How to feed SpaceMIT K3 / Milk-V Jupiter 2 better

**Role:** research note for the decode throughput effort. Complements `codex_recs_1.md` (implementation roadmap) and `docs/HARDWARE.md` (measured map).  
**Question:** is there a path past “CPU `vle8` + plain `vmadot` + DRAM stream” that actually uses this hardware better?  
**Status:** hypotheses + external evidence + in-repo vendor evidence. **Fill measured numbers into the tables as probes land.** Do not treat this file as proof until board results are written in.

---

## 1. Executive synthesis

Three layers matter. There is **no** credible evidence of a secret 30–60 TOPS decode path past LPDDR. There **is** strong evidence the custom pure-C path uses only a **subset** of IME-2, while SpacemiT’s own kernels use native int4/i2 ISA forms, N32 panel kernels, in-loop scale folding, and a persistent A100 worker model.

| Layer | Missing / underused relative to vendor | Likely impact on decode tok/s |
|-------|----------------------------------------|-------------------------------|
| **Kernel/ISA** | `vnpack4`, typed `vmadot … i4`, `vmadotsu.hp`, i2×i8, N32K32 panels, in-register fp16-scale fold | Potentially large removal of current fold/dispatch overhead; **not** a reduction below W4 weight bytes |
| **Runtime** | Persistent A100 workers, spin barriers, tile queue, pack-once | Large for MoE (many small GEMVs) |
| **Product** | Spec / ngram / M-batch so one W stream yields more accepted tokens | Largest *useful* tok/s (external proof) |

Marketing “>10 tok/s @ 30B” is **MoE + good software and/or accepted-token amplification**, not a magic matmul. External Jupiter 2 measurements report roughly 6.5–7.2 raw decode tok/s for a ~3B-active quality MoE and 15.5 tok/s only on a copy-heavy ngram-speculative workload with 81% acceptance.

### Answer to “can this take us further after the current path?”

**Yes, but not every path is equally credible.** Once `codex_recs_1.md` PR1–PR3 have produced a trustworthy profile, the best next kernel experiment is a minimal port of the active vendor `gemm_kernel_i8i4_m1` shape. It directly combines three things the current engine lacks:

1. one activation group reused across an N32 output panel;
2. native `.hp` int4 operations; and
3. per-group fp16 scale application into fp32 accumulators before the next K group.

That can remove much of the scalar group-fold path and reduce activation loads and output-store traffic. It cannot evade reading the W4 weights from memory, so the end state is still bounded by packed-weight bandwidth. After kernel/runtime parity, M-batching and ngram/speculative verification are the credible ways to go beyond single-token bandwidth limits.

Do not make TCM/ai_dma the automatic next project. It advances only if cold-stream overlap measurements beat direct DRAM streaming end to end.

---

## 2. External ground truth (same SoC class)

Source: [Tao of Mac — MilkV Jupiter 2 / SpacemiT K3](https://taoofmac.com/space/reviews/2026/06/11/1830) (Jun 2026), SpacemiT / fork llama.cpp stacks.

| Workload | Stack | Decode (reported) |
|----------|--------|-------------------|
| TinyLlama 1.1B Q2_K | `llama.cpp-tools-spacemit` | ~36.6 tok/s |
| Qwen3 0.6B | SpacemiT llama.cpp | ~43–55 tok/s |
| Qwen3.6-28B-REAP-A3B (~3B active) | same | ~6.5–7.2 tok/s |
| Same MoE + ngram draft (~81% accept) | same | ~15.5 tok/s |
| Gemma 4 E2B QAT | same | ~12.9 tok/s |

**BW-consistent story:** ~3B active × ~0.5–0.625 B/weight ≈ 1.5–1.9 GB/token, before attention, lm-head, padding, and metadata. The external 6.5–7.2 tok/s result is a useful software target, not proof that the same rate follows from a simple DRAM-GB/s division. If pure-C MoE remains far below that on the **same GGUF, context, thread map, and sampling settings**, the gap is primarily software path rather than undiscovered silicon peak.

**Also reported (and important negatives):**

- X100 draft + A100 target MTP: **no** decode win (no free fast fabric between clusters).
- Decode tracks **1 / active parameters** (memory-bound), not peak IME TOPS.
- Linlon V5 / Zhouyi NPU appears in DT; **no public driver/SDK** → dead for now.
- PowerVR OpenCL exposes `integer_dot_product`; interesting for vision/research, not the measured LLM path.

---

## 3. What this repo already proved

See `docs/HARDWARE.md`, `bench/EP_INVESTIGATION.md`, `bench/*`.

| Fact | Implication for “feed” |
|------|------------------------|
| A100 unlock: `/proc/set_ai_thread` then affinity | Required for any IME-2 userspace path |
| IME-2: 4 units, pairs (8,9)…(14,15); VLEN 1024; tile 8×8×16 int8 | Resource is units + BW, not core count alone |
| Register-fed ~14.6 TOPS int8 | **Not** decode ceiling |
| L1-fed GEMM ~2.3 TOPS; vendor EP ~2.8 TOPS @ ~1 core | Real matmul ceiling class; no 60 TOPS path |
| EP uses `/dev/tcm`, `/dev/ai_dma`, mmap ring/doorbell | Offload exists; RE is large; still ~2.8 TOPS class |
| Decode is weight-stream bound | Bytes/token and achieved GB/s dominate |
| q4-in-q8 interleave is **correct** and beats int8 BW | Good intermediate design; may not be native-optimal |

---

## 4. Path A — Vendor-shaped IME-2 M1×N32 int4 kernel (strongest next kernel hypothesis)

### 4.1 What the custom engines do today

- Requant to signed int4, **software** pack two K16 tiles into 128 B.
- Unpack with `vsll` / `vsra`.
- Plain `vmadot` (int8-shaped operands).
- M=1 decode pads 7 of 8 tile rows with zeros.

### 4.2 What vendor kernels in-tree already do

Evidence: `reference/spacemit-backend/ime2_kernels.cpp`.

**Native nibble / typed-dot operations:**

```asm
vnpack4.vv   v8, v3, v3, 3          # lo4 of A
vnpack4.vv   v10, v24, v24, 3       # hi4 of A
vmadotsu     v16, v10, v4, i4       # typed int4 MAC
vmadotu      v16, v8, v4, i4
vpack.vv     ...                    # rearrange half-results
```

**Half-path multi-acc form and low-bit modes:**

```asm
vmadotsu.hp  v24, v2, v8, v0, 0, i8
# ... and i2 × i8 kernels with vmadotsu.hp ...
```

**B layout and scale fold (comments and active assembly in the same file):**

```text
A:  M1 × K32 int8     (256-bit) + fp32 scale + int16 asum
B:  N8 × K32 int4     (1024-bit full VLEN; e.g. vl4r.v load)
    + fp16 scales for N32 panels
```

The important local evidence is stronger than the public narrative: in `gemm_kernel_i8i4_m1`, the ordinary typed-`i4` implementation is under `#if 0`, while the `.hp` implementation in the `#else` branch is active. It processes N in panels of 32, loads four vector registers of B data, loads 32 fp16 B scales, executes the typed int4 dot sequence, converts/scales inside the K-group loop, and retains an fp32 N32 accumulator until the final store.

The `vnpack4` operations split the int8 activation into nibble contributions used to reconstruct its signed value against packed int4 B. They do **not** further compress the already-int4 weight stream.

### 4.3 Why this might beat q4-in-q8 interleave

| | Custom interleave | Vendor-shaped M1×N32 int4 |
|--|-------------------|--------------------|
| Output panel | N8 | N32 (four N8 blocks) |
| Activation use | two padded 8×16 A loads per N8/K32 | one M1K32 activation group reused across N32 |
| Weight bytes | 128 B per N8K32 | 512 B per N32K32 — **same 0.5 B/weight** |
| Dot/fold | two int8-shaped `vmadot`, store 8×8 int32 each group, scalar fp32 fold | typed `.hp` int4 sequence, fp16 B-scale multiply, fp32 accumulation inside K loop |
| Scale stream | separate fp32 `gs[]` = 0.125 B/weight | fp16 group scales = 0.0625 B/weight |
| Final output | group partial stores plus scalar reduction | one final N32 fp32 store |

Hypothesis: interleave is a **correct compatibility shim**; the vendor shape wins primarily through N32 activation reuse and an in-register group-scale fold, with a secondary scale-bandwidth win from fp16. It does **not** halve W4 data again. This distinction matters: if the current full-token profile is already dominated by cold packed-weight reads, the native kernel’s ceiling gain will be limited; if scalar fold/dispatch dominates, it can be substantial.

### 4.4 Probe plan (falsifiable)

| ID | Probe | Pass / fail | Result (fill in) |
|----|--------|-------------|------------------|
| A1 | Assemble+run `vnpack4.vv`, typed `vmadotsu … i4`, and `vmadotsu.hp` on A100 | No illegal insn; record compiler/binutils and encodings | |
| A2 | Extract/port the minimal active M1×N32K32 `.hp` kernel and its exact A/B pack | Correct against fp32/grouped oracle | |
| A3 | Hot A/B: vendor shape vs custom interleave at identical N/K/data | Split kernel, fold, and total time | |
| A4 | Cold/model-stream A/B with identical W4 values and scales | Report µs, **physical bytes/s**, useful weights/s, correctness | |
| A5 | Full-token substitution on the same cached model | tok/s and bucket changes | |

**Do not** rewrite production engines until A1–A4 land. Do not compare the two kernels using only “effective int8 GB/s”; report actual bytes read for packed weights and scales.

### 4.5 Risk / benefit

| Benefit | Risk |
|---------|------|
| Removes scalar group fold and reuses A across N32 | Encoding may not match Bianbu as; exact register/pack contract is delicate |
| Aligns pack format with SpacemiT llama.cpp / ONNX | Different cache format; version requant caches |
| fp16 scales reduce scale bytes | Rounding/quality differs from current fp32 scales |
| i2 path for selected tensors | Severe quality cliffs; separate research after int4 parity |

---

## 5. Path B — Runtime feed (persistent A100 team)

### 5.1 Vendor / SpacemiT llama.cpp pattern

- Persistent pthreads on A100 (the external report describes **six workers on harts 8–13**; verify the referenced fork/configuration locally rather than treating six as universally optimal).
- `spine_barrier_t` (atomic spin barrier) — see `reference/spacemit-backend/spine_barrier.h`.
- Shared threadpool/tile work; workers remain alive between operations.
- TCM paths in `reference/spacemit-backend/ime.cpp` use `rvv::memcpy1d` and `spine_barrier_wait` to stage panels. This is local evidence for CPU-copy/barrier staging, not proof that `ai_dma` is used by this llama.cpp path.

Tao of Mac reports 36.6–55 tok/s on small Qwen/TinyLlama cases and attributes part of the result to the persistent worker model. Treat attribution as an external observation; the same source reports only about 6.5–7.2 raw decode tok/s on the ~3B-active quality MoE.

### 5.2 Custom engines today

- `#pragma omp parallel` **per** `lin_mm`.
- Re-quant of activations **inside** each linear.
- Hot `malloc` of group partials (to be removed per `codex_recs_1.md`).

### 5.3 Hypothesis

For MoE (router + 8×3 expert GEMVs + attn), **dispatch overhead and re-pack** dominate a non-trivial fraction of token time. Microkernel ILP cannot fix that.

### 5.4 Probe plan

| ID | Probe | Result (fill in) |
|----|--------|------------------|
| B1 | Empty OpenMP region + barrier cost × the measured number of `lin_mm` calls | µs / token |
| B2 | Many-small GEMV (MoE-shaped) OpenMP-per-call vs persistent pool | |
| B3 | Full-token profile: pack count, parallel-region count | |
| B4 | 4-across (8,10,12,14) vs 6-worker vs 8-paired on **cold** W | tok/s, GB/s |

Aligns with codex PR2/PR5/PR8 — measure first, then persistent pool only if B1–B2 show material overhead.

---

## 6. Path C — TCM + ai_dma (real, gated)

### 6.1 What is real

- ~3 MB on-chip TCM; pair-associated blocks (order ~8 × 384 KB).
- `libspine_tcm.so` / `spine_tcm.h` API in-tree.
- EP opens tcm + **ai_dma** + aidma_list + dma_msi; submission via **mmap ring** (no per-xfer ioctl storm).
- llama.cpp spacemit path often uses **CPU `memcpy1d`** into TCM rather than full DMA RE.

### 6.2 What is oversold

| Claim | Caution |
|-------|---------|
| ~58 GB/s spine_tcm read | Likely **warm cache / L2**; codex_recs requires cold-stream proof |
| Docs ~5.4 GB/s A100 TCM | Scratchpad class, not second DRAM |
| Fit 8 MoE experts in TCM | One expert gate+up+down W4 ≈ **2.4–3 MB** (with scales) → nearly all of TCM; **panels / double-buffer**, not whole expert set |

### 6.3 Interesting pattern (not “infinite SRAM”)

```text
hart pair unit:
  buf A: compute from TCM
  buf B: DMA or memcpy next weight panel from DRAM
  spine barrier → swap
```

Attempts to hide DRAM latency behind IME work. It helps only if copy and compute genuinely overlap and measured **full-token** time drops. CPU `memcpy1d` consumes load/store and memory bandwidth; it is not automatically free just because another hart performs it.

### 6.4 Probe plan

| ID | Probe | Result (fill in) |
|----|--------|------------------|
| C1 | First-touch TCM read BW; working set > L2; cold vs warm | GB/s |
| C2 | DRAM→TCM memcpy BW + latency | |
| C3 | Overlap: partner memcpy while compute on other hart; compare direct-DRAM kernel | full GEMV µs, total DRAM bytes, worker CPU time |
| C4 | ai_dma ring: objdump EP + one scripted submit (research branch) | |

**Gate:** do not build production decode on TCM until C1–C3 are written. Prefill offload via EP remains the lower-risk path for large matmuls (~2.8 TOPS class).

---

## 7. Path D — Product-level feed (same silicon, more useful tokens)

These do not raise peak TOPS; they raise **accepted tokens per second** or **tokens per weight stream**.

| Idea | Mechanism | External signal |
|------|-----------|-----------------|
| **M-batch independent sequences** | One W stream, M≤8 rows of IME tile | Tile geometry free rows today |
| **Expert-grouped MoE** | Stream expert e once for all tokens that selected e | Serving / multi-seq |
| **Ngram / draft spec** | Verify many drafts per W pass | ~15.5 tok/s on 28B-REAP class |
| **Native GGUF→IME pack** | Avoid dequant→requant→custom pack | Vendor cache format |
| **fp16 group scales** | 0.5625 vs 0.625 B/weight | Vendor B scales are fp16 |
| **i2 on selected tensors** | Extreme density | Vendor i2×i8 kernels exist |

The cited external experiment found no benefit from placing an assistant drafter on X100 while the target used A100. That result does not prove there is literally no fast fabric; it proves that tested scheduling did not improve end-to-end decode. Keep draft/verify within one coordinated runtime unless a new end-to-end measurement says otherwise.

---

## 8. Dead or weak paths (do not prioritize)

| Path | Verdict |
|------|---------|
| Secret 60 TOPS real matmul | Killed by EP ~2.8 TOPS + L1 ceiling ~2.3 |
| Linlon V5 NPU userspace | No driver/SDK |
| More register-fed `vmadot` peak | Orthogonal to decode BW |
| Cross-cluster draft/verify for free speedup | Measured no win |
| Raw uncached `/dev/tcm` CPU `vle8` | ~0.4 GB/s class — anti-pattern |

---

## 9. Ranked research agenda

| Rank | Focus | Decides | Owner suggestion |
|------|--------|---------|------------------|
| **R0** | Finish codex timing buckets + same-GGUF SpacemiT baseline | Establish the gap and its location | Required gate |
| **R1** | Assemble/run probe for `vnpack4` / typed `i4` / `.hp` | Is Path A available in this toolchain? | Small board binary |
| **R2** | Minimal active vendor M1×N32 kernel + correctness oracle | Can it reproduce grouped W4 output? | Standalone benchmark |
| **R3** | Cold vendor-shape vs `q4_gemv` A/B | Keep interleave or migrate pack? | Physical-byte and time table |
| **R4** | Full-token kernel substitution | Does the micro win survive engine glue? | Engine feature flag |
| **R5** | Persistent workers vs OpenMP (B-probes) | Whether PR8 is warranted | |
| **R6** | M-batch / expert grouping / ngram spec | More tokens per W stream | Product path |
| **R7** | Cold TCM + pair double-buffer (C-probes) | Whether staging beats direct DRAM | Research only |

---

## 10. Relationship to `codex_recs_1.md`

| codex theme | This research note |
|-------------|-------------------|
| Measure first; don’t assume mid-loop store vanishes | Still true; Path A is **additional** ISA surface |
| W4 ≈ 0.625 B/weight with fp32 scales | Path D: fp16 scales as later density knob |
| TCM 58 GB/s not streaming | Path C gate |
| M>1 not validated in `q4_gemv` | Path D depends on fixed M-oracle |
| Persistent pool only after overhead measured | Path B |
| Prefill via EP ~2.8 TOPS | Unchanged |

**Do not** reorder codex PR1–PR3 for Path A/C. Small illegal-instruction probes may run at any time, but the result that matters is a cold, same-data A/B after the profiler exists. Merge ISA changes only after A1–A5. Path C remains behind the stronger cold-stream gate.

---

## 11. In-repo and external source index

| Source | Use |
|--------|-----|
| `reference/spacemit-backend/ime2_kernels.cpp` | Native int4/i2/`.hp`/`vnpack4` asm |
| `reference/spacemit-backend/spine_tcm.h`, `spine_barrier.h` | TCM API, worker sync |
| `reference/spacemit-backend/repack.cpp` | GGUF → interleaved/native pack |
| `docs/HARDWARE.md` | Measured topology / ceilings |
| `bench/EP_INVESTIGATION.md` | EP ~2.8 TOPS, ai_dma devices |
| `bench/q4_gemv.c` | Custom interleave oracle |
| [spacemit-com/riscv-ime-extension-spec](https://github.com/spacemit-com/riscv-ime-extension-spec) | Public IME proposal (int4/fp types) |
| [Tao of Mac K3 review](https://taoofmac.com/space/reviews/2026/06/11/1830) | End-to-end tok/s, worker/TCM narrative |
| SpacemiT / community llama.cpp K3 forks | Production baseline |

---

## 12. Results log (append only)

Copy a row per probe run. Prefer the fuller template in `codex_recs_1.md` §18 for engine tokens.

| Date | Probe ID | git SHA | Setup | Metric | Notes |
|------|----------|---------|-------|--------|-------|
| 2026-07-26 | A1 | 31504f6 | A100 hart 8, gcc 15.2.0 (Bianbu), binutils 2.46, `-march=rv64gcv_zvfh_xsmtvdotii` | illegal? N | `vnpack4.vv`, `vmadotsu`/`vmadotu ...i4`, `vmadotsu.hp`/`vmadotu.hp` all assemble+run with the *same* march token already used for plain `vmadot` — no new extension flag needed. `bench/vendor_ime_probe.c`. Side finding: vendor B int4 is **unsigned nibble (0-15) with a baked-in zp=8** (`vmv.v.i v28,8` + asum·zp correction), NOT signed two's-complement like our current `pack_w_int4` — A2's pack routine must differ from ours, can't reuse it. |
| 2026-07-26 | A2 | 31504f6 | Hand-ported `gemm_kernel_i8i4_m1` (.hp branch) from `reference/spacemit-backend/ime2_kernels.cpp`, own A/B pack per kernel comments | correct? PARTIAL/UNVERIFIED | Kernel executes deterministically (`bench/vendor_ime_a2_probe.c`) and single-element probes disproved the naive `raw-zp*asum` hypothesis (real: `raw+zp*asum`), but then found the **actual production A-pack** (`quantize_a_row_i8_hp` in `rvv_kernels.cpp`) uses a two-level fp16 scale (per-32 relative ratio x per-256 shared average) + a **pre-scaled `-asum*8` fp16 term**, not the fp32-scale/raw-int16-asum layout the kernel's own inline comments imply. The 5768-line reference file has multiple gemm_kernel/quantize_a_row variant pairs (RVV-intrinsic and raw-asm, HP and non-HP, per-quant-type); I paired the wrong two. **Do not trust this hand port's correctness** — needs either full call-graph tracing in `ime.cpp` to find the true pair, or `objdump` of the actually-compiled `libggml-cpu.so` as ground truth instead of the (possibly stale) reference source snapshot. |
| 2026-07-26 | A5-equiv | 31504f6 | Real vendor binary `/root/llama.cpp/build/bin/llama-bench` (SpacemiT fork, build 308f61c, IME2 on), same GGUF (`Qwen3-30B-A3B-Q4_0.gguf`), decode-only (`-p 0 -n 16`) | tok/s | **t=4: 11.71±0.19 tok/s. t=8: 12.89±0.03 tok/s** (2 reps each, reproducible) vs our pure-C engine's **1.49 tok/s** at t=4 — **~8x gap**. This sidesteps A2's fragile hand-port entirely: the real compiled vendor kernel was already on the board. `t=16` and the combined `pp+tg` benchmark mode both crash (`malloc(): invalid size`) — `/dev/tcm_sync_mem` doesn't exist on this board (`open() failed, errno=2`, silently falls back to heap; that fallback path appears to have a bug under more threads/back-to-back benchmarks). t=4/t=8 decode-only is stable. |
| 2026-07-26 | A2-corrected | ed8b89d | Traced `ime.cpp`'s dispatch (line ~276): for Q4_0/Q4_1, INTER_SIZE==256, count_m<4, the real call chain is `gemm_kernel_i8i4_hp` → **`gemm_kernel_i8i4_hp_m1`** (ime2_kernels.cpp:2883) — NOT `gemm_kernel_i8i4_m1`, which was A2's original wrong pairing. Paired with the real `quantize_a_row_i8_hp` (rvv_kernels.cpp:1989, ported to scalar C) for A and `make_block_q4_0x32` (repack.cpp:292, also ported) for B. | correct? **VERIFIED PASS** | `bench/vendor_ime_a2_full.c`: verbatim asm transcription of `gemm_kernel_i8i4_hp_m1`, faithful ports of both real pack routines, validated against an independent scalar dequant reference (built from the same quantized nibbles/scales, isolating kernel correctness from quant noise). N=32,K=256 random test: **max rel diff 2.3%**, consistent with expected fp16 intermediate-accumulation noise (the kernel accumulates in fp16 before final fp32 widen), not a bug. Byte offsets independently confirmed: the asm's `+272`/`+288` constants exactly match `q8_hp_blk_size(256,true,true)=290B`'s derived layout. Real B nibble-pairing (from `make_block_q4_0x32`) is adjacent-pair `{2j,2j+1}`, **not** the native-ggml `{j,j+16}` pairing A1/A2's first attempt assumed — that was the root cause of the original mismatch. |
| 2026-07-26 | A3 | ed8b89d | Hot, in-cache, kernel-only (no pack overhead), identical N=32 K=256, 200k reps, both on A100 hart 8 | ns/call | **ours (4x `gemv_nb_int4_grouped` N8K256): 2972.8 ns/call. vendor (`gemm_kernel_i8i4_hp_m1`, N32K256 in one call): 446.4 ns/call. 6.66x faster.** Close to the ~8x full-token gap from A5-equiv, meaning kernel microarchitecture (not ggml dispatch/threading overhead) is the dominant factor — confirms porting this kernel into our engine is the right lever. |
| 2026-07-26 | A5 (real) | 3131fef | `qwen_moe_hp.c` full engine integration (see A2-corrected/A3 rows), run against the real 30B-A3B model, nt=4 | tok/s + correctness | **6.19 tok/s, ' Tokyo' PASS, coherent generation matches the original engine exactly.** Up from our 1.49 tok/s (**4.16x**), still below the real vendor binary's 11.71-12.89 (A5-equiv). Requant into the new format took **1104.5s (~18.4 min)** — much slower than the old format's ~2 min parallel requant (real cost: more fp16 conversions + 8-subblock nesting per weight group); cache save 92.8s. Buckets at the time looked like `linear(kernel)` 38.4ms / `rest` 100.3ms=62% — **this bucket split was WRONG, see PR8 row below.** |
| 2026-07-26 | PR8 + bucket fix | (pending commit) | Implemented the persistent spin-dispatch pool (threads created once, generation-counter dispatch instead of `#pragma omp parallel` per `Lin` call) per the A5 row's hypothesis. **Result: 6.19→6.53 tok/s, only ~6% gain — nowhere near the ~16 tok/s projected.** Investigating why led to a real bug, not a wrong hypothesis about the hardware: the new engine's `lin_mm()` wrapper (used for `o`, all 8 `ed[e]`, and `lm` — over a third of per-layer Lin calls) had **zero timing instrumentation**, unlike the old engine's version. All of that real linear-algebra time was silently falling into the `rest` bucket the whole time, making it look like 62%-of-wall "mystery overhead" when it was actually mostly real, correctly-attributed work. Fixed the instrumentation (ported the old engine's `_ta/_tb` pattern into the new `lin_mm`) and re-measured: **act-pack 16.6ms, linear(kernel) 58.7ms, attention 16.9ms, rest 59.3ms, wall 152.4ms → 6.56 tok/s.** `linear` and `rest` are now roughly tied as the two biggest buckets; `rest` is genuine scalar C (router matvec O(n_exp·d)=262144 mults/layer unvectorized, per-head RoPE+qk-norm ~4600 sin/cos calls/layer, SwiGLU ~6144 `expf` calls/layer), not dispatch overhead. **Retracting the "~16 tok/s, beats vendor" projection from the A5 row above** — it was built on a bucket count that was silently missing a third of the real work. |
| | R3 | | model=… | tok/s | spacemit llama vs pure-C |
| | | | | | |

---

## 13. Bottom line

1. **Decode remains weight-bandwidth-bound at the end state**, but the current 1.36 tok/s engine has substantial non-stream overhead. Use the same-model vendor result and timing buckets to set the target; do not promise 7–12 tok/s from arithmetic alone.
2. The strongest next kernel path is **(A) the vendor-shaped N32 native-int4 kernel with in-loop scale folding**, followed by **(B) measured runtime improvements** and **(D) batching/spec**. This is not a hidden matrix unit beyond EP’s ~2.8 TOPS.
3. Your **q4-in-q8 interleave remains a valid baseline**; treat vendor kernels as the **hypothesis to beat or match** on cold-stream GB/s and full-token time.
4. Record probe numbers here; promote winners into `codex_recs_1.md` as ordered PRs only after they falsify or confirm the tables above.

---

*Living document. Update §12 and the “Result (fill in)” cells; do not rewrite history of failed probes — keep them so we do not re-chase dead paths.*
