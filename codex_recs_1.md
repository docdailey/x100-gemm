# codex_recs_1 — Make the SpaceMIT K3 sing

This is an implementation roadmap for Claude Code (or any engineer) working on
the pure-C decode engines in this repository. It refines `grok_recs_1.md` using
the current code and measured hardware, and separates proven facts from
hypotheses that still need an on-board experiment.

The goal is **maximum useful tokens/second**, not the largest synthetic TOPS
number.

## Executive direction

The K3 is two different machines sharing memory:

```text
X100 harts 0–7   general RVV work, attention, normalization, routing, packing
A100 harts 8–15  VLEN=1024, four IME-2 units, one unit shared by each hart pair
                  pairs: (8,9), (10,11), (12,13), (14,15)
```

For decode, model weights dominate traffic. Treat the four IME-2 units as a
streaming W4A8 GEMV engine and use X100 capacity for everything surrounding the
linear layers. The default A100 placement should begin with harts
`8,10,12,14`; paired harts are an experiment, not an assumption.

The shortest credible path to higher decode throughput is:

1. Measure a real token and establish where time and bytes go.
2. Quantize/pack each activation once and reuse it across compatible linears.
3. Remove allocation and avoidable scalar work from every GEMV invocation.
4. Optimize the required group-scale fold, based on measurements.
5. Measure 4-unit versus 8-hart scheduling on production-sized, cold streams.
6. Validate and ship M-batched execution so one weight stream serves several
   tokens or sequences.
7. Add persistent scheduling only if measured OpenMP overhead warrants it.
8. Treat TCM/DMA and custom prefill work as separate, evidence-gated projects.

Do not start by chasing 14.6 or 60 TOPS. Those figures describe register-fed or
marketing ceilings, not ordinary model decode.

---

## 1. Facts to preserve

The following are sufficiently established to guide implementation:

- A100 access requires writing `"0"` to `/proc/set_ai_thread` before affinity.
- A100 VLEN is 1024 bits.
- One IME-2 `vmadot` tile is 8×8×16 int8, accumulating into an 8×8 int32 tile.
- Four IME-2 units are shared by four hart pairs.
- Four harts placed across the pairs are the efficient register-fed operating
  point; eight harts add much less compute throughput in that benchmark.
- Decode is principally a weight-streaming workload.
- Interleaved W4 packing is correct and avoids a shuffle:

  ```text
  packed[j] = low_nibble(tile0[j]) | high_nibble(tile1[j])
  low  <- (packed << 4) arithmetic>> 4
  high <- packed arithmetic>> 4
  ```

- The vendor EP reaches roughly 2.8 TOPS on a large int8 prefill-like matmul.
- The current custom IME-2 GEMM is useful research, but it is not the preferred
  production prefill path today.

Primary evidence lives in:

- `docs/HARDWARE.md`
- `bench/EP_INVESTIGATION.md`
- `bench/peak_ime2*.c`
- `bench/ime2_l1_ceiling.c`
- `bench/ime2_interleave.c`
- `bench/q4_gemv.c`
- `src/gemm_ime2_i8.c`

When a new measurement contradicts this document, keep the raw result and
update the model. Do not bend results to preserve the roadmap.

---

## 2. Corrections to the earlier roadmap

### 2.1 Group-scale folding cannot simply eliminate every store

Production W4 uses a different scale for every output channel and every 32
values of K. An int32 result for each group must therefore be converted and
weighted before the accumulators are reused for the next group.

The float-fold sketch in `grok_recs_1.md` still performs one 256-byte
`vse32` per group. Moving that store from a `Kp*64` heap buffer to a 64-element
stack buffer reduces live storage, but does not eliminate the store instruction
or its L1 traffic. The current K=2048 partial buffer is about 16 KiB per thread
and will normally be cache-resident.

The valid optimization question is:

> What is the cheapest correct way to turn each 8×8 int32 group result into the
> eight M=1 fp32 output accumulators?

Candidate implementations must be benchmarked:

1. Existing full partial buffer plus scalar fold — correctness baseline.
2. One reusable 8×8 int32 scratch tile plus immediate scalar fold.
3. One scratch tile plus RVV int32→fp32 conversion and vector FMA with scales.
4. Per-channel/full-K quantization for selected tensors, only if a proper
   quality evaluation passes.

Do not describe option 2 as “no mid-loop C store.” Its potential wins are a
smaller working set, no heap allocation, and a more optimizable fold.

### 2.2 W4 is 0.625 byte/weight in the current representation

Packed weights cost 0.5 byte/weight. The engine additionally stores one fp32
scale per 32 weights:

```text
packed weight       16.0 bytes / 32 weights
group scale          4.0 bytes / 32 weights
total                20.0 bytes / 32 weights
effective             0.625 byte / weight
```

Ignore allocator and page overhead for the first-order model, but do not ignore
the scale array. Current W4 is about 1.6× smaller than raw W8, not 2×. Replace
tok/s ceilings based on `params/2` with `params*0.625` unless the scale format
changes.

Possible later improvement: fp16 group scales reduce the representation to
0.5625 byte/weight. This is only worth doing after testing numerical quality and
measuring whether scale traffic matters.

### 2.3 M=2…8 is not validated by the current benchmark

`bench/q4_gemv.c` packs M rows, but its correctness path extracts and checks
only row zero and allocates W4/W8 result arrays for only N values. Before using
that benchmark as proof of MTP or multi-sequence support, fix it to:

- allocate `M*N` outputs for all implementations;
- extract `ct[m*8+n]` for every active row;
- compare all `M*N` results against the scalar reference;
- poison inactive rows and output buffers to catch accidental dependence;
- test M = 1, 2, 4, 7, and 8;
- test multiple N/K shapes, including tails or explicitly reject unsupported
  tails.

### 2.4 The 58 GB/s TCM number is not yet a streaming result

`bench/tcm_bw3.c` rereads the same pair-local block thousands of times. A
cacheable mapping can therefore turn the test into an L2 benchmark. Do not use
58 GB/s as sustainable TCM bandwidth until a cold/streaming experiment defeats
the cache explanation.

Required TCM experiment:

- report the exact acquired block size and cache hierarchy assumptions;
- compare first pass with warm passes;
- rotate across a working set larger than A100 L2;
- evict or overwrite between samples;
- report CPU loads, DMA copy bandwidth, and simultaneous compute bandwidth;
- verify behavior on all four blocks/pairs;
- distinguish CPU-readable cacheable mapping bandwidth from DMA→IME feed
  bandwidth.

Until then, TCM is research, not a P0 decode dependency.

---

## 3. Performance model to use

For single-token decode:

```text
token_time ≈ weight_bytes / achieved_weight_Bps
           + activation_quant_pack
           + scale_fold
           + attention_and_KV
           + normalization_and_elementwise
           + scheduling_and_barriers
           + sampling
```

For the current group-32 W4 representation:

```text
weight_bytes ≈ active_weights * 0.625
```

This is still an approximation. Some weights remain fp32, tensor dimensions are
padded to IME tiles, and the engine may retain both source GGUF data and repacked
weights. Record actual allocated bytes when possible.

For M independent sequences processed together:

```text
linear weight bytes per batch ≈ one weight stream
useful linear outputs          ≈ M
```

This can approach M× aggregate linear throughput only when the engine actually
batches the sequences at the same layer and weight, and when attention,
activation work, memory capacity, and sampling do not become dominant. Report
both aggregate tok/s and per-sequence latency.

For MoE speculative verification, count the union of selected experts, not
simply `M * active_experts` and not optimistically one token’s expert set:

```text
bytes ≈ shared_attention_weights + router
      + union(selected_experts_across_M)
      + lm_head
```

---

## 4. P0 — Build measurement into the engines

Before changing architecture, make `qwen_ime4` and `qwen_moe` report a useful
per-token breakdown. Wall-clock timers are adequate initially.

Measure at least:

| Bucket | Included work |
|---|---|
| activation | absmax, quantization, tile packing, memset |
| IME kernel | packed weight loads, unpack, vmadot, int32 store |
| scale fold | int32→fp32, group scales, final activation scale |
| OpenMP | parallel-region/barrier overhead not already charged |
| attention | QK, softmax, weighted V, KV movement |
| elementwise | RMSNorm, RoPE, SiLU, residuals |
| router | router matvec, softmax, top-k |
| lm-head | final projection |
| total | complete decode token |

Also report:

- model and quantization;
- prompt position/context length;
- thread count and exact hart map;
- active parameters or estimated weight bytes;
- effective packed-weight GB/s;
- first-token/prefill time separately from steady decode;
- aggregate and per-sequence tok/s for M-batch.

Use a warm-up token, but keep a cold-start measurement as a separate number.
Do not combine packing/requant-cache creation with steady decode.

### Acceptance gate

- Timed buckets sum to within 5% of total token time.
- Repeated steady-state runs are stable enough to distinguish a 5–10% change.
- Output token IDs are recorded for a fixed prompt and seed.

---

## 5. P0 — Prequantize once, reuse often

This is the safest immediate engine win.

Today, `lin_mm()` quantizes its input internally. That repeats work for linears
sharing the same input:

- Q, K, and V share normalized hidden state;
- expert gate and up share normalized hidden state;
- a fused gate/up scheduling region can share the same activation tile;
- in some layouts, multiple selected experts can share the same activation
  tile.

Split the API:

```c
typedef struct {
    int8_t *tile;
    float scale;
    int K;
} PackedActivation;

void pack_activation_q8_m1(const float *x, int K, PackedActivation *px);
void lin_mm_prepacked(const Lin *l, const PackedActivation *px,
                      float *y, int nt, ThreadScratch *scratch);
```

Then schedule:

```text
pack hn once → q, k, v
pack hn once → selected expert gates and ups
pack each post-SiLU expert vector once → that expert’s down projection
```

The Q/K/V outputs may have different N, but share K and the packed activation.

### Important constraints

- Tile memory must remain valid until every consumer finishes.
- M=1 packing zeroes inactive rows once, not once per consumer.
- Preserve scalar rounding semantics initially. Vectorized quantization is a
  separate change so numerical differences are attributable.
- Reject or explicitly handle K not divisible by 32.

### Acceptance gate

- Bit-identical packed activation versus the old scalar path.
- Numerically identical linears for fixed packed weights.
- One activation pack for Q/K/V, verified by counters.
- One activation pack for expert gate/up input, verified by counters.
- Measured token-time reduction, not just a faster isolated pack function.

---

## 6. P0 — Remove hot-path allocation

Both production W4 engines allocate a group partial buffer inside each OpenMP
region and free it after each linear. Replace this with persistent aligned
per-thread scratch.

Suggested structure:

```c
typedef struct {
    _Alignas(64) int32_t group_tile[64];
    _Alignas(64) float fold[8];
    /* optional old-path buffer retained for A/B testing */
    int32_t *all_groups;
    size_t all_groups_capacity;
} ThreadScratch;
```

Allocate scratch at model/engine initialization, one instance per worker. Size
it from the largest K in the model only if the old full-buffer path remains.

Also audit allocations in attention and tokenization. Page faults or allocator
locks must not be part of steady decode.

### Acceptance gate

- No `malloc`, `calloc`, `realloc`, or `free` during a warmed decode token.
- Scratch has no false sharing between workers.
- AddressSanitizer on a non-IME host-side test catches no scratch bounds error.

---

## 7. P1 — Optimize the correct group-scale fold

Keep the existing grouped implementation as the numerical oracle. Add variants
behind an enum or CLI switch so the same executable can A/B them.

### Variant A: immediate scalar fold

For every group:

1. Clear two IME accumulators.
2. Load the two activation K tiles and packed W4 tile.
3. Unpack and execute two `vmadot`s.
4. Sum to one 8×8 int32 tile.
5. Store into one reusable 256-byte aligned scratch tile.
6. For M=1, fold only the first eight int32 values using the eight group scales.

This removes the large partial buffer and second traversal, while retaining the
required per-group store.

### Variant B: immediate RVV fold

After the int32 tile store, use RVV to:

- load the first eight int32 results;
- widen/convert them to fp32;
- load eight scales in an N-contiguous layout;
- FMA into eight fp32 accumulators.

Current scale storage is `[N][Kp]`, which makes eight output-channel scales for
one group strided. Consider repacking scales to `[Nb][Kp][8]`. That layout is
more natural for immediate vector folding and preserves sequential group access
inside an N block.

This scale-layout change should be versioned in the requant cache header so an
old cache cannot be misread silently.

### Variant C: per-channel/full-K experiments

Per-channel W4 allows full-K int32 accumulation and one final store, but changes
quantization quality. Treat this as a tensor-specific quality experiment, not a
universal performance fix.

Test independently on:

- attention Q/K/V/O;
- dense gate/up/down;
- MoE expert gate/up/down;
- lm-head.

### Acceptance gate

- Random-tile unit test against the existing grouped implementation.
- Full-model fixed-prompt output check.
- Short-generation comparison.
- Prefer perplexity or KL/logit comparison over a single “Tokyo” token.
- Report kernel time, fold time, total linear time, and full-token time.

Do not merge a numerically different variant based only on first-token parity.

---

## 8. P1 — Vectorize activation quantization and packing

Only after prepacked reuse lands, optimize the remaining pack operation:

1. RVV absolute-max reduction.
2. Calculate activation scale.
3. RVV fp32→int conversion with controlled rounding and saturation.
4. Narrow/store int8 values.
5. Place each 16-value K segment into row zero of its 128-byte IME tile.

Benchmark two approaches:

- vector quantize to contiguous int8, then scatter/copy 16-byte segments;
- direct strided/tiled stores if instruction cost is lower.

Zeroing the seven inactive M rows writes 8× more activation bytes than useful
for M=1. This is a small absolute buffer, but test whether one pre-zeroed buffer,
targeted row-zero writes, or a padded template copy is cheapest.

### Acceptance gate

- Define whether exact scalar rounding is required.
- Compare every packed byte on random and adversarial values.
- Include zero, denormals, infinities, NaNs, saturation edges, and all-negative
  vectors.
- Show a full-token win; small-N linears should be reported separately.

---

## 9. P1 — Determine the right A100 worker configuration

Test topology with real GEMVs and full tokens. Pure register-fed results do not
decide a memory-stalled workload.

Configurations:

```text
4-across: 8,10,12,14
8-paired: 8,9,10,11,12,13,14,15
1-unit:   8
1-pair:   8,9
```

For every configuration record:

- exact N, K, M and group format;
- packed weight bytes;
- cold or warm status;
- GEMV latency and achieved packed-weight GB/s;
- full-token latency;
- CPU utilization;
- power-supply stability and throttling indicators.

Avoid a benchmark that rereads a small W buffer enough times to become a cache
benchmark. Include model-sized streams or rotate among buffers larger than the
aggregate relevant cache.

Make the hart stride/configuration a CLI option. Choose the default from
full-token results. Never silently map `nt > 4` with the four-hart stride formula
and place multiple software workers on the same hart.

---

## 10. P2 — Validate and implement M-batched linears

M-batching is the major product-level opportunity because an 8-row IME tile is
mostly idle during single-stream decode.

Land it in two stages.

### Stage 1: independent sequences

Batch up to eight sequences at the same model layer:

```text
X tile: [kb][m][16], m < M
W tile: streamed once
C tile: consume all ct[m*8+n], m < M
activation scale: one per M row
weight scale: one per output and K group
```

The group fold must compute:

```text
y[m,n] += int32_group[m,n] * x_scale[m] * w_scale[n,group]
```

This is an outer product of M activation scales and eight output scales applied
to the int32 tile. Design the RVV fold with M=8 in mind, even if M=1 lands first.

Report:

- aggregate tok/s;
- per-sequence tok/s;
- median and tail token latency;
- attention cost versus linear cost;
- memory use per active sequence.

### Stage 2: speculative/MTP verification

Reuse the M-batch linear path, but separately solve:

- causal attention for the draft span;
- KV commit/rollback;
- acceptance logic;
- MoE expert-union scheduling;
- draft-model overhead.

Do not quote effective speculative tok/s from an M-tile microbenchmark alone.
Measure accepted tokens per wall-clock second end to end.

### Acceptance gate

- All `M*N` outputs pass the corrected integer kernel oracle.
- M=1 remains performance- and numerically-compatible.
- Independent-sequence full-model batch produces the same results as processing
  each sequence separately within the declared floating-point tolerance.

---

## 11. P2 — MoE scheduling

After activation reuse, restructure MoE around shared work:

1. Pack normalized hidden state once.
2. Compute router and select experts.
3. Sort selected expert IDs only if measurement shows locality benefit.
4. Execute gate and up using the same packed activation.
5. Fuse SiLU×up as an RVV elementwise pass.
6. Pack that expert’s intermediate once and execute down.
7. Accumulate weighted expert output without clearing/reallocating buffers.

Potential next step: schedule gate and up in one A100 parallel region. They use
different weights but share activation and dimensions. This saves scheduling
overhead; it does not reduce weight bytes.

For batched MoE, group tokens by expert. Stream an expert’s weights once for all
tokens that selected it. The useful metric is tokens served per expert-weight
stream, and the enemy is the union of sparsely selected experts.

Avoid claiming cache benefit merely from sorting expert IDs: model-sized expert
weights generally will not remain in cache. Sorting may still improve sequential
addressing, readahead, or future staging.

---

## 12. P2/P3 — Persistent scheduling, only after measurement

OpenMP parallel regions may already reuse a runtime worker team internally.
Measure empty-region, barrier, and real-linear overhead before replacing the
control plane.

Progression:

1. Reuse per-thread scratch while retaining current OpenMP regions.
2. Combine obviously related linears into one parallel region.
3. If overhead remains material, create a forward-scoped persistent A100 team.
4. Only create a generation-scoped custom job queue if forward-scoped regions
   remain insufficient.

A persistent design needs:

- immutable or double-buffered job descriptors;
- a monotonic epoch with correct memory ordering;
- fixed worker count and pin map;
- clear ownership of activation and output buffers;
- shutdown/error propagation;
- a serial/reference fallback;
- watchdog diagnostics for barrier deadlocks.

Acceptance: demonstrate the overhead bucket shrinks and full-token latency
improves. Complexity alone is not success.

---

## 13. Use both clusters deliberately

The A100 and X100 clusters can run concurrently, but transformer layer
dependencies limit naive overlap. Do not claim that layer L+1 normalization can
run before layer L residual output exists.

Good X100 work candidates:

- RMSNorm;
- RoPE;
- attention QK and weighted-V work;
- softmax;
- router/top-k;
- SiLU and residual operations;
- activation packing, if transferring ownership does not add synchronization;
- sampling/tokenization.

Useful overlap opportunities are more likely with:

- multiple independent sequences;
- pipelined attention for one sequence while A100 serves another;
- speculative draft/verify orchestration;
- DMA/staging, if the vendor path is understood.

Build an execution timeline from measured timestamps before introducing a
cross-cluster queue.

---

## 14. Prefill strategy

Prefill is not decode GEMV and should have its own backend decision.

Current direction:

- Use the vendor EP for large supported matmuls when integration cost is
  acceptable; approximately 2.8 TOPS has been measured.
- Continue custom `gemm_ime2_i8` work only when it offers a concrete integration,
  datatype, shape, or dependency advantage.
- If improving custom GEMM, prioritize cheap vector C handling and MN/cache
  blocking based on measured working-set behavior.
- Do not extrapolate the 14.6 TOPS register-reuse loop to real packed matrices.

Keep time-to-first-token, prompt tokens/s, and decode tokens/s as separate
reported metrics.

---

## 15. TCM and ai_dma research gate

Do not build production decode around TCM until all of these are answered:

1. What bandwidth comes from the physical TCM after defeating L1/L2 caching?
2. What is DRAM→TCM DMA bandwidth and setup latency?
3. Can DMA overlap IME consumption without stealing the same bottleneck?
4. Are blocks pair-private, and what synchronization is required?
5. Can double buffering fit useful weight panels plus activations and outputs?
6. Does staging reduce full-token time, not merely a microbenchmark load time?

One Qwen3-30B-A3B expert’s gate+up+down W4 payload is roughly 2.36 MB before
scale overhead, or about 2.95 MB with fp32 group-32 scales. That nearly consumes
the entire nominal 3 MB TCM, leaving no comfortable double buffer. Down-only or
panel staging is more plausible than whole-expert double buffering.

Reverse engineering the vendor DMA ring is a separate project with explicit
scope. Prefer the vendor EP for supported prefill operations rather than
rebuilding it merely to match its measured ceiling.

---

## 16. Quality and correctness gates

A single expected first token is a smoke test, not a quality evaluation.

Use four levels:

1. **Kernel exactness:** packed integer GEMV versus scalar integer reference.
2. **Linear numerical test:** grouped dequantized output versus current engine;
   report max absolute, mean absolute, relative error, and cosine similarity.
3. **Model determinism:** fixed prompt/seed tokens and logits versus baseline.
4. **Quality:** perplexity or a representative evaluation set before changing
   quantization granularity or scale precision.

Changes to scheduling and scratch ownership should preserve numerical output.
Changes to rounding, group layout, scale precision, or accumulation order must
declare and quantify expected differences.

Cache files need:

- magic;
- format version;
- tensor dimensions;
- quantization/group format;
- scale datatype and layout;
- source model identity or checksum;
- checked reads and writes.

Never load a cache after changing scale layout without bumping its version.

---

## 17. Concrete PR sequence

### PR1 — Correct benchmarks and add instrumentation

- Fix `bench/q4_gemv.c` validation for every M row.
- Add decode timing buckets and packed-byte counters.
- Add hart-map reporting.
- Add repeatable result records.

Exit criterion: trustworthy M=1 and M>1 kernel results plus a full-token profile.

### PR2 — Prepacked activation API and scratch ownership

- Split pack from `lin_mm`.
- Reuse Q/K/V and expert gate/up activation tiles.
- Allocate per-worker scratch once.
- Remove steady-token allocation.

Exit criterion: same outputs, fewer pack calls, measured token improvement.

### PR3 — Group-fold A/B implementations

- Preserve existing grouped-buffer oracle.
- Add immediate scalar fold.
- Repack scales to `[Nb][Kp][8]` and add immediate RVV fold.
- Version the packed cache.

Exit criterion: correctness gates pass and the fastest measured implementation
becomes the default.

### PR4 — RVV activation quantization

- Vector absmax, conversion, saturation, and packing.
- Retain a scalar reference path.

Exit criterion: declared rounding behavior and full-token win.

### PR5 — Topology and scheduling selection

- CLI-selectable 4-across and 8-paired maps.
- Production-shaped cold-stream benchmarks.
- Combine related linears in a parallel region if justified.

Exit criterion: default selected from full-token results.

### PR6 — M-batched independent sequences

- Extend packed activation and group fold to M≤8.
- Add engine scheduler for sequences at the same layer.
- Report throughput and latency.

Exit criterion: equivalence to separate inference and real aggregate tok/s gain.

### PR7 — Batched MoE expert scheduling

- Group token work by expert.
- Reuse each expert weight stream across its selected tokens.

Exit criterion: measured reduction in expert weight bytes per output token.

### PR8 — Persistent pool, if still warranted

- Implement only if PR1 profiling still attributes material time to scheduling.

### Research branches

- TCM/DMA cold-stream characterization.
- Speculative/MTP end-to-end prototype.
- Vendor EP prefill integration.
- fp16 group scales.
- Tensor-selective per-channel W4.

---

## 18. Benchmark record template

Keep a CSV or Markdown table with at least:

| Field | Value |
|---|---|
| git SHA | |
| board / firmware | |
| power supply | |
| model and source quant | |
| packed format/version | |
| M / context / token index | |
| A100 thread count | |
| exact hart map | |
| kernel/fold variant | |
| active weight bytes | |
| linear ms | |
| quant+pack ms | |
| fold ms | |
| attention ms | |
| scheduling ms | |
| total token ms | |
| achieved packed GB/s | |
| aggregate tok/s | |
| per-sequence tok/s | |
| correctness/quality result | |

Always compare against the immediately preceding working implementation on the
same board state. Record raw runs, not only the best run.

---

## 19. Working rules for Claude Code

When implementing this roadmap:

1. Make one performance hypothesis per change.
2. Preserve a selectable reference path until the replacement passes.
3. Add a benchmark or counter that can falsify the hypothesis.
4. Run correctness before performance.
5. Report full-token improvement alongside microkernel improvement.
6. Do not silently change quantization, cache formats, or hart placement.
7. Do not generalize a warm-cache result into a DRAM/TCM claim.
8. Do not use synthetic TOPS as the primary decode metric.
9. Keep prefill and decode conclusions separate.
10. Update `docs/HARDWARE.md` only with reproducible measurements and commands.

## If only three things get done

1. **Fix measurement and reuse packed activations** across Q/K/V and expert
   gate/up.
2. **Remove hot allocation and optimize the required group fold** with a
   scale layout suited to RVV, choosing the winner from on-board measurements.
3. **Validate and ship M-batched execution**, first for independent sequences,
   so each streamed weight tile produces useful work in more than one of the
   IME tile’s eight rows.

That is the most defensible route from the current prototype to a decoder that
uses the K3 as the hardware is actually built: X100 for orchestration and
nonlinear work, four A100 IME-2 units for packed linear streams, and batching to
turn scarce memory bandwidth into multiple useful tokens.

---

## 20. Findings — Claude Code implementation pass (2026-07-25)

Status of the pure-C engines against this roadmap. Board: `root@192.168.68.88`
(static), A100 hart 8, `csrr vlenb` = 128 → VLEN 1024 confirmed. `nt=4` = harts
8/10/12/14 (4-across). All runs pinned via `/proc/set_ai_thread` + affinity.
Correctness so far is **smoke-test only** (` Tokyo` + a few facts) — no perplexity
or logit comparison yet (§16 gate not met).

### 20.1 What is validated (coherent generation)

Prompt `[785,6722,315,9625,374,12095,13,576,6722,315,6323,374]` = "The capital of
France is Paris. The capital of Japan is"; greedy; expect next token 26194 (' Tokyo').

| Engine | Model / source quant | Weight fmt | first tok | Generation | tok/s (M=1, nt=4) | load |
|---|---|---|---|---|---|---|
| `qwen_ime.c`  | Qwen3-4B / Q8_0 | **W8A8 per-channel** | ' Tokyo' PASS | "…Tokyo. …Germany is Berlin. …Italy is Rome. …Spain is" | **3.85** | ~43 s |
| `qwen_ime4.c` | Qwen3-4B / Q8_0 | **W4A8 per-channel** | FAIL (garbage) | — | — | ~144 s |
| `qwen_ime4.c` | Qwen3-4B / Q8_0 | **W4A8 per-32-group** | ' Tokyo' PASS | "…Tokyo. …Germany is Berlin." | (not timed) | ~144 s |
| `qwen_moe.c`  | Qwen3-30B-A3B / Q4_0 (unsloth) | **W4A8 per-32-group** | ' Tokyo' PASS | "…Tokyo. …Brazil is Brasília. …Canada is Ottawa." | **1.36** | ~956 s (requant) |

30B config parsed from GGUF: 48 layers, d=2048, heads 32/4, hd=128, **128 experts /
8 active, moe_ffn=768**, `tie_word_embeddings=false`.

### 20.2 Three bugs found and fixed (all pre-perf, correctness)

1. **Per-channel W4 is too lossy to run.** Same dense-4B code: W8A8 per-channel →
   ' Tokyo'; **W4A8 per-channel → garbage** ("...?", "BallBall..."). Switching to
   **per-32-group W4** scales (matching Q4_0) → ' Tokyo'. This directly confirms
   §2.1/§7: for W4 the group fold is *required*, not optional. Implementation:
   `gemv_nb_int4_grouped` resets accumulators and does one `vse32` per 32-K group
   (= one interleaved 2-K-block iteration); `lin_mm` folds `int32_group * x_scale *
   gs[out][group]` in scalar C. This is the §7 "Variant A-ish" baseline oracle.
2. **Untied lm_head.** The 30B ships a separate `output.weight`; using `token_embd`
   as lm_head gave first-token garbage even with correct hidden states. Fix: use
   `output.weight` when `gguf_has()` finds it, else tie.
3. **Q6_K (ggml type 14) dequant missing.** The 30B `output.weight` is Q6_K; the
   4B-Q4_0 also carries Q6_K tensors (that GGUF aborted on "dequant type 14"). Added
   the 256-value super-block (210-byte: ql[128]+qh[64]+scales[16]+d) dequant.
   Also present and validated: Q4_0 (type 2), Q4_1 (type 3, used by `ffn_down_exps`),
   Q8_0 (8), F32/F16.

Early per-tensor dequant stats (blk.0, prints in <1 s, catches dequant bugs before
the ~16 min requant): `ffn_gate_exps` (Q4_0) std 0.0227, `ffn_down_exps` (Q4_1) std
0.0246, `output.weight` (Q6_K) std 0.0272 — all mean≈0. This is a cheap, useful
guard and should stay.

### 20.3 Where the 1.36 tok/s goes (first-order, NOT yet bucket-profiled)

Confirms §2.2 and §3: at group-32 W4 = **0.625 B/weight**, ~3B active →
~1.9 GB/token. At the measured ~13 GB/s packed-W4 rate that's ~150 ms of pure
weight streaming, yet the token is ~735 ms. **~585 ms is glue/overhead** — exactly
the P0 targets:

- `lin_mm` re-quantizes the normed hidden for **every** linear (Q/K/V redo the same
  `hn`; each expert's gate+up redo the same `hn`). §5 prepack-reuse is the largest
  safe win and is not done yet.
- Per-matmul `malloc`/`free` of the group `part` buffer inside the OpenMP region
  (§6) — ~1200 matmuls/token.
- Scalar group fold with `[N][Kp]` scale layout (strided for RVV) — §7.
- No per-token timing buckets yet (§4) — the ~585 ms split is a hypothesis, not a
  measurement. **Instrumentation is the true next step; the above is not yet
  falsifiable.**

### 20.4 Requant cache + NAS (implemented)

Requant is single-threaded and ~16 min, so `qwen_moe.c` has `cache_save`/`cache_load`
(config header + all packed `Wq`+`gs` + embeddings). Cache path is argv[4], default
`/mnt/jupiter2/qwen3-30b-a3b.imecache`. **Board has a persistent sshfs mount**
`/mnt/jupiter2` → `willy@192.168.69.133:/storage/milkv_jupiter2` (UGOS SFTP is
chrooted to the *share* root, so `/storage/...` not `/volume2/storage/...`); in
`/etc/fstab` with `x-systemd.automount`; board-root ed25519 key authorized on the NAS.
Cache being populated now (~20 GB).
- **Caveat vs §16:** the current cache header stores dims/config but **not** a scale-
  layout version or model checksum. Per §7/§16, this must be versioned *before* any
  `[Nb][Kp][8]` scale-layout change, or an old cache will be silently misread. TODO.

### 20.5 Corrections this pass makes to earlier optimism

- Not "int4 halves bytes": **1.6×** smaller than W8 (group scales), per §2.2.
- The "68 tok/s @ 30B M=8" figure elsewhere in this repo is a **synthetic,
  glue-free, perfect-overlap ceiling** — the real M=1 full-token number is 1.36.
  Do not quote 68 as a decode result.
- The `bench/{decode_layer,moe_decode}.c` throughput harnesses use a fixed
  pre-packed activation and a persistent pool with no attention/quant/fold — they
  measure the weight-stream ceiling, not a real token. Useful for the bandwidth
  model, misleading as tok/s.

### 20.6 Immediate next (agreed P0 order)

1. Per-token timing buckets in `qwen_moe`/`qwen_ime4` (§4) — establish the real split.
2. Prepacked activation API; reuse across Q/K/V and expert gate/up (§5).
3. Persistent per-thread scratch; remove hot-path alloc (§6).
   Re-measure after each; keep the grouped fold as the numerical oracle.
Then §7 fold A/B (+ `[Nb][Kp][8]` scales + cache version bump) and §10 M-batch.

## 21. Findings — PR2 closed: activation reuse + persistent scratch (2026-07-26)

Session was interrupted (host power loss) right after the P0.1 baseline landed;
resumed from the K3-local cache (valid, `ENDIMEC`-footed, survived the outage intact)
and implemented steps 2–3 of §20.6 / PR2 in full.

### 21.1 What changed

- `lin_mm` split into `pack_act()` (quantize x -> int8 tile + scale) and
  `lin_mm_packed()` (matmul against a pre-packed tile). `lin_mm` itself is now a
  thin wrapper (pack + `lin_mm_packed`) kept for call sites with a unique input
  (`o` proj, each expert's `down` proj, `lm_head`).
- `forward()` now packs `hn` **once** per layer for Q/K/V (3 calls -> 1 pack) and
  packs the ffn-norm'd `hn` **once** for all selected experts' gate+up (16 calls ->
  1 pack), via a second persistent buffer `xt2` (kept separate from the transient
  `xt` used by `o`/`down`/`lm_head`, which would otherwise be clobbered mid-loop by
  each expert's `down`-proj pack). Packs/layer: **28 -> 11**, matching the §20.6
  estimate exactly.
- `lin_mm_packed`'s `part[]` accumulator (int32, `Kp*64*4` bytes) is now a
  `static __thread` buffer grown once to the largest `Kp` needed and reused for the
  rest of the process, instead of `malloc`/`free` on every (Lin, thread) call.

### 21.2 Measured, same cache, same prompt, nt=4 (12-token decode, ' Tokyo' PASS throughout)

| Stage | act-pack | linear(kernel+fold) | attention | rest | wall/tok | tok/s |
|---|---|---|---|---|---|---|
| P0.1 baseline (pre-crash) | 60.6 ms | 603.2 ms | 16.8 ms | 60.2 ms | 741.8 ms | 1.35 |
| + activation reuse (PR2 step 2) | 17.0 ms | 579.5 ms | 16.8 ms | 59.5 ms | 673.9 ms | 1.48 |
| + persistent scratch (PR2 step 3) | 17.2 ms | 575.9 ms | 16.7 ms | 59.7 ms | 670.5 ms | 1.49 |

Net: **+10.3% decode tok/s** (1.35 -> 1.49), all from removing steady-token
overhead — no kernel/fold change, no cache format change, output unchanged.

### 21.3 Correction to §20.3's glue hypothesis

§20.3 guessed ~585 ms/token of "glue" outside a ~150 ms weight-stream cost, based
on a bandwidth-only estimate. The buckets say otherwise: **glue (act-pack +
attention + rest) was only ~138 ms/token even before this pass** (60.6+16.8+60.2);
it's now ~94 ms (17.2+16.7+59.7, 14% of wall). The other ~600 ms was always
*inside* `linear(kernel+fold)` itself — i.e. inside the per-`Lin` GEMV kernel call
and its scalar per-group scale fold, not sitting outside it as separate glue. Scratch
removal barely moved `linear` (579.5 -> 575.9 ms), confirming malloc/free was never
the story there either.

**Implication:** `linear(kernel+fold)` is now **86% of wall-clock** (575.9/670.5 ms)
and is the only bucket left with room — this is exactly §7/PR3 (group-fold A/B +
`[Nb][Kp][8]` scale repack) and research_feed_paths.md's Path A (vendor-shaped
N32 kernel with in-loop fold), not a new branch. R0's gate (§9 of
research_feed_paths.md: "finish codex timing buckets") is now satisfied with real
numbers, not the earlier bandwidth-only guess.

### 21.4 Not yet done — explicitly stopping here per working rule

Per §19 / PROGRESS.md: P0 (buckets + reuse + scratch) is complete and measured;
**not** auto-advancing into PR3 (fold A/B) or Path A (vendor kernel) without a
branch decision. `research_feed_paths.md` §9's ranked agenda and §12 results log
still apply for that decision.

Toolchain note (affects any int4/int8 IME kernel here): the vmadot asm functions
must carry `__attribute__((optimize("no-tree-vectorize")))` (or the file built
`-fno-tree-vectorize`) — gcc auto-vectorizing the surrounding C (esp. the `ct→y`
copy) emits RVV that collides with the asm's vector state → heap corruption. Also,
the identical object miscompiles under **rustc static-link with -fPIC** (unbounded
loop); pure gcc `-fno-pie`/non-PIC is correct. Prefer building these engines as
standalone C (Python via ctypes), not linked into a rustc PIE.

## 22. Findings — Path A vendor kernel, ported, verified, integrated (2026-07-26)

Picked up the §21.4 branch decision: research_feed_paths.md Path A (vendor-shaped N32 int4
kernel), following its A1-A5 probe plan.

### 22.1 A1-A4: port the real kernel, not a guess

First attempt (`gemm_kernel_i8i4_m1`, hand-paired with a guessed A-pack from the kernel's own
inline comments) executed without crashing but was **unverified** — single-element hardware
probes disproved the naive zero-point hypothesis, then surfaced that the reference file
(`reference/spacemit-backend/ime2_kernels.cpp`, 5768 lines) has multiple kernel/pack variant
pairs and the wrong two had been paired. Traced `ime.cpp`'s actual dispatch (Q4_0/Q4_1,
INTER_SIZE==256, count_m<4) to the real chain: `gemm_kernel_i8i4_hp` → **`gemm_kernel_i8i4_hp_m1`**
— a materially different kernel (scale-fusion baked into the dot instruction via a packed operand,
fp16 accumulation) — paired with the real `quantize_a_row_i8_hp` (A) and `make_block_q4_0x32` (B).
Root cause of the first mismatch: real B nibble-pairing is adjacent `{2j,2j+1}`, not the
native-ggml `{j,j+16}` pairing assumed from reading the asm alone. **Faithfully transcribed +
verified against an independent scalar dequant oracle: max rel diff 2.3%**, consistent with
fp16-accumulation noise (`bench/vendor_ime_a2_full.c`). Hot kernel-only A/B at identical N32K256:
vendor is **6.66x faster** than our `gemv_nb_int4_grouped` (`bench/vendor_ime_a2_full.c`'s timing
section) — close enough to the ~8x full-token gap against the real vendor binary
(research_feed_paths.md A5-equiv: 11.71-12.89 tok/s vs our 1.49) to confirm kernel
microarchitecture, not ggml dispatch overhead, is what's worth chasing.

Lesson for future kernel ports in this repo: reference source with `#if 0`/`#else` branches and
multiple similarly-named functions is not reliable to hand-pair by reading comments — trace the
actual dispatch call graph (or `objdump` the compiled `.so`) before trusting a port's correctness.
"It executes without a SIGILL" is not "it's correct."

### 22.2 A5: full engine integration (`qwen_moe_hp.c`)

New file, not a modification of the working/committed `qwen_moe.c` — this repo's existing pattern
of parallel engine variants (`qwen_ime.c`/`qwen_ime4.c`/`qwen_moe.c`) made a separate binary the
natural "feature flag" split, so the original engine is never at risk. Same GGUF reader, model
struct, `forward()`, attention, MoE routing, and the §21 P0.2 shared-activation-pack structure —
only the GEMV+weight-pack layer changed. New incompatible cache format (`IMEC` ver=2). All of this
model's Lin shapes (d=2048, qd=4096, kvd=512, moe_ffn=768, vocab=151936) are exact multiples of
256(K)/32(N) — zero remainder/padding handling needed anywhere.

**Result on the real 30B-A3B model, nt=4: 6.19 tok/s, `' Tokyo'` PASS, coherent generation
matching the original engine exactly. 4.16x over the 1.49 tok/s baseline**, though still below
the real vendor binary's 11.71-12.89. Requant into the new format is slow (**1104.5s / ~18.4
min**, vs ~2 min for the old format's simpler pack) — a real, not-yet-optimized cost from the
extra fp16 conversions and 8-subblock nesting per weight group.

### 22.3 The buckets pointed at PR8 — then turned out to be measuring the wrong thing

First read: `linear(kernel)` fell 15x (575.9→38.4ms, matching A3's hot-timing prediction) but
`rest` grew to 100.3ms = 62% of wall (was 59.7ms/9%). Read that as OpenMP fork-join overhead
(`lin_mm_hp` opens a fresh `#pragma omp parallel` team per `Lin` call, 1392 spawns/token) and
projected that fixing it would land ~16 tok/s — beating the vendor binary.

**That projection was wrong, and not because the hardware behaved unexpectedly.** Implemented
PR8 (persistent spin-dispatch pool: threads spawned once, generation-counter dispatch instead of
per-call `#pragma omp parallel`) and got only 6.19→6.53 tok/s — a ~6% gain, not the 2.4x implied
by the projection. Investigating found a real bug: the new engine's `lin_mm()` wrapper (used for
`o`, all 8 `ed[e]`, and `lm` — over a third of per-layer `Lin` calls) had **zero timing
instrumentation**, unlike the old engine's version. All of that call's real work — pack *and*
kernel — was silently falling into `rest` the whole time, making a mostly-real, correctly-doing-
its-job bucket look like a 62%-of-wall dispatch-overhead mystery. Fixed the instrumentation
(ported the `_ta/_tb` pattern from the old `lin_mm`) and re-measured with the pool still active:

| | act-pack | linear(kernel) | attention | rest | wall | tok/s |
|---|---|---|---|---|---|---|
| Old engine (§21, post-P0.2/P0.3) | 17.2ms | 575.9ms | 16.7ms | 59.7ms (9%) | 670.5ms | 1.49 |
| New engine, first (buggy) reading | 5.1ms | 38.4ms | 16.8ms | 100.3ms (62%) | 161.6ms | 6.19 |
| New engine, pool + instrumentation fixed | 16.6ms | **58.7ms** | 16.9ms | **59.3ms (39%)** | 152.4ms | **6.56** |

`linear` and `rest` are now roughly tied as the two biggest buckets, and `rest` is genuine scalar
C — router matvec (128 experts x d=2048 = 262144 unvectorized mults/layer), per-head RoPE + q/k
rmsnorm (~4600 `sinf`/`cosf` calls/layer across nh=32+nkv=4 heads), SwiGLU (~6144 `expf`
calls/layer) — not dispatch overhead waiting to be collapsed. The persistent pool is a real, worth
keeping ~6% win; the "beats the vendor binary" claim from the first reading is retracted.

**Lesson**: a bucket that looks too large after a big kernel win is exactly the moment to check
whether it's actually being measured, not just theorize about the hardware. "The buckets say X"
is only as trustworthy as the instrumentation producing them.

Not yet done: profiling `rest` itself finely enough to pick a next target (router vectorization is
the obvious first guess — 262144 unvectorized mults/layer is the single largest identified item —
but per §19/working-rule, that's a measurement to take, not an assumption to build on), and
speeding up the new format's slow requant (~18.4 min). Stopping here per the same working rule as
§21.4 — this is the next branch decision, not an auto-advance.

### 22.4 Splitting `rest` for real, then two safe fixes (2026-07-26)

Followed §22.3's "not yet done" directly: split `rest` into `rope+qknorm`/`router`/`swiglu`/
`rest(other)`. `rest(other)` came out to 2.1ms — confirms the split is essentially complete, not
another undercount waiting to bite. Two fixes, both exact (no lossy approximation, unlike a
SwiGLU/sigmoid approximation would be):

1. **RoPE table caching.** `rope()` recomputed `powf`+`sinf`+`cosf` per head per layer — up to
   1728x/token (nh+nkv heads x 48 layers) — despite the cos/sin values depending only on
   `(hd,pos,rope_base)`, identical across every head and every layer for a given token. Split into
   `rope_table()` (build once per `forward()` call) and `rope_apply()` (per-head rotation only).
   **13.2→2.0ms (6.6x).**
2. **Router matvec vectorization.** 128 experts x d=2048 (~1MB/layer), previously plain scalar
   `for(i)s+=rw[i]*hn[i]`. RVV `vfmacc_vv_f32m1` accumulate + `vfredusum` reduction (`vdot_f32`),
   vector-length-agnostic. **29.4→18.7ms — only 1.6x**, well short of the ~8-32x a VLEN=1024 A100
   hart's lane count would suggest for compute-bound work. Reads as memory-bandwidth-bound: the
   router matrix is ~1MB/layer streamed fresh from DRAM with poor cache locality, so vectorizing
   the multiply-add doesn't touch the actual bottleneck (the load, not the FMA).

Net: **wall 154.4→132.6ms, 6.56→7.54 tok/s.** Correctness held — `' Tokyo'` PASS, identical
continued generation (Brasília, Ottawa) — confirming `vfredusum`'s unordered-reduction rounding
differences didn't flip any expert-selection tie. **Cumulative from the original 1.49 tok/s P0
baseline: 5.06x.**

Fresh bucket ranking: `linear(kernel)` 58.5ms (44% of wall, still the single biggest item) >
`router`≈`attention` 18.7ms each > `act-pack` 17.3ms > `swiglu` 14.2ms (untouched) > `rope` 2.0ms
> `rest(other)` 2.2ms. SwiGLU is next-biggest untouched, but it's a qualitatively different kind
of fix than the two above: there's no native RVV transcendental for `expf`, so a real speedup
means either a polynomial sigmoid approximation or a lookup table — both carry numerical/quality
risk that caching and exact vectorization didn't. The single-prompt `' Tokyo'`-coherence check
used throughout this session is not enough validation for that; needs a broader
quality/perplexity-style check before shipping, not just "still generates something plausible."

### 22.5 8-hart pinning: fixed a real collision bug, then measured that it doesn't matter here

`pin_once`'s formula (`8+(tn*2)%8`) was not arbitrary — `docs/HARDWARE.md` documents 4 IME-2
units, each shared by a core pair (8,9)(10,11)(12,13)(14,15), and measures using both cores of one
pair as *contended* (7.31 TOPS for 2 units via pairs vs 13.09 for 4 units one-per-pair). The
formula correctly picks one-per-unit (8,10,12,14) for nt=4 — but it only works by coincidence:
for nt=8 it collides (tn=4 maps back to hart 8, same as tn=0), silently oversubscribing half the
units instead of using all 8 harts. Replaced with a lookup table (`{8,10,12,14,9,11,13,15}`,
indexed `tn%8`) — one-per-unit first, then paired partners, collision-free for any nt in 1..8.
Regression-tested at nt=4: 7.51 vs 7.54 tok/s (noise, confirms no behavior change).

**nt=8, tested for real on this workload: 7.27 tok/s — slightly worse than nt=4's 7.51.**
`linear(kernel)` went 58.7→62.9ms, i.e. *up*, not down. This matches two independent signals
already in hand: the router finding (vectorizing compute didn't help much because the bottleneck
is memory bandwidth, not FLOPs) and `docs/HARDWARE.md`'s own peak-TOPS table, where going from
4-across-pairs to all-8-cores gains only +11% even for pure register-fed compute with zero memory
traffic. Our actual per-call GEMV work is tiny and the shared LPDDR5 bus is the real constraint;
doubling threads adds contention without touching that. Correctness held at nt=8 either way
(`' Tokyo'` PASS, identical generation), so the fix itself is verified safe to use — it's just
that more threads is not the lever here. **nt=4 remains the right default for this engine.**

### 22.6 Router-as-Lin fp16 experiment: validated safe, zero speedup, two new toolchain gotchas

The router is itself a 128xd `Lin`, same shape family as everything else in this engine, but kept
full fp32 (~1MB/layer). Tried weight-only fp16 (activation stays fp32, matching this repo's
W-lower/A-higher convention) as a lower-risk step before considering int4. Unlike RoPE-caching and
router-vectorization (exact, zero numerical risk), this perturbs values feeding a **discrete**
top-8 selection — quantization noise could flip which experts get chosen, not just perturb a
smooth activation — so it needed an explicit expert-selection comparison against the fp32
reference, not just "still says Tokyo."

Two real bugs surfaced before getting a trustworthy result, both now folded into "Toolchain
gotchas" (PROGRESS.md) as general lessons for any future RVV code in this file:

1. **SIGSEGV before any reachable code ran.** `main()` crashed before its first `fprintf`, even
   with explicit `fflush`. Traced via `dmesg` (SEGV_MAPERR at a stack-like address) and `strace`
   (crash lands immediately after `bind_ai()`'s syscalls return, before `sched_setaffinity` is even
   called) to gcc's `-O3` RVV auto-vectorizer mangling a **completely unrelated, plain-scalar**
   `f32->f16` conversion loop (no asm, no custom instructions) — the existing "vmadot + autovec
   collide" gotcha turned out to apply to any hot loop near vector code, not just custom-extension
   asm. Fix: isolate it into its own `__attribute__((noinline,optimize("no-tree-vectorize")))`
   function (`router_f16_build`). Bisection method: binary-search which lines mattered by
   selectively removing code and re-testing — removing the malloc+loop (keeping the struct field)
   fixed it; removing just the *user* of that data (`vdot_f16w_f32a`, keeping the loop) did not.
2. **Wrong output, no crash** (`' 乾坤'` instead of `' Tokyo'`) once #1 was fixed. RVV widening
   instructions (`vfwcvt.f.f.v`) read the **active vtype at the instruction's execution**, not at
   the time the source register was loaded — had a `vsetvli` for an unrelated `e32,m1` load
   sitting between the `e16,mf2` load and the widening convert that was supposed to interpret it,
   so the convert silently used the wrong vtype. Fixed by moving the convert to immediately follow
   the matching-width load, verified against `gemm_kernel_i8i4_hp_m1`'s already-proven ordering.

**Result after both fixes**: `' Tokyo'` PASS, identical generation. Validation: **0/1344
expert-set mismatches** (12 prefill + 16 decode positions x 48 layers) vs the fp32 reference.
Logit deltas were reported at `%.5f` (0.00000) — that precision can mask real differences; treat
"0.00000" as "small" not "zero" until re-measured in scientific notation. **The router bucket
didn't move — 18.6ms fp16 vs 18.7ms fp32.**

**Self-correction (flagged in review): the "stop here" conclusion this section originally reached
was premature and has been retracted.** Two problems with it: (1) this experiment used a
standalone hand-written `vdot_f16w_f32a`, run **single-threaded on the main thread only** — not
the actual `lin_mm_hp`/pool-dispatch path (nt=4, parallel across N32 panels) every other `Lin` in
this engine goes through. A real HP-format router `Lin`, packed the same way as q/k/v/etc. and run
through `lin_mm_hp`, would get both the proven ~446ns/call kernel (A3) *and* multi-threading —
neither of which this test exercised. (2) "Loop-carried latency is the bottleneck" was written as
a finding but is an **unverified hypothesis** — it was never isolated (e.g. via independent
accumulators, or by actually threading the comparison). **Do not treat W4/HP-router as closed.**
Real next step: pack router weights in the real HP `Lin` format, run through `lin_mm_hp`, and
compare all router logits + top-8 sets + full generation + timing against fp32 — only conclude
"not worth it" if that apples-to-apples test also shows no material improvement. Kept the fp16
path in the codebase regardless (validated at the granularity tested, harmless, real if small
memory saving: 512KB vs 1MB/layer).

### 22.7 Router-as-HP-Lin, done for real — result is genuinely mixed, not a clean win

Followed through on §22.6's "next step": router weights packed via `lin_new_hp` (real int4 vendor
format, identical to every other `Lin` in this engine) and computed via `lin_mm_hp` (pooled,
multi-threaded, the proven ~446ns/call kernel from A3) — `ly->router` (fp32) kept only as the
validation reference. Two real bugs surfaced before the result could be trusted, both instructive:

1. **First validation run: 1344/1344 (100%) expert-set mismatches.** Not a quantization surprise —
   a real bug in the validation harness. The top-8 selection's argmax sentinel (`bv=-1`) is only
   safe when every candidate is guaranteed positive, which holds for post-softmax probabilities but
   not for raw logits — the fp32 reference comparison used raw logits, observed running as low as
   ~-5, so the sentinel silently rejected valid negative candidates and corrupted the comparison.
   Fixed (`bv=-1e30f`).
2. **A second, separate instrumentation bug**, same *class* as §21's `lin_mm` gap: the router-bucket
   timer had been stopped immediately after `lin_mm_hp`, but the pre-HP-Lin baseline's `router`
   bucket also bundled in softmax + top-8-select + renormalize. Comparing a narrower window against
   a wider one is not apples-to-apples. Fixed by moving the stop-timer to match the original
   boundary exactly. **Lesson repeated from §21**: when a number moves further than expected after
   a change, check what's actually being measured before trusting the hardware explanation.

**Result after both fixes**: 791/1344 (58.9%) expert-set mismatches remain — this part is real, not
a bug. Added a finer metric (avg experts differing per mismatch, not just binary match/no-match):
**1.38/8 on average** — most mismatches are a single near-tie expert swap, not wholesale different
routing. Max abs/rel logit deltas (8.45 / 403%) are outlier-dominated across 172032 individual
logit comparisons and not representative of the typical case. Generation stayed coherent
(`' Tokyo'` PASS) on the 28-token test used throughout this session.

**Timing: router 18.7→12.5ms (-33%), wall 132.6→128.5ms (-3%), 7.51-7.54→7.78 tok/s (+3-4%).** Real,
but modest — not the ~40x the first (buggy) reading implied.

**This does not cleanly satisfy the stated bar** ("keep only if quality holds and the bucket falls
materially"): the speedup is real but modest, and quality does not cleanly hold — a majority of
routing decisions get at least one expert swapped, even if usually a minor one. **Left as an open,
unresolved decision** rather than picking a side: revert to fp32, keep as-is (generation still
looked fine), or first check whether swapped experts typically carry low renormalized softmax
weight (which would mean low actual output impact despite the raw mismatch rate) — any of these
needs broader evaluation (longer generations, more prompts, ideally perplexity) to actually settle,
not another short coherence check on 28 tokens.

**Caveat on §22.6's fp16 result**: that experiment's validation used the same vulnerable sentinel
pattern. Given fp16's much smaller rounding error the "0/1344 mismatches" result is still plausibly
correct, but it was never verified against this specific bug, and the code has since been replaced
so it can't be re-checked directly.

**Decision recorded**: fp32 restored as the router default; int4 HP routing kept behind
`g_router_mode==1`, not promoted. See §22.8 for the int8 alternative that followed.

### 22.8 Vendor int8 M1 router: found, validated, wired in — materially better than int4

Per the review's item 4 ("try a vendor-style W8 HP router... could retain matrix-engine/pool speed
while reducing routing error"): `gemm_kernel_i8i8_m1` (ime2_kernels.cpp:4773) is real, dispatched
for `count_m<4` via the same pattern as the int4 kernel. Genuinely simpler, not just a wider
version of the same trick:

- Plain typed `vmadot ...,i8` — signed x signed, **no zero-point trickery at all**. int8's range is
  wide enough to store signed values directly; int4 needed the unsigned-nibble+implicit-zp=8 hack
  specifically because 4 bits isn't enough range otherwise.
- Pairs with the **simple** A-format (fp32 scale + int16 asum + 32B int8 data, 38B/K32-group) that
  §22.7's *first* (wrong) int4 attempt assumed. That format wasn't a wrong guess in the abstract —
  it just belonged to a different kernel (`gemm_kernel_i8i8_m1`, not `gemm_kernel_i8i4_hp_m1`).
- B-pack ground truth from `make_block_q8_0x32` (repack.cpp:357): plain row-major `memcpy` per row.
  int8 doesn't need int4's nibble-interleave trick at all — one byte per value, no packing puzzle.
- Structural difference from int4: the loop is **flat** over K32-groups (`k_blks = K/32`), not
  int4's nested 256-wide-superblock-of-8-subblocks (`k_blks = K/256`). Easy to get wrong if porting
  by pattern-matching against the int4 kernel instead of reading this kernel's own loop structure.

Validated standalone before touching production (same discipline as A1-A2, `bench/
vendor_ime_i8_probe.c` + `bench/vendor_ime_i8_full.c`): **max abs diff 0.00000, max rel diff
0.00001 — essentially bit-exact against an independent dequant oracle, correct on the first pass**
(no reordering bug this time — the byte layout was simple enough to get right from ground truth
alone, unlike int4's `vfwcvt` ordering trap). Hot kernel-only timing: 791.7 ns/call vs int4's
446.4ns/call (A3) — ~1.77x slower per call, consistent with streaming 2x the weight bytes.

**Wired into production** as a third `g_router_mode` (0=fp32 default, 1=int4, 2=int8), reusing the
existing pool dispatch infrastructure (`HpWork` generalized with a `kind` field, `lin_mm_i8`
alongside `lin_mm_hp`) rather than duplicating it. One validate run now reports int4-vs-fp32 and
int8-vs-fp32 independently from the same 1344 comparisons, so both tradeoffs are directly
comparable from a single pass — no need to re-run and hope conditions matched.

**Result on the real model: int8 is a materially better tradeoff than int4.**

| | int4-HP | int8-M1 | fp32 |
|---|---|---|---|
| expert-set mismatches | 811/1344 (60.3%) | **105/1344 (7.8%)** | — reference — |
| avg experts differ/mismatch | 1.38/8 | 1.16/8 | — |
| max abs logit delta | 8.45 | **0.576** | — |
| router bucket | 12.5ms (-33%) | 16.0ms (-14%) | ~18.7-21.9ms |
| tok/s | 7.78 | 7.69 | 7.36-7.54 |

int8 gives ~8x fewer perturbed routing decisions and a ~15x smaller worst-case logit delta than
int4, for about half the speedup (14% vs 33% off the router bucket). Generation stayed coherent
(`' Tokyo'` PASS) in every mode tested.

**Still experimental, not promoted to default.** 7.8% mismatch is much closer to "quality holds"
than int4's 60.3%, but it isn't clean zero, and the same limitation applies as it did for int4: a
28-token single-prompt test can't fully settle whether even this level of perturbation is benign.
Unlike int4 (where the modest speedup plus high perturbation made the call fairly easy), int8's
tradeoff is closer and worth an explicit decision now that both are quantified on the actual model
— not left as another open item to revisit blind.

### 22.9 Attention vectorized (RVV) — reused the router's own primitives, bigger win than the router got

With the router default settled at fp32 (§22.7-22.8, both quantized modes kept experimental), the
standing review item was: "otherwise move to activation packing or attention." Chose attention over
activation-packing (`pack_A_hp`/`pack_act_hp`) because attention's cost scales with context length —
a fixed win here compounds as generations get longer, unlike a fixed per-call packing cost.

The attention inner loop (`forward()`) was still fully scalar: a triple-nested loop computing the
QK dot product per (head, position) pair, then the AV weighted-sum accumulation, both by hand with
no vectorization at all — despite the router matvec next to it having already been vectorized in
§22.4. Two changes, both **exact vectorization of the same math, zero approximation risk** (same
class as the RoPE-cache and router-RVV fixes in §22.4, not a lossy approximation like SwiGLU would
require):

- **QK dot product**: reused `vdot_f32` as-is — the same RVV `vfmacc_vv_f32m1` +
  `vfredusum_vs_f32m1_f32m1` helper written for the router matvec in §22.4, now called once per
  (head, position) pair instead of a hand-written scalar accumulation loop.
- **AV weighted accumulation**: no existing helper matched the `y[i] += scale*x[i]` (axpy) shape, so
  added `vaxpy_f32(float*y, const float*x, float scale, int n)` — vector-length-agnostic RVV
  `vfmacc_vf_f32m1` (scalar-broadcast multiply-accumulate), the vector-length-agnostic loop pattern
  already established in this file (`__riscv_vsetvl_e32m1` driving the tail).

```c
static void vaxpy_f32(float*y,const float*x,float scale,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl), vy=__riscv_vle32_v_f32m1(y+i,vl);
        vy=__riscv_vfmacc_vf_f32m1(vy,scale,vx,vl);
        __riscv_vse32_v_f32m1(y+i,vy,vl); i+=vl; }
}
```

Neither change needed the `noinline,optimize("no-tree-vectorize")` isolation from §22.6's toolchain
gotchas — both are plain e32m1 same-width RVV (load/fmacc/store), not a widening convert, and the
existing `vdot_f32` already proved this same instruction shape compiles and runs correctly here.

**Result: attention 18.7ms→9.1-9.2ms (-51%), 7.51-7.54→7.68-7.89 tok/s.** Two independent runs on
the board, both `' Tokyo'` PASS with identical continued generation (Brasília, Ottawa):

| Run | attention | wall | tok/s |
|---|---|---|---|
| 1 | 9.2ms | 130.3ms | 7.68 |
| 2 | 9.1ms | 126.8ms | 7.89 |

The 9.1-9.2ms spread is run-to-run noise, not a regression — both are consistent with the same
~51% cut from the 18.7ms baseline that had held since §22.4.

**Notably bigger win than the router's own RVV pass got (1.6x, §22.4)**, despite reusing the exact
same `vdot_f32` primitive. The difference is access pattern, not the vectorization itself: the
router matvec streams ~1MB/layer from a memory-bandwidth-bound gather (§22.4's conclusion — the
1.6x ceiling was DRAM streaming, not the scalar multiply), while attention's QK/AV arrays are dense,
small (`hd`-sized per head), and sequentially accessed — squarely compute-bound at this scale, so
the same instruction-level vectorization has far more room to help.

Bucket ranking is now: `linear(kernel)` 59.0ms (~47% of wall, unchanged) > `router` 22.1ms (fp32) >
`act-pack` 17.4ms > `swiglu` 14.0ms (still untouched, still deferred pending broader quality
validation) > `attention` 9.1ms (no longer tied with router) > `rope` 2.0ms > `rest(other)` 2.1ms.
Activation-packing remains the other, still-unaddressed half of this review item — a candidate for
a follow-up pass, not yet attempted.

### 22.10 Activation-packing vectorized — ported the real vendor RVV code, not a hand-rolled one, closes the review item

With attention done (§22.9), activation-packing (`pack_A_hp`/`pack_act_hp`) was the other,
still-unaddressed half of "otherwise move to activation packing or attention." Unlike attention,
this one wasn't a case of reusing an already-proven exact-math primitive (`vdot_f32`/`vaxpy_f32`) —
it needed a genuinely new quantization routine: find each 32-wide subblock's max-abs value, derive
a per-subblock scale plus a per-256-wide-block averaged scale, and round+clamp 256 floats to int8.
Float-to-int rounding is exactly the kind of thing that bit for this file before (the `vfwcvt`
vtype-timing bug in §22.6) — hand-writing a new vectorized quantizer from intuition risked a *third*
instance of "compiles clean, runs clean, silently wrong," the hardest class of bug to catch.

**So: don't hand-write it — trace the real vendor implementation, same as every kernel port this
session.** `pack_A_hp` was already commented as "a portable scalar port of `quantize_a_row_i8_hp`"
(`rvv_kernels.cpp:1989`) — meaning the *actual RVV vendor source* for this exact function was known
to exist and had simply never been ported, only its scalar behavior copied by hand in an earlier
session. Read it directly off the board (`/root/llama.cpp/ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp`):
it branches on `__riscv_vlenb()` — a `vlenb==128` path (VLEN=1024, this board's A100 cores, one
32-wide subblock fits exactly in one `e32m1` vector register) and a `vlenb==32` path (VLEN=256,
X100, `e32m4`/4x wider LMUL to cover the same 32 elements). Since every call site in this engine
runs on the main thread, which `main()` already pins to hart 8 (A100) before the decode loop,
**only the `vlenb==128` branch applies here** — ported that one, intrinsics-for-intrinsics, into
`bench/vendor_ime_actpack_probe.c`:

```c
size_t vl=__riscv_vsetvl_e32m1(32);
vfloat32m1_t v_a=__riscv_vle32_v_f32m1(a+kk*32,vl);
vfloat32m1_t v_a_abs=__riscv_vfabs_v_f32m1(v_a,vl);
vfloat32m1_t v_a_max=__riscv_vfredmax_vs_f32m1_f32m1(v_a_abs,__riscv_vfmv_v_f_f32m1(0.0f,vl),vl);
float max_abs_a=__riscv_vfmv_f_s_f32m1_f32(v_a_max);
/* ... scale_temp[kk]=max_abs_a/127.0f, scale_avg accumulation, unchanged from the scalar port ... */
vfloat32m1_t v_a_scale = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
vint16mf2_t  v_a_quant = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);      /* float -> int16, rounds */
vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);     /* narrow int16 -> int8 */
vint16m1_t   v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, __riscv_vmv_v_x_i16m1(0,vl), vl);
```

**Validation methodology**: not "does this look right" — byte-for-byte comparison against the
currently-shipping scalar `pack_A_hp` (copied verbatim into the probe as the oracle), across
200,000 random trials spread across 5 input distributions (generic random, tiny-magnitude,
one-dominant-outlier, all-zero, alternating-sign) chosen to stress the edge cases a quantizer is
most likely to get wrong.

**First run: 100% mismatch, every distribution.** Root cause wasn't the port — `__riscv_vlenb()`
reported **32** (X100/VLEN=256), not the expected 128, even though the probe explicitly pinned to
hart 8 via `sched_setaffinity`. Traced this to an infra gotcha, not a code bug (full writeup in
`PROGRESS.md`'s Toolchain gotchas): this SSH session's shell has `Cpus_allowed` capped to harts 0-7
by its cgroup, and `sched_setaffinity(CPU_SET(8))` alone **silently no-ops** when requesting a hart
outside that mask — no error was checked, so the probe just kept running on an X100 hart with
VLEN=256, where `__riscv_vsetvl_e32m1(32)` clamps to `vl=8` and only a quarter of each subblock got
processed. The fix was already sitting in `qwen_moe_hp.c`'s own `main()`, one line before its own
pinning call: `bind_ai()` — writes `"0"` to `/proc/set_ai_thread`, the vendor driver interface that
actually grants a thread access to harts 8-15. Adding the same `bind_ai()` call to the probe (in
the same order, immediately before `sched_setaffinity`) fixed it: `vlenb=128`, matching production.

**Second run, with `vlenb=128` confirmed: 20.0000% mismatch, exactly and only the all-zero-row
distribution (`mismatched_records=40000` of 200000, `20000/5`).** A real, narrow divergence: the
scalar port initializes its running max as `float amax=1e-6f` (a defensive floor against
division-by-zero for an all-zero input), while the vendor RVV code starts its `vfredmax` reduction
from a literal `0.0f` with no floor. For a genuinely all-zero subblock this makes `scale_temp[kk]`
either a tiny nonzero value (scalar) or exactly `0.0f` (RVV) — different stored bytes, but
**numerically inconsequential**: an all-zero activation block contributes exactly 0 to the matmul
either way, since the quantized data is 0 regardless of which scale accompanies it. Matched the
scalar port's floor in the RVV version anyway (one `if` before the divide) — not because leaving it
out would have been wrong in practice, but because "byte-identical" is a strictly stronger,
easier-to-trust guarantee than "equivalent in the cases I thought to check," and it cost nothing.

**Third run: byte-identical, 0/200000 mismatches, 0 total byte diffs, all 5 distributions.** Hot
kernel-only timing (2M reps, warm cache, matching A3's methodology): **6009.7ns/call scalar vs
1542.4ns/call RVV — 3.90x.**

**Wired into production directly** — replaced `pack_A_hp`'s body in `qwen_moe_hp.c`, no feature
flag needed (byte-identical to the code it replaced, not an approximation, same reasoning as the
attention change in §22.9). Added the same `noinline,optimize("no-tree-vectorize")` guard as every
other RVV function in this file, defensively (this code doesn't touch `vmadot` state at all, so it
likely isn't strictly required, but costs nothing and matches the file's established pattern).

**Result: act-pack 17.4ms→2.3ms (-87%), 7.68-7.89→8.84-9.25 tok/s.** The production win is bigger
than the isolated 3.90x hot-call ratio would suggest — plausibly because the RVV version also
issues far fewer scalar loads/stores/branches per call, which helps more under realistic
non-warmed-cache conditions than a tight warm-cache repeat-call microbenchmark can capture. This is
the single largest per-token win of the session so far, larger than the attention change (§22.9,
-51% off a smaller bucket). Two runs on the board, both `' Tokyo'` PASS, identical continued
generation (Brasília, Ottawa, then Canberra/Cairo/New Delhi on the longer 40-token run).

Bucket ranking is now: `linear(kernel)` 58.6-58.7ms (~52% of wall, dominant) > `router` 18.3-18.9ms
(fp32) > `swiglu` 14.0ms (still untouched) > `attention` 9.2-14.7ms (grows with context, no longer
fixed) > `act-pack` 2.3ms (down from 17.4ms, no longer top-3) > `rope` 2.0ms > `rest(other)` 2.1ms.
With attention and activation-packing both closed out, `linear(kernel)` is now the clear, sole
largest bucket and hasn't been revisited since A3's hot-timing validation — the natural next target.
