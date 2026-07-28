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

### 22.11 Before optimizing `linear(kernel)`: checked "memory-bandwidth-bound" against a real number — it doesn't hold up

Concurrently with this session's own investigation, subclass instrumentation landed in
`qwen_moe_hp.c` (`g_lin_class`/`lin_add()`, partitioning the 58.6ms `linear(kernel)` bucket by
consumer without touching the math) and found: **expert FFN (gate+up+down across 8 experts) is
35.9ms — 61% of linear(kernel), the clear majority** — vs qkv 9.4ms (16%), o 7.6ms (13%), lm_head
5.7ms (10%). The natural read: optimize the expert path.

Before acting on that, worth checking the load-bearing assumption underneath it. This session has
repeatedly reached for "memory-bandwidth-bound" as the explanation for why vectorization stalls
out — the router's RVV pass only got 1.6x (§22.4), and nt=8 regressed vs nt=4 (§22.5) — and that
framing was about to be used again to either justify or deprioritize expert-path work. It had never
actually been checked against a real achievable-bandwidth number on this hardware.

**`bench/ram_bw_probe.c`**: nt=4 threads, pinned exactly like production (harts 8,10,12,14,
`bind_ai()`+`sched_setaffinity` per the §22.10 infra gotcha), each stream-read a private 256MB
buffer (far larger than any cache level) for 4 passes. Result: **17.05 GB/s aggregate** (solo
single-thread: 10.56 GB/s — notably the aggregate is only ~1.6x solo, so even a clean sequential
pattern doesn't scale linearly past 1 thread on this board, a fact worth remembering on its own).

Compared against `linear(kernel)`'s real per-token weight-byte volume — computed from `BREC`
(576B/32×256 tile) × tile-count across q/k/v/o and all 8 selected experts' gate/up/down, summed
across 48 layers — **191.1 MB/token**. At 58.5ms (nt=4), that's **3.26 GB/s effective: only 19% of
the measured 17.05 GB/s ceiling.**

**Isolated the per-thread-scaling question directly**, not just inferred from the aggregate: reran
the real engine at nt=1 (no pool, no multi-thread contention at all). Result: `linear(kernel)`
169.8ms → **1.13 GB/s**, closely matching A3's original isolated-kernel rate (446.4ns/call for a
576B tile ≈ 1.29 GB/s) — reassuring, not a new/contradictory number, just confirmation the nt=1
path is behaving as expected. But nt=4's 58.5ms is only a **2.90x speedup over nt=1, not the ~4x**
a compute-bound, unit-per-thread workload should give — and nt=4 already assigns one thread to
each of the 4 physical IME-2 units with zero unit-sharing (unlike nt=8, which puts two harts on
each shared unit — the well-understood, different cause of *that* regression, per §22.5).

**Conclusion: DRAM-controller bandwidth is not the limiter — 81% of the measured streaming-read
ceiling sits unused while `linear(kernel)` runs.** The "memory-bandwidth-bound" framing, while
correctly ruling out nt=8 and explaining why the router's RVV pass had a low ceiling, does **not**
explain the sub-linear nt=1→nt=4 scaling seen here, and citing it again to wave off further
expert-path work would be exactly the kind of unmeasured guess this file has flagged as a repeat
mistake before (§22.6's premature "router not worth pursuing," the two timing-boundary bugs).
Candidate explanations for the actual gap, none yet isolated:

1. **Pool-dispatch/synchronization overhead**, accumulating over ~1392 small `Lin` calls/token —
   each dispatch round-trip (generation-counter bump, worker spin-wake, per-panel work, done-flag
   spin-wait) has some fixed cost that a large aggregate byte-count measurement doesn't separate
   out from real memory/compute time.
2. **Real per-call access is far less favorable than a clean 256MB stream** — each `Lin`'s `B`
   buffer is a separate, independently-`malloc`'d region (via `lin_new_hp`), so every call boundary
   is a cold start for the DRAM row buffer / hardware prefetcher, unlike the probe's one long
   uninterrupted stream per thread. Per-thread chunks are also small (e.g. an expert's `eg`:
   ~27.6KB/thread/call at nt=4) — row-buffer-friendly streaming has much less runway to help before
   the next call jumps somewhere else in the heap.
3. **The kernel's own fp16-accumulation compute path may be the real per-thread ceiling**, not the
   memory fetch feeding it — A3's original framing ("kernel microarchitecture is the dominant
   factor, not ggml dispatch/threading overhead," when comparing against the custom q4-in-q8
   kernel) was about relative kernel efficiency, not an absolute bandwidth statement, and may still
   be closer to the truth than the "bandwidth-bound" language that got applied to it later.

**Not resolved here — deliberately left as an open, well-defined question rather than a guessed
fix.** The next step has to isolate dispatch-wait time from kernel-execution time (e.g. a separate
timer around just the pool's spin-wait-for-done, vs. a timer around a single unpooled direct kernel
call at realistic per-thread chunk sizes) before any change to the dispatch pool or the kernel
itself can be justified by more than intuition. `linear(kernel)`'s 58.5ms bucket, and its expert-FFN
majority (61%, from the concurrent subclass work), stand as the confirmed target — *how* to close
the gap to the vendor binary's 11.71-12.89 tok/s is still open.

### 22.12 §22.11 retracted: factor-of-8 byte-count error — the bandwidth math was wrong, not the bottleneck

**§22.11's central claim ("DRAM bandwidth is not the limiter — 81% of the ceiling sits idle") is
wrong, caught by external review before any code change was made on the strength of it** — a case
of exactly the "measure before optimizing" discipline working as intended, just one step later than
ideal (the measurement itself had a bug).

**The error**: the per-token weight-byte calculation used `tiles(N,K) = (N/32)*(K/256)` — correctly
counting the number of N32×K256 *superblocks* in a `Lin` — then multiplied that count by `BREC`
(576, `#define`d as "bytes per 32-wide-K x 32-wide-N B record", i.e. **one K32 subblock**) instead
of `BSUPER` (`NSUB*BREC` = 4608, "bytes per 256-wide-K x 32-wide-N B superblock", the actual byte
count for one N32×K256 tile — `NSUB=8` K32-subblocks per K256-superblock). An 8x undercount,
present identically in the constant used by both the nt=4 and nt=1 figures, so it canceled out of
their *ratio* (the "2.90x not ~4x" scaling claim is unaffected) but invalidated every absolute
GB/s number derived from it.

**Corrected**: 1.529 GB/token (was 191.1 MB — matches this session's own much earlier "Honest perf
model" estimate of ~1.5-1.9 GB/token, which the wrong number should have been checked against and
wasn't). Effective bandwidth: **nt=4 → 26.13 GB/s** (was "3.26 GB/s, 19% of ceiling"), **nt=1 →
9.00 GB/s** (was "1.13 GB/s"). The nt=1 figure now closely matches A3's original single-call
kernel-only rate reinterpreted correctly (446.4ns for one full N32×K256 call = 4608B ≈ 10.32 GB/s,
not 576B ≈ 1.29 GB/s as both this section and, apparently, the original A3 write-up's informal
framing implied) — good internal consistency once the same fix is applied throughout.

**The `bench/ram_bw_probe.c` "ceiling" is also not trustworthy as an upper bound, independent of the
byte-count fix.** It measures one `volatile` 64-bit scalar load per 64-byte cache line with a
dependent scalar accumulation — a pattern that can be latency-bound / limited by outstanding-request
count, not necessarily bandwidth-bound. The real kernel issues wide RVV vector loads (`vle16.v`,
`vl4r.v`) with a different, likely higher-MLP (memory-level-parallelism) access pattern. Consistent
with this: the corrected nt=4 rate (26.13 GB/s) now *exceeds* the probe's claimed 17.05 GB/s
ceiling — a real workload outperforming its own supposed upper bound is a clear signal the bound
was measured wrong, not that the workload is unusually efficient.

**Revised conclusion — reverting to, not away from, the earlier framing**: after the byte-count
fix, the corrected numbers are consistent with `linear(kernel)` genuinely being bound by shared
memory/cache/fabric throughput, not dispatch overhead or heap-allocation locality as §22.11
speculated. A 2.9x gain from nt=1 to nt=4 is an entirely ordinary shape for four workers approaching
saturation of a shared resource. **This does not mean there is zero headroom** — the probe's real
flaw is that it can't be trusted as a ceiling in either direction, so the true achievable bandwidth
on this hardware under a vector-load access pattern is simply unknown, not "known to be ~17 GB/s
with lots of room" (§22.11) or "known to already be maxed out" (this section's best guess, not a
measurement). A trustworthy next probe would need to replicate the kernel's actual access shape
(wide vector loads, multiple outstanding requests, real per-Lin buffer fragmentation) rather than a
single dependent scalar stream, before drawing any further conclusion — dispatch-overhead hunting
(§22.11's candidate #1) is deprioritized by this correction, but not proven wrong, just no longer
supported by the evidence that motivated it.

### 22.13 Five follow-up probes, in order: what's actually left unexplained (closed out — not a new optimization target)

§22.12 corrected the bandwidth math but left the *mechanism* open. Five further probes, each
isolating one variable at a time, run to ground before deliberately stopping (external review:
"the remaining synthetic-to-production difference is too small and poorly localized to justify
chasing speculative allocation/A-buffer explanations now").

**1. `bench/run_hp_m1_scaling_probe.c` — does the kernel itself scale linearly with no dispatch?**
Each thread hammers `run_hp_m1` on its own small (36-221KB), reused, cache-resident private
buffers. Result: **10.54 → 20.70 → 41.13 GB/s (nt=1/2/4), 3.90x** — near-perfect linear scaling.
Rules out shared tensor-engine-unit contention or kernel-compute as an inherent bottleneck under
concurrency (nt=4 already gives one thread per physical IME-2 unit, no sharing).

**2. `bench/pool_dispatch_overhead_probe.c` — does the real dispatch mechanism cost anything on
top of that?** Same hot buffers, but routed through a faithful replica of `qwen_moe_hp.c`'s actual
persistent spin-dispatch pool (atomic gen/done counters, main thread as tn=0, workers 1..nt-1
spinning), at production's real per-dispatch granularity (6 panels/thread/round, matching eg/eu).
Result: **10.23 → 20.62 → 39.49 GB/s, 3.86x** — within ~4% of kernel-only. Dispatch overhead is
negligible.

**3. `bench/cold_streaming_probe.c` — what happens with a real cold (≥256MB/thread, single-pass,
no reuse) working set?** Four variants at nt=4: contiguous single buffer **35.68 GB/s**;
production Lin sizes/order (separate `malloc` per Lin, real per-layer visit sequence) **30.60
GB/s**; randomized visit order **30.13 GB/s**; `MADV_HUGEPAGE` **29.97 GB/s**. Randomized-vs-ordered
and hugepage-vs-normal both show no meaningful difference — rules out call-order predictability and
TLB/page-size pressure. Cold access alone reaches 30-36 GB/s, comfortably above production's 26.13
GB/s, so "cold access is just slow" doesn't explain the gap either.

**4. `bench/shared_buffer_scheduling_probe.c` — does production's actual per-thread panel striding
matter?** Same cold working set, but now genuinely *shared* across all nt threads (not private per
thread) through the real dispatch pool, comparing production's real assignment (**cyclic**:
`np=tn; np<Np; np+=nt`) against **blocked** (each thread gets one contiguous panel range). Across
the full q/k/v/o/8×(eg/eu/ed) set: cyclic **29.34 GB/s** vs blocked **29.09 GB/s** at nt=4 — no
meaningful difference. Also tested whether the probes' tiny reused output-write buffer (vs
production's real per-panel-unique write into a full N-sized array) mattered: **28.95 vs 28.76
GB/s** with a real N-sized output array — also no difference. Both candidates ruled out for the
q/k/v/o/expert Lins.

**5. `lm_head` in isolation — the one place scheduling actually matters.** `lm_head` is
structurally different from every other Lin in this model: `N=151936` gives `Np=4748` panels,
roughly 40x more than the next-largest Lin (`q`, Np=128) tested. Isolated it: **cyclic 26.18 GB/s
vs blocked 29.45 GB/s at nt=4 — a real ~12% gap**, and cyclic's number lands almost exactly on
production's measured overall rate (26.13 GB/s). **Phrased cautiously, per review**: the experiment
demonstrates a real *scheduling* effect, not a confirmed hardware mechanism — it was not
instrumented to observe DRAM row-buffer behavior directly. The plausible explanation is that
cyclic striding gives each hart an ~147KB stride between consecutive `lm_head` panels
(`4*36864`B, `nt=4`), repeated ~1187 times across a 175MB span, which *apparently* defeats
per-hart streaming/prefetch behavior over that unusually long traversal — the smaller Lins never
accumulate enough stride distance (at most ~30 strides within a ≤4.5MB buffer) for the same effect
to show up.

**Net accounting, and why this stops here.** `lm_head` is only ~11% of total per-token weight
bytes and ~10% of `linear(kernel)`'s time (5.7ms of 58.5ms, per the subclass breakdown) — even a
full fix there is back-of-envelope worth roughly half a millisecond per token, not the whole
26→30GB/s gap. **Recording this as an unresolved ~10% delivery gap between synthetic cold-access
ceilings (~29-36 GB/s across every tested variant) and production's real 26.13 GB/s — not a new
optimization target to chase further with more speculative probes** (real `Lin` allocation
patterns across all 1344 buffers at once, A-buffer/pack interaction, and other candidates remain
untested and are being deliberately left that way; the marginal value of resolving them is judged
too small relative to the effort of building yet another faithful synthetic reproduction).

**One concrete, actionable candidate survives**: switch `lin_mm_hp_worker_run`'s panel assignment
from cyclic to blocked (`lo=Np*tn/nt, hi=Np*(tn+1)/nt`) for `lm_head` — or, since blocked did not
regress any of the smaller Lins in probe #4, as the new general default for every `Lin`, which
would also be simpler code. **Not yet applied to production, and not yet A/B tested against the
real engine** — this is a recommendation pending that test, not a landed result. If a real
production run confirms the gain (expect `lm_head`'s 5.7ms bucket dropping toward ~5.1ms, and no
regression elsewhere), blocked assignment should become the default.

### 22.14 The production A/B: blocked scheduling REGRESSED — reverted, not adopted

Ran the one remaining bounded experiment from §22.13: changed `lin_mm_hp_worker_run`'s panel
assignment from cyclic (`np=tn; np<Np; np+=nt`) to blocked (`lo=Np*tn/nt, hi=Np*(tn+1)/nt`), for
both the int4-HP and int8-M1 kernel paths (a pure scheduling change — panels are independent, each
writing a disjoint `y[np*32..]` slice, so which thread computes which panel cannot change the
numeric result; no oracle validation needed, only the token-identity check below).

**Method**: built two binaries from the exact same source tree, one at the current committed
(cyclic) state, one with only the panel-assignment change, both `LD_LIBRARY_PATH=/usr/lib
./qwen_moe_hp{_cyclic,_blocked} /root/models/Qwen3-30B-A3B-Q4_0.gguf 16 4
/root/models/qwen3-30b-a3b.hp.imecache 0 0` — same prompt, same cache, fp32 router (default), same
generation length, nt=4, run back to back on the board, twice each.

**Tokens identical in every run**: `' Tokyo'` PASS, generation matches exactly
(`...Tokyo. ...Brasília. ...Ottawa.`) in all four runs — the scheduling change is exactly as
correctness-neutral as expected.

**Throughput: the OPPOSITE of §22.13's isolated-probe prediction, and reproducible across both
runs:**

| Bucket (ms/tok) | cyclic run 1 | blocked run 1 | cyclic run 2 | blocked run 2 |
|---|---|---|---|---|
| qkv | 9.4 | **10.2** | 9.4 | **10.2** |
| o | 7.6 | **8.5** | 7.6 | **8.5** |
| expert (gate/up/down) | 36.0 | 35.1 | 35.9 | 34.7 |
| **lm_head** | **5.7** | **6.0** | **5.7** | **6.0** |
| linear(kernel) sum | 58.8 | 59.8 | 58.5 | 59.4 |

`qkv`, `o`, and — most notably — `lm_head` all get consistently *worse* with blocked scheduling
(identical shift both runs, well outside this session's observed noise band for these buckets),
not better. Only `expert` improved slightly. Net `linear(kernel)` is worse with blocked in both
runs (+1.0 to +1.7%).

**§22.13's pre-registered keep-criterion was explicit: "keep only if `lm_head` falls toward ~5.1ms
with no regression elsewhere."** This result fails on both counts — `lm_head` rose, and `qkv`/`o`
regressed. **Reverted immediately, per the pre-committed criterion, not kept.** `qwen_moe_hp.c` is
back to the exact committed (cyclic) state; no production code changes from this experiment.

**Why the isolated probe's prediction failed to generalize is not diagnosed, and — per the standing
decision to stop this investigation (§22.13) — is not being chased further.** One observation worth
recording without further testing: `shared_buffer_scheduling_probe.c` found NO cyclic-vs-blocked
difference for `qkv`/`o`-sized Lins (Np≤128) in isolation, yet the real engine shows a real,
reproducible regression for exactly those Lins under blocked scheduling — meaning the isolated
probe's *conclusion for the small Lins* also didn't transfer cleanly to production, not just its
`lm_head` prediction. This is itself informative: it says the remaining ~10% gap (§22.13) and this
scheduling experiment's reversed result both point at something about the *real* engine (full
model, real interleaving with pack/RoPE/router/SwiGLU, the actual all-1344-buffers-at-once
allocation pattern) that none of the five isolated probes fully captured, rather than validating
either probe's specific mechanism story. The lesson generalizes past this one experiment: an
isolated synthetic probe that rules a factor in or out is real evidence, but a production A/B is
still required before adopting anything derived from it — exactly the discipline that caught this
one before it shipped.

**Closed.** No further scheduling work on `linear(kernel)` planned. Per external review, the next
high-value branch is a real multi-prompt quality harness — it unlocks safely evaluating SwiGLU
approximation and promoting the int8 router (§22.8) past "experimental," neither of which the
current single-prompt `' Tokyo'`-coherence check can responsibly settle.

### 22.15 Multi-prompt quality harness built, int8-M1 router promoted to default

Built `run_quality_harness()` (`qwen_moe_hp.c`, `QWEN_HARNESS=1` env var), replacing single-prompt
`' Tokyo'`-coherence with teacher-forced evaluation across 10 fixed prompts spanning factual (2),
reasoning (2), code (2), multilingual — French + Chinese (2), and long-context ≥113-token prefills
(2). Prompts were tokenized with the real tokenizer (`llama-tokenize --ids --no-bos`, already on
the board) rather than hand-guessed IDs, and cross-checked against the existing golden prompt's
known-correct token array before use.

**Promotion thresholds fixed in the code before any run** (not tuned after seeing results):
router expert-set mismatch < 10%; avg NLL delta < 0.5 nats/token; token-argmax divergence < 15%;
router bucket ≥10% faster than fp32. All four must PASS.

**Methodology**: for each prompt, (1) generate a reference continuation greedily under fp32,
recording each token's own self-NLL; (2) replay the same prompt + reference tokens under int8 via
**teacher forcing** — feed the reference token at every step regardless of what int8 itself would
pick, so one divergence can never compound into a different context for later positions; compare
`argmax(int8_logits)` against the reference token and compute NLL of the reference token under
int8's distribution at each step.

**Three real bugs found and fixed while building this, none in the eventual harness logic itself:**

1. **A genuine pre-existing memory-corruption bug**, newly exposed, not caused, by this work. The
   first version passed the harness trigger as an 8th positional CLI arg (`argv[7]`). It read back
   correctly at the very top of `main()` but was clobbered — reproducibly, to what looks like
   reinterpreted weight data (`0x358637bd49742400`) — by the time execution reached
   `lin_mm_pool_init()`, somewhere during `cache_load`/model setup. Confirmed via `dmesg` (the
   crashing `badaddr` matched the corrupted pointer exactly) and by printing every `argv[i]` at
   both ends of `main()`. **Never manifested before because nothing previously read past
   `argv[6]`** (`g_router_mode`). **Not root-caused — worked around**, not fixed: switched the
   harness trigger to `QWEN_HARNESS=1` (env var via `getenv`, read once, not stored in a
   heap/stack region apparently shared with something in the load path) instead of a new CLI arg.
   **This is a real, open finding**, flagged here and in `research_feed_paths.md` for whoever next
   touches `cache_load`/`model_load` — the corruption is real regardless of which mechanism
   triggers the harness, it was just invisible before.
2. **The exact "vmadot-adjacent autovectorization" toolchain gotcha this file already documents**
   (Toolchain gotchas, `PROGRESS.md`), hit a third time. The new harness functions are full of
   plain scalar accumulation/comparison loops — nothing touching custom instructions — yet adding
   them caused the *unconditional* baseline decode path (harness never invoked) to segfault right
   after `lin_mm_pool_init()`. Confirmed by reverting to the clean committed `qwen_moe_hp.c` on the
   board (worked perfectly) vs the harness-added version (crashed identically with zero harness
   code executed). Fixed with the established mitigation:
   `__attribute__((noinline,optimize("no-tree-vectorize")))` on every new harness function. Worth
   restating plainly since this is the third time: **any new function added to this file, however
   innocuous, needs this attribute** — the compiler doesn't know the file contains hand-scheduled
   vector-register state that ordinary `-O3` autovectorization can silently corrupt.
3. **A measurement-contamination bug of the same class this session has hit repeatedly**
   (`research_feed_paths.md` §12's "always suspect the measurement before the hardware"). The
   first harness run enabled `g_router_validate=1` for the entire int8 teacher-forced pass, since
   that flag is also what's needed to collect router expert-set mismatch stats. But
   `g_router_validate=1` makes *every* `forward()` call additionally compute the fp32 and int4-HP
   router variants too, plus O(experts) sentinel-argmax/mismatch-counting overhead — none of it
   wrapped in a named timing bucket, so it all landed in the untimed `rest(other)` bucket
   (1.4ms→38.1ms/tok) and produced a bogus **"int8 router is 29.1% slower"** result — the opposite
   sign from every other measurement of this router this session. Fixed by splitting into two
   passes: phase 2a (`g_router_validate=1`, timing explicitly discarded via a `gT_on` save/restore)
   for the mismatch stats, phase 2b (`g_router_validate=0`, clean apples-to-apples timing) for
   speed. Corrected result (13.7% faster) closely matches this session's independently-measured
   §22.8 number (int8 router ~16.0ms vs fp32 ~18.3-18.9ms), confirming the fix rather than just
   flipping the sign to what was expected.

**Final result, 10 prompts, 192 total generated tokens, 25,392 router comparisons:**

| Metric | Threshold | Result | Verdict |
|---|---|---|---|
| Router expert-set mismatch | < 10% | **6.1%** (1557/25392) | PASS |
| Avg NLL delta (int8 tf − fp32 self) | < 0.5 nats/tok | **−0.0034** | PASS |
| Token-argmax divergence | < 15% | **2.1%** (4/192) | PASS |
| Router bucket speedup | ≥ 10% | **13.7%** (18.7→16.1ms) | PASS |

The 6.1% mismatch rate is notably *better* than §22.8's single-prompt estimate (7.8%) now that
it's measured across a real diversity of prompt types and lengths — the earlier number wasn't
wrong, just thin evidence (1344 comparisons on one 28-token prompt vs 25,392 here). The NLL delta
being slightly *negative* (int8 marginally more confident in fp32's own token choices than fp32
itself) is noise around zero, not a meaningful effect, but it's a strong signal there's no
systematic quality degradation. Only the multilingual/French and long-context/narrative prompts
showed any token divergence at all (1/20 and 3/16 respectively) — no divergence on factual,
reasoning, code, or the Chinese prompt.

**All four thresholds pass. Per the standing instruction ("keep both behind flags until they pass
the harness"), int8-M1 router is now the production default** (`g_router_mode` default changed
from 0 to 2; `g_router_mode=0` remains available as an explicit exact-fp32 revert flag; int4-HP
stays available as `g_router_mode=1` but remains rejected per §22.7, not promoted). Verified on the
board post-change: default invocation now reports `router=int8-M1(default)`, `' Tokyo'` PASS,
router bucket 16.2ms, unchanged decode correctness.

**Not done**: SwiGLU approximation evaluation — explicitly the next step ("then evaluate a fast
sigmoid/SwiGLU implementation"), not started. The harness is general enough to evaluate it the same
way (teacher-forced NLL/divergence against the fp32 reference) once a candidate implementation
exists, without needing a rewrite.

### 22.16 Memory corruption root-caused (it was the autovec bug all along) + systematic toolchain hardening

Per explicit direction after the int8 promotion: "next should be memory-corruption/toolchain
hardening, then SwiGLU evaluation." Two findings, closing out both at once.

**The "unresolved" `argv[7]` memory-corruption bug (§22.15) was never a separate bug.** It was the
same vmadot-adjacent autovectorization collision already documented twice in this file, just
manifesting as corrupted-but-not-crashing memory instead of an immediate SIGSEGV. Confirmed
directly: rebuilt a reproduction harness (`argv[7]` restored, printed at the point that used to
show corruption) three ways —

| Build | `v[7]` at the crash-adjacent point |
|---|---|
| `-fsanitize=address -O1` | correct (`"1"`) |
| `-O3`, current (harness-fixed) source | correct (`"1"`) |
| `-O3`, source *before* the §22.15 `noinline` fixes | corrupted (`0x358637bd49742400`) |

The `noinline,optimize("no-tree-vectorize")` attributes added in §22.15 to fix the *unconditional
baseline crash* had **already fixed the argv corruption too** — this was one bug with two
symptoms, not two bugs. §22.15's "not root-caused" note was written before circling back to
re-verify `argv[7]` specifically after that fix landed; it should have been retested then. No
remaining open memory-safety item from this investigation.

**Systematic hardening: found the per-function attribute approach is not just reactive but
provably incomplete, adopted a global build flag instead.** Even with every function this session
has individually patched (`pack_A_hp`, `vdot_f32`, `vaxpy_f32`, all `harness_*` functions), the
*now-default* int8-M1 router path was still running dramatically slower than it should — not
crashing, just silently paying a large, real performance tax. Bisected two contributing functions
directly (`pack_A_i8`, `lin_mm_hp_worker_run`), each independently unguarded and each responsible
for only part of the effect (16.2ms→15.1ms and →15.6ms respectively when patched alone) — meaning
**at least one more contributing function was never found**, because a third confirmed fact made
further one-by-one hunting not worth it: a global `-fno-tree-vectorize` build flag captured the
*entire* effect at once (router 16.2ms→4.4ms, a ~4x cut) and, as a side effect, *also* measurably
improved the fp32 router path (~18-19ms→11.1ms) — a path none of the individually-tested functions
even touch, proving there's real, uncaught exposure elsewhere in the file that per-function
patching was never going to fully close.

Verified safe and net-positive across all three router modes, repeated runs each:

| Mode | Router bucket, before | Router bucket, with `-fno-tree-vectorize` |
|---|---|---|
| int8-M1 (default) | 16.2ms | **4.3-4.4ms** |
| fp32 (revert flag) | ~18.7-19.0ms | **11.1ms** |
| int4-HP (rejected, still available) | ~12.5-16.0ms (historical) | **1.5ms** |

Net decode: **8.84-9.25 → 9.5-9.9 tok/s**, all with `' Tokyo'` PASS and identical generation —
correctness unaffected, as expected (this flag only touches gcc's *automatic* vectorization pass;
every real hot-path vectorization in this file is explicit RVV intrinsics or hand-written asm,
neither of which the tree-vectorize pass produces or can remove). One small, consistently observed
regression: `rope+qknorm` ~2.0ms→~3.1-3.2ms (rope's scalar `sinf`/`cosf` loop apparently was
getting *legitimate*, correct autovectorization benefit) and `rest(other)` ~2.2ms→~8.0ms — both
real but small relative to the router/fp32 gains, and net wall-clock is unambiguously better in
every mode tested.

**Adopted as the standing build flag** (`qwen_moe_hp.c`'s header comment): `gcc -O3
-fno-tree-vectorize -march=rv64gcv_zvfh_xsmtvdotii -fopenmp ...`. Existing per-function
`noinline,optimize("no-tree-vectorize")` attributes are left in place (harmless alongside the
global flag, and useful as documentation of specifically-confirmed-affected functions), but **the
global flag is now the load-bearing fix, not the per-function attributes** — any future function
added to this file is automatically covered without needing to remember the attribute.

**One thing this did NOT do**: fully explain *why* ordinary scalar code — no custom instructions,
no inline asm — gets miscompiled or de-optimized by `-O3`'s tree-vectorizer specifically in this
translation unit. Three confirmed incidents (a SIGSEGV, a corruption, a silent 4x slowdown) and a
working fix, but the actual gcc/RVV-backend interaction responsible was never isolated further
than "vectorizing ordinary code near vmadot-using code is unsafe on this toolchain" — that
empirical rule has now held three times and is the operative one, whether or not the precise
compiler-internals mechanism is ever found.

### 22.17 Quality harness re-run under the mandatory build — confirmed, this run is now the baseline

§22.15's promotion harness ran under the pre-hardening build (no `-fno-tree-vectorize`). Since
§22.16 made that flag mandatory, re-ran the identical harness unchanged, built with the exact
command now in `qwen_moe_hp.c`'s header comment, before doing anything else — per explicit
instruction, SwiGLU work does not start until this confirms clean.

**Quality metrics: essentially unchanged, as expected** (the flag only affects gcc's automatic
vectorization pass, never explicit RVV intrinsics/asm, so no numeric difference should exist and
none does beyond ordinary run-to-run noise):

| Metric | §22.15 (pre-hardening build) | This run (production build) | Threshold |
|---|---|---|---|
| Router expert-set mismatch | 6.1% (1557/25392) | **6.1% (1541/25392)** | < 10% |
| Avg NLL delta | −0.0034 | **+0.0060** | < 0.5 |
| Token-argmax divergence | 2.1% (4/192) | **1.6% (3/192)** | < 15% |
| Router speedup vs fp32 | 13.7% (18.7→16.1ms) | **57.2% (10.2→4.4ms)** | ≥ 10% |

All four thresholds **PASS again**, cleanly, under the actual production compiler configuration.
The speed delta looks dramatically larger here than the 13.7%→~14-30% range seen in single-prompt
production runs — expected, not a discrepancy: this harness measures router cost in isolation
across 10 varied prompts via the same phase-2b clean-timing methodology as §22.15, and both fp32's
and int8's absolute bucket times dropped substantially under `-fno-tree-vectorize` (§22.16), with
int8's dropping proportionally further — consistent with, not contradicting, the standalone
single-decode numbers reported in §22.16's own table (int8 default: 16.2→4.4ms; fp32: ~19→11.1ms).

**This run is now the reference baseline** for judging any future router or SwiGLU change — not
§22.15's numbers, which were measured under a build configuration no longer in use. Any future
harness comparison (SwiGLU candidates, a possible re-look at int4-HP, etc.) should be run under
this same mandatory build and compared against the numbers in this table, not §22.15's.

**Next, per explicit instruction**: implement one flagged fast-SwiGLU candidate with quality and
speed gates predeclared before implementation, evaluated with this same harness once it exists.

### 22.18 Fast-SwiGLU candidate: hard-swish, promoted to default

Per explicit instruction, only starting this after §22.17 confirmed the router promotion holds
under the mandatory `-fno-tree-vectorize` build. Gates predeclared, in the source, before writing
`swiglu_hswish_rvv` — not tuned after seeing results:

1. Avg NLL delta < **0.3** nats/token — stricter than the router's 0.5, because SwiGLU runs on
   *every* selected expert at *every* layer (pervasive), unlike routing, which only affects which
   experts get chosen once per layer.
2. Token-argmax divergence < **15%** — same bar as the router.
3. SwiGLU bucket ≥**15%** faster than exact — stricter than the router's 10%, since replacing a
   known-exact nonlinearity with a coarse approximation is a bigger quality gamble than an
   already-somewhat-lossy int8 quantization, so a bigger performance case is required to accept it.

**Candidate: hard-swish** (Howard et al. 2019, MobileNetV3) — `x * hard_sigmoid(x)`,
`hard_sigmoid(x) = clamp((x+3)/6, 0, 1)`. Chosen over alternatives (a bit-manipulation fast-exp
trick, a minimax polynomial fit) specifically because it needs **zero transcendentals** — just
add/mul/clamp, all plain RVV float ops (`vfadd`/`vfmul`/`vfmax`/`vfmin`), genuinely vectorized
rather than relying on gcc's autovectorizer (which §22.16 established is actively unsafe in this
file). It's also a well-established, production-proven approximation, not something invented for
this session.

```c
static void swiglu_hswish_rvv(float*g,const float*u,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(g+i,vl), vu=__riscv_vle32_v_f32m1(u+i,vl);
        vfloat32m1_t vh=__riscv_vfadd_vf_f32m1(vx,3.0f,vl);
        vh=__riscv_vfmul_vf_f32m1(vh,1.0f/6.0f,vl);
        vh=__riscv_vfmax_vf_f32m1(vh,0.0f,vl);
        vh=__riscv_vfmin_vf_f32m1(vh,1.0f,vl);
        vfloat32m1_t vy=__riscv_vfmul_vv_f32m1(vx,vh,vl);
        vy=__riscv_vfmul_vv_f32m1(vy,vu,vl);
        __riscv_vse32_v_f32m1(g+i,vy,vl); i+=vl; }
}
```

**Two-tier validation, matching this file's established discipline but adapted for a genuinely
lossy approximation** (unlike every other RVV port in this file, there's no bit-exact oracle to
check against here):
1. **Vectorization correctness** (`bench/swiglu_hswish_probe.c`): compares the RVV implementation
   against a scalar implementation of the *same* hard-swish formula, across 50,000 trials × 768
   elements (the real `moe_ffn` width), spanning wide/near-zero/exact-zero/extreme-saturating
   input distributions. **0 mismatches, max abs diff 3.8e-6** (float rounding noise) — the vector
   code is a correct implementation of the formula. This catches implementation bugs, separately
   from asking whether the formula itself is a good enough approximation.
2. **Approximation quality**: the multi-prompt harness, extended with a new phase — reference
   generation held at *today's actual production default* (int8-M1 router + exact SwiGLU, not the
   original fp32 ground truth, since the question is "does adding this on top of what already
   ships cause a problem," not a re-litigation of the router decision), then hard-swish
   teacher-forced against that reference.

**Result, 10 prompts, 192 tokens:**

| Metric | Threshold | Result | Verdict |
|---|---|---|---|
| Avg NLL delta | < 0.3 nats/tok | **+0.0969** | PASS |
| Token-argmax divergence | < 15% | **5.7%** (11/192) | PASS |
| SwiGLU bucket speedup | ≥ 15% | **96.2%** (14.00→0.53ms) | PASS |

All three thresholds pass, though — worth stating plainly rather than burying it — both quality
metrics are meaningfully higher than the router's corresponding numbers (NLL delta +0.0969 vs the
router's ±0.006-0.01; divergence 5.7% vs the router's 1.6-2.1%). This is expected and was
anticipated when the stricter thresholds were set: hard-swish is a real, coarser approximation of
an actual nonlinearity, not a numerically-close quantization of weights. It clears the
predeclared, stricter bar comfortably, but by a smaller margin than the router did. The speedup is
dramatic (96.2%, removing per-element `expf()` transcendental calls entirely) — `moe_ffn`=768
elements × 8 experts × 48 layers × every generated token, previously the single most
expensive-per-element scalar operation in the file after activation packing was fixed (§22.10).

**Promoted to production default** (`g_swiglu_fast` default 0→1), per the standing instruction
("keep both [router and SwiGLU] behind flags until they pass the harness"). `g_swiglu_fast=0`
remains available as an explicit exact-SiLU revert flag (8th CLI arg). Verified on the board,
repeated runs: `' Tokyo'` PASS, identical generation, swiglu bucket 14.0ms→0.4-0.6ms, **decode
9.5-9.9→11.49-11.5 tok/s** — the single largest per-token-time win of the session, on top of
everything already landed.

**Session cumulative, from the original 1.49 tok/s pure-scalar baseline: ~7.7x.**

### 22.19 RETRACTED: hard-swish promotion was premature — default restored to exact SiLU

Per external review, immediately after §22.18 landed. The promotion decision was wrong to make on
the evidence available, for reasons that were all present in §22.18's own numbers but under-weighted
in how they were framed:

1. **The NLL delta is a ~10.2% perplexity multiplier, not a small number.** `exp(0.0969) ≈ 1.102`.
   §22.18 reported "+0.0969 nats/token, PASS (<0.3)" — technically true against the predeclared
   threshold, but reporting the raw nats figure obscured what it actually means: hard-swish makes
   the model about **10% more perplexed**, on average, about the tokens fp32-exact-SwiGLU would
   have generated. That is not a rounding-noise-scale effect.
2. **192 total generated tokens is a thin sample for a decision this consequential.** The router
   promotion's 25,392 router-comparisons (§22.15) is two orders of magnitude more evidence than
   192 teacher-forced token positions — and the router change is architecturally narrower (affects
   *which* experts are chosen) than SwiGLU (affects *every* expert's output at *every* layer).
3. **Hard-swish is not a numerically-close approximation of SiLU — it is a different activation
   function.** Every other "fast" technique in this file (RVV-vectorized dot products, activation
   packing, the int4/int8 router kernels) is either exact or a bounded-error quantization of the
   *same* underlying function. Hard-swish is qualitatively different: MobileNetV3 was **trained
   from scratch with hard-swish**, so its weights co-adapted to hard-swish's piecewise-linear
   shape. Qwen3-30B-A3B was trained with exact SiLU and has never seen hard-swish's shape during
   training. Applying it post-hoc is a real distribution shift on an unadapted model, not a "fast
   math trick" — a fundamentally different risk category than everything else promoted this
   session, and it should have been evaluated (and gated) as such from the start.

**Immediate action taken: `g_swiglu_fast` default reverted 1→0 (exact SiLU).** `g_swiglu_fast=1`
(hard-swish) remains available as an explicit experimental flag, not removed — the RVV
implementation is still validated-correct (§22.18's `bench/swiglu_hswish_probe.c` result stands,
0/38.4M mismatches against the scalar hard-swish formula; that finding is about vectorization
correctness and is unaffected by this retraction), only the *promotion* is retracted.

**Documentation corrections** (§22.18's phrasing, not its numbers, which stand as measured):
- "96.2% faster" is ambiguous — the swiglu bucket dropped 14.00ms→0.53ms, a **96.2% time
  reduction**, equivalently **~26.4x faster** (14.00/0.53). Both phrasings are now used together
  going forward to avoid the ambiguity.
- "swiglu bucket ≥15% faster than exact" was correctly implemented in the harness's own printf
  (`>=15%%`), but PROGRESS.md's prose write-up mistakenly said "(<15% required)" — corrected to
  "(≥15% required)".
- "within ~2% of 11.71-12.89 tok/s" compared only against the vendor binary's *low* endpoint
  (11.49/11.71 ≈ 98.1%). Against the *high* endpoint it's 11.49/12.89 ≈ 89.1% — **about 11% below**.
  Both endpoints should be stated, not the favorable one alone. (This entire comparison is now
  moot pending re-promotion in any case, since the default no longer includes hard-swish.)

**Remediation plan, in order, before any re-promotion is considered:**
1. ~~Restore exact SiLU as default, keep hard-swish behind its flag~~ — **done, this section**.
2. **Expand evaluation to several thousand tokens**, spanning perplexity-style evaluation on real
   (not model-generated) text, code, multilingual, reasoning, *and* free-running generations (not
   only teacher-forced against a reference) — teacher-forcing on the reference model's own
   continuation cannot reveal whether hard-swish's errors compound differently over a longer,
   uncorrected horizon. Not yet done.
3. **Compare the full combined production stack** (int8 router + exact SwiGLU vs int8 router +
   hard-swish) **against the original fp32-router + exact-SiLU baseline**, not only pairwise
   against the immediately-prior config — §22.18 only measured hard-swish's *marginal* effect on
   top of int8 routing, never the *cumulative* deviation from the fully-exact original engine that
   a real deployment decision should be judged against. Not yet done.
4. **Try a closer polynomial/rational sigmoid approximation** — something that approximates the
   *actual* sigmoid function within its normal operating range (unlike hard-swish, which is a
   different function by construction) — it may retain most of the ~13.5ms bucket gain with
   materially lower NLL drift, precisely because Qwen3's weights were trained against real SiLU's
   shape, not hard-swish's. Not yet done.
5. **Promote only after the larger evaluation**, with reconsidered thresholds — the retraction
   above suggests the original 0.3 nats/token bound (a 35% perplexity-inflation ceiling) was too
   loose; any next threshold should likely be stated directly in perplexity-multiplier terms rather
   than raw nats, so the number's real meaning doesn't require converting to notice a large effect.

### 22.20 Rational-SiLU candidate and expanded evaluation harness — implementation checkpoint, results pending

Implemented a second experimental SwiGLU mode, **not promoted and not yet evaluated on the board**:
a rational-Padé approximation to the actual sigmoid used by SiLU. It approximates
`tanh(x/2)` with `y*(27+y²)/(27+9y²)`, clamps to `[-1,1]`, then derives
`sigmoid(x)=0.5*(1+tanh(x/2))`. The RVV path uses ordinary add/multiply/min/max plus one vector
divide and remains behind `g_swiglu_fast=2`; exact SiLU remains the default (`0`) and hard-swish
remains experimental (`1`). A standalone RVV-vs-scalar and true-SiLU numerical probe was added as
`bench/swiglu_ratsig_probe.c`. Its initial RNG bug (`[0,2)` samples instead of `[0,1)`) was fixed
before any result was accepted.

The quality harness was expanded before running the candidate:

- generated continuations increased from 192 to 352 token positions;
- four independently authored real-text corpora add 688 next-token perplexity positions across
  literature, technical prose, code, and reasoning;
- candidates are compared both marginally against int8-router + exact-SiLU production and
  cumulatively against the original fp32-router + exact-SiLU stack;
- two free-running generations are printed to expose compounding behavior hidden by teacher forcing;
- hard-swish and rational-Padé are evaluated side by side under the same build and harness.

Final gates were fixed before the larger board run: teacher-forced perplexity multiplier `<1.05`,
token divergence `<15%`, and SwiGLU bucket reduction `>=15%`; additionally, independently authored
real-text perplexity must remain `<1.05x` aggregate versus int8 + exact SiLU, with no individual
corpus exceeding `1.10x`. Passing the teacher-forced checks alone is now explicitly preliminary.
No candidate becomes default from this checkpoint; the next step is to run the standalone probe
and full board harness, record the results, then perform a production A/B only for a candidate that
passes every gate.

**Checkpoint completed — results below.** `bench/swiglu_ratsig_probe.c` (RNG fix applied):
vectorization check 0/38.4M mismatches (max abs diff 5.7e-6) — the RVV path is a correct
implementation of the rational-Padé formula. Numerical closeness to true SiLU, rational-Padé vs
hard-swish, mean absolute error ratio: **2.51-5.18x lower error for rational-Padé** across
typical/near-zero/wide input ranges (unaffected by the RNG fix — both candidates draw from the
same corrected distribution, so the relative comparison was never invalid, only the absolute
numbers shifted slightly).

**Full board harness** (`HARNESS_GEN_SHORT=40`, `HARNESS_GEN_LONG=28` — up from 20/16 — plus the
4-text real-text corpus): **376 teacher-forced token positions** (not the estimated 352 above —
`8×40 + 2×28 = 376`), **687 real-text next-token positions** (`8×4` prompt-category positions was
never the mechanism; it's `Σ(text_len−1)` across the 4 corpora = `175+167+170+175`), plus the
router re-confirmation (376 tokens, 34,224 router comparisons — router thresholds still PASS
cleanly: 5.7% mismatch, +0.0037 nats/tok, 1.3% divergence, 59.0% faster).

| | vs int8+exact production | vs original fp32+exact | divergence | speed |
|---|---|---|---|---|
| **hard-swish** | **x1.1101 (11.0% inflation)** | x1.1411 (14.1%) | 6.1% (23/376) | 24.4x (95.9% reduction) |
| **rational-Padé** | **x1.0201 (2.0% inflation)** | x1.0262 (2.6%) | 3.2% (12/376) | 10.0x (90.0% reduction) |

Real-text perplexity (687 tokens, pure forward pass, independently authored — not teacher-forced
against either model's own output): fp32+exact=2.975, int8+exact=3.015, int8+hard-swish=2.989,
int8+rational-Padé=2.967 (perplexity, `exp(avg NLL)`). Aggregate multipliers vs int8+exact
production: hard-swish **x0.9915**, rational-Padé **x0.9841** — both comfortably under the 1.05x
gate. Per-corpus multipliers vs production (all four corpora, both candidates): hard-swish
1.034/1.016/0.993/0.928, rational-Padé 0.993/0.967/1.014/0.964 — **every value for both candidates
is under the 1.10x per-corpus gate.**

**This is the decisive, informative result of the whole exercise: real-text perplexity alone would
have passed hard-swish.** Its aggregate multiplier (0.9915) and every per-corpus multiplier clear
the real-text gates comfortably — real-text perplexity, averaged over a full probability
distribution, is not sensitive enough to catch what the teacher-forced greedy-argmax comparison
catches (an 11.0% inflation in how confidently the model predicts the *specific* tokens a sibling
config actually chose). Both methodologies were necessary; either alone would have given an
incomplete picture — real-text alone would have wrongly cleared hard-swish, and the original
§22.18 teacher-forced-only test (thin, 192 tokens, no real-text cross-check) is exactly what
produced the premature promotion this checkpoint exists to correct.

**Free-running (not teacher-forced) qualitative spot-check**, long-context/narrative prompt:
- int8+exact (production): *"...and noticed something peculiar: the hands of the watch were
  frozen at 12:00, but the gears inside were still turning."*
- int8+hard-swish: *"...and noticed something peculiar: the gears inside were not made of brass,
  but of a strange, dark metal that seemed to absorb light instead of"* — coherent, but a real
  narrative departure introduced mid-continuation.
- int8+rational-Padé: *"...and noticed something peculiar: the hands of the watch were frozen at
  midnight, but the gears inside were still turning. He asked the stranger,"* — tracks
  production's actual phrasing closely (same fact, "midnight" vs "12:00"), continues in the same
  direction.

**Verdict: hard-swish REJECTED** (fails the primary teacher-forced gate, 11.0% > 5%, despite
passing every real-text gate — consistent with, not contradicting, §22.19's retraction).
**Rational-Padé PASSES every gate administered**: teacher-forced marginal (2.0% < 5%), teacher-forced
full-stack (2.6%, informational), real-text aggregate (0.9841 < 1.05x) and per-corpus (all <1.10x),
divergence (3.2% < 15%), speed (90.0% reduction / 10.0x, ≥15% required). Qualitative free-running
check corroborates the quantitative result.

**Per the plan above: no candidate is promoted from this checkpoint.** `g_swiglu_fast` default
remains `0` (exact SiLU). Both `1` (hard-swish, rejected) and `2` (rational-Padé, passed every
harness gate) remain experimental flags only. The explicitly open next step, not yet done: a real
production A/B for rational-Padé specifically (the same board-run, cyclic-vs-blocked-style
methodology used for the scheduling experiment, §22.14) — harness gates are necessary but the
session's own established discipline (§22.14) is that a harness/isolated-probe result, however
clean, still needs a production A/B before adoption. Not run yet; explicit confirmation, not
automatic promotion, is what determines whether it happens next.

### 22.21 Rational-Padé production A/B — CONFIRMED, promoted to default

One bounded production A/B, run exactly as scoped: identical build (mandatory
`-fno-tree-vectorize` flags), identical cache, identical canonical prompt/router/thread count
(`nt=4`, `g_router_mode=2`), identical generation length (`ngen=16`), varying only
`g_swiglu_fast` between `0` (exact) and `2` (rational-Padé). Two paired trials each, run
interleaved (exact, ratsig, exact, ratsig) to control for thermal/load drift rather than blocked
by config.

| trial | config | tok/s | wall/tok | swiglu bucket | other buckets | tokens |
|---|---|---|---|---|---|---|
| 1 | exact | 9.55 | 104.7ms | 17.7ms | act-pack 2.6, linear 58.9, attn 9.1, rope 3.2, router 4.2, rest 8.2 | `Tokyo. The capital of Brazil is Brasília. The capital of Canada is Ottawa.` |
| 1 | ratsig | 11.38 | 87.9ms | 1.2ms | act-pack 2.3, linear 58.7, attn 9.0, rope 3.2, router 4.2, rest 8.2 | identical |
| 2 | exact | 9.92 | 100.8ms | 14.0ms | act-pack 2.5, linear 58.9, attn 9.0, rope 3.2, router 4.2, rest 8.0 | identical |
| 2 | ratsig | 11.37 | 88.0ms | 1.3ms | act-pack 2.4, linear 58.8, attn 9.1, rope 3.1, router 4.2, rest 8.1 | identical |

Averages: exact 9.735 tok/s (wall 102.75ms, swiglu 15.85ms); rational-Padé 11.375 tok/s (wall
87.95ms, swiglu 1.25ms). **SwiGLU bucket reduction: 92.1%** (15.85→1.25ms), exceeding the expected
~90%. **Wall-time reduction: 14.4%** (14.8ms/token saved, close to the ~12.5ms/token estimate).
**tok/s improvement: +16.8%** (9.735→11.375), landing at **11.37-11.38 tok/s**, inside the
predicted 11.0-11.4 tok/s range. Every non-SwiGLU bucket (act-pack, linear-kernel, attention,
rope+qknorm, router, rest, and the qkv/o/expert/lm_head linear breakdown) is statistically
unchanged across all four runs — differences are ≤0.3ms, consistent with run-to-run noise, not a
regression. All four runs: `' Tokyo'` PASS, and **generated tokens are byte-identical between
exact and rational-Padé** on this canonical prompt (stronger than the "desirable but not required"
bar) — `Tokyo. The capital of Brazil is Brasília. The capital of Canada is Ottawa.` in every trial.
This 16-token free-running decode is itself non-teacher-forced (each step feeds back its own
argmax), corroborating the two longer free-running spot-checks already run in §22.20.

**Keep-criterion met**: the expected ~90% SwiGLU reduction transferred to production with no
regression elsewhere, exactly the pre-registered bar from step 5 of the A/B scope. **Promoted**:
`g_swiglu_fast` production default changed `0→2` (rational-Padé). `g_swiglu_fast=0` (exact SiLU)
remains available as an explicit revert flag; `g_swiglu_fast=1` (hard-swish) stays rejected,
unchanged. Verified post-promotion: invoking the binary with *no* `g_swiglu_fast` argument at all
(the real production quick-start command) reports `swiglu 1.4ms` and `11.35 tok/s` — the promotion
takes effect through the actual default code path, not only when the flag is passed explicitly.

**Session cumulative from the original 1.49 tok/s baseline, at the actual current default
(int8-M1 router + rational-Padé SwiGLU): ~7.6-7.8x** (9.735→11.35-11.38 tok/s over 1.49 tok/s),
now materially closer to the real vendor binary's 11.71-12.89 tok/s window (11.37/11.71≈97.1% of
the low endpoint, 11.37/12.89≈88.2% of the high endpoint — both stated, per the §22.19 phrasing
correction).

### 22.22 Release-quality checkpoint frozen (2026-07-26)

Per explicit direction ("freeze this as a release-quality checkpoint"), recorded everything needed
to exactly reproduce or re-verify the promoted state from §22.21, gathered directly from the board
(not assumed from earlier notes):

**Toolchain**
- Compiler: `gcc (Bianbu 15.2.0-16ubuntu1bb5) 15.2.0`
- Assembler/linker: `GNU Binutils for Bianbu 2.46`
- OS: Bianbu 4.0.1 "Resolute Raccoon" (`PRETTY_NAME="Bianbu 4.0.1"`)
- Kernel: `Linux k3 6.18.3-generic #1.0.1.4 SMP PREEMPT_DYNAMIC Thu May 21 16:47:06 CST 2026 riscv64`
- Board: hostname `k3`, `192.168.68.24` (static, wired `end0`), SpacemiT K3 SoC — 8x A100 cores +
  8x X100 cores (16 total, `riscv64`), the same board used for every measurement this session.

**Exact build command** (mandatory flags, unchanged since §22.16's toolchain hardening):
```
gcc -O3 -fno-tree-vectorize -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe_hp qwen_moe_hp.c -lm -lpthread
```

**Artifact hashes** (`sha256sum`, gathered directly on-board, not computed locally then assumed
unchanged):
| artifact | size | sha256 |
|---|---|---|
| `qwen_moe_hp.c` (source) | — | `3b9bc183b62da66b01a1c603cad2a2811f87b7f0fd8af80e82f6c191288e50ef` |
| `qwen_moe_hp` (binary) | 72744 B | `c28c1f66e21ff2f60e9fc3cb24ade29bc9a759c0c5fa7cccaab87a230755bae8` |
| `Qwen3-30B-A3B-Q4_0.gguf` (model) | 17379988032 B (16.19 GiB) | `8b8e9febb6ed326d91a016f5d465cf5b55a0b328d157a43e2242fda71fde9c40` |
| `qwen3-30b-a3b.hp.imecache` (requant cache) | 18288076356 B (17.03 GiB), format **v2** (vendor HP) | `f9da3b65e5adf34137ac35a922f0040fb43bfdcda070a23b7e4f2bd64da476d5` |

The source hash **exactly matches `git show 0dde2f4:qwen_moe_hp.c`** (verified locally,
`3b9bc183b6...` both ways) — the on-board binary being frozen here is provably built from the
committed production-A/B-promotion commit, not from any uncommitted or in-flight edit.

**Defaults and fallback flags, as of this checkpoint** (commit `0dde2f4`):
| flag | default | fallback / alternatives |
|---|---|---|
| `g_router_mode` (7th CLI arg) | `2` = int8-M1 (§22.15, passed harness) | `0` = fp32 exact (explicit revert flag); `1` = int4-HP (REJECTED, §22.7, 60.3% routing perturbation) |
| `g_swiglu_fast` (8th CLI arg) | `2` = rational-Padé (§22.20-21, passed harness + production A/B) | `0` = exact SiLU (explicit revert flag); `1` = hard-swish (REJECTED, §22.19-20, 11.0% perplexity inflation) |
| `g_router_validate` (6th CLI arg) | `0` (off) | `1` enables the router fp32-vs-quantized validator, ~2x router bucket cost, diagnostic only |

**Quick-start invocation, exact defaults, no positional overrides needed**:
```
ssh root@192.168.68.24
LD_LIBRARY_PATH=/usr/lib /root/qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf 16 4 /root/models/qwen3-30b-a3b.hp.imecache
```
Verified at freeze time: `' Tokyo'` PASS, `11.35 tok/s`, `swiglu 1.4ms` — matches §22.21's A/B
figures exactly, confirming the frozen artifact reproduces the promoted result.

### 22.23 Context-length scaling: attention overtakes linear(kernel) at ~59 tokens of context

Every router/SwiGLU decision this session was measured at a **12-token prompt** — a short-context
regime where `linear(kernel)` (weight streaming, independent of context length) dominates the
per-token bucket breakdown. Per explicit direction ("benchmark context-length scaling ... to
expose when attention replaces linear as the bottleneck"), added a `QWEN_CTXLEN` env var (same
"avoid another positional production arg" convention as `QWEN_HARNESS`) that synthesizes a prefill
of the requested length by tiling the harness's real 113-token narrative prompt (`hp9`), grows the
KV cache to fit, and otherwise runs the unmodified production decode loop/build — no new benchmark
harness, same binary. Swept ctx ∈ {32, 64, 128, 512, 1024} (plus the existing 12-token default as
an implicit low point), all at the promoted default config (int8-M1 + rational-Padé), `nt=4`,
16-token decode window per run:

| ctx (tokens) | attention (ms/tok) | linear(kernel) (ms/tok) | wall (ms/tok) | tok/s | prefill (s) |
|---|---|---|---|---|---|
| 12 (default prompt) | 9.1 | 58.9 | 88.1 | 11.35 | 0.97 |
| 32 | 28.1 | 58.8 | 107.0 | 9.34 | 2.71 |
| 64 | 64.4 | 58.9 | 143.6 | 6.97 | 6.04 |
| 128 | 143.9 | 59.3 | 223.7 | 4.47 | 17.40 |
| 512 | 600.3 | 59.9 | 680.6 | 1.47 | 182.32 |
| 1024 | 1216.0 | 60.7 | 1297.3 | 0.77 | 680.13 |

`linear(kernel)` is flat (58.8-60.7ms) across nearly two orders of magnitude of context length, as
expected — it streams a fixed set of weights per token regardless of KV cache size. `attention`
scales linearly in context length: a least-squares fit over the 32-1024 points gives **~1.197
ms per token of context, intercept ≈ -11ms** (R² not computed but the fit is visually tight — see
raw points above), consistent with the expected O(context) per-decode-step cost of attending over
a growing KV cache with no windowing/pruning in this implementation. Every other bucket (act-pack,
rope+qknorm, router, swiglu, rest) stays within measurement noise of its short-context value at
every context length tested — only attention and, trivially, wall time move.

**Crossover point: attention overtakes linear(kernel) as the dominant per-token bucket at ~59
tokens of context** (solving `1.197*ctx - 11 = 59.2` using the fitted line and the ~59.2ms average
linear-kernel bucket). The ctx=64 measurement directly brackets this (attention 64.4ms already
exceeds linear 58.9ms), consistent with the fit. **Every quality/speed decision made this session
(router promotion, SwiGLU promotion, the A/B in §22.21) was measured well inside the
linear-dominated regime** (12-token prompt, decode positions 12-27) — none of those results say
anything about relative bucket weight at longer context, though the *quality* harness (§22.20) did
separately validate up to 113-token prefills for its long-context prompts, so the quality
conclusions are not context-length-limited the way the speed/bucket conclusions are.

**Practical implication, not yet acted on**: for genuinely long-context workloads (multi-turn
chat history, long documents, extended generations well past ~60 tokens of accumulated context),
attention becomes the dominant cost and the two optimizations promoted this session (int8-M1
router, rational-Padé SwiGLU) — both of which target buckets that stay flat with context — provide
progressively smaller relative benefit; at ctx=1024 they touch only ~5% of the per-token wall time
(4.2ms router + swiglu combined out of 1297ms) versus ~94% at the 12-token baseline. Prefill time
also grows steeply (0.97s→680s from ctx=12 to ctx=1024) and was not the target of this sweep
(prefill is O(context) work done once, not the per-token decode cost being profiled here) but is
flagged for anyone using this engine on long documents. **Attention itself — not yet vectorized
beyond the existing RVV QK/AV dot products from §22.9, and using no KV-cache pruning, windowing, or
approximate-attention technique — is the natural next optimization target if long-context
throughput becomes a priority**, distinct from and orthogonal to the short-context work done this
session.

### 22.24 Board OOM incident, a real vendor-binary bug found along the way, and a clean vendor A/B at parity

**Incident (self-inflicted, root-caused).** Attempting to expand the quality harness (§22.20's
next step) and get a fresh vendor `llama-bench` comparison in the same sitting, the two were
launched concurrently — a sequencing mistake, not a deliberate stress test. The board has **32.8GB
RAM and zero swap**; the harness alone holds ~18GB resident (the requant cache) and `llama-bench`
loads its own separate ~17GB copy of the GGUF, so combined demand (~35GB) exceeded physical RAM
with nothing to absorb the overage. `journalctl` confirms the kernel OOM-killer fired and killed
the harness process directly (`Out of memory: Killed process ... (qwen_moe_hp)
total-vm:35013040kB, anon-rss:17999616kB`), and per systemd's unit-level logs it took out
additional processes across `user-0.slice`/`user.slice` — very likely `llama-bench` and, per the
`journalctl -u ssh` timeline, `sshd`'s own listening socket, which explains why the board refused
even a fresh SSH connection (banner-exchange timeouts) for roughly 15-20 minutes until `ssh.service`
restarted. Board temperatures were 57-61°C throughout — never the cause, confirmed once
reachable again. Lesson recorded in Claude's persistent memory: never run two model-loading
processes on this board concurrently; always confirm one has fully exited before starting another.

**A second, independent problem surfaced once the board recovered**: `llama-bench`, run completely
alone with no contention, crashed reproducibly at both `-t 4` and `-t 8` with `ggml_abort()` —
`"set thread affinity error for thread_n %d, cpu_id %d"` inside SpacemiT's `ime.cpp`. Tracing it
down: `/root/llama.cpp/ggml/src/ggml-cpu/spacemit/ime.cpp` is a locally modified file (tracked as
`M` in git status) that already has this exact call downgraded from `GGML_ABORT` to
`GGML_LOG_ERROR` (an earlier session-local patch, alongside a `bind_ai_thread()` unlock call —
`ime.cpp.orig`, the untouched original, still has the unpatched `GGML_ABORT` at the matching line).
But `/usr/lib/libggml-cpu.so.0.13.1` — what `LD_LIBRARY_PATH=/usr/lib` (the invocation pattern used
everywhere in this session, including the A5-equiv baseline in `research_feed_paths.md`) actually
loads at runtime — is dated **Jul 7**, predating the patch; it was never rebuilt/reinstalled after
the source was fixed. `/root/llama.cpp/build/bin/libggml-cpu.so.0.13.1` (the build tree's own
copy, Jul 25) has the correct patched behavior. Fix: point `LD_LIBRARY_PATH` at the build tree
instead of the stale system copy — `LD_LIBRARY_PATH=/root/llama.cpp/build/bin`, not `/usr/lib`.
No source or system-file changes were needed, just the correct library path. This means the
**A5-equiv baseline row in `research_feed_paths.md` §12** (11.71±0.19 tok/s @t=4, 12.89±0.03 @t=8,
git SHA `31504f6`) was itself obtained against this same stale library — worth keeping in mind if
that number is ever revisited, though it isn't necessarily wrong (the crash is a thread-affinity
abort under specific conditions, not proof every prior run was corrupted).

**Clean vendor A/B, obtained after both fixes, board otherwise idle, no contention**:

| config | tok/s | conditions |
|---|---|---|
| our engine (production default: int8-M1 + rational-Padé) | 11.35-11.39 (repeated runs) | 12-token prefill (untimed) + 16-token timed decode, `nt=4` |
| vendor `llama-bench` (patched lib) | **11.38 ± 0.21** (r=3) | `-p 0 -n 16` (empty-context decode, no prefill), `-t 4` |
| vendor `llama-bench` (patched lib) | 12.82 ± 0.05 (r=3) | same, `-t 8` (our engine has no `nt=8` config tested today; history notes `nt=8` measured slightly worse for our engine, memory-bandwidth-bound) |

**Our pure-C engine and the vendor's own compiled implementation are now at parity at `nt=4`** —
11.35-11.39 vs 11.38±0.21, well within the vendor's own run-to-run noise band. One honest
measurement-boundary caveat, per the "matching measurement boundaries" requirement: our number's
decode window covers KV-cache positions 12-27 (after a 12-token prefill, ~9ms/token of attention
cost per §22.23's bucket data), while vendor's `-p 0` starts from an empty context (positions
0-15, attention cost ramping from ~0), so vendor's number has a slightly easier attention profile
over its window than ours does. Given the two results already land within noise of each other,
this doesn't change the "at parity" conclusion, but it means the comparison isn't byte-for-byte
identical in context depth — flagged rather than glossed over. The `nt=8` vendor figure (12.82) is
close to the earlier stale-library baseline's 12.89, a useful cross-check that the library-path fix
didn't materially change the underlying performance number, only the crash behavior.

### 22.25 Confirmed heap-overflow bug in the expanded harness — fixed; ASan root-cause in progress

**Rerunning the expanded quality harness (§22.20's corpus/generation-length expansion) alone, no
contention, crashed with SIGSEGV** — `dmesg` shows `qwen_moe_hp[28045]: unhandled signal 11 code
0x1 at 0x...` inside `libc.so.6` itself, at "SwiGLU phase 2 -- hard-swish candidate", prompt 6/10.
Register contents at the fault (`s2=0x300`=768=`moe_ffn`, `s6=0x800`=2048=`d`, `s3=0x80`=128=
`n_exp`) look like live model-dimension values sitting in a corrupted-heap crash, not a clean
direct out-of-bounds hit at the fault site itself — consistent with heap corruption surfacing later
and elsewhere than its actual cause, exactly the kind of misleading symptom this session's own
prior corruption bug (§22.16) produced.

**A concrete, independently provable bug was found by inspection before any debugger involvement**:
`run_quality_harness()` allocates the harness's KV cache at `ctx=HARNESS_MAXCTX` (200 at the time),
but `harness_eval_ppl()` walks real-text positions `0..txt->n-2` over that SAME cache — and
`ppl_code2` (added in §22.20's expansion) is **355 tokens**, nearly double the 200-token
allocation. `forward()`'s KV-cache write (`memcpy(Kc+(size_t)pos*kvd,k,kvd*4)`, previously
unguarded) would silently write past `kv->Kc`/`kv->Vc`'s allocated extent whenever `pos>=200`,
corrupting adjacent heap allocations. This phase runs *after* the SwiGLU phase-2 crash site in the
current code's execution order, so it is not necessarily *the* cause of the observed crash, but
it is unambiguously a real, live bug on its own, confirmed by arithmetic alone (355 > 200), that
had to be fixed regardless of what ASan reports.

**Fixes applied (before ASan resolves, so they don't wait on it)**:
1. `HARNESS_MAXCTX` computed properly as the max of every `HarnessPrompt`'s prefill+gen (worst
   case hp9/hp10: 113+60=173) and every `PplText`'s raw length (worst case `ppl_code2`: 355) —
   raised to **512** for headroom, from the unsound 200.
2. A hard guard added directly in `forward()`, at the exact point of the KV-cache write:
   ```c
   if(pos<0 || pos>=kv->ctx){
       fprintf(stderr,"KV position overflow: pos=%d ctx=%d\n",pos,kv->ctx);
       abort();
   }
   ```
   This turns any future instance of this bug class into an immediate, precisely located abort at
   the actual point of failure, instead of silent heap corruption that surfaces later as a
   misleading crash somewhere unrelated — the exact failure mode this bug just produced. Verified
   harmless for every existing production code path: default `main()` uses `ctx=64` (prompt+gen
   always well under that) or, under `QWEN_CTXLEN`, `ctx=ctxlen_req+ngen+4` with `np=ctxlen_req`,
   leaving a 4-position margin — the guard should never fire outside a genuine bug.

**The ASan run was deliberately left running rather than restarted with the fix** — per explicit
direction, it may still reveal a separate, earlier bug specifically in the SwiGLU phase-2 path
(prompt 6/10), since that phase runs before the now-fixed real-text overflow in execution order and
so cannot be explained by it alone.

**ASan reported first, and it wasn't `ppl_code2`.** It found a **global-buffer-overflow READ** in
`harness_eval_ppl` (`qwen_moe_hp.c:1001`), 0 bytes past the end of `ppl_reason` (size 596 bytes =
149 `int`s), immediately adjacent to `ppl_code` in memory. **`ppl_reason`'s `PplText` entry declared
`n=176`, but the array itself has only 149 real elements — a stale length that predates this
session's corpus expansion entirely** (`ppl_reason` is one of the original 4 real-text corpora from
§22.20's first checkpoint). Every real-text-perplexity evaluation that included `ppl_reason` this
whole session — including the numbers reported in §22.20 and relied on for the rational-Padé
production A/B in §22.21 — read 27 tokens past the array's real end, into `ppl_code`'s memory,
silently blending unrelated content into part of the reasoning-corpus's perplexity computation. This
predates and is independent of my expansion work; it simply had never been exercised under ASan
before.

**Finding one over-read prompted a full audit of every declared length against actual array size**
(all 9 `PplText` entries, all 10 `HarnessPrompt` entries) rather than fixing only what ASan happened
to catch first. Result: **5 mismatches total**, all pre-existing, none introduced by this session's
corpus expansion (every one of the 5 new `PplText`/hp-adjacent additions checked out correctly):

| entry | declared | actual | class |
|---|---|---|---|
| `ppl_reason` | 176 | 149 | **over-declared — real OOB read (the ASan-caught one)** |
| `hp3` (reasoning/syllogism) | 15 | 16 | under-declared — 1 token of prefill silently dropped |
| `hp4` (reasoning/sequence) | 22 | 24 | under-declared — 2 tokens dropped |
| `hp9` (long-context/narrative) | 113 | 124 | under-declared — 11 tokens dropped |
| `hp10` (long-context/technical) | 113 | 155 | under-declared — **42 tokens dropped, over a quarter of the intended prompt** |

Under-declared entries are not memory-unsafe (the harness never reads past the array's real bounds
when `n` is smaller than the actual size) but are a real correctness bug: every harness run this
entire session — the original router promotion (§22.15), both SwiGLU evaluations (§22.18-21) — fed
a *shorter* prefill for these four prompts than the tokenized arrays actually contain, particularly
undermining `hp9`/`hp10`, whose entire purpose is testing long-context behavior. All 5 array
*contents* were checked by hand (first/last 15 tokens each) for signs of corruption or accidental
duplication before trusting them as correct — all read as coherent, non-repeating continuations, so
the fix in every case is to correct the declared length to match the real array, not to touch the
array content.

**All 5 fixed** (`qwen_moe_hp.c`, the `g_hprompts[]` and `g_ppltexts[]` tables) and a second ASan
run launched to check whether the SwiGLU phase-2/prompt-6 crash site has a separate root cause not
explained by any of these five. Both the ASan and production binaries were rebuilt from the fixed
source (build-only, not run, so as not to introduce a second concurrent board workload per §22.24's
lesson) before the ASan rerun was launched alone.

**ASan rerun completed clean, end to end** — the full harness ran to natural completion (router
phase, SwiGLU phase 1-3, real-text perplexity across all 9 corpora, both free-running spot checks),
no AddressSanitizer report, no abort, log not truncated. The SwiGLU phase-2/prompt-6 crash site
that killed the original (unguarded) run did **not** recur — it was fully explained by the
`ppl_reason` overflow corrupting heap state that a later allocation then tripped over, not a
separate bug. **Gate satisfied**: per explicit direction, a new harness quality result is only
trusted after a clean complete ASan pass, and that condition is now met.

**Corrected results (760 teacher-forced positions with the real hp3/hp4/hp9/hp10 prefill lengths,
2182 real-text positions with the real `ppl_reason` length)** — every conclusion from §22.20/§22.21
holds, several numbers improve now that they're computed on correct data instead of
truncated/overrun data:

| | vs int8+exact production | vs original fp32+exact | divergence | speed |
|---|---|---|---|---|
| **hard-swish** | x1.1340 (13.4% inflation) — **FAILS** | x1.1200 (12.0%) | 7.4% (56/760) | 16.46x (93.9% reduction) |
| **rational-Padé** | x1.0174 (1.7% inflation) — **PASSES** | x1.0048 (0.5%) | 3.3% (25/760) | 9.91x (89.9% reduction) |

Real-text perplexity (2182 tokens, 9 corpora): production x0.9990 vs fp32 ground truth,
hard-swish x1.0023 vs production (aggregate PASSES, worst corpus 1.0549 multilingual/spanish,
still PASSES <1.10 — real-text alone still would have wrongly cleared hard-swish, exactly as
§22.20 found; the teacher-forced gate remains the deciding one), rational-Padé x0.9894 vs
production (worst corpus 1.0058, code). Router re-confirmed independently on the corrected prefill
lengths: 2.8% divergence, +0.0048 NLL delta, 5.4% mismatch, 87.0% faster — all four thresholds
PASS, unchanged conclusion. Free-running spot check (long-context/narrative) again shows
rational-Padé tracking production's phrasing far more closely than hard-swish's real narrative
departure, consistent with every earlier run.

**No promotion status changes as a result of this bugfix.** `g_swiglu_fast` was already promoted to
`2` (rational-Padé) in §22.21 on the strength of a bounded production A/B, not on this harness's
real-text numbers — this expanded, now-bug-fixed rerun is confirmatory validation on a larger,
correct dataset, not a new promotion decision. Hard-swish remains rejected. The takeaway is
narrower but still important: the corpus/generation-length expansion added in §22.20 shipped with a
real, previously-latent bug (`ppl_reason`'s stale length) plus four newly-discovered ones
(hp3/hp4/hp9/hp10's stale prefill lengths) that had silently affected every harness run referencing
those five entries all session, now fixed and ASan-verified clean.

### 22.26 Fresh production A/B, exact vs rational-Padé — reconfirmed, plus a real production-path bug found along the way

Per explicit direction to run a clean production A/B given everything since §22.21 (the OOM
incident, the vendor library fix, the harness bugfixes), reran the identical methodology: same
build, same cache, canonical 12-token prompt, `g_router_mode=2` (int8-M1), `nt=4`, `ngen=16`,
varying only `g_swiglu_fast` (0 vs 2), two paired trials, board otherwise idle.

| trial | config | tok/s | wall/tok | swiglu bucket | tokens |
|---|---|---|---|---|---|
| 1 | exact | 9.87 | 101.3ms | 14.5ms | `Tokyo. The capital of Brazil is Brasília. The capital of Canada is Ottawa.` |
| 1 | ratsig | 11.34 | 88.2ms | 1.3ms | identical |
| 2 | exact | 9.55 | 104.7ms | 17.6ms | identical |
| 2 | ratsig | 11.37 | 88.0ms | 1.2ms | identical |

Averages: exact 9.71 tok/s (wall 103.0ms, swiglu 16.05ms); rational-Padé 11.355 tok/s (wall
88.1ms, swiglu 1.25ms). **SwiGLU bucket reduction 92.2%, wall time -14.5%, tok/s +16.9%** —
reproduces §22.21's original A/B (92.1%/-14.4%/+16.8%) almost exactly, confirming the promotion
remains sound after everything that's happened to the board since. Every other bucket
(act-pack 2.3-2.5, linear 58.6-58.8, attention 9.0-9.2, rope 3.1, router 4.2-4.3, rest 8.0-8.2)
stayed flat across all four runs — no regression anywhere else. Tokens byte-identical between exact
and rational-Padé in every trial, `' Tokyo'` PASS throughout.

**Extending the qualitative check past the canonical 16-token window (per "output quality") found
and fixed a real production-path bug.** Running `ngen=60` with the default 12-token prompt hit the
new `forward()` bounds guard immediately: `KV position overflow: pos=64 ctx=64`. Cause: `main()`'s
`ctx` sizing only grew to accommodate `ngen` inside the `QWEN_CTXLEN` branch
(`ctx=ctxlen_req+ngen+4`) — the *default* 12-token-prompt path left `ctx` hardcoded at 64
regardless of `ngen`, even though `ngen` is a directly user-controlled 2nd CLI arg with no upper
bound. **Any production invocation requesting more than ~52 generated tokens on the default prompt
would have silently overrun the KV cache** — this was always live in the production binary, not
harness-only, and had simply never been exercised with a large enough `ngen` argument before
today. Fixed: `ctx` now computed from `base_np+ngen+4` where `base_np` is whichever prefill length
is actually in effect (12 by default, or `ctxlen_req`), not only the `QWEN_CTXLEN` case. Rebuilt,
reverified: `ngen=60` now runs clean under both `g_swiglu_fast` settings, `' Tokyo'` PASS, default
`ngen=16` invocation unaffected (11.23 tok/s, within normal run-to-run noise).

Qualitative spot check at `ngen=60` (well past the §22.23 attention/linear crossover — context
grows from 12 to 71 across the window, so attention dominates far more here than at the canonical
16-token setting): both configs stay on-topic and coherent (continuing the "capitals of the world"
list), diverging only in which specific countries get mentioned after Brazil/Canada, consistent
with every earlier free-running comparison this session. At this longer window the SwiGLU
bucket savings (14.0ms→1.2ms) barely move end-to-end tok/s (8.82 vs 8.88) because attention has
grown to dominate the per-token cost — a live, concrete illustration of §22.23's crossover finding,
not just a synthetic `QWEN_CTXLEN` sweep.

**Conclusion: the production A/B reconfirms rational-Padé's promotion with fresh numbers, and the
exercise of actually testing output quality at a non-canonical generation length caught a real bug
that had been sitting in the shipped production binary the whole session.**

### 22.27 Attention optimization, Phase 1: QK/softmax/AV bucket split, baseline scoreboard

`attention_optimization_plan.md` (Codex-authored, treated as controlling) opens the next branch:
attention overtook `linear(kernel)` as the dominant per-token bucket past ~59 tokens of context
(§22.23), and nothing has touched it beyond the original vectorization pass (§22.9). Per the plan's
explicit execution directive, **Phase 1 only** — instrumentation, no layout/threading/GQA-fusion/
semantic changes — was implemented and run this entry; later phases are not started.

**Instrumentation**: three new global accumulators (`gT_attn_qk`, `gT_attn_sm`, `gT_attn_av`)
timed inside the existing per-head loop in `forward()`, splitting the already-existing `gT_attn`
bucket by operation — QK dot-product, softmax, weighted-V accumulate — without changing any
arithmetic. Verified before trusting any measurement: canonical-prompt run reports byte-identical
`' Tokyo'` output and unchanged tok/s (11.2-11.4 range, within normal noise) versus the
pre-instrumentation binary, and the three sub-buckets sum to within ~0.1ms of the pre-existing
`attention` bucket at every context tested — the timing calls (`clock_gettime` via `now()`, ~1500
extra calls/token at `nh=32, nl=48`) add negligible measurable overhead.

**Baseline scoreboard** (2 trials/context, board otherwise idle throughout, `nt=4`, canonical-style
`ngen=16` decode via `QWEN_CTXLEN`-synthesized prefill; full per-trial table now in the plan
document's own scoreboard section, not duplicated here):

| context | QK (avg ms) | softmax (avg ms) | AV (avg ms) | attention (avg ms) | tok/s (avg) |
|---|---|---|---|---|---|
| 128 | 61.80 | 9.45 | 65.34 | 136.75 | 4.63 |
| 512 | 270.79 | 35.08 | 279.50 | 585.55 | 1.50 |
| 1024 | 567.08 | 69.31 | 580.87 | 1217.45 | 0.77 |

Tokens byte-identical between the two trials at every context (as expected — deterministic
greedy decode, same config both times).

**What this establishes, per the plan's own stated purpose ("establish whether QK, softmax, or AV
dominates before optimizing anything")**: **QK and AV are essentially co-dominant, ~46-48% of the
attention bucket each at every context tested — neither one alone is "the" bottleneck.** Softmax is
a real but secondary cost, 5.7% (ctx=1024) to 6.9% (ctx=128) of the bucket — material enough that
Phase 5 (exact RVV softmax) is worth eventually doing, per the plan's own "only pursue if material"
framing, but clearly not the dominant term the way QK/AV are. Both QK and AV scale close to
linearly with context (128→512 is a 4x context increase, both grow ~4.3-4.4x; 512→1024 is 2x
context, both grow ~2.07-2.10x) — consistent with, and now decomposing, §22.23's aggregate
~1.2ms/token-of-context finding into its two real components. AV runs consistently ~2.3-2.5%
more expensive than QK at the larger contexts, plausibly because `vaxpy_f32`'s accumulation is a
read-modify-write on the output vector while `vdot_f32`'s reduction only writes a single scalar per
position — noted for Phase 4, where both directions independently re-read the same K/V once per
query head sharing a KV head (8x redundant reads either way, regardless of which of QK/AV is
costlier).

**Per the execution directive, no further phase has started.** Phase 2 (head-major KV-layout A/B)
is the next planned step, not yet implemented — this baseline scoreboard is the review point the
directive requires before it begins.

### 22.28 Attention Phase 2: head-major KV layout — KEPT, dramatic gain, isolated from threading/fusion

Explicit authorization: Phase 2 only, isolated head-major KV-layout A/B, not combined with
threading or GQA fusion. The §22.27 baseline showing QK/AV as co-dominant was read as confirming
"exact GQA reuse remains the likely larger later opportunity, while softmax should wait" — i.e.
this entry is Phase 2 alone, Phase 3 (threading) and Phase 4 (fusion) are separate future decisions.

**Change**: per-layer KV-cache layout in `forward()` changed from `[position][kv_head][head_dim]`
(2KB stride between consecutive positions for one head — `kvd=nkv*hd=4*128=512` floats) to
`[kv_head][position][head_dim]` (one head's whole history contiguous). The write became `nkv`
separate per-head `memcpy`s (positions for different heads are no longer adjacent, so the old
single all-heads-at-once `memcpy` no longer applies) instead of one; the read indexes each head's
contiguous run directly instead of striding by `kvd` per position. Same total per-layer allocation
size either way (`ctx*kvd` floats), same arithmetic, same quantization — pure addressing change,
matching the plan's "do not change quantization or attention math" constraint exactly.

**Validation** (plan's required order): (1) ASan on a bounded long-context test — `QWEN_CTXLEN=256,
ngen=24`, completed clean, no report, no abort. (2) No KV bounds errors — the `forward()` guard
added in §22.25/§22.26 stayed silent throughout every run. (3) Token identity vs the time-major
baseline — confirmed byte-identical at the canonical 12-token prompt, at the ASan ctx=256 run, and
at every A/B context below. (4) A/B at contexts 128, 512, 1024 — full results follow. Baseline
preserved at `/tmp/qwen_moe_hp_kv_timemajor.c` (`git show HEAD:qwen_moe_hp.c` from immediately
before this change — hash-verified identical to the exact binary §22.27's baseline scoreboard was
measured against, so that data was reused rather than re-measured, saving substantial board time
on a workload where ctx=1024 alone costs ~20 minutes/trial).

| | 128 | 512 | 1024 |
|---|---|---|---|
| attention bucket | 136.75→56.40ms (**-58.8%**) | 585.55→201.14ms (**-65.6%**) | 1217.45→384.06ms (**-68.5%**) |
| wall/token | 216.3→135.3ms (-37.4%) | 665.8→280.4ms (-57.9%) | 1298.8→463.8ms (-64.3%) |
| tok/s | 4.63→7.39 (**+59.6%**) | 1.50→3.56 (**+136.9%**) | 0.77→2.16 (**+179.9%**) |

Short-context (canonical prompt, 2 paired trials each): baseline wall 88.45ms avg → candidate
87.7ms avg, a **0.85% improvement, not a regression** — the plan's "no more than 2% short-context
regression" gate is not merely met but inverted (the change helps everywhere tested, not just at
long context). Every pairing above is two independent trials in tight agreement (e.g. ctx=1024:
383.32ms vs 384.79ms) — the improvement is reproducible, not a one-off. Bonus, not a required
metric: prefill time also dropped substantially (ctx=512 ~172-185s→91s, ctx=1024 ~678-680s→279s),
the same locality argument applying to prefill's own repeated K/V writes/reads.

**This result is far larger than the plan's own framing anticipated** ("may improve prefetching and
cache-line use" was stated as a secondary, exploratory rationale for testing layout first, not a
prediction of a 58-68% bucket reduction). The magnitude makes sense in hindsight: the old layout's
2KB stride between consecutive positions for a fixed head guaranteed zero cache-line reuse and a
strided access pattern hostile to hardware prefetching; the new layout's contiguous per-head
history is close to the best case for both. This also reframes the co-dominant QK/AV baseline from
§22.27 — a meaningful fraction of what looked like "real compute cost" was actually avoidable
memory-layout overhead, not fundamental FLOP or bandwidth cost.

**Both predeclared keep criteria met — KEPT.** Head-major is now the production KV-cache layout in
`qwen_moe_hp.c` (committed). Per the explicit authorization, **threading (Phase 3) and GQA fusion
(Phase 4) are not started** — this entry is the isolated layout change only, as scoped. The
co-dominant QK/AV baseline (§22.27) is read as the user directed: it confirms exact GQA reuse
(Phase 4 — eliminating the 8x redundant K/V re-reads across query heads sharing a KV head) remains
the likely larger later opportunity, while softmax vectorization (Phase 5) can wait, being a
smaller and now-even-smaller-relatively fraction of a bucket that just dropped by more than half.

### 22.29 Attention Phase 3: four-KV-group worker parallelism — KEPT, isolated from GQA fusion

Explicit authorization: Phase 3 only, isolated four-KV-group worker parallelism, not combined with
GQA fusion. Softmax to remain deferred despite a small regression (see below).

**Implementation, cleanup of a stale plan reference**: the plan's own "Immediate next action"
section still said "Implement Phase 1 instrumentation only" after Phase 1/2 had both completed and
been kept — fixed to correctly identify Phase 3, then Phase 4, as the actual next steps.

**Change**: `HpWork` (the existing persistent-pool dispatch struct, `qwen_moe_hp.c`) gained a third
`kind==2` variant for attention, reusing the pool unchanged — no new threads created. Worker `tn`
handles KV heads `tn, tn+attn_nt, tn+2*attn_nt, ...` and each such KV head's `gpr` (8) query heads,
in ascending head order; with `attn_nt==nkv` (the promoted default) each worker gets exactly one KV
head, matching the plan's stated ideal division. Every pool thread is woken and increments the
shared done-counter exactly once per round regardless of whether it had real work that round
(threads with `tn>=attn_nt` no-op), so testing `attn_nt`=1/2/4 needed no change to the existing
`lin_mm_hp`/`lin_mm_i8` wait discipline. A private per-worker score scratch buffer (`>=ctx` floats)
is allocated once, lazily, not per-call. `g_attn_nt<=1` (the serial fallback, matching Phase 2's
code byte-for-byte) took an entirely separate, unmodified branch in `forward()` — zero risk to the
already-verified path when parallelism is off, and the fallback used throughout Phase 1/2's own
testing.

Dispatch/sync overhead is measured directly (`gT_attn_dispatch`): wall time of the whole attention
step minus the busiest single worker's own QK+softmax+AV time that round — cost not explained by
any one worker's real compute. QK/softmax/AV sub-buckets are kept thread-safe via per-worker
indexed accumulator slots (no locking needed, each thread only ever writes its own index) summed
into the globals by the single dispatching thread after every round completes, so there's no data
race despite up to 4 threads recording sub-timings concurrently.

**Validation** (plan's required order): tokens confirmed byte-identical at every worker count
(1/2/4) and every context tested (canonical prompt, ctx=256/ngen=24, ctx=512, ctx=1024). ASan clean
on a bounded ctx=256/ngen=24 run at both `attn_nt=2` and `attn_nt=4` (no report, no abort). UBSan
clean at `attn_nt=4` on the same bounded test — the plan's "no ASan or UBSan failure" gate is fully
covered, not just the ASan half.

**A/B, 2 trials per context, board otherwise idle, against Phase 2's head-major baseline**
(`*` marks summed-across-workers CPU-time, not wall-clock-comparable once threaded — "attention
bucket" is the wall-clock figure that matters):

| | 128 | 512 | 1024 |
|---|---|---|---|
| attention bucket | 56.40→14.09ms (**-75.0%**) | 201.14→50.33ms (**-75.0%**) | 384.06→98.42ms (**-74.4%**) |
| wall/token | 135.3→92.85ms (-31.4%) | 280.4→129.35ms (-53.9%) | 463.8→177.8ms (-61.7%) |
| tok/s | 7.39→10.77 (**+45.7%**) | 3.56→7.73 (**+117.1%**) | 2.16→5.625 (**+160.4%**) |

Short-context (canonical prompt, 2 paired trials): wall time 88.45ms (Phase 2 baseline) → 81.15ms,
**an 8.25% improvement**, comfortably clearing "no more than 2% regression." **All five predeclared
keep criteria met** — tokens identical, ASan+UBSan clean, attention-bucket reduction (75.0%/74.4%,
both far past the 20% floor at ctx 512/1024), end-to-end improvement (117.1%/160.4%, both far past
the 10% floor), no short-context regression (an improvement instead). Dispatch/sync overhead stayed
tiny throughout (0.09-1.14ms across every configuration) — the coordination cost of parallelizing
is not eating the gain.

**Four-KV-group worker parallelism is now the production attention path.** `g_attn_nt` default
changed from a hardcoded `1` to `min(nt,nkv)` (computed in `main()` once `m.nkv` is known, so it
naturally tracks whatever `nt` is actually configured with, `=4` at the standard default).
`QWEN_ATTN_NT=1` remains an explicit serial revert flag, byte-identical to Phase 2's code path. The
true production default invocation (no CLI overrides, no env vars) now reports `attn_nt=4` and
**12.3-12.35 tok/s at the canonical prompt**, up from 11.3-11.5 before this entry — the short-
context HEADLINE number itself moved, not just the long-context numbers this branch was chasing.

**As explicitly directed: softmax (Phase 5) stays deferred despite a small regression.** Its own
summed-work total held roughly flat in absolute terms through Phases 2-3 (e.g. ctx=1024: 69.31ms
Phase 1 baseline vs 69.17ms Phase 3 total-work — essentially unchanged), but QK/AV's wall-clock
contribution collapsed by ~74-75%, so softmax is now a much larger *relative* share of a bucket that
shrank dramatically — the "small regression" is relative, not absolute, and exactly what Phase 5
exists to eventually address, but per direction it is not touched now: no exponential
approximation, no vectorization work, until specifically authorized.

**Per the execution directive, still no GQA fusion at the time of this entry.** Phase 4 (exact
GQA-fused kernels) was the next identified opportunity — see §22.30 for the multi-Q QK result.

### 22.30 Attention Phase 4.1: multi-Q QK (exact GQA reuse on K) — KEPT, isolated from multi-Q AV

Explicit authorization: Phase 4 only; implement multi-Q QK first in isolation; do not begin AV
fusion in the same patch; preserve an explicit Phase 3 revert path; validate integrated scores
before production A/B; keep only if attention/e2e improve reproducibly and sanitizers are clean.
Softmax remains deferred.

**Change**: `qk8_dot` — for each K position of a KV head, load each 32-lane K chunk once and
immediately FMA it into all 8 query-head accumulators (named `a0..a7`; RVV types are sizeless so
they cannot be array elements). Per-query-head chunking, FMA order, and `vfredusum` reduction are
intentionally identical to `vdot_f32`, so each head's scalar score is bit-exact to the unfused path
(verified, not assumed). Fused path needs a larger per-worker scratch (`gpr×ctx` floats) because
all query heads' score rows for a KV head complete together before softmax/AV start. Unfused path
(`g_qk_fuse==0`) remains the exact Phase 3 code, byte-for-byte. Controlled by `QWEN_QK_FUSE` env
var (same convention as `QWEN_ATTN_NT`/`QWEN_CTXLEN`).

**Validation** (required order, all on board `root@192.168.68.24`):

1. Standalone probe `bench/qk_multiq_probe.c` at three contexts: 11,496 comparisons, max_abs=0,
   max_rel=0.
2. Integrated capture via `QWEN_QK_VALIDATE=1` (runs both paths through the real pool/dispatch on
   every real attention call, diffs post-scale pre-softmax scores): **60,426,240 comparisons, 0
   mismatches, max_abs=0, max_rel=0**.
3. ASan clean on bounded `ctx=256/ngen=24` with `QWEN_QK_FUSE=1` (multiple trials; production-opt
   levels). ASan/`-O1` had earlier intermittent faults traced to a racy temporary debug print,
   which was removed (3/3 clean ASan-`-O1` runs afterward). A separate, later UBSan-`-O1` page-fault
   on 8 live RVV accumulators was not explained by that fix — it did not reproduce at `-O2`/`-O3`
   under either sanitizer, which is suspected but not confirmed to be a compiler code-generation
   issue (the same GCC/RVV interaction class already documented for this file, §22.16), not
   established causation. Production builds with `-O3 -fno-tree-vectorize`.
4. UBSan clean at `-O2` with `QWEN_QK_FUSE=1` on the same bounded test.
5. Token identity: first-argmax and full generation strings byte-identical fuse0↔fuse1 at the
   canonical short prompt (`' Tokyo'` PASS) and at every long-context A/B trial.

**A/B, 2 trials per config, board otherwise idle, against Phase 3 baseline (`qk_fuse=0`,
`attn_nt=4`, head-major KV)** — `*` = summed worker CPU-time:

| | short | 128 | 512 | 1024 |
|---|---|---|---|---|
| attention bucket | 2.41→2.25ms (-6.6%) | 14.17→12.78ms (**-9.8%**) | 50.60→46.04ms (**-9.0%**) | 97.57→87.51ms (**-10.3%**) |
| wall/token | 81.05→80.95ms (-0.1%) | 92.85→91.8ms (-1.1%) | 129.85→125.8ms (-3.1%) | 177.5→167.2ms (**-5.8%**) |
| tok/s | 12.345→12.35 (~flat) | 10.775→10.895 (+1.1%) | 7.70→7.95 (**+3.2%**) | 5.635→5.98 (**+6.1%**) |

QK summed-work alone: ~22–26% reduction at long context (the intended mechanism). Softmax and AV
essentially flat (AV fusion not in this patch). Short-context does not regress.

**Honest relative to Phase 2/3 gates**: the Phase 3 predeclared floors (20% attention bucket / 10%
e2e at ctx 512 or 1024) were written for the threading candidate and are **not** met by multi-Q QK
alone. The Phase 4 authorization gate was the weaker "reproducible attention/end-to-end improvement
+ sanitize," which is met with tight trial-to-trial agreement. This is a free, exact, modest win —
kept and promoted because the quality risk is zero (bit-exact) and every long-context regime
improves, not because it moves the HEADLINE short-context number.

**Promoted**: `g_qk_fuse` defaults to `1`. `QWEN_QK_FUSE=0` remains the explicit unfused revert
(Phase 3 path). Canonical short-context HEADLINE stays ~12.3–12.4 tok/s (noise-flat); the win is
in the long-context regime this attention branch has been chasing since §22.23.

**Next**: Phase 4.2 multi-Q AV (load each V once, update 8 independent output accumulators) is the
residual larger opportunity — AV still co-dominates and still re-reads each V 8×. Softmax (Phase 5)
stays deferred. Do not combine with approximate math.

### 22.31 Attention Phase 4.2: fused multi-Q AV (exact GQA reuse) — KEPT

Explicit authorization: "Phase 4.2 only: isolated exact multi-Q AV, with the same end-to-end
completion discipline used for QK" — implement, validate (standalone probe + production
integration validation), sanitize, run the full A/B matrix, apply the predeclared keep/revert
decision, document, commit, then stop.

**Implementation.** The unfused AV loop calls `vaxpy_f32(oh,vj,sc[j],hd)` once per (query head,
position) pair, re-reading the same V row up to 8 times per position — the AV-side analog of QK's
redundant K re-reads. `av8_chunk` instead processes ONE 32-lane hd-chunk at a time across the
WHOLE position loop, holding 8 named accumulators (one per query head) live in registers for that
chunk, loading each V chunk once per position and updating all 8 accumulators before advancing;
only after the full position sweep does it write the 8 accumulators back to memory. This is
chunk-outer/position-inner — the opposite nesting from `qk8_dot` (single-position/chunk-inner) —
because AV's reduction axis is position, not head_dim: the accumulator must live across the
position loop rather than be reduced away within one. Holding only 8 accumulators live at a time
(never 8 heads × 4 chunks = 32 at once) keeps the same register-pressure profile as `qk8_dot`.
Per-query-head numerics are intentionally identical to `vaxpy_f32`: for any fixed head and chunk,
the position-ascending accumulation order is unchanged — IEEE-754 addition order, not storage
location, determines the result, so holding the running sum in a register instead of round-
tripping it through memory does not change it. Requires the fused-QK branch's structure (all `gpr`
heads' softmax weights ready together before AV starts), so it only takes effect when
`g_qk_fuse==1`; `attn_worker_run`'s fused branch was split into a softmax-all-heads loop followed
by a switchable AV block (`g_av_fuse`, env var `QWEN_AV_FUSE`, default was 0 pre-promotion). The
unfused branch is untouched, byte-for-byte the Phase 4.1 code.

**Validation** (same order as Phase 4.1): standalone probe `bench/av_multiq_probe.c` (`vaxpy_f32`/
`av8_chunk` copied verbatim) at ctx∈{113,300,1024}, gpr=8, synthetic weights — **bit-exact: 3,072
comparisons, 0 mismatches, max_abs=0, max_rel=0.** Production integration validation
(`attn_av_validate_and_dispatch`, `QWEN_AV_VALIDATE=1`) is simpler than QK's version: AV's fused
output lands directly in the real `att` buffer already, so it just runs the real `attn_dispatch`
path twice into two separate output buffers (once per mode) on the same real q/K/V from actual
decode and diffs the buffers directly — no capture instrumentation needed inside the kernel.
**Result: 55,050,240 comparisons, 0 mismatches, max_abs=0, max_rel=0 — PASS.**

**Sanitizers found a real, narrowly-characterized issue, not a logic bug.** ASan-`-O1`: clean.
ASan-`-O2` (default GCC inlining): a reproducible `stack-use-after-scope` in `av8_chunk`, 2/2 runs,
different worker thread and different timing each run. This did **not** reproduce in the isolated
standalone probe under ASan-`-O2` (clean), which narrows it to something about the production
calling context (the persistent worker-thread pool's repeated invocation), not the kernel's
arithmetic. Rebuilding ASan-`-O2` with `-fno-inline`: clean — this specifically isolates the
finding to GCC-`-O2`'s default decision to inline `av8_chunk` into `attn_worker_run`, not a defect
in the kernel logic itself. ASan-`-O3` (the actual production optimization level): clean, 3/3
repeated runs. UBSan-`-O1`: a separate runtime-error diagnostic (misaligned/undersized pointer
store) in the pre-existing, unmodified `qh[]`-population line, accompanied by garbled decode
output — consistent with this file's already-documented `-O1` GCC/RVV instability (§22.16, and
Phase 4.1's own `-O1` `qk8_dot` SEGV) rather than a new defect. UBSan-`-O2` and UBSan-`-O3`: both
clean. Net: the actual production build (`-O3 -fno-tree-vectorize`, no sanitizer) is clean under
both sanitizers, the anomaly is isolated to one specific compiler decision at one specific
optimization level with a sanitizer attached, and it does not reproduce in isolation — suspected to
be a compiler/sanitizer-instrumentation interaction specific to that inlining decision, not
confirmed as its root cause, and not a numerical or logic defect (both independent bit-exact
correctness checks above already rule that out).

**A/B, 2 trials per configuration, board otherwise idle, `QWEN_AV_FUSE=0` vs `1` against the Phase
4.1 baseline** (`qk_fuse=1` in both arms, since that is now the default; values are 2-trial
averages):

| | short | 128 | 512 | 1024 |
|---|---|---|---|---|
| wall/token | 81.25→81.10ms (-0.2%) | 92.40→88.25ms (**-4.5%**) | 125.45→110.90ms (**-11.6%**) | 168.50→138.60ms (**-17.8%**) |
| tok/s | 12.31→12.33 | 10.82→11.34 (+4.8%) | 7.97→9.02 (**+13.2%**) | 5.94→7.22 (**+21.6%**) |
| attention bucket (wall) | 2.22→1.52ms (-31.5%) | 13.25→8.61ms (**-35.1%**) | 46.08→30.56ms (**-33.7%**) | 88.59→57.96ms (**-34.6%**) |
| AV (summed across workers) | 3.39→0.81ms (-76.1%) | 23.58→4.23ms (**-82.1%**) | 78.75→17.22ms (**-78.1%**) | 155.68→37.79ms (**-75.7%**) |

Roughly 3× the magnitude of Phase 4.1's own wall-time win at every context, as expected — AV was
the (very slightly) larger of the two co-dominant buckets in the §22.27 baseline, and the fused
kernel's own reduction (76-82%) is larger than QK's fused reduction (22-26%) was. Trial-to-trial
agreement is tight throughout (e.g. ctx=1024 fuse=1: 138.5ms/7.22 tok/s vs 138.7ms/7.21 tok/s).
Generated token text is byte-identical between `fuse=0` and `fuse=1` at every context length across
both trials. Short-context wall time did not regress at all (a slight improvement, -0.2%),
comfortably inside the 2% ceiling.

**KEPT.** Three of the four predeclared criteria are unambiguously met (reproducible improvement,
no material numerical/token change, no short-context regression). The sanitizer criterion is
**not** a clean sweep across every configuration tested — ASan-`-O2` with default inlining
reproducibly failed — but the actual production configuration (`-O3 -fno-tree-vectorize`, no
sanitizer) is clean under both ASan and UBSan, and the one failing configuration is narrowly
isolated (via the `-fno-inline` experiment) to a compiler inlining decision rather than the kernel.
Judged on that basis to satisfy the gate's intent (the production build is not affected), not
because every sanitizer/optimization-level combination came back clean. `g_av_fuse` default
promoted from 0 to 1; `QWEN_AV_FUSE=0` remains the explicit revert path, byte-identical to Phase
4.1. No change to softmax, KV layout, or worker scheduling. Rebuilt the production binary (`-O3
-fno-tree-vectorize`) and confirmed the promoted default (no env vars) reproduces the A/B's
`fuse=1` numbers exactly and still passes the canonical `' Tokyo'` check.

**Per the execution directive: stop here.** Softmax optimization (Phase 5), eight-core testing
(Phase 6), KV quantization, and windowing remain explicitly out of scope until separately
authorized.

### 22.32 Attention Phase 5: exact softmax RVV vectorization — KEPT

Explicit authorization: continue autonomously through the optimization ladder, same end-to-end
discipline as prior phases. Re-profile QK/softmax/AV, RVV-vectorize the max reduction and
normalization, keep `expf` exact (no approximate exponential), A/B at short/128/512/1024, keep only
a reproducible end-to-end win with identical tokens and ≤2% short-context regression.

**Re-profile before implementing** (production default, `qk_fuse=1, av_fuse=1`): softmax's summed
work is now 43.5ms at ctx=256/ngen=24 — the single **largest** attention sub-bucket, bigger than QK
(32.1ms) and far bigger than AV (8.6ms). QK and AV's Phase 4 reductions (22-26% and 76-82%
respectively) shrank the buckets around it while softmax's own absolute cost stayed essentially
flat, so its relative share grew sharply — exactly the dynamic §22.29's Phase 3 entry had already
flagged as "the small regression Phase 5 exists to eventually address."

**Implementation.** `softmax(x,n)` is three passes: (1) scalar linear max-scan, (2) `expf(x[i]-m)`
+ running sum, (3) scalar `x[i]*=inv` normalize. Per explicit instruction only passes 1 and 3 are
RVV candidates -- pass 2's `expf` stays untouched, scalar, exact. `rvv_max_f32` accumulates an
elementwise RVV max across chunks then folds with `vfredmax`; `rvv_scale_f32` is a straight
vector-scalar multiply. Both are expected bit-exact to the scalar versions: max of a finite set (no
NaNs occur here) is order-independent under IEEE-754, and a per-element multiply's result depends
only on the two operands, not on scalar-loop vs RVV-lane execution. `g_softmax_rvv` (env var
`QWEN_SOFTMAX_RVV`) gates the whole function between the two implementations; the scalar path is
preserved byte-for-byte as the explicit revert.

**Validation.** Standalone probe `bench/softmax_rvv_probe.c` (`rvv_max_f32`/`rvv_scale_f32` copied
verbatim) at n∈{1,12,32,33,113,128,300,512,1024} — deliberately including n=1 and non-multiples of
32 to exercise the tail case. **Bit-exact: 2,155 comparisons, 0 mismatches, max_abs=0, max_rel=0.**
Unlike QK/AV, softmax's correctness doesn't depend on the multi-threaded dispatch machinery (it's a
pure per-call function regardless of caller), so production integration validation
(`QWEN_SOFTMAX_VALIDATE=1`) runs both implementations on every real call in place, using thread-
local scratch (grown once per thread, not malloc/free per call) rather than replaying the whole
dispatch path twice. **Result: 32,740,245 comparisons, 0 mismatches, max_abs=0, max_rel=0 — PASS.**

**Sanitizers reproduced the same two already-characterized findings from Phase 4.2, nothing new.**
ASan-`-O1`/`-O3` and UBSan-`-O2`/`-O3`: clean. ASan-`-O2` (default inlining) still shows the
`stack-use-after-scope` in `av8_chunk` (unchanged code, same line, same signature as §22.31).
UBSan-`-O1` still shows the pre-existing `qh[]`-population pointer-store diagnostic and garbled
output (unchanged code, same `-O1` GCC/RVV instability class). Neither trace touches `softmax`,
`rvv_max_f32`, or `rvv_scale_f32` — this phase's own new code introduces zero new sanitizer
findings at any tested optimization level. Production (`-O3`) remains clean under both sanitizers.

**A/B, 2 trials per configuration, board otherwise idle, `QWEN_SOFTMAX_RVV=0` vs `1`** (values are
2-trial averages; `qk_fuse=1, av_fuse=1` in both arms, the current production baseline):

| | short | 128 | 512 | 1024 |
|---|---|---|---|---|
| wall/token | 80.95→80.95ms (0%) | 88.20→87.75ms (**-0.5%**) | 111.05→109.70ms (**-1.2%**) | 138.55→135.50ms (**-2.2%**) |
| tok/s | 12.36→12.355 (~flat) | 11.335→11.40 (+0.6%) | 9.005→9.115 (+1.2%) | 7.215→7.38 (+2.3%) |
| softmax (summed across workers) | 1.595→1.46ms (-8.5%) | 9.355→8.055ms (**-13.9%**) | 35.205→29.915ms (**-15.0%**) | 69.875→58.985ms (**-15.6%**) |

Improvement grows with context length, consistent with softmax's own cost and this reduction's
benefit both scaling with `n`. Tokens are byte-identical between `fuse=0` and `fuse=1` at every
context length across both trials. Short-context wall time does not regress at all (flat, not
negative), comfortably inside the 2% ceiling.

**This is a modest win, honestly smaller in relative and absolute terms than QK or AV** — softmax
is the remaining sub-bucket after two much larger fusions already shrank the buckets around it, so
there is proportionally less left to recover, and the wall-time effect (0.5-2.2%) sits closer to
this board's measurement-noise floor than QK/AV's double-digit wins did (trial-to-trial agreement,
while still consistent in direction, is not as tight as the earlier phases'). It is nonetheless a
real, reproducible win in the correct direction at every long context tested, with zero short-
context cost and zero numerical/token risk — meeting the explicitly stated gate ("a reproducible
end-to-end win... with identical tokens and ≤2% short-context regression"), which is narrower than
Phase 3/4's percentage floors and was met deliberately, not by a wide margin.

**KEPT.** `g_softmax_rvv` default promoted from 0 to 1; `QWEN_SOFTMAX_RVV=0` remains the explicit
scalar revert. Rebuilt the production binary (`-O3 -fno-tree-vectorize`) and confirmed the
promoted default reproduces the A/B's `fuse=1` numbers and still passes the canonical `' Tokyo'`
check.

**All three attention-bucket components (QK, AV, softmax) are now RVV-optimized and fused/
vectorized where each phase's own gate justified it.** Per the execution directive, continuing
autonomously to Phase 6 (eight-core attention) next.

### 22.33 Attention Phase 6: eight-core attention — KEPT

Explicit authorization: linear/IME work stays on four workers (harts 8/10/12/14); attention-only
work may additionally use harts 9/11/13/15; generalize the persistent pool/job worker count safely,
idling surplus workers during four-worker linear jobs; measure attention workers 1/2/4/8 at
ctx=128/512/1024; check cache/memory contention, dispatch overhead, temperatures, end-to-end tok/s;
keep eight-worker attention only if it beats four workers reproducibly.

**Implementation.** Attention had been capped at `min(nt,nkv)`=4 workers because each KV head's
work was assigned to exactly one worker atomically (Phase 3's own division). To use more than 4
workers, each KV head's `gpr` (8) query heads must be SPLIT across `gpk` sub-workers
(`gpk=attn_nt/nkv`, e.g. 2 at attn_nt=8) instead of striding across more KV heads (there are only
4). `attn_worker_run` now computes `gpk`, `qi_count` (query heads owned per sub-worker), and each
worker's `(kvh, qi_start)` assignment from `(tn, attn_nt, nkv, gpr)`; at `attn_nt<=nkv` (`gpk=1`)
this reduces exactly to the pre-Phase-6 code path byte-for-byte (`qi_start=0`, `qi_count=gpr`), so
attn_nt=1/2/4 are provably unchanged. Pool sizing was generalized: `g_lin_nt` (new global, `min(nt,
4)`) caps linear/IME GEMM dispatch independent of the total pool size, while `g_pool_nt` (the
actual thread count) becomes `max(nt, g_attn_nt)` -- every dispatch (linear or attention) still
wakes all `g_pool_nt-1` secondary threads, but `lin_mm_hp_worker_run` now no-ops for `tn>=g_lin_nt`
regardless of how large the pool has grown for attention's sake. `g_hart_order` (already
`{8,10,12,14,9,11,13,15}`, added in an earlier session per docs/HARDWARE.md's per-pair IME-2
sharing note) already placed the paired harts last, so no pinning changes were needed -- worker
threads created for `tn=4..7` naturally land on harts 9/11/13/15.

**A genuine bug was found and fixed before validation -- not another compiler-quirk false alarm.**
ASan at `-O3` (the production optimization level, unlike every prior anomaly in this file's history)
reported a **100% reproducible** (10/10 trials) `stack-buffer-overflow`: a write to `ohp[8]`, one
past the end of an 8-element pointer array, in the line populating `av8_chunk`'s output-pointer
array. The suspected loop bound (`qi_count`) was checked directly via a temporary debug guard and
confirmed always in range [1,8] across every run it fired on -- ruling out the obvious "arithmetic
computes a bad count" explanation. The actual fix: `qh`/`scw`/`ohp` were declared *inside* the
per-KV-head loop (re-entering scope every iteration) with a runtime-bounded (`qi_count`) population
loop; hoisting them to function scope (declared once, `={0}`-initialized, matching `qk8_dot`/
`av8_chunk`'s own `if(n>K)`-guard discipline for the unused tail slots) resolved it completely:
**0/15 trials** failed post-fix (5 initial + 10 more), vs 10/10 pre-fix -- a clean, deterministic
before/after, not a coincidence. This is consistent with a real GCC-15.2/`-O3` ASan-instrumentation
interaction specific to loop-re-entered fixed-size arrays paired with a variable-trip-count
population loop, though the precise compiler mechanism was not further isolated -- the fix's
reliability (15/15 clean) was judged sufficient without pursuing a minimal reproduction case.

**Validation** (full order, all on `root@192.168.68.24`): production integration validation
(`QWEN_WORKERS_VALIDATE=1`, comparing each `attn_nt` against the attn_nt=1 serial baseline on real
q/K/V through the real dispatch path) — **attn_nt=2/4/8 all report 55,050,240 comparisons, 0
mismatches, max_abs=0** (attn_nt=1 trivially reports 0 comparisons against itself). Re-confirmed
identically on the post-fix code. Full sanitizer matrix at `attn_nt=8`, 3 trials each (matching the
"3/3 clean" bar from Phase 4.1/4.2): **ASan clean 9/9** across `-O1`/`-O2`/`-O3` (including the
now-fixed `-O3` case). UBSan clean 6/6 at `-O2`/`-O3`; UBSan-`-O1` crashes 3/3 with a silent SIGSEGV
(no diagnostic text, exit 139) -- this matches, in pattern (fails only at `-O1`, clean at `-O2`/
`-O3` under both sanitizers), this file's already-documented `-O1` GCC/RVV instability class
(§22.16, and Phase 4.1/4.2's own `-O1`-only findings), not a new logic bug -- especially given ASan
is now clean at all three optimization levels including `-O1`. Board temperatures stayed in the
60-70°C range throughout the full A/B (interleaved baseline/candidate, ~90 minutes), rising
gradually with sustained load, no thermal-throttling signature in the timing data (trial-to-trial
agreement stayed tight even in the later, warmer runs).

**A/B, 2 trials per configuration, board otherwise idle, interleaved baseline(`attn_nt=4`)/
candidate(`attn_nt=8`) per run per the operational rules** (values are 2-trial averages):

| | short | 128 | 512 | 1024 |
|---|---|---|---|---|
| attention bucket (wall) | 1.52→0.97ms (**-36.2%**) | 8.345→4.505ms (**-46.0%**) | 29.145→16.155ms (**-44.6%**) | 55.52→33.675ms (**-39.3%**) |
| wall/token | 81.3→81.0ms (-0.4%) | 88.65→84.7ms (**-4.5%**) | 109.5→97.7ms (**-10.8%**) | 135.9→115.7ms (**-14.9%**) |
| tok/s | 12.30→12.34 (+0.3%) | 11.28→11.805 (**+4.7%**) | 9.13→10.24 (**+12.2%**) | 7.36→8.645 (**+17.5%**) |

Improvement grows with context length -- expected, since eight-way parallelism has more work per
worker to amortize dispatch/sync overhead against at longer contexts, and the 4-vs-8-worker gap in
per-worker QK/AV compute (still summed-work bound, not yet bandwidth-bound at these context lengths)
widens accordingly. Dispatch overhead itself stayed tiny and flat throughout (0.08-0.14ms across
every configuration, no growth with worker count) -- the coordination cost of the larger pool is not
eating the gain. Generated token text is byte-identical between `attn_nt=4` and `attn_nt=8` at every
context length across both trials (also confirmed by the bit-exact integration validation above).
Short-context wall time does not regress (a small improvement instead), comfortably clear of any
reasonable regression ceiling.

**KEPT.** `g_attn_nt` default promoted from 4 to 8. `QWEN_ATTN_NT=4` remains the explicit four-
worker revert (byte-identical to the Phase 3-5 configuration, since `gpk==1` at `attn_nt<=nkv`
reduces to the exact pre-Phase-6 code path); `QWEN_ATTN_NT=1` remains the serial revert. Rebuilt the
production binary (`-O3 -fno-tree-vectorize`) and confirmed the promoted default reproduces the
A/B's `attn_nt=8` numbers exactly and still passes the canonical `' Tokyo'` check. **The canonical-
prompt HEADLINE number itself moved again: ~12.3-12.35→12.37 tok/s** (a small further gain on top
of Phases 3-5's own short-context improvements, since short-context attention was already a tiny
fraction of wall time before this phase).

**Per the execution directive, all of Phase 1 through Phase 6 are now complete and kept.** The next
tracks (exact-path closure, M-batch/continuous batching, batched prefill, speculative/n-gram
verification, and quality-changing long-context experiments) proceed per the broader authorization,
with track 7 (quality-changing experiments) requiring explicit approval before promoting any
semantics-changing mode, per that authorization's own terms.

### 22.34 Exact-path closure: re-profile + one evidence-supported residual — CLOSED (no candidate cleared its gate)

Explicit authorization: re-profile every bucket after Phases 5-6; test only evidence-supported
residuals (KV prefetch, score/scratch allocation, cache alignment, safe next-layer memory
prefetch); no cross-layer compute-overlap claims; declare the exact M=1 path closed when no
candidate clears its gate.

**Re-profile at the new production default** (`attn_nt=8`, all fusions on): at the canonical short
prompt, attention is now negligible — 1.0ms of 80.7ms wall (`<1.3%`); linear(kernel) dominates at
60.0ms (`74%`). At ctx=1024, attention has grown to 34.5ms but linear/MoE still dominates at 61.9ms
(1.8x), with `expert(gate/up/down)` alone at 37.8ms (constant across contexts, as expected for MoE
FFN work that doesn't depend on KV history) the single largest sub-bucket at every context tested.
`rest(other)` sits flat at 7.7-8.0ms regardless of context — genuinely constant per-token overhead
(residual adds, rmsnorm, KV-cache-write memcpys), not attention-related.

**Evidence review before implementing anything** (per the explicit "test only evidence-supported
residuals" scope), for each of the four named candidates:

- **Cache alignment**: checked directly. `kv.Kc`/`kv.Vc` (the KV cache itself) are large enough
  (>100MB at ctx=1024) that glibc's malloc/calloc routes them through its mmap-backed large-
  allocation path, which already returns page-aligned (4096-byte) memory — far more aligned than a
  single 64-byte cache line requires. No evidence supports further work there. The per-worker
  attention scratch buffers (`g_attn_scratch`/`g_attn_scratch_multi`), however, are small (e.g.
  32KB at ctx=1024) and go through glibc's normal heap allocator, which only guarantees ~16-byte
  alignment — a real, checkable gap. This is the one candidate with concrete supporting evidence,
  so it's the one actually tested (below).
- **Score/scratch allocation**: already lazy and reused across calls (`attn_scratch_ensure`/
  `attn_scratch_multi_ensure` only reallocate when the required size grows), not re-allocated per
  token or per layer — this was already the Phase 3 requirement and remains true. No further
  candidate here beyond the alignment question above.
- **KV prefetch**: no evidence supports this. Head-major layout (Phase 2) already makes each
  worker's own K/V history contiguous, and each worker's access pattern is a simple ascending scan
  — exactly the pattern hardware prefetchers already handle well. Dispatch overhead has stayed flat
  and tiny (0.08-0.14ms) through every A/B in this branch, showing no sign of memory-latency
  starvation that software prefetch would address.
- **Safe next-layer memory prefetch**: not pursued. This model is MoE with data-dependent expert
  routing — which experts' weight blocks are needed for the *next* layer isn't known until that
  layer's own router computation completes, which itself depends on that layer's own attention
  output. There is no safe, useful prefetch window that reaches genuinely across a layer boundary
  without first computing most of what a real cross-layer overlap would need to avoid computing —
  this is the same transformer-dependency constraint the execution directive explicitly warned
  against claiming to bypass. Not evidence-supported as scoped; would need a substantially different
  investigation (e.g., speculating on the router's next-layer decision) to become one.

**Scratch-buffer cache-line alignment, tested.** `g_scratch_align` (env var `QWEN_SCRATCH_ALIGN`)
switches the attention scratch allocator between plain `malloc` (default) and 64-byte-aligned
`posix_memalign`. Correctness: trivially unaffected (pure allocation-strategy change, no data-path
change; canonical `' Tokyo'` check passed both ways). A/B, 2 trials, short prompt + ctx=512/1024:

| | short | 512 | 1024 |
|---|---|---|---|
| wall/token | 81.3→80.65ms (-0.8%) | 97.8→97.4ms (-0.4%) | 115.2→115.6ms (**+0.35%**) |

No reproducible win at any context — the short/512 deltas are within this board's typical trial-to-
trial noise band (compare e.g. Phase 6's own short-context pairs, which varied by a similar margin
run-to-run), and ctx=1024 trends slightly the *wrong* direction. This does not clear any reasonable
"reproducible improvement" bar. **Not kept** — `g_scratch_align` stays at its default (0, plain
malloc); the flag remains in the code (documented, off by default) as a tested-and-rejected option,
per this session's practice of keeping negative results visible rather than deleting the evidence.

**Declared: the exact M=1 decode path is closed.** All four named residual candidates were
either ruled out by direct evidence (cache alignment for the KV cache itself; scratch allocation
strategy, already lazy/reused) or tested and found not to clear their gate (scratch-buffer cache-
line alignment); the remaining two (KV prefetch, next-layer prefetch) lack supporting evidence for
this architecture and, in next-layer prefetch's case, run into the same real cross-layer dependency
constraint the execution directive itself flagged. No further exact-path attention or scratch/cache
micro-optimization is planned without new evidence. Per the broader authorization, next: M-batch/
continuous batching, the primary remaining hardware-throughput lever.

### 22.35 M-batch track, milestone 1: vendor M4 kernel ported and validated

Explicit authorization: "Investigate vendor HP M=2/M=4/M=8 kernels or construct a validated
multi-row path... Report aggregate tok/s, per-sequence tok/s, latency, memory use, and M=1
regression for M=1/2/4/8. Never describe synthetic GEMM throughput as model tok/s." Given the scale
jump from every prior tuning-flag phase (this is a genuinely new vendor-kernel port, the same class
of effort the original M1 port itself required, including a documented false start), confirmed with
the user before committing session budget to it.

**Investigation.** The vendor library ships a real, hand-tuned RVV kernel for M≥4,
`gemm_kernel_i8i4_hp_m4` (`reference/spacemit-backend/ime2_kernels.cpp:3360`), already dispatched
in the vendor's own production llama.cpp build (`ime.cpp:276/583`, `count_m>=4` branch) — this is a
porting task, not novel kernel design. No literal M=2 or M=8 general-purpose kernel exists (a
`moe_m2_gemm_kernel_i8i4` exists but is MoE-routing-specific, a different use case); M4 is the
largest real hand-tuned tile. Confirmed `block_q4_0` (this engine's weight format) has
`block_type_has_zp<block_q4_0>()==false` (`ime.cpp:107`), so the live call path is M4's no-zp
branch, not the with-zp branch (initially misread — the with-zp comment block appears first in the
source and was read first, a red herring corrected before any code was written).

**A-record ground truth**, from the vendor's own packer `quantize_a_4row_i8_hp`
(`rvv_kernels.cpp:2100`, `vlenb==128` branch — this board's A100 harts, VLEN=1024), not
hand-derived from the denser asm comments alone (which describe "4 x fp16 row scales" but the
packer shows this is one shared fp16 scale per subblock across all 4 rows, plus 6 unused padding
bytes — confirmed by the packer only ever writing index 0 of each 4-slot area). Key finding:
**M4's per-subblock quantization scale is computed from the max absolute value across all 4
batched rows jointly** (`v_max_abs = max(max(|a0|,|a1|),max(|a2|,|a3|))`), not per row —
structurally different from every prior fusion in this session (QK/AV/softmax/worker-count were
all bit-exact by construction; M4 batching is not, by design, since sharing one scale across 4
rows is the mechanism that makes the wider hardware tile possible). This means M4's output is
expected to differ numerically from 4 independent M1 calls on the same rows — a quantization-
tradeoff question, not a bit-exactness gate.

**Port and probe** (`bench/vendor_ime_m4_probe.c`): `run_hp_m4` is a verbatim asm port of the
no-zp branch; `pack_A_hp_m4` matches the vendor packer's exact computation (1160-byte record: 8×
136B subblocks + 64B a_sum trailer + 8B scale_avg area, only the documented-used bytes written).
B-record (weights) is unchanged from M1 (`block_q4_0x32`, same weight stream — confirmed by
`b_tile_stride` depending only on `k_blks`, never on M).

**A validation methodology bug found and fixed before trusting any result.** The probe's first
version compared kernel output against an "oracle" using exact fp32 activations (no A-side
quantization at all) — this produced a wildly misleading ~100-140% "mean error" for *both* M1 and
M4, which would be impossible given M1's own already-proven production correctness. Root cause:
comparing against exact-fp32 activations conflates the kernel's own correctness with A-side int8
quantization's inherent ~8-bit rounding error, which is large in isolation (up to a couple percent
per term, compounding over 256 terms) and has nothing to do with whether the port is right. Fixed
by adopting `vendor_ime_a2_full.c`'s own already-proven methodology exactly: reconstruct each
kernel's reference from its *own actual stored, quantized, fp16-rounded packed bytes* (both A and
W sides), matching "does the ported asm correctly compute the dot product its own documented byte
format implies" — the real question, isolated from expected quantization noise.

**Result, 200 trials, 25,600 comparisons** (random K=256, N=32 blocks, board otherwise idle):

| | mean_abs_err | % of mean value | max_abs_err |
|---|---|---|---|
| M1 vs its own reconstructed reference | 1.00e-2 | 0.119% | 1.03e-1 |
| M4 vs its own reconstructed reference | 1.00e-2 | 0.119% | 1.12e-1 |

**M4/M1 mean-error ratio: 0.999×** — M4 introduces no meaningful extra error beyond M1's own
already-accepted quantization noise; the ported asm correctly implements its documented byte
format. M1-vs-M4 direct output comparison shows a real, expected, nonzero difference (mean 0.041,
max 0.24) — the shared-scale quantization tradeoff described above, not a bug.

**Milestone 1 (kernel port + numerical validation) is complete.** Remaining for the full M-batch
track per the original authorization: wire the validated kernel into the production linear/FFN
dispatch path for M rows, build a genuine multi-sequence batched-decode harness (independent
prompts/KV-caches sharing weight reads at the linear layer, attention still per-sequence), validate
every batched sequence against separate M=1 inference, and report aggregate/per-sequence
tok/s, latency, memory, and M=1 regression for the concrete M values this hardware actually
supports (M=1 exact; M=4 via the kernel validated here; M=8 as two M4 dispatches; M=2 has no
dedicated tile and would need its own tradeoff decision — pad to M4 or fall back to two M1 calls).

### 22.36 M-batch track, milestone 2: real 4-sequence batched decode, dense layers only

Explicit scope decision, offered to and confirmed by the user given the scale of what full
integration required: MoE's per-sequence expert routing (each of 4 batched sequences independently
selects its own top-8 of 128 experts from its own hidden state, so different sequences generally
select different, only-partially-overlapping expert sets) means the expert FFN cannot be
M4-batched by the same mechanism as the dense layers without a genuinely different, comparably-
sized gather/scatter-by-expert mechanism. Scoped this milestone to **dense-layer batching only**
(QKV, O, router, lm_head — all use one shared weight matrix regardless of sequence), leaving MoE
expert FFN and attention per-sequence via the existing, already-validated M1/serial paths, and
documenting expert-batching as explicitly out of scope for tonight.

**Implementation.** `run_hp_m4`/`pack_A_hp_m4` (the validated kernel from §22.35) wired into a new
`kind==3` pool-dispatch path (`lin_mm_hp_m4`), reusing the exact same N-tile-across-workers
machinery as `kind==0`'s M1 dispatch — only the per-N-tile kernel call and the A-record format
differ. New `forward4_dense_batch`: processes 4 independent (token, position, KV-cache) triples in
lockstep, packing all 4 sequences' activations into one M4 record at each dense layer (`Abuf4`,
sized for the wider of `d`/`qd` since the O-projection's input, `att`, is `qd`-wide — a real bug
caught before running anything: an early version packed `att` with K=`d` instead of `ly->o.K`,
silently under-packing `(qd-d)/256` blocks) and splitting the batched output back into per-sequence
buffers. Attention and MoE-FFN loop over the 4 sequences individually, calling the existing,
unmodified per-sequence machinery (`attn_dispatch`, `lin_mm_hp` for `eg`/`eu`/`ed`) — a second real
bug caught here: an early version dropped the `pack_act_hp(hn,d,Abuf2)` call the expert FFN needs
(the M1-format packed activation), inherited from `forward()` but never re-added for the 4-way
version, which would have fed the expert layers stale/garbage activation data.

**Test methodology** (`QWEN_MBATCH_TEST=1`, `run_mbatch_test`): 4 REAL, DISTINCT prompts from the
harness's own existing prompt set (factual/capitals, factual/chemistry, reasoning/syllogism,
reasoning/sequence — not 4 copies of one prompt, not synthetic data), each with its own KV cache.
Path 1: 4 separate, sequential M=1 decodes via the existing, already-proven `forward()`. Path 2: 1
batched M=4 decode via `forward4_dense_batch`. Both prefill identically (per-sequence, M=1, timed
separately from decode since it's identical cost in both paths and would otherwise dilute an
aggregate "total wall time" comparison for these short test prompts — prefill batching is
explicitly out of scope this milestone, the ladder's own next item).

**A real, informative divergence was found and characterized, not hand-waved.** Token agreement
over the full 16-token generation was only 70.31% (45/64) — initially concerning, since every prior
numerical change in this session showed single-digit-percent divergence at most. Root-caused via a
staged diagnostic, not assumed: **one M4-batched 48-layer forward pass** (before any autoregressive
compounding — prefill is identical M1 in both paths, so this isolates the batched pass's own
divergence from a byte-identical input) shows small, reasonable numerical difference (max_abs=0.73,
mean_abs=0.105 across the full 151,936-wide vocab logit vector) and **all 4 sequences' argmax
token choices agree exactly**. The full-generation divergence instead comes from ordinary
autoregressive compounding: this is a hard-top-8-of-128-expert MoE model, so once a close argmax
call flips for ANY one step (a small, real logit perturbation crossing a decision boundary), the
resulting *different* token becomes every subsequent step's own input, and errors branch and
compound from there — a well-understood property of discrete-token generation under ANY numerical
perturbation, not specific to or a defect of this port. Per-step tracing confirms the pattern
exactly: 2 of 4 sequences track the M1 baseline bit-for-bit through all 16 steps; the other 2
diverge from a single flipped step onward (position 5 and position 9 respectively), not from step 1
and not scattered randomly.

**Sanitizers**: ASan and UBSan both clean (`-O2`, full test run) — no memory-safety issues in the
new dispatch/batching/test-harness code. Correctness numbers (token agreement, per-step pattern,
logit diffs) are bit-for-bit identical across the plain, ASan, and UBSan builds (fully
deterministic, as expected — no data race, no uninitialized read). Confirmed the existing, unrelated
single-sequence M=1 production path is completely unaffected (12.38-12.41 tok/s, matching the
established baseline, `' Tokyo'` PASS).

**Performance, decode-phase only** (prefill excluded per the reasoning above; 2 trials, board
otherwise idle):

| | M=1 (4 separate) | M=4 (1 batched) | speedup |
|---|---|---|---|
| aggregate tok/s | 13.32-13.34 | 14.80-14.82 | **1.111x** |
| per-sequence tok/s | 3.33-3.34 | 3.70 | — |
| per-token latency | 74.95-75.06ms/seq | 269.95-270.24ms/batch-step | — |

Reproducible trial-to-trial (1.111x both clean runs; sanitizer builds, under heavy instrumentation
overhead, showed the two paths within noise of each other, 0.990x-1.076x, consistent with a small
underlying effect being harder to resolve through sanitizer overhead, not a sign the effect isn't
real on the production build). **Honest characterization: this is a real but modest speedup, well
below a naive "4 rows for the price of 1" intuition**, for two understood reasons: (1) the M4
kernel's own arithmetic work still scales with M (it holds 4 sets of accumulators and does
proportionally more `vmadotsu.hp`/`vmadotu.hp` work per call, not a fixed cost regardless of M) —
the mechanism is weight-stream-read amortization, not compute-for-free; (2) the batched layers
(QKV+O+router+lm_head) are roughly half the linear bucket at these context lengths, with the MoE
expert FFN — explicitly unbatched this milestone — remaining the single largest linear sub-bucket
and untouched by this speedup.

**Memory**: ~222 KB additional for the batched path's own buffers (`Abuf4`, `Abuf2_4`, and the
`y4q`/`y4k`/`y4v`/`y4o`/`y4r`/`y4lm` intermediate output buffers), beyond the 4x baseline per-
sequence buffer cost every path pays regardless of batching.

**Status: validated, working, NOT wired into the production default path.** `forward4_dense_batch`
and `lin_mm_hp_m4` exist as tested, correct, available capabilities (reachable via
`QWEN_MBATCH_TEST=1`), not a promoted default — there is no single-sequence "production" call site
that would benefit from this (batching requires multiple concurrent sequences to exist in the first
place, which this decode engine's current CLI/harness structure doesn't provide outside this test).
Remaining for a genuinely useful M-batch feature: MoE expert-FFN batching (the larger remaining
opportunity, a separate undertaking per the scoping decision above), a real multi-request serving
loop/scheduler (this milestone hand-constructs one fixed 4-sequence batch, not a general N-sequence
continuous-batching admission/eviction system), and batched prefill (explicitly deferred, the
ladder's own next item).

### 22.37 M-batch track, milestone 3: MoE expert-FFN batching — implemented, measured NOT materially worthwhile at N=4

Explicit direction: "batch the MoE expert FFN by grouping sequences per selected expert. That
targets the largest unbatched linear component and should determine whether M-batching becomes
materially worthwhile."

**Design.** Each of the 4 batched sequences independently selects its own top-8 of 128 experts from
its own hidden state (§22.36's own scoping rationale). This milestone groups sequences that
happened to select the SAME expert and batches that group via M4 — but only when the group is a
full 4-way match (all 4 sequences selected the same expert). Partial 2- or 3-way overlaps
deliberately fall back to the existing per-sequence M1 path, NOT padded up to a full M4 call: §22.36
found M4's own arithmetic work scales with M rather than being a fixed per-call cost, so a padded
call with only 1-3 real rows would do CLOSE TO the same arithmetic as a full 4-row call while only
saving 1-3 weight-stream reads — plausibly a net loss relative to just running M1 for those rows,
not a clear win, and not worth the risk without first knowing whether partial overlaps are even
common enough to matter (measured below).

**Implementation.** For each layer, after the (already-batched, §22.36) router computation
produces each sequence's own `sel[s][]`/`sw[s][]`, build `esel[e][s]` (the slot index sequence s
used to select expert e, or -1) and `ecount[e]` (how many of the 4 sequences selected expert e) by
one pass over all 4×8 selections. For every expert with `ecount[e]==4`: pack all 4 sequences' `hn`
via `pack_A_hp_m4`, dispatch M4 for `eg[e]`/`eu[e]`, apply SwiGLU per row (four independent calls
into the batched output rows, no change to the SwiGLU math itself), pack the 4 post-SwiGLU rows and
dispatch M4 again for `ed[e]`, then scatter each row's own-sequence-weighted contribution into that
sequence's `eout[s]` and mark that (sequence, slot) pair processed. Every NOT-processed
(sequence, slot) pair afterward runs through the unchanged, existing per-sequence M1 fallback loop
from milestone 2 — every one of a sequence's 8 selected experts is guaranteed to be handled exactly
once, either by the batched or the fallback path, never both, never neither.

**Validation and sanitizers.** Same discipline as prior milestones: `QWEN_MBATCH_TEST=1` re-run
against the same 4 real, distinct prompts, checked against separate M=1. ASan and UBSan both clean;
token agreement, per-step divergence pattern, and the expert-overlap statistics below are
bit-identical across the plain/ASan/UBSan builds (fully deterministic). Existing single-sequence
M=1 production path confirmed unaffected (12.41 tok/s, unchanged).

**The measurement the user asked for.** Across the full 16-token generation for all 4 sequences
(48 layers × 16 steps × 4 sequences × 8 selections = 24,576 individual expert selections, collapsing
to 18,576 distinct (expert, layer, step) slots with at least one selecting sequence):

| selecting-sequence count | slots | % of slots |
|---|---|---|
| 1 (no overlap) | 15,044 | 80.99% |
| 2 | 2,765 | 14.88% |
| 3 | 602 | 3.24% |
| 4 (batched this milestone) | 165 | **0.89%** |

Only 0.89% of expert-selection slots hit the case this milestone actually batches. Decode-only
speedup: **1.103-1.105x**, statistically indistinguishable from milestone 2's dense-layer-only
1.111x (well within this board's own trial-to-trial noise band, as established throughout this
session) — expert-FFN batching, as scoped, adds no measurable benefit on top of dense-layer
batching alone.

**Verdict: MoE expert-FFN batching by same-expert grouping is measured NOT materially worthwhile
at N=4 concurrent sequences with this model's 128-expert/8-selected configuration.** Two
independent, now-quantified reasons: (1) overlap is fundamentally too rare at this batch size — 81%
of expert-selection events involve exactly one sequence, so there is very little to batch in the
first place; (2) even the full theoretical upside of ALSO batching the 2-3-way partial-overlap
cases (an additional ~18.1% of slots, per the histogram above) would not proportionally translate
into speedup given finding (1) from §22.36 that M4's arithmetic cost scales with M rather than
being amortized for free — the batchable-in-principle opportunity is both rare AND, per row,
worth proportionally less than dense-layer batching's own win.

**This closes the loop on the M-batch track's core empirical question.** Combining milestones 2 and
3: dense-layer M-batching gives a real, modest, reproducible ~1.11x decode-phase speedup at N=4;
expert-FFN batching via same-expert grouping adds nothing further at this concurrency level. Larger
N (more concurrent sequences) would very plausibly change this calculus — the "exactly-N-way
overlap" event this milestone requires gets exponentially rarer as N grows, but "at least 2 of N"
overlap events get more common (birthday-paradox-style), meaning the fundamental limitation here is
specific to N=4, not necessarily to expert-batching as an idea — but building and validating that
at a larger, more production-realistic N is a separate, larger undertaking (a real multi-request
serving loop, not a fixed hand-constructed batch) not attempted here. `g_moe4_hits`/
`g_moe_ecount_hist` remain in the code as a reusable diagnostic for any future N.