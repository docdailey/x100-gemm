# Long-Context Attention Optimization Plan

> **Execution directive for Claude Code:** Start with Phase 1 instrumentation only. Do not implement KV-layout changes, attention threading, GQA fusion, approximate math, or attention-semantics changes until the Phase 1 baseline scoreboard is complete and documented. After that review point, test layout and exact GQA reuse before proposing any semantics change.

## Objective

Improve decode performance beyond approximately 64 tokens of context without changing model quality or destabilizing the current short-context release, which runs at approximately 11.35 tok/s.

Measured motivation:

- Attention overtakes linear work at approximately 59 tokens of context.
- Context 128: approximately 4.47 tok/s.
- Context 512: approximately 1.47 tok/s.
- Context 1024: approximately 0.77 tok/s.
- Linear work remains approximately 59–61 ms/token while attention grows approximately linearly with context.
- Current attention is exact RVV but single-threaded and uses a time-major KV layout: `[position][kv_head][head_dim]`.
- Qwen GQA geometry is 32 query heads, 4 KV heads, and 8 query heads per KV head.

## Scope: three different performance regimes

This plan addresses long-context decode, but the measurements expose three separate engineering problems:

| Regime | Current wall | Correct architectural interpretation |
|---|---|---|
| Short decode, roughly context <=60 | Linear weight streaming, approximately 59–61 ms/token | The exact M=1 path is already near vendor parity. Larger gains require amortizing weight traffic with batching or accepted-token speculation, not reopening the closed linear-scheduling investigation. |
| Long decode, roughly context >60 | Full-history attention grows linearly per decoded token | First recover avoidable implementation loss through layout, GQA reuse, and parallelism. Windowed or sparse attention can subsequently cap the curve, but is a separate quality-changing product experiment. |
| Long prefill | The current token-at-a-time implementation becomes unusably slow | This is a separate track requiring batched/chunked prefill kernels and prefix reuse. Across an entire prompt, naive causal attention work is O(context^2), not O(context). |

The current attention slope is approximately 1.2 ms per context position per decoded token across all 48 layers. That describes this implementation, not a proven DRAM floor. The implementation rereads each KV head for all eight associated query heads, uses a strided time-major layout, and runs attention on one thread. Phases 1–4 must determine how much of the slope is avoidable before declaring attention bandwidth-bound.

### Product-level tracks that are intentionally outside this first branch

After the exact attention work is measured, keep these as separate proposals with their own correctness and quality gates:

1. **Sliding-window or sink-plus-window attention.** This changes which tokens the model can attend to and therefore requires a real long-context quality harness. A 256- or 512-token window caps growth but does not, by itself, imply 10 tok/s: the measured linear and fixed costs remain, and the optimized attention slope must be known first.
2. **KV-cache quantization.** This reduces cache footprint and possibly traffic, but adds dequantization and changes numerical behavior. Do not attempt it until profiling shows the optimized exact path is actually limited by KV traffic.
3. **M-batch/continuous batching.** This is the primary aggregate-throughput lever for short-context multi-user service because weights can be reused across multiple activation rows. Report aggregate tok/s and per-sequence latency separately.
4. **Speculative or n-gram verification.** This only amortizes weight traffic if verification uses a genuine multi-token/M-row path and acceptance is high. Do not infer accepted-token speed from a draft microbenchmark.
5. **Prefix KV caching.** This avoids rebuilding a repeated prefix during prefill, but normal dense decode still attends to the cached prefix. It does not reduce per-token full-history attention unless paired with sparse/windowed attention.
6. **Batched/chunked prefill.** Chunking is useful for memory and scheduling, but chunking alone does not eliminate total causal-attention work. The important implementation change is processing prompt tokens in matrices/tiles instead of repeatedly invoking the M=1 decode path.

Do not plan to overlap layer L attention with layer L+1 expert computation: the transformer data dependency prevents layer L+1 from starting until layer L attention, output projection, residual, and FFN have produced its input. Safe prefetching may overlap memory movement with current-layer computation, but any claimed pipeline gain must first identify a genuinely independent stage.

## Safety and process rules

1. Never run two model-loading processes simultaneously.
2. Use a global board lock (`flock`) for every benchmark and harness invocation.
3. Preserve:
   - the mandatory `-fno-tree-vectorize` build flag;
   - the int8-M1 router default;
   - the rational-Padé SwiGLU default;
   - the hard KV-position bounds guard.
4. Change one attention factor at a time.
5. Give every candidate an exact baseline/candidate production A/B.
6. Do not introduce approximate softmax in this branch.
7. Append every result, including failures, to:
   - `codex_recs_1.md`;
   - `research_feed_paths.md`;
   - `PROGRESS.md`.

## Phase 1 — Split the attention bucket

Instrument the following components without changing any math:

- QK dot-product time;
- softmax time;
- weighted-V accumulation time;
- attention dispatch/synchronization time, if dispatch is added later.

Run identical builds at:

- context 128;
- context 512;
- context 1024.

Use at least two measured decode trials per context. Record wall time, tok/s, all existing buckets, and the new attention sub-buckets.

The purpose is to establish whether QK, softmax, or AV dominates before optimizing anything.

## Phase 2 — KV-cache layout A/B

The current layout makes one KV head traverse positions with a 2 KB stride:

```text
[position][kv_head][head_dim]
```

Implement an exact head-major candidate:

```text
[layer][kv_head][position][head_dim]
```

Update both KV writes and K/V reads. Do not change quantization or attention math.

Why test this first:

- Each head's history becomes contiguous.
- It may improve prefetching and cache-line use.
- It is independent of thread scheduling and easy to revert.

Validation:

1. Run ASan on a bounded long-context test.
2. Confirm there are no KV bounds errors.
3. Confirm generated token IDs are identical to the baseline.
4. A/B at contexts 128, 512, and 1024.

Keep the change only if attention improves reproducibly and short-context wall time does not regress by more than 2%.

## Phase 3 — Parallelize across the four KV-head groups

Reuse the existing persistent four-worker pool. Do not create threads per layer.

The ideal work division for this model is:

- worker 0: KV head 0 and its 8 query heads;
- worker 1: KV head 1 and its 8 query heads;
- worker 2: KV head 2 and its 8 query heads;
- worker 3: KV head 3 and its 8 query heads.

This preserves GQA locality better than cyclic query-head assignment.

Requirements:

- Give each worker a private score buffer of at least `ctx` floats.
- Allocate scratch once, not once per layer or token.
- Ensure each worker writes only its own output-head range.
- Add an attention job type to the persistent pool.
- Measure attention dispatch/wait separately.
- Preserve the exact per-head operation order.

Test attention worker counts 1, 2, and 4. Do not test eight cores yet.

Predeclared keep criteria:

- generated tokens remain identical;
- no ASan or UBSan failure;
- at least 20% attention-bucket reduction at context 512 or 1024;
- at least 10% end-to-end improvement at context 512 or 1024;
- no more than 2% short-context regression.

Revert the candidate if these gates fail.

## Phase 4 — Exact GQA-fused kernels

Only after measuring the layout and four-worker results, address duplicated KV traffic.

The current code scans the same K and V history eight times—once for every query head sharing a KV head.

Implement two separate exact GQA-group candidates.

### 4.1 Multi-Q QK

- Process the eight query heads belonging to one KV head together.
- Load each K vector or chunk once.
- Update eight independent dot-product accumulators.

### 4.2 Multi-Q AV

- Process eight output heads together.
- Load each V vector or chunk once.
- Update eight independent output accumulators with their respective softmax weights.

VLEN=1024 provides 32 fp32 lanes, while `head_dim=128` requires four vector chunks. Structure the AV loop so output accumulators remain in RVV registers across the position loop where register pressure permits.

Keep QK and AV as separate commits and separate A/Bs so their effects remain attributable.

Validation must compare candidate attention output with the original implementation using:

- maximum absolute error;
- maximum relative error;
- per-layer comparisons over deterministic synthetic data;
- final token identity;
- long-context production A/B results.

Use a tight numerical threshold appropriate for a changed floating-point reduction order. If generated tokens change, investigate before accepting the candidate.

## Phase 5 — Exact softmax optimization

Only pursue this if Phase 1 shows softmax is material.

- RVV-vectorize the maximum reduction.
- RVV-vectorize normalization.
- Keep `expf` exact initially.
- Do not add an exponential approximation without a separate, predeclared quality study.

If softmax is a small fraction of attention, stop instead of optimizing it merely because it is visible.

## Phase 6 — Optional eight-core attention experiment

Attempt this only after the four-worker implementation is correct and measured.

Attention does not use IME-2, so the paired A100 harts might provide additional RVV throughput. Test an attention-only eight-worker pool pinned in this order:

```text
8, 10, 12, 14, 9, 11, 13, 15
```

Compare attention scaling at 1, 2, 4, and 8 workers for contexts 512 and 1024. Shared-cache or memory contention may make eight workers slower, so retain it only when end-to-end performance improves reproducibly.

Do not change the four-thread linear/IME configuration.

## Required scoreboard

Phase 1 baseline filled in 2026-07-26 (2 trials/context, board otherwise idle, one job at a time;
raw trials kept, not just averages — see `codex_recs_1.md` §22.27 for full methodology/discussion).
`nt=4`, `ngen=16`, `QWEN_CTXLEN` synthesizing the prefill. All later rows pending, not started —
per the execution directive, no layout/threading/fusion candidate is implemented yet.

| Candidate | Context | QK ms | Softmax ms | AV ms | Attention ms | Wall ms | tok/s | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline (T1) | 128 | 60.93 | 9.40 | 57.37 | 127.86 | 207.5 | 4.82 | (ref) |
| Baseline (T2) | 128 | 62.68 | 9.49 | 73.32 | 145.65 | 225.2 | 4.44 | identical to T1 |
| Baseline (avg) | 128 | 61.80 | 9.45 | 65.34 | 136.75 | 216.3 | 4.63 | — |
| Baseline (T1) | 512 | 268.69 | 35.04 | 283.32 | 587.22 | 667.4 | 1.50 | (ref) |
| Baseline (T2) | 512 | 272.90 | 35.12 | 275.69 | 583.88 | 664.3 | 1.51 | identical to T1 |
| Baseline (avg) | 512 | 270.79 | 35.08 | 279.50 | 585.55 | 665.8 | 1.50 | — |
| Baseline (T1) | 1024 | 566.90 | 69.27 | 580.86 | 1217.22 | 1298.5 | 0.77 | (ref) |
| Baseline (T2) | 1024 | 567.25 | 69.34 | 580.88 | 1217.67 | 1299.0 | 0.77 | identical to T1 |
| Baseline (avg) | 1024 | 567.08 | 69.31 | 580.87 | 1217.45 | 1298.8 | 0.77 | — |
| Head-major KV (T1) | 128 | 22.03 | 10.75 | 23.74 | 56.69 | 135.5 | 7.38 | identical to baseline |
| Head-major KV (T2) | 128 | 21.66 | 10.78 | 23.51 | 56.11 | 135.1 | 7.40 | identical to baseline |
| Head-major KV (avg) | 128 | 21.85 | 10.77 | 23.63 | 56.40 | 135.3 | 7.39 | — |
| Head-major KV (T1) | 512 | 73.96 | 40.34 | 86.30 | 200.79 | 279.8 | 3.57 | identical to baseline |
| Head-major KV (T2) | 512 | 75.23 | 40.33 | 85.75 | 201.49 | 280.9 | 3.56 | identical to baseline |
| Head-major KV (avg) | 512 | 74.60 | 40.34 | 86.03 | 201.14 | 280.4 | 3.56 | — |
| Head-major KV (T1) | 1024 | 140.84 | 79.79 | 162.50 | 383.32 | 463.2 | 2.16 | identical to baseline |
| Head-major KV (T2) | 1024 | 140.30 | 79.91 | 164.36 | 384.79 | 464.4 | 2.15 | identical to baseline |
| Head-major KV (avg) | 1024 | 140.57 | 79.85 | 163.43 | 384.06 | 463.8 | 2.16 | — |
| Four KV-group workers (T1) | 128 | 20.54* | 9.41* | 21.59* | 13.95 | 92.7 | 10.79 | identical to baseline |
| Four KV-group workers (T2) | 128 | 21.36* | 9.40* | 21.41* | 14.23 | 93.0 | 10.75 | identical to baseline |
| Four KV-group workers (avg) | 128 | 20.95* | 9.41* | 21.50* | 14.09 | 92.85 | 10.77 | — |
| Four KV-group workers (T1) | 512 | 78.16* | 35.06* | 79.39* | 50.83 | 129.7 | 7.71 | identical to baseline |
| Four KV-group workers (T2) | 512 | 75.52* | 34.96* | 78.41* | 49.83 | 129.0 | 7.75 | identical to baseline |
| Four KV-group workers (avg) | 512 | 76.84* | 35.01* | 78.90* | 50.33 | 129.35 | 7.73 | — |
| Four KV-group workers (T1) | 1024 | 149.86* | 69.17* | 154.29* | 98.34 | 177.6 | 5.63 | identical to baseline |
| Four KV-group workers (T2) | 1024 | 150.94* | 69.16* | 153.41* | 98.50 | 178.0 | 5.62 | identical to baseline |
| Four KV-group workers (avg) | 1024 | 150.40* | 69.17* | 153.85* | 98.42 | 177.8 | 5.625 | — |
| Multi-Q QK (T1) | 128 | 16.65* | 9.37* | 20.36* | 12.70 | 91.5 | 10.93 | identical to Phase 3 |
| Multi-Q QK (T2) | 128 | 16.58* | 9.40* | 20.78* | 12.86 | 92.1 | 10.86 | identical to Phase 3 |
| Multi-Q QK (avg) | 128 | 16.62* | 9.39* | 20.57* | 12.78 | 91.8 | 10.895 | — |
| Multi-Q QK (T1) | 512 | 59.88* | 35.17* | 77.08* | 45.74 | 125.0 | 8.00 | identical to Phase 3 |
| Multi-Q QK (T2) | 512 | 60.51* | 35.19* | 79.22* | 46.34 | 126.6 | 7.90 | identical to Phase 3 |
| Multi-Q QK (avg) | 512 | 60.20* | 35.18* | 78.15* | 46.04 | 125.8 | 7.95 | — |
| Multi-Q QK (T1) | 1024 | 113.65* | 69.68* | 152.64* | 87.55 | 167.2 | 5.98 | identical to Phase 3 |
| Multi-Q QK (T2) | 1024 | 113.70* | 69.63* | 152.72* | 87.47 | 167.2 | 5.98 | identical to Phase 3 |
| Multi-Q QK (avg) | 1024 | 113.68* | 69.66* | 152.68* | 87.51 | 167.2 | 5.98 | — |
| Fused multi-Q AV | 512 | | | | | | | |
| Fused multi-Q AV | 1024 | | | | | | | |

**Baseline reading**: QK and AV are essentially co-dominant (~46-48% of the attention bucket each,
at every context tested), softmax is a small but non-trivial 5.7-6.9% (shrinking slightly as a
fraction as context grows, since QK/AV scale with context and softmax's cost per call is
`O(pos+1)` for the max/exp/normalize passes but only `O(hd)`-comparable per-element work, not the
`O(hd)` dot-product work QK/AV do at every position) — material enough that Phase 5 is worth doing
eventually, but neither QK nor AV can be ignored in favor of it. QK+softmax+AV sums track the
`attention` bucket tightly (gap ≤0.2ms at every context, the zero-init/loop overhead the three
sub-timers don't cover) — the instrumentation is not introducing distorting overhead. Scaling from
128→512 (4x context) and 512→1024 (2x context) is close to linear for both QK and AV, consistent
with `codex_recs_1.md` §22.23's aggregate finding, now decomposed into which two components carry
that slope. AV runs measurably (~2.5% at ctx=1024) more expensive than QK despite comparable FLOP
counts — plausibly the read-modify-write accumulation into the output vector (`vaxpy_f32` reads
*and* writes `oh`) versus QK's read-only reduction to a scalar (`vdot_f32` only writes one `sc[j]`
per position) — relevant to Phase 4's fused-kernel motivation, since both directions redundantly
re-read the same K/V once per query head sharing a KV head (8x redundant reads either way).

**Phase 2 result (2026-07-26): KEPT — result far exceeds the plan's own modest framing.** The
head-major layout was expected to "improve prefetching and cache-line use" as a secondary,
possibly-marginal effect; the actual measured effect is dramatic, not marginal — the old layout's
2KB stride between consecutive positions for one head guaranteed zero cache-line reuse and a
strided (not sequential) access pattern; the new layout makes one head's whole K/V history
contiguous, ideal for the hardware prefetcher.

| | 128 | 512 | 1024 |
|---|---|---|---|
| attention bucket | 136.75→56.40ms (**-58.8%**) | 585.55→201.14ms (**-65.6%**) | 1217.45→384.06ms (**-68.5%**) |
| wall/token | 216.3→135.3ms (-37.4%) | 665.8→280.4ms (-57.9%) | 1298.8→463.8ms (-64.3%) |
| tok/s | 4.63→7.39 (**+59.6%**) | 1.50→3.56 (**+136.9%**) | 0.77→2.16 (**+179.9%**) |

Short-context (canonical 12-token prompt, 2 paired trials): wall time 88.45ms→87.7ms, a **0.85%
improvement, not a regression** — comfortably clears the "no more than 2% regression" gate with
room to spare. Both predeclared keep criteria met: attention improves reproducibly (tight
trial-to-trial agreement at every context, see table above) and short-context does not regress.
ASan clean on a bounded ctx=256/ngen=24 run (no report, no abort); tokens byte-identical to the
time-major baseline at every context tested (128, 512, 1024, plus the ASan ctx=256 run) and at the
canonical short prompt. Bonus, not a required metric: prefill time also improved substantially
(ctx=512: ~172-185s→91s; ctx=1024: ~678-680s→279s), consistent with the same locality argument
applying to the prefill loop's own repeated K/V writes and reads.

**Head-major is now the production KV-cache layout** (`qwen_moe_hp.c`, committed). The time-major
baseline is preserved at `/tmp/qwen_moe_hp_kv_timemajor.c` on the board (built from the exact
pre-Phase-2 commit) as the revert reference, matching this session's established pattern (e.g. the
§22.14 scheduling A/B) of keeping a scratch comparison copy rather than committing it to the repo.
Full writeup in `codex_recs_1.md` §22.28.

**Phase 3 result (2026-07-26): KEPT — explicit authorization, Phase 3 only, isolated from GQA
fusion.** `*` in the scoreboard above marks QK/softmax/AV values that are **summed CPU-time across
all participating workers**, not directly wall-time-comparable once threaded (they naturally scale
up with worker count since more workers means more total concurrent compute-seconds, even though
wall-clock drops) — the **Attention ms** column remains the true wall-clock figure and the one
comparable across every phase.

| | 128 | 512 | 1024 |
|---|---|---|---|
| attention bucket | 56.40→14.09ms (**-75.0%**) | 201.14→50.33ms (**-75.0%**) | 384.06→98.42ms (**-74.4%**) |
| wall/token | 135.3→92.85ms (-31.4%) | 280.4→129.35ms (-53.9%) | 463.8→177.8ms (-61.7%) |
| tok/s | 7.39→10.77 (**+45.7%**) | 3.56→7.73 (**+117.1%**) | 2.16→5.625 (**+160.4%**) |

Short-context (canonical 12-token prompt, 2 paired trials): wall time 88.45ms (Phase 2 baseline,
attn_nt=1) → 81.15ms, an **8.25% improvement** — comfortably clears "no more than 2% regression."
All five predeclared keep criteria met: tokens identical at every worker count (1/2/4) and context
tested; ASan clean on a bounded ctx=256/ngen=24 run at attn_nt=2 and attn_nt=4; UBSan clean at
attn_nt=4; attention-bucket reduction 75.0%/74.4% at ctx=512/1024, both far past the 20% floor;
end-to-end tok/s improvement 117.1%/160.4%, both far past the 10% floor; short-context improved
rather than regressed. Dispatch/sync overhead is tiny (0.09-1.14ms across every configuration
tested) — parallelization is not paying a meaningful coordination tax.

**Four-KV-group worker parallelism is now the production attention path**: `g_attn_nt` defaults to
`min(nt,nkv)` (=4 at the standard `nt=4` config), set once `m.nkv` is known in `main()`.
`QWEN_ATTN_NT=1` remains an explicit serial revert flag, byte-identical to the Phase 2 code path.
Full writeup in `codex_recs_1.md` §22.29.

**As explicitly directed: softmax (Phase 5) remains deferred despite a small regression.** Softmax's
own summed-work total grew slightly in absolute terms through Phases 2-3 while QK/AV's wall-clock
contribution collapsed (e.g. ctx=1024: 69.31ms at the Phase 1 baseline → 69.17ms total-work at
Phase 3 — essentially flat in absolute terms, but now a much larger *relative* share of a bucket
that shrank by 74%, since QK/AV no longer dwarf it the way they did in Phase 1). This is exactly
the kind of change Phase 5 exists to eventually address, but per direction it stays deferred: no
exponential approximation, no work here, until specifically authorized.

**Phase 4.1 result (2026-07-27): KEPT — multi-Q QK only, isolated from multi-Q AV.** Explicit
authorization: Phase 4 only; first implement multi-Q QK in isolation; do not fuse AV in the same
patch; preserve `QWEN_QK_FUSE=0` as the Phase 3 revert path. Softmax remains deferred.

`qk8_dot` loads each K chunk once and updates all 8 query-head accumulators with identical per-head
FMA/reduction order to `vdot_f32` (bit-exact by construction). Fused path uses a larger per-worker
scratch (`gpr×ctx`) because all query heads' scores for a KV head finish together before softmax.
Standalone probe (`bench/qk_multiq_probe.c`) and production integration validation
(`QWEN_QK_VALIDATE=1`, 60,426,240 real-dispatch comparisons) both report max_abs=0 / max_rel=0.

ASan and UBSan clean at `-O2` (production uses `-O3`); UBSan at `-O1` hit a compiler
register-pressure crash on 8 live RVV accumulators that does not reproduce at `-O2`/`-O3` — same
class of GCC/RVV interaction already documented for this file. Tokens byte-identical fuse0↔fuse1 at
every context and the canonical short prompt.

| | short | 128 | 512 | 1024 |
|---|---|---|---|---|
| attention bucket | 2.41→2.25ms (-6.6%) | 14.17→12.78ms (**-9.8%**) | 50.60→46.04ms (**-9.0%**) | 97.57→87.51ms (**-10.3%**) |
| wall/token | 81.05→80.95ms (-0.1%) | 92.85→91.8ms (-1.1%) | 129.85→125.8ms (-3.1%) | 177.5→167.2ms (**-5.8%**) |
| tok/s | 12.345→12.35 (~flat) | 10.775→10.895 (+1.1%) | 7.70→7.95 (**+3.2%**) | 5.635→5.98 (**+6.1%**) |

QK summed-work alone drops ~22–26% at long context (the intended effect); AV is essentially
unchanged (as expected — not fused yet); softmax flat. Short-context does not regress. Honest
relative to Phase 2/3: this is a **modest exact win**, not a dramatic one — the Phase 3
predeclared 20% attention / 10% e2e floors (written for the threading candidate) are **not** met
here. The Phase 4 authorization gate was "reproducible attention/end-to-end improvement + sanitize,"
which is met. **KEPT and promoted**: `g_qk_fuse` defaults to 1; `QWEN_QK_FUSE=0` is the explicit
revert. Full writeup in `codex_recs_1.md` §22.30.

**Next: Phase 4.2 (multi-Q AV) remains the larger residual opportunity** — AV is still co-dominant
and still re-reads each V vector 8× per KV head. Softmax (Phase 5) stays deferred.

## Stop conditions

Stop this branch when any of the following becomes true:

- the next candidate fails its predeclared production gate;
- attention reaches a demonstrated KV-traffic limit;
- another bucket becomes dominant;
- further work requires approximate math without an approved quality experiment.

Do not combine layout, threading, and fused kernels into one patch.

## Immediate next action

Phases 1–3 and Phase 4.1 (multi-Q QK) are complete and kept — see the scoreboard above and
`codex_recs_1.md` §22.27–§22.30. **Next: Phase 4.2 only — multi-Q AV** (load each V once, update
eight independent output accumulators). Keep QK fusion, head-major layout, four-worker scheduling,
fp32 KV storage, and softmax unchanged. Softmax (Phase 5) remains explicitly deferred.
