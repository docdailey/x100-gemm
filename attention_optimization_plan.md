# Long-Context Attention Optimization Plan

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

| Candidate | Context | QK ms | Softmax ms | AV ms | Attention ms | Wall ms | tok/s | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 128 | | | | | | | |
| Baseline | 512 | | | | | | | |
| Baseline | 1024 | | | | | | | |
| Head-major KV | 128 | | | | | | | |
| Head-major KV | 512 | | | | | | | |
| Head-major KV | 1024 | | | | | | | |
| Four KV-group workers | 128 | | | | | | | |
| Four KV-group workers | 512 | | | | | | | |
| Four KV-group workers | 1024 | | | | | | | |
| Fused multi-Q QK | 512 | | | | | | | |
| Fused multi-Q QK | 1024 | | | | | | | |
| Fused multi-Q AV | 512 | | | | | | | |
| Fused multi-Q AV | 1024 | | | | | | | |

## Stop conditions

Stop this branch when any of the following becomes true:

- the next candidate fails its predeclared production gate;
- attention reaches a demonstrated KV-traffic limit;
- another bucket becomes dominant;
- further work requires approximate math without an approved quality experiment.

Do not combine layout, threading, and fused kernels into one patch.

## Immediate next action

Implement Phase 1 instrumentation only. Establish QK, softmax, and AV costs at contexts 128, 512, and 1024 before changing the KV layout or attention implementation. Then perform the isolated head-major KV-layout A/B described in Phase 2.
