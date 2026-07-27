# PROGRESS — K3 pure-C IME-2 decode engine (session log, live)

Last update: 2026-07-27. Durable state so work survives a session kill.

**Release-quality checkpoint frozen (2026-07-26, `codex_recs_1.md` §22.22)**: commit `0dde2f4`
(int8-M1 router + rational-Padé SwiGLU, the current HEADLINE default) — compiler, exact build
command, and sha256 of the source/binary/model/cache are all recorded there for exact
reproducibility. Board-side re-verified at freeze time: `' Tokyo'` PASS, 11.35 tok/s.

**Context-length scaling swept (2026-07-26, `codex_recs_1.md` §22.23)**: all this session's
speed/bucket decisions were measured at a 12-token prompt, a linear(kernel)-dominated regime.
Swept ctx ∈ {32,64,128,512,1024} via a new `QWEN_CTXLEN` env var — `attention` scales linearly with
context (~1.2ms/token of context) while every other bucket stays flat. **Crossover where attention
overtakes linear(kernel): ~59 tokens of context.** Past that, the int8-M1/rational-Padé speedups
(both target flat buckets) contribute a shrinking fraction of wall time — only ~5% at ctx=1024.
Attention itself (RVV QK/AV dot products, no pruning/windowing) is flagged as the natural next
target for long-context throughput, not started.

**Board outage, root-caused (2026-07-26): OOM-killer, not thermal.** Launched the expanded quality
harness and the vendor `llama-bench` A/B concurrently (a sequencing mistake) — each loads its own
~17-18GB model/cache, and combined demand (~35GB) exceeded the board's **32.8GB RAM with zero
swap**. `journalctl` confirms the kernel OOM-killer killed the harness process
(`qwen_moe_hp total-vm:35013040kB, anon-rss:17999616kB`) and took sshd down with it — the board was
unreachable (SSH banner-exchange timeouts) for ~15-20 min until sshd's unit restarted. Temps were
57-61°C throughout, never the cause. **Lesson recorded** (also in Claude's persistent memory): never
run two model-loading processes on this board concurrently — always confirm one has fully exited
before starting another.

**Vendor `llama-bench` A/B redone, sequentially, plus a real vendor-binary bug found and fixed
(2026-07-26, `codex_recs_1.md` §22.24).** `llama-bench` crashed reproducibly even run alone
(`ggml_abort` on a thread-affinity error) — traced to `LD_LIBRARY_PATH=/usr/lib` loading a stale
(Jul 7) system copy of `libggml-cpu.so` that predates a local source patch already present in the
checkout; the build tree's own copy (Jul 25) has the fix. No source changes needed, just point
`LD_LIBRARY_PATH` at `/root/llama.cpp/build/bin` instead of `/usr/lib`. With that fixed, board
otherwise idle: **our engine 11.35-11.39 tok/s vs vendor 11.38±0.21 tok/s at nt=4 — at parity.**
(vendor nt=8: 12.82±0.05, close to the earlier stale-library baseline's 12.89, confirming the fix
didn't change the underlying performance number, only the crash.)

**Expanded harness rerun crashed (SIGSEGV) — root-caused with ASan, 5 real bugs found and fixed,
clean rerun confirms every conclusion (2026-07-26, `codex_recs_1.md` §22.25).** Rerunning the
corpus/generation-length expansion (alone, no contention) crashed with a heap-corruption segfault.
Root-caused with AddressSanitizer (same methodology as §22.16): `ppl_reason`'s declared length
(176) didn't match its real array size (149) — a **stale bug predating this session's expansion
entirely** — causing a genuine out-of-bounds heap read in every real-text-perplexity run that
included it, this whole session. Auditing every other declared length against its actual array
found 4 more (all pre-existing, none from this session's additions): `hp3`/`hp4`/`hp9`/`hp10` were
all under-declared, silently truncating the prefill actually fed to the model — `hp10` by 42
tokens, over a quarter of its intended length. Also raised `HARNESS_MAXCTX` (200→512, computed
properly from the real max prompt+corpus length) and added a hard bounds guard directly in
`forward()`'s KV-cache write, so any future instance of this bug class aborts immediately at the
fault site instead of corrupting the heap silently. **A second ASan run, after all 5 fixes, completed
clean end-to-end** — no report, full harness finished naturally. Corrected results confirm every
conclusion unchanged (hard-swish still fails at 13.4% inflation, rational-Padé still passes at
1.7%/0.5% inflation, router promotion re-confirmed) — no promotion status changes, since
rational-Padé was already promoted on the strength of §22.21's production A/B, not this harness's
numbers. This was purely a validate-the-validator exercise, and it found and fixed real, previously
undetected bugs in the harness itself.

**Fresh production A/B reconfirms rational-Padé; a real production-path bug found and fixed along
the way (2026-07-26, `codex_recs_1.md` §22.26).** Reran §22.21's exact/rational-Padé A/B fresh,
given everything since (OOM incident, vendor lib fix, harness bugfixes): **92.2% SwiGLU bucket
reduction, wall -14.5%, tok/s +16.9%** (9.71→11.355 tok/s avg) — reproduces §22.21 almost exactly,
promotion confirmed sound. Extending the check to `ngen=60` (beyond the canonical 16-token window)
immediately tripped the new `forward()` guard: `main()`'s KV-cache sizing only grew for `ngen` under
`QWEN_CTXLEN`, leaving the *default* prompt path's `ctx` hardcoded at 64 regardless of `ngen` — any
production invocation requesting more than ~52 generated tokens would have silently overrun the KV
cache. **This was live in the shipped production binary all session**, not harness-only. Fixed
(`ctx` now sized from whichever prefill length is actually in effect); reverified clean at
`ngen=60` under both SwiGLU configs, default `ngen=16` unaffected. At `ngen=60`'s longer context,
SwiGLU's savings barely move tok/s (8.82 vs 8.88) because attention now dominates — a live
confirmation of §22.23's crossover finding.

**Attention optimization branch opened — Phase 1 (instrumentation + baseline) complete
(2026-07-26, `attention_optimization_plan.md`, `codex_recs_1.md` §22.27).** New QK/softmax/AV
sub-buckets added inside the existing attention timer, verified byte-identical output and
negligible overhead. Baseline (2 trials/context): **QK and AV are co-dominant, ~46-48% of the
attention bucket each at every context tested (128/512/1024) — neither is "the" bottleneck.**
Softmax is secondary but non-trivial (5.7-6.9%). Both QK and AV scale close to linearly with
context, decomposing §22.23's aggregate ~1.2ms/token-of-context finding. Per the plan's execution
directive, **no layout/threading/fusion work has started** — this baseline is the required review
point before Phase 2 (head-major KV layout) begins.

**Attention Phase 2 KEPT — head-major KV layout, dramatic gain (2026-07-26, `codex_recs_1.md`
§22.28).** Explicit authorization: Phase 2 only, isolated from threading/GQA fusion. Per-layer KV
cache changed from `[position][kv_head][head_dim]` (2KB stride between positions for one head) to
`[kv_head][position][head_dim]` (one head's history contiguous) — pure addressing change, same
math/quantization. **Attention bucket: -58.8% (ctx=128), -65.6% (ctx=512), -68.5% (ctx=1024).
tok/s: +59.6%, +136.9%, +179.9% respectively** — far beyond the plan's own modest "may improve
prefetching" framing; the old layout's 2KB stride was hostile to hardware prefetching, and a
meaningful fraction of what looked like real attention compute cost was actually avoidable
memory-layout overhead. Short-context (canonical prompt): wall time *improved* 0.85%, not
regressed — both predeclared keep criteria (attention improves reproducibly, short-context doesn't
regress >2%) cleared with large margins. ASan clean on a bounded long-context run, tokens
byte-identical to the time-major baseline at every context tested. **Head-major is now the
production KV-cache layout.** Threading (Phase 3) and GQA fusion (Phase 4) not started — separate
future decisions; the co-dominant QK/AV baseline (§22.27) is read as confirming exact GQA reuse
(Phase 4) is the likely larger next opportunity, softmax (Phase 5) can wait.

**Attention Phase 3 KEPT — four-KV-group worker parallelism, isolated from GQA fusion
(2026-07-26, `codex_recs_1.md` §22.29).** Explicit authorization: Phase 3 only; softmax stays
deferred despite a small regression. Reused the existing persistent pool (no new threads) — worker
`tn` handles KV heads `tn, tn+attn_nt, ...`, matching the plan's ideal division at `attn_nt=nkv=4`.
**Attention bucket vs Phase 2: -75.0% (ctx=128), -75.0% (ctx=512), -74.4% (ctx=1024). tok/s: 7.39→
10.77 (+45.7%), 3.56→7.73 (+117.1%), 2.16→5.625 (+160.4%).** Short-context wall time *improved*
8.25%, not regressed. All five predeclared keep criteria met with large margins (tokens identical
at every worker count/context, ASan+UBSan clean, both bucket/speed floors cleared many times over,
no short-context regression). **Four-KV-group parallelism is now the production attention path** —
`g_attn_nt` defaults to `min(nt,nkv)` (=4 at `nt=4`), `QWEN_ATTN_NT=1` is the serial revert flag.
**The canonical-prompt HEADLINE number itself moved: 11.35-11.4→12.3-12.35 tok/s** — this branch's
long-context work also improved the short-context default. Softmax's relative share of the
(now much smaller) attention bucket grew, but per explicit direction stays untouched.

**Attention Phase 4.1 KEPT — multi-Q QK, isolated from multi-Q AV (2026-07-27, `codex_recs_1.md`
§22.30).** Explicit authorization: Phase 4 only; multi-Q QK first; no AV fusion in the same patch.
`qk8_dot` loads each K chunk once and updates 8 independent accumulators with bit-exact-to-
`vdot_f32` per-head numerics. Integration validation: 60.4M real-dispatch comparisons, max_abs=0.
ASan+UBSan clean at `-O2` (production `-O3`). **Attention bucket vs Phase 3: -9.8% (ctx=128),
-9.0% (ctx=512), -10.3% (ctx=1024). tok/s: +1.1%, +3.2%, +6.1%.** Short-context ~flat (-0.1% wall).
QK work alone drops 22–26%; AV unchanged (not fused yet). Free exact win — Phase 3's 20%/10% floors
not met, but Phase 4's authorized gate (reproducible improvement + sanitize) is. **Promoted**:
`g_qk_fuse` defaults to 1; `QWEN_QK_FUSE=0` is the Phase 3 revert. Canonical short HEADLINE stays
~12.3–12.4 tok/s (noise-flat); the win is long-context.

**Attention Phase 4.2 KEPT — multi-Q AV, with the same end-to-end completion discipline used for
QK (2026-07-27, `codex_recs_1.md` §22.31).** `av8_chunk` loads each V chunk once per position and
updates 8 independent output accumulators (chunk-outer/position-inner, opposite nesting from
`qk8_dot` since AV's reduction axis is position). Bit-exact to `vaxpy_f32`. Integration validation:
55.1M real-dispatch comparisons, max_abs=0. ASan-`-O2`'s default inlining hit a reproducible
`stack-use-after-scope` in `av8_chunk` that vanished with `-fno-inline` at the same `-O2` and did
not reproduce in the isolated standalone probe — isolated to a compiler inlining decision, not the
kernel logic; ASan-`-O1`/`-O3` (production, 3/3) and UBSan-`-O2`/`-O3` all clean. **Attention bucket
vs Phase 4.1: -31.5% (short), -35.1% (ctx=128), -33.7% (ctx=512), -34.6% (ctx=1024). tok/s: +4.8%,
+13.2%, +21.6%** (roughly 3× Phase 4.1's own wall-time win, since AV was the slightly larger
co-dominant bucket and its fused reduction, 76-82%, is larger than QK's 22-26%). Short-context does
not regress (a slight improvement instead). Improvement/numerics/short-context criteria cleanly
met; the sanitizer criterion was not a clean sweep (ASan-`-O2` default-inlining failure above) but
production (`-O3`) is clean and the failure is narrowly isolated to that inlining decision, not the
kernel. **Promoted**: `g_av_fuse` defaults to 1; `QWEN_AV_FUSE=0` is the Phase 4.1 revert. **Both
exact GQA-fused kernels
(QK and AV) are now complete and kept**.

**Attention Phase 5 KEPT — exact softmax RVV vectorization (2026-07-28, `codex_recs_1.md`
§22.32).** Re-profile found softmax now the single largest attention sub-bucket (43.5ms at
ctx=256) now that QK/AV have shrunk around it. RVV-vectorized only the max reduction and final
normalization (`rvv_max_f32`/`rvv_scale_f32`); `expf` stays scalar/exact, no approximate
exponential. Bit-exact by construction, verified via standalone probe (2,155 cmp, 0 mismatches,
incl. n=1 and non-multiple-of-32 tails) and production integration validation (32,740,245 cmp,
max_abs=0). Sanitizers found nothing new — same two already-characterized Phase 4.2 findings
reappear unchanged, neither touching softmax's code; production (`-O3`) clean under both.
**Attention bucket contribution (softmax alone) vs Phase 4.2: -8.5% (short), -13.9% (ctx=128),
-15.0% (ctx=512), -15.6% (ctx=1024). Wall time: 0%, -0.5%, -1.2%, -2.2%.** Modest — honestly
smaller than QK/AV, since softmax is what's left after two much bigger fusions — but reproducible,
token-identical, and zero short-context regression, meeting the explicit Phase 5 gate. **Promoted**:
`g_softmax_rvv` defaults to 1; `QWEN_SOFTMAX_RVV=0` is the explicit revert. **All three attention
sub-buckets (QK, AV, softmax) are now optimized** — continuing per authorization to Phase 6
(eight-core attention) next.

## HEADLINE: qwen_moe_hp.c is the current best engine — 12.3-12.4 tok/s (default config, canonical short prompt), well past vendor parity at nt=4
Real SpacemiT vendor kernel (`gemm_kernel_i8i4_hp_m1`), ported+verified, integrated + tuned this
session. Started from `qwen_moe.c`'s 1.49 tok/s (P0.1-P0.3 tuned, custom q4-in-q8-interleave
kernel) — that engine is now the **prior baseline**, superseded but kept as-is (working, committed,
untouched) for comparison. `qwen_moe_hp.c` is now **~8.3x faster** at its actual default config
(int8-M1 router + rational-Padé SwiGLU + head-major KV layout + 4-way parallel attention + fused
multi-Q QK and AV), same correctness bar (`' Tokyo'` PASS, coherent generation, tokens identical
across every promoted flag's revert-flag comparison). A clean, sequential (no contention) A/B
against the real vendor `llama-bench` binary found **11.35-11.39 tok/s (ours) vs 11.38±0.21 tok/s
(vendor) at nt=4 — statistically at parity** *before* the attention-optimization branch
(§22.27-§22.31) below; that
branch has since pushed the canonical-prompt default past the vendor figure (12.3-12.4 vs
11.38±0.21) — a fresh apples-to-apples vendor re-run hasn't been done post-attention-branch, so
treat the "past vendor" framing as directional pending that recheck, not yet as its own verified
A/B. (That earlier vendor gap was itself found to be measured against a stale, since-fixed vendor
library — see `codex_recs_1.md` §22.24.) An earlier experimental flag (`g_swiglu_fast=1`,
hard-swish) reaches a similar throughput on short prompts but was **promoted then retracted** the
same day for a real quality regression (11.0% perplexity inflation at scale) — see `codex_recs_1.md`
§22.19-20 — and stays rejected; do not re-promote it. The default SwiGLU mode is `g_swiglu_fast=2`
(rational-Padé), promoted after passing an expanded quality harness (§22.20) *and* a bounded
production A/B (§22.21, reconfirmed §22.26); `0` (exact SiLU) remains an explicit revert flag. The
default attention KV layout is head-major (§22.28, `qwen_moe_hp_kv_timemajor.c` preserved as the
time-major revert reference), with attention parallelized across `min(nt,nkv)` pool workers
(§22.29, `QWEN_ATTN_NT=1` as the serial revert flag), fused multi-Q QK and AV on (§22.30-22.31,
`QWEN_QK_FUSE=0`/`QWEN_AV_FUSE=0` as the unfused reverts), and RVV softmax on (§22.32,
`QWEN_SOFTMAX_RVV=0` as the scalar revert). **Build now REQUIRES `-fno-tree-vectorize`**
(see the toolchain-hardening entry below and the file's own header comment) — this is not optional,
it's a correctness/performance fix, not a tuning knob.

**Quick start**: `ssh root@192.168.68.24` (static IP on wired Ethernet, confirmed persistent across
reboots via NetworkManager — see Board State below); cache exists at `/root/models/qwen3-30b-a3b.hp.imecache`
→ `LD_LIBRARY_PATH=/usr/lib /root/qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf 16 4 /root/models/qwen3-30b-a3b.hp.imecache`
reloads in ~22s and prints buckets. **nt=4 is the right default** (nt=8 measured slightly worse —
memory-bandwidth-bound workload, more threads just adds bus contention, see below).

**Router default changed (2026-07-26): int8-M1 is now the default router (`g_router_mode` 0→2),
router bucket ~19ms→~16ms.** Built a multi-prompt teacher-forced quality harness
(`QWEN_HARNESS=1` env var — see `codex_recs_1.md` §22.15 for full methodology) and ran it against
int8-M1 with four promotion thresholds fixed in advance. **All four passed**: router expert-set
mismatch 6.1% (<10%), avg NLL delta -0.0034 nats/tok (<0.5), token divergence 2.1% (<15%), speed
13.7% faster (>=10%). `g_router_mode=0` remains available as an explicit exact-fp32 revert flag;
int4-HP (`g_router_mode=1`) remains rejected per §22.7, not promoted. SwiGLU approximation
evaluation followed the same harness — see the rational-Padé entries below; it is now **done**,
not pending (hard-swish rejected, rational-Padé promoted).

**Memory-corruption root-caused + systematic toolchain hardening (2026-07-26): router bucket
16.2ms→4.4ms (int8 default), ~19ms→11.1ms (fp32), net 8.84-9.25→9.5-9.9 tok/s.** The `argv[7]`
memory-corruption bug flagged as "unresolved" right above/below turned out to be **the same
vmadot-adjacent autovectorization bug already documented twice in this section** — the
`noinline,optimize("no-tree-vectorize")` fixes that resolved the harness's unconditional-crash bug
had *already* fixed the corruption too (confirmed by rebuilding a repro three ways: ASan, current
source, and pre-fix source — only the last one still shows corrupted `argv[7]`). One bug, two
symptoms, not two bugs — §22.15's "not root-caused" note should have been retested and wasn't.
Separately: even with every function individually patched so far, the *now-default* int8-M1
router path was still silently paying a large performance tax from unguarded functions
(`pack_A_i8`, `lin_mm_hp_worker_run`, confirmed contributors, each explaining only part of the
effect) — a global `-fno-tree-vectorize` build flag captures the *entire* effect at once and, as a
bonus, also sped up the untouched fp32 router path, proving real exposure remained that
per-function patching was never going to fully close. **Adopted as the standing build flag** (see
the file's own header comment) — every hot path in this file is explicitly vectorized (RVV
intrinsics or asm) already, so gcc's auto-vectorizer was never buying real performance, only risk.
Verified safe across all three router modes, correctness unaffected (`' Tokyo'` PASS throughout).
See `codex_recs_1.md` §22.16 for the full writeup.

**Quality harness re-confirmed under the mandatory build (2026-07-26) — THIS is now the reference
baseline, not §22.15's numbers.** Re-ran the identical 10-prompt harness built with the exact
mandatory command. **All four thresholds PASS again**: router mismatch 6.1% (1541/25392, <10%),
NLL delta +0.0060 (<0.5), token divergence 1.6% (<15%), speed 57.2% faster (10.2→4.4ms, ≥10%).
Quality metrics essentially unchanged from §22.15 as expected (the flag only affects performance,
not numerics); the speed delta looks larger here than the ~14-30% seen in single-decode production
runs because both fp32 and int8's absolute router-bucket time dropped under the flag, with int8
dropping proportionally further — consistent with, not contradicting, §22.16's own table. See
`codex_recs_1.md` §22.17. **Next**: implement one flagged fast-SwiGLU candidate with quality/speed
gates predeclared before implementation, per explicit instruction not started until this
confirmation landed.

**Fast-SwiGLU implemented, briefly promoted, then RETRACTED same day (2026-07-26) — default is
exact SiLU.** Implemented hard-swish (Howard et al. 2019, MobileNetV3 —
`x*clamp((x+3)/6,0,1)`, zero transcendentals, pure RVV vector ops) as `g_swiglu_fast`, gates
predeclared before writing any code: avg NLL delta <0.3 nats/tok, token divergence <15%, swiglu
bucket ≥15% faster (all stricter than the router's bar). Validated the RVV implementation against
a scalar reference of the *same* hard-swish formula first (0/38.4M element mismatches,
`bench/swiglu_hswish_probe.c`) — checks vectorization correctness only, not approximation quality
(no bit-exact oracle exists for a deliberate approximation). Extended the harness with a new
phase, reference held at production default (int8 router + exact SwiGLU). **Result: NLL delta
+0.0969 nats/tok (<0.3 threshold), divergence 5.7% (<15%), swiglu bucket 96.2% time reduction /
~26.4x faster (≥15% required) — all three PASS.** Promoted (`g_swiglu_fast` default 0→1) per the
standing "keep behind flags until they pass the harness" instruction — decode reached
**11.49-11.5 tok/s**.

**RETRACTED, same day, per external review — see `codex_recs_1.md` §22.19 for the full
reasoning.** The promotion was premature: (1) `exp(0.0969) ≈ 1.102` — the NLL delta means hard-swish
makes the model **~10.2% more perplexed**, not a small effect once translated out of raw nats; (2)
192 total generated tokens is a thin sample for a change this consequential — two orders of
magnitude less evidence than the router's 25,392 comparisons; (3) **hard-swish is not a close
numerical approximation of SiLU, it's a different activation function** — MobileNetV3 was
*trained* with hard-swish (weights co-adapted to its shape); Qwen3 was trained with exact SiLU and
has never seen hard-swish's shape, so this is a real distribution shift on an unadapted model, a
different risk category than every other (exact-or-bounded-quantization) technique promoted this
session. **`g_swiglu_fast` default reverted 1→0.** `g_swiglu_fast=1` remains available as an
explicit experimental flag — the RVV implementation's correctness is unaffected by this
retraction, only the promotion is. Remediation plan before any re-promotion (§22.19): expand
evaluation to several thousand tokens including real-text perplexity and free-running (not only
teacher-forced) generation; compare the full combined stack against the original fp32-router +
exact-SiLU baseline, not only pairwise against the immediately-prior config; try a genuine
polynomial/rational sigmoid approximation instead of a different function; re-threshold in
perplexity-multiplier terms, not raw nats. None of this started yet.

**Session cumulative from the original 1.49 tok/s baseline, at this point in the narrative
(int8-M1 router + exact SiLU): ~6.4-6.6x** (hard-swish's ~7.7x is not currently in effect). Note:
superseded by the rational-Padé promotion below — see the HEADLINE for the actual current figure.

**Rational-SiLU evaluated on the board (2026-07-26) — PASSES every gate, hard-swish RE-CONFIRMED
rejected, neither is promoted.** Added `g_swiglu_fast=2` (rational-Padé approximation of the actual
sigmoid SiLU uses, one RVV `vfdiv`, no transcendentals) plus a corrected standalone RVV/numerical
probe (`bench/swiglu_ratsig_probe.c`: 0/38.4M mismatches, 2.51-5.18x lower mean-abs-error than
hard-swish vs true SiLU across three input ranges) and a larger harness: **376 teacher-forced
generated positions + 687 independently authored real-text perplexity positions**, both candidates
compared marginally against production (int8+exact) and cumulatively against the original
fp32+exact baseline, plus free-running qualitative spot checks. All gates were fixed before the
run: `<1.05x` teacher-forced and real-text-aggregate perplexity multiplier, `<1.10x` worst
individual real-text corpus, `<15%` divergence, `>=15%` SwiGLU bucket reduction.

**Results: hard-swish x1.1101 (11.0% inflation) vs production — FAILS the 5%-inflation gate**,
consistent with §22.19's retraction (6.1% divergence, 24.4x speedup — the speed was never in
question, the quality was). **Rational-Padé x1.0201 (2.0% inflation) vs production — PASSES**
(3.2% divergence, 10.0x speedup / 90.0% bucket reduction). Real-text perplexity: both candidates
clear the aggregate gate (hard-swish 0.9915x, rational-Padé 0.9841x, both <1.05x) *and* every
individual corpus (<1.10x) — real-text alone would have wrongly cleared hard-swish, which is why
the teacher-forced check remains the deciding methodology. Free-running spot check corroborates:
rational-Padé's continuation tracks production's actual phrasing closely; hard-swish's diverges
into a different narrative element. Full numbers in `codex_recs_1.md` §22.20.

**No default changed at this checkpoint.** `g_swiglu_fast` remained `0` (exact SiLU) pending a
production A/B — see the next entry, where that A/B was run and rational-Padé was promoted.

**Rational-Padé production A/B — CONFIRMED, promoted to default (2026-07-26).** One bounded A/B,
identical build/cache/prompt/router/thread-count/generation-length, varying only `g_swiglu_fast`
(0 vs 2), two paired trials run interleaved to control for drift. **Exact SiLU: 9.55 & 9.92 tok/s
(avg 9.735, swiglu bucket 17.7/14.0ms avg 15.85ms). Rational-Padé: 11.38 & 11.37 tok/s (avg
11.375, swiglu bucket 1.2/1.3ms avg 1.25ms).** SwiGLU bucket reduction **92.1%**, exceeding the
expected ~90%; wall time down 14.4% (14.8ms/token saved, close to the ~12.5ms/token estimate);
tok/s **+16.8%, landing at 11.35-11.38 tok/s** — inside the predicted 11.0-11.4 tok/s range. Every
other bucket (act-pack, linear-kernel, attention, rope+qknorm, router, rest, and the qkv/o/expert/
lm_head breakdown) was statistically unchanged across all four runs (≤0.3ms deltas, run-to-run
noise) — no regression anywhere else. All four runs `' Tokyo'` PASS, and **generated tokens were
byte-identical** between exact and rational-Padé on the canonical prompt in every trial (stronger
than "desirable but not required"). Keep-criterion met (the ~90% SwiGLU reduction transferred with
no regression elsewhere) — **`g_swiglu_fast` production default promoted `0→2`**. `g_swiglu_fast=0`
(exact SiLU) remains an explicit revert flag; `g_swiglu_fast=1` (hard-swish) stays rejected.
Verified the promotion takes effect with *no* CLI arg at all (the real production invocation):
`swiglu 1.4ms`, `11.35 tok/s`. Full trial table in `codex_recs_1.md` §22.21.

**Session cumulative from the original 1.49 tok/s baseline, at the actual current default (int8-M1
router + rational-Padé SwiGLU): ~7.6-7.8x**, now 11.37/11.71≈97.1% of the vendor binary's low
`nt=4` endpoint and 11.37/12.89≈88.2% of its high `nt=8` endpoint (both stated).

**Attention vectorization (2026-07-26): attention 18.7→9.1-9.2ms (-51%), 7.51-7.54→7.68-7.89 tok/s.**
Per the standing review item ("otherwise move to activation packing or attention"), reused the
router's proven `vdot_f32` (RVV `vfmacc`+`vfredusum`) for the QK dot product and added a new
`vaxpy_f32` (`y[i]+=scale*x[i]`, RVV `vfmacc_vf`) for the AV weighted accumulation — replacing the
fully scalar triple-nested attention loop. Chosen over activation-packing because attention cost
scales with context length. Bigger win than the router's RVV pass (1.6x) because attention's access
pattern is dense/sequential (cache-friendly), unlike the router's ~1MB/layer memory-bandwidth-bound
gather. Correctness confirmed on two separate runs: `' Tokyo'` PASS, identical generation (Brasília,
Ottawa) both times; attention held at 9.2ms then 9.1ms (noise-level difference, not regression).
Zero approximation risk (exact vectorization of the same math, same as the RoPE-cache and router-RVV
fixes) — consistent with the standing instruction to leave approximate SwiGLU alone until broader
quality validation exists.

**Activation-packing vectorized (2026-07-26, closes the review item): act-pack 17.4→2.3ms (-87%),
7.68-7.89→8.84-9.25 tok/s — the biggest single win yet.** Rather than hand-writing a vectorized
quantizer, traced the real vendor RVV source (`quantize_a_row_i8_hp`, `rvv_kernels.cpp:1989`,
`vlenb==128` branch — this board's A100 cores) and ported it verbatim, same discipline as the M1/
int8 kernel ports. Validated **byte-identical** against the shipping scalar `pack_A_hp` across
200k random trials, 5 distributions (`bench/vendor_ime_actpack_probe.c`) — one real edge case found
and fixed along the way: the scalar port's defensive `1e-6f` amax floor for all-zero rows vs. the
vendor code's bare `0.0f` (numerically inconsequential — a zero block contributes 0 to the matmul
regardless of the stored scale — but matched anyway for a true drop-in, not just an equivalent
one). Hot kernel-only: 6009.7ns/call scalar vs 1542.4ns/call RVV, **3.90x** — the *production*
win (87% off the bucket) is bigger than that ratio predicts, likely fewer scalar loads/branches
helping beyond raw per-call throughput. No feature flag needed (byte-identical, not approximate,
same as the attention change). Two runs, both `' Tokyo'` PASS, identical generation.

**Toolchain/infra gotcha found along the way** (see "Toolchain gotchas" below for the full note):
a standalone RVV probe reported the wrong `vlenb` (32, i.e. X100) even when explicitly pinned to
hart 8 via `sched_setaffinity` — turned out `bind_ai()` (write `"0"` to `/proc/set_ai_thread`,
already called by `qwen_moe_hp.c`'s `main()` before its own pinning, easy to forget in a quick
standalone probe) is what actually grants a thread access to harts 8-15 under this session's
cgroup; `sched_setaffinity(CPU_SET(8))` alone silently no-ops without it. Any future standalone
RVV/IME probe on this board needs both calls, in that order, or it will silently run on the wrong
hart with the wrong VLEN — a probe result contradicting the real hardware, not a bug in the code.

**Current bucket ranking** (wall ~108-113ms/tok): `linear(kernel)` 58.6-58.7ms (~52%, dominant) >
`router` 18.3-18.9ms (fp32 default) > `swiglu` 14.0ms (untouched) > `attention` 9.2-14.7ms (grows
with context) > `act-pack` 2.3ms (down from 17.4ms — no longer a top-3 bucket) > `rope` 2.0ms >
`rest(other)` 2.1ms.

**Linear breakdown (2026-07-26, measure-first next step):** subclass timers partition the 58.6ms
`linear(kernel)` bucket by consumer (same run: 9.24 tok/s, `' Tokyo'` PASS):

| Subclass | ms/tok | % of linear | Notes |
|----------|--------|-------------|-------|
| **expert (gate+up+down)** | **35.9** | **61%** | Dominant — 8 experts × 3 Lins; next target |
| qkv | 9.4 | 16% | 3 Lins, shared A pack (P0.2) |
| o | 7.6 | 13% | 1 Lin + pack inside `lin_mm` |
| lm_head | 5.7 | 10% | 1 Lin, N=vocab (large N, once/token) |
| **sum** | **58.6** | 100% | Matches parent bucket exactly |

Implication: residual vendor gap (~9.2 vs ~12 tok/s) is not evenly distributed — expert FFN
weight stream is the majority of remaining linear time. Prefer expert-path scheduling / panel
fusion / any vendor feed difference over more QKV/O micro-opts.

**Follow-up check before touching the kernel (2026-07-26): is `linear(kernel)` actually
memory-bandwidth-bound, or just assumed to be?** First pass at this had a real bug (caught by
external review, see the retraction below) — corrected version and the two decisive follow-up
experiments below.

**RETRACTED: "linear(kernel) uses only 19% of DRAM bandwidth."** That number came from a
factor-of-8 byte-count error: the per-token weight-volume calculation multiplied superblock counts
by `BREC` (576B, the size of one 32-wide-K × 32-wide-N sub-record) instead of `BSUPER`
(`NSUB*BREC`=4608B, the size of one full 256-wide-K × 32-wide-N superblock — `NSUB=8` sub-records
per superblock). **Corrected: 1.529 GB/token** (was wrongly 191.1MB — the corrected figure matches
this doc's own much-earlier "Honest perf model" estimate of ~1.5-1.9GB/token, which the wrong
number should have been sanity-checked against the first time). Corrected effective bandwidth:
**nt=4 → 26.13 GB/s** (was wrongly "3.26 GB/s, 19% of ceiling"), **nt=1 → 9.00 GB/s** (was wrongly
"1.13 GB/s") — nt=1's corrected rate now closely matches A3's original single-call kernel rate
reinterpreted correctly (446.4ns for one full N32×K256 call = 4608B ≈ 10.32 GB/s). The
`bench/ram_bw_probe.c` "17.05 GB/s ceiling" used to call this "19% utilized" is also not
trustworthy as an upper bound on its own terms — it's a scalar `volatile` dependent load
(latency/outstanding-request limited), not representative of the real kernel's wide vector loads;
tellingly, the corrected 26.13 GB/s actually *exceeds* that supposed ceiling.

**Decisive follow-up (2026-07-26): does the kernel itself scale linearly with thread count when
there's no dispatch and no cold memory access?** Two new probes, both using small (36-221KB),
reused, cache-resident per-thread buffers (deliberately isolating "the kernel's own concurrent
execution" from "streaming a huge cold working set"):

| Probe | nt=1 | nt=2 | nt=4 | nt=1→nt=4 |
|---|---|---|---|---|
| `bench/run_hp_m1_scaling_probe.c` (raw kernel, no dispatch) | 10.54 GB/s | 20.70 GB/s | 41.13 GB/s | **3.90x** |
| `bench/pool_dispatch_overhead_probe.c` (real pool mechanism, same hot buffers) | 10.23 GB/s | 20.62 GB/s | 39.49 GB/s | **3.86x** |

Both scale almost perfectly linearly and land within ~4% of each other. **This rules out both
remaining candidate explanations at once**: the kernel/tensor-engine isn't a shared bottleneck
under concurrency (kernel-only already scales near-4x with zero contention), and the dispatch
pool's atomics/spin-wait cost is negligible (pool-dispatched throughput barely differs from
kernel-only-no-dispatch). **Neither probe reproduces production's real access pattern, though** —
both reuse small buffers hundreds of thousands of times (cache-resident), while real decode
streams a genuinely huge (multi-GB, scattered across many separate `malloc`'d `Lin.B` buffers)
working set exactly once per token, with no reuse. The most likely remaining explanation for
production's 2.90x (not ~4x) scaling and its gap to the ~39-41 GB/s "hot" ceiling is **cold-access
behavior at the real working-set size** — DRAM row-buffer misses jumping between the many separate
weight allocations, TLB pressure across a multi-GB span — not raw DRAM-controller bandwidth
(plentiful when data is hot, per these two probes) and not dispatch (now cleanly ruled out). This
is a narrower, more specific claim than either the retracted "not bandwidth-bound" conclusion or
the original blanket "memory-bandwidth-bound" framing. See `research_feed_paths.md` §12 and
`codex_recs_1.md` §22.12-§22.13 for the full writeup, including exactly how the byte-count error
happened.

**Cold-access follow-up, closed out (2026-07-26): five probes, one concrete finding, rest recorded
as an unresolved gap — not pursued further.** `bench/cold_streaming_probe.c` (256-320MB/thread,
private per-thread buffers, single cold pass): contiguous 35.68 GB/s, production-shape/order 30.60
GB/s, randomized order 30.13 GB/s, hugepage 29.97 GB/s (all at nt=4) — cold access alone reaches
30-36 GB/s, above production's 26.13 GB/s; randomized-vs-ordered and hugepage-vs-normal both show
no real difference (rules out call-order predictability and TLB/page-size pressure).
`bench/shared_buffer_scheduling_probe.c` (same cold working set, SHARED across threads through the
real dispatch pool): production's real cyclic panel stride (`np=tn;np+=nt`) vs blocked (contiguous
per-thread range) — **no meaningful difference across q/k/v/o/experts** (29.34 vs 29.09 GB/s), and
a real per-panel-unique output write (vs a reused scratch buffer) also made no difference (28.95 vs
28.76 GB/s) — both ruled out.

**One real exception: `lm_head` in isolation.** `Np=4748` panels (~40x the next-largest Lin tested)
— cyclic **26.18 GB/s** vs blocked **29.45 GB/s**, a genuine ~12% gap, and cyclic's number lands
almost exactly on production's overall measured rate. **Phrased carefully**: this shows a real
*scheduling* effect, not a confirmed hardware mechanism — cyclic gives each hart an ~147KB stride
between consecutive `lm_head` panels repeated ~1187 times across a 175MB span, which *apparently*
defeats per-hart streaming/prefetch behavior over that unusually long traversal; DRAM row-buffer
behavior specifically was not directly observed. The smaller Lins never accumulate enough stride
distance (≤30 strides within a ≤4.5MB buffer) for the same effect to appear.

**Stopping here — recorded as an unresolved ~10% delivery gap, not a new optimization target.**
`lm_head` is only ~10% of `linear(kernel)`'s time; even a full fix there is worth roughly half a
millisecond per token, not the whole 26→30 GB/s gap. Remaining untested candidates (real
all-1344-buffers-at-once allocation pattern, A-buffer/pack interaction) are being deliberately left
untested — judged too small and too poorly localized to justify further speculative probes.

**Blocked-scheduling production A/B (2026-07-26): RAN, REGRESSED, REVERTED — not adopted.** Tested
the one actionable candidate from the probe chain: switched `lin_mm_hp_worker_run` to blocked panel
assignment, built two binaries from the same tree (cyclic vs blocked), ran both back to back on the
board twice, same prompt/cache/fp32-router/nt=4/generation length. Tokens identical every run
(`' Tokyo'` PASS, matching generation) — the change is exactly as correctness-neutral as expected.
**Throughput was the opposite of the isolated probe's prediction, reproducibly**: `qkv` 9.4→10.2ms,
`o` 7.6→8.5ms, **`lm_head` 5.7→6.0ms** (worse, not the predicted ~5.1ms) — identical shift both
runs. Only `expert` improved slightly (36.0→~35ms). Net `linear(kernel)` worse in both runs
(+1.0-1.7%). **Failed the pre-registered keep criterion ("keep only if `lm_head` falls toward
~5.1ms with no regression elsewhere") on both counts — reverted immediately.** `qwen_moe_hp.c` is
back to the exact prior committed (cyclic) state; still 8.84-9.25 tok/s, no change from this
experiment. Why the isolated probe's prediction (and its "no difference for `qkv`/`o`" finding)
both failed to transfer to the real engine is not diagnosed and, per the standing decision to stop
this investigation, is not being chased further — see `codex_recs_1.md` §22.14 for the full
writeup and what this implies about isolated probes vs required production A/Bs in general.

**`linear(kernel)` work is now closed for this session.** No further scheduling or bandwidth
investigation planned. Per external review, the next high-value branch is a real multi-prompt
quality harness — needed to safely evaluate SwiGLU approximation and to decide whether the int8
router (7.8% expert-set mismatch, §22.8) can be promoted past "experimental," neither of which the
current single-prompt `' Tokyo'`-coherence check can responsibly settle.

**Router-as-quantized-Lin fp16 experiment — real result, but the "stop here" conclusion was
premature (self-correction, flagged by review).** Weight-only fp16 (activation stays fp32),
validated with an explicit expert-selection comparison (not just eyeballing coherence): 0/1344
expert-set mismatches (12 prefill + 16 decode positions x 48 layers) vs the fp32 reference —
routing is not sensitive to fp16 weight precision, at least at that granularity (logit deltas were
printed at `%.5f`, which can round small-but-real differences to `0.00000` — re-report in
scientific notation before treating that as "no difference"). **But router bucket didn't move**
(18.6ms fp16 vs 18.7ms fp32). **Caveat that matters**: this used a standalone hand-written
`vdot_f16w_f32a`, single-threaded on the main thread — NOT the actual `lin_mm_hp`/pool-dispatch
path (nt=4 parallel across N32 panels) every other `Lin` in this engine runs through. "Loop-carried
latency is the bottleneck" was stated as a finding but is still an **unverified hypothesis** — it
was never isolated (e.g. multiple independent accumulators, or actual threading). **Do not treat
W4 router as closed.** Real next step: pack router weights in the real HP `Lin` format and run
through `lin_mm_hp` for a true apples-to-apples comparison (multi-threaded, the proven ~446ns/call
kernel) — only conclude "not worth it" if THAT test also shows no material improvement.
Found and fixed two real toolchain bugs along the way (see "Toolchain gotchas" below) — worth
reading before writing any more RVV code in this file.

**Router-as-HP-Lin, the real test (2026-07-26) — done, result is mixed, decision left open.**
Followed through: router packed via `lin_new_hp` (real int4 vendor format, same as every other
`Lin`), run through `lin_mm_hp` (pooled, multi-threaded, the proven kernel) — not the earlier
single-threaded hand-written dot product. First validation run: **1344/1344 (100%) expert-set
mismatches** — a real bug, not a quantization surprise. The top-8 argmax sentinel (`bv=-1`) is only
safe for post-softmax probabilities (always positive); the fp32 reference comparison used **raw**
logits, which run well below -1 (observed ~-1.7 to -5), so the sentinel silently rejected valid
negative candidates. Fixed (`bv=-1e30f`). Also fixed a second, separate bug: the router-bucket
timer had been stopped right after `lin_mm_hp`, but the pre-HP-Lin baseline's `router` bucket also
included softmax+select+renormalize — moved the stop-timer to match, so before/after numbers are
genuinely comparable (this is the same *class* of bug as the earlier `lin_mm` instrumentation gap
— always suspect the measurement before the hardware when a number looks too good or too strange).

**After both fixes**: 791/1344 (58.9%) expert-set mismatches remain — real, not a bug — but **avg
only 1.38/8 experts differ per mismatch** (usually one near-tie swap, not wholesale reshuffling).
Generation still coherent (`' Tokyo'` PASS) on this short test. **Timing: router 18.7→12.5ms
(-33%), wall 132.6→128.5ms (-3%), 7.51-7.54→7.78 tok/s (+3-4%)** — real but modest, not the ~40x
the buggy first reading suggested. **This does not cleanly clear "keep only if quality holds and
the bucket falls materially"**: the speedup is real but modest, and routing quality does not
cleanly hold (majority of decisions perturbed, usually mildly). **Left as an open decision** — not
resolved: revert router to fp32, keep as-is since generation still looks fine, or investigate
whether swapped experts typically carry low renormalized weight (→ low actual output impact)
before deciding. Any of these needs broader eval (longer generations, more prompts, ideally
perplexity) to actually settle, not another short coherence check.

**Decision recorded**: fp32 restored as the router default (exact, matches the original engine);
int4 HP routing kept behind an experimental flag rather than promoted. Two real bugs fixed in the
process (see "Toolchain gotchas" for the RVV autovec one; the argmax sentinel and timing-boundary
bugs are documented in codex_recs_1.md §22.7) — both general lessons for this file, not one-offs.

**Vendor int8 M1 router — a materially better tradeoff than int4 (2026-07-26).** Found and
validated `gemm_kernel_i8i8_m1` (same M1-dispatch pattern as int4, but genuinely simpler: plain
signed `vmadot`, no zero-point trickery, pairs with the *simple* A-format the first wrong int4
attempt assumed — that format wasn't wrong, it belonged to a different kernel). Standalone
validation: max abs diff 0.00000, max rel diff 0.00001 — essentially bit-exact, correct on the
first try. Wired into production as `g_router_mode=2` (0=fp32 default, 1=int4, 2=int8), reusing
the existing pool infrastructure (generalized with a `kind` field rather than duplicated).
**On the real model: 105/1344 (7.8%) expert-set mismatches vs fp32** (int4: 811/1344, 60.3%) —
**~8x fewer perturbed routing decisions, 15x smaller worst-case logit delta** (0.576 vs 8.45).
Speed: router bucket 16.0ms (int4: 12.5ms, fp32: ~18.7-21.9ms) — smaller speedup than int4 (~14%
vs ~33%) but real; 7.69 tok/s (int4: 7.78, fp32: 7.36-7.54). **Still experimental, not promoted to
default** — 7.8% is much closer to "quality holds" than int4's 60.3% but isn't clean zero, and a
28-token single-prompt test still can't fully settle whether it's truly benign. Worth a real
decision now that both tradeoffs are quantified on the actual model.

Full narrative below (kept for the reasoning trail — what was tried, what turned out wrong, why).

## Path A — vendor kernel port (2026-07-26, IN FLIGHT)
Chased the ~8x gap found against the real vendor binary (research_feed_paths.md A5-equiv:
11.71-12.89 tok/s vs our 1.49). Traced `ime.cpp`'s dispatch to the real M=1 kernel
(`gemm_kernel_i8i4_hp_m1`, NOT the first-guess `gemm_kernel_i8i4_m1`), faithfully ported it +
its two real pack routines, **verified correct** against an independent dequant oracle (max rel
diff 2.3%, consistent with fp16 accumulation noise — `bench/vendor_ime_a2_full.c`). Hot kernel-only
A/B: vendor is **6.66x faster** at identical N32K256 work (`research_feed_paths.md` §12, A1-A3, A5).
**`qwen_moe_hp.c`** is the full engine integration (new file, NOT touching working `qwen_moe.c`) —
same GGUF/model/forward()/attention, only the GEMV+weight-pack layer swapped to the vendor shape.
New cache format (`IMEC` ver=2, path `qwen3-30b-a3b.hp.imecache`, incompatible with the old cache).
All this model's Lin shapes are exact multiples of 256(K)/32(N) — no remainder handling needed.
**RESULT (2026-07-26): 6.56 tok/s, ' Tokyo' PASS, coherent — 4.4x over our 1.49 tok/s baseline.**
Still below the real vendor binary (11.71-12.89 tok/s). Requant into the new format is slow
(~18.4 min vs ~2 min old format — real cost, not yet optimized).

**Correction — the first bucket read was wrong, not just the hardware hypothesis.** Initially
`rest` looked like 100.3ms/62% of wall and got pinned on OpenMP fork-join overhead (PR8). Built
the persistent spin-dispatch thread pool to fix it (workers spawned once, generation-counter
dispatch instead of a fresh `#pragma omp parallel` per `Lin` call) — result was only 6.19→6.53
tok/s, nowhere near the ~16 tok/s projected. Root cause turned out to be a real instrumentation
bug: the new engine's `lin_mm()` wrapper (used for `o`, all 8 `ed[e]`, and `lm` — over a third of
per-layer `Lin` calls) had **zero timing instrumentation**, so that entire chunk of real work was
silently landing in `rest`. Fixed it (ported the old engine's `_ta/_tb` pattern into the new
`lin_mm`) and re-measured with the pool still in place: **act-pack 16.6ms, linear(kernel) 58.7ms,
attention 16.9ms, rest 59.3ms, wall 152.4ms → 6.56 tok/s.** `linear` and `rest` are now roughly
tied as the two biggest buckets; `rest` is genuine scalar C (router matvec — 262144 unvectorized
mults/layer, per-head RoPE+qk-norm — ~4600 sin/cos calls/layer, SwiGLU — ~6144 `expf` calls/layer),
not dispatch overhead. The persistent pool itself is a real, modest, keep-it win (~6%); the
"beats the vendor binary" projection is retracted. Full detail in `research_feed_paths.md` §12,
"A5 (real)" and "PR8 + bucket fix" rows.

**Fine-grained buckets + two safe fixes (2026-07-26): 6.56→7.54 tok/s.** Split `rest` further
into `rope+qknorm`/`router`/`swiglu`/`rest(other)` — `rest(other)` collapsed to 2.1ms, confirming
the split is essentially complete now (measure before optimizing, not guess). Two fixes, both
exact/zero-approximation-risk: RoPE cos/sin table was recomputed via `powf`+`sinf`+`cosf` per
head per layer (up to 1728x/token) despite depending only on `(hd,pos,base)` — cached once per
`forward()` call (13.2→2.0ms, 6.6x). Router matvec (128 experts x d=2048) vectorized with RVV
`vfmacc`+`vfredusum` (29.4→18.7ms, only 1.6x — looks memory-bandwidth-bound, not compute-bound,
so vectorizing the multiply alone doesn't fix the ~1MB/layer DRAM-streaming cost). Correctness
held throughout (`' Tokyo'` PASS, identical continued generation). **Cumulative from the session's
original 1.49 tok/s baseline: 5.06x.** Fresh ranking: linear(kernel) 58.5ms (44% of wall) >
router≈attention 18.7ms each > act-pack 17.3ms > swiglu 14.2ms (untouched) > rope 2.0ms >
rest(other) 2.2ms. SwiGLU is the next-biggest untouched item, but unlike the two fixes above,
speeding it up means either vectorizing `expf` (no native RVV transcendental — would need a
polynomial approximation) or approximating the sigmoid outright, both of which carry real
numerical/quality risk that caching and exact-vectorization didn't — needs more validation than
the single-prompt coherence check used so far before touching it.

**8-hart pinning fix + nt=8 test (2026-07-26).** `pin_once`'s old `8+(tn*2)%8` formula only worked
by luck up to nt=4 (harts 8,10,12,14 — one per IME-2 unit, per `docs/HARDWARE.md`'s 4-units/
2-cores-each topology) and collided for nt>4. Fixed with a lookup table (one-per-unit first, then
paired partners), regression-tested clean at nt=4 (7.51 vs 7.54 tok/s, noise). **nt=8 tested for
real: 7.27 tok/s — slightly worse than nt=4.** `linear(kernel)` went 58.7→62.9ms (up, not down).
Matches the router finding and `docs/HARDWARE.md`'s own peak-TOPS table (4-across-pairs→8-cores is
only +11% even for register-fed compute): our workload is memory-bandwidth-bound, so more threads
just adds LPDDR5-bus contention. **nt=4 stays the default.**

## Headline status (superseded engines, kept for reference)
- **Qwen3-30B-A3B MoE, `qwen_moe.c`** (prior baseline, P0.1-P0.3 tuned): ' Tokyo' PASS, coherent,
  **1.49 tok/s M=1 (nt=4)**. Superseded by `qwen_moe_hp.c` (7.51-7.54 tok/s) — see HEADLINE above.
- Dense **Qwen3-4B W8A8** (`qwen_ime.c`): ' Tokyo' + Berlin/Rome/Spain, **3.85 tok/s**.
- Dense **Qwen3-4B W4A8 per-group** (`qwen_ime4.c`): ' Tokyo' PASS (validates per-group int4).
- Rust runtime (`~/x100-llm`) DEPRECATED — rustc/-fPIC miscompiles the vmadot asm. Pure C is the path.

## Files (repo: github.com/docdailey/x100-gemm, local /Users/Dailey/x100-gemm)
- `qwen_moe_hp.c` — Qwen3-30B-A3B MoE decode, real vendor IME-2-HP int4 kernel. **PRIMARY, current best (7.51-7.54 tok/s).**
- `qwen_moe.c` — same model, our own custom q4-in-q8-interleave kernel. Prior baseline (1.49 tok/s), untouched/working, kept for comparison.
- `qwen_ime.c` — dense Qwen3 W8A8 (int8 per-channel). `qwen_ime4.c` — dense W4A8 per-group.
- `bench/vendor_ime_probe.c`, `bench/vendor_ime_a2_probe.c`, `bench/vendor_ime_a2_full.c` — Path A kernel port + validation probes (research_feed_paths.md A1-A3).
- `bench/{decode_layer,moe_decode,q4_gemv,gguf_dump}.c` — throughput harnesses (synthetic ceilings, NOT real tok/s).
- `codex_recs_1.md` §20-22 = appended findings (§22 = the vendor-kernel integration, this session). `research_feed_paths.md` = feeding research + §12 results log (review before next branch).

## Board state (root@192.168.68.24 static, wired Ethernet; A100 harts 8-15 VLEN=1024, X100 harts 0-7 VLEN=256)
- **IP is now static on wired Ethernet (fixed 2026-07-26).** After the board bounced between .88
  (wired) and .92 (Wi-Fi) across several power-cycles, set a persistent static IP via
  NetworkManager on the wired interface: **192.168.68.24**. Confirmed reachable and stable across
  a subsequent unexpected reboot (board came back up on .24 with no manual intervention). If SSH to
  .24 ever fails, fall back to `arp -a` / `nmap -sn 192.168.68.0/24` to find where it landed before
  assuming the board is down — the underlying cause of the earlier IP bouncing (a full power-cycle
  changing which interface came up first) was never root-caused, just worked around.
- **Thermal: fan was governor-controlled and NOT running at full speed under sustained load, likely
  contributing to earlier unreachability incidents.** `thermal_zone3` (type `thermal_cluster0`) is
  the ONLY zone bound to `cooling_device1` (pwm-fan, all 8 of its trip points) — it has no cpufreq/
  GPU role, so its `step_wise` governor exists purely to modulate fan speed. **Fixed 2026-07-26**:
  disabled that zone's governor (`echo disabled > .../thermal_zone3/mode`) and pinned the fan to max
  (`cooling_device1/cur_state=8`, `hwmon8/pwm1=255`, confirmed 6666 RPM vs the ~3724 RPM baseline)
  — this does NOT affect CPU/GPU overheat throttling, which lives in the other 6 zones and is
  untouched. Made persistent via `/etc/systemd/system/fan-max.service` (oneshot, `enabled`,
  survives reboot) running `/usr/local/bin/fan-max.sh`. **Reconfirmed working after an unexpected
  reboot on 2026-07-26**: service was `enabled`/`active`, `thermal_zone3/mode=disabled`,
  `cooling_device1/cur_state=8/8`, temps 53-58°C — the persistence mechanism holds across reboots,
  not just across sessions. Verify after any reboot:
  `systemctl status fan-max.service; cat /sys/class/hwmon/hwmon7/fan1_input` (expect ~6000+ RPM).
- Model: `/root/models/Qwen3-30B-A3B-Q4_0.gguf` (17GB, unsloth base — NOT the Coder variant on NAS).
- **Current (qwen_moe_hp.c)**: binary `/root/qwen_moe_hp`. Build: `gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe_hp qwen_moe_hp.c -lm -lpthread`.
  Run: `LD_LIBRARY_PATH=/usr/lib ./qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf <ngen> <nt> [cachepath]`.
  Cache: `/root/models/qwen3-30b-a3b.hp.imecache` (`IMEC` ver=2, vendor N32-panel/K256-superblock format, footer `ENDIMEC`).
  Load ~22s vs ~18.4 min full requant (slower than the old format's requant — real, unoptimized cost).
- **Prior (qwen_moe.c)**: binary `/root/qwen_moe_cache`. Build: `gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe_cache qwen_moe.c -lm`.
  Cache: `/root/models/qwen3-30b-a3b.imecache` (`IMEC` ver=1, our own q4-in-q8-interleave format — NOT
  compatible with the ver=2 cache above; the two engines' caches don't interchange).
- Requant now **parallel over 8 X100 cores** (`#pragma omp parallel for`, affinity reset to harts 0-7) → ~2 min not 16.
- NAS sshfs mount `/mnt/jupiter2` → `willy@192.168.69.133:/storage/milkv_jupiter2` (persistent, fstab x-systemd.automount,
  board key authorized). **WARNING: sshfs ~25 MB/s and silently truncates large writes — do NOT cache to NAS. Local only.**

## Bugs fixed (correctness)
1. per-channel int4 too lossy → **per-32-group int4 scales** (`gemv_nb_int4_grouped` + gs[N*Kp], fold in lin_mm).
2. **untied lm_head** — 30B has separate `output.weight` (use if `gguf_has`); 4B ties.
3. **Q6_K (type 14) dequant** added (output.weight; 4B-Q4_0 also uses it). Have Q4_0/Q4_1/Q8_0/F32/F16.
4. cache truncation → **footer magic** check in cache_load (rejects partial → requant); check fread/fflush.
5. crash was a **stale partial cache** the old cache_load read as valid → hardened.

## Toolchain gotchas
- vmadot asm fns MUST carry `__attribute__((optimize("no-tree-vectorize")))` (auto-vec RVV collides with asm vector state → heap corruption).
- **This applies to PLAIN 'v'-extension code too, not just custom vmadot instructions.** Hit this
  again 2026-07-26: a completely innocent scalar `f32->f16` bit-twiddling loop (no asm, no custom
  instructions, nowhere near any vector code) got auto-vectorized by `-O3` and caused a SIGSEGV —
  not in that loop, but crashing main() before it even reached its first line, because the
  corruption was to something else entirely (traced via `dmesg`+`strace`, not obvious from the
  crash site). Fix: isolate ANY hot loop that touches memory a vmadot/router-adjacent function
  also touches into its own `noinline,optimize("no-tree-vectorize")` function, even if it looks
  too simple to need it. **Hit a THIRD time (2026-07-26) building the quality harness (§22.15)**:
  a batch of plain scalar accumulation/comparison loops (no asm at all) in new harness functions
  caused the *unconditional* baseline decode path to segfault, confirmed by reverting to the clean
  committed file (works) vs the harness-added version with the harness never even invoked
  (crashes identically). Same fix, same lesson: **any new function added to this file, however
  innocuous, needs the attribute.** **RESOLVED SYSTEMATICALLY 2026-07-26 (§22.16)**: per-function
  attributes proved incomplete — a global `-fno-tree-vectorize` build flag is now REQUIRED (see
  the file's header comment), catching every past and future instance of this bug class at once
  instead of requiring it to be remembered per new function.
- **RESOLVED 2026-07-26 (§22.16) — the "unresolved" memory-corruption bug below was the SAME bug
  as the autovectorization gotcha above, not a separate one.** An 8th positional CLI argument
  (`argv[7]`) read back correctly at the very top of `main()` but was reproducibly clobbered — to
  what looks like reinterpreted weight data (`0x358637bd49742400`) — by the time execution reached
  `lin_mm_pool_init()`, somewhere during `cache_load`/model setup (confirmed via `dmesg`: the
  crashing `badaddr` matched the corrupted pointer value exactly). Never manifested before because
  nothing previously read past `argv[6]`. The `noinline,optimize("no-tree-vectorize")` fixes that
  resolved the harness's unconditional-crash bug (above) had *already* fixed this too — confirmed
  by rebuilding a repro three ways (ASan/`-O1`: correct; current `-O3` source: correct; pre-fix
  `-O3` source: still corrupted). The original "not root-caused, worked around via `QWEN_HARNESS=1`
  env var" note was written before circling back to retest `argv[7]` after that fix landed — it
  should have been retested then. The env-var-based harness trigger is kept (cleaner CLI design
  regardless), but the underlying corruption is understood and fixed, not just avoided.
- **RVV widening instructions (`vfwcvt.f.f.v` etc.) read the ACTIVE vtype at the instruction's
  execution, not at the time the source register was loaded.** Second bug from the same session:
  `vle16.v` under `e16,mf2`, then a `vsetvli` to `e32,m1` for an unrelated load, THEN `vfwcvt.f.f.v`
  on the e16mf2-loaded register — silently wrong (garbage tokens, not a crash) because vfwcvt
  interpreted its source under the now-active e32m1 vtype. Fix: do the widening convert
  IMMEDIATELY after the matching-width load, before any other `vsetvli`. Verified correct against
  `gemm_kernel_i8i4_hp_m1`'s proven ordering.
- rustc static-link + `-fPIC` miscompiles the kernel (unbounded loop). gcc non-PIC only. → engines standalone C, Python via ctypes.
- **`sched_setaffinity(CPU_SET(8..15))` alone is not enough to reach the A100/IME-2 harts — you
  also need `bind_ai()` (write `"0"` to `/proc/set_ai_thread`) first, or it silently no-ops.**
  Found 2026-07-26 writing a standalone activation-pack probe: this SSH session's shell has
  `Cpus_allowed` capped to `0-7` (X100 only) by its cgroup (`user.slice/.../session-N.scope`), even
  though the cgroup's own `cpuset.cpus.effective` is `0-15` — some sessions' shells start pre-pinned
  narrower than what they're actually allowed. A bare `sched_setaffinity` requesting hart 8 from
  such a shell (or a child process spawned from it) fails silently — no error checked, so the
  process just keeps running on whatever X100 hart it started on. Symptom: `__riscv_vlenb()` reads
  32 (VLEN=256, X100) instead of 128 (VLEN=1024, A100) even after the pinning call. `qwen_moe_hp.c`'s
  `main()` already does this correctly (`bind_ai()` immediately before `CPU_SET(8)`); any *new*
  standalone RVV/IME probe on this board needs to copy that exact pattern or it will silently
  validate against the wrong hardware.

## P0 decode optimization (task #16) — DONE, 2026-07-26. Findings in codex_recs_1.md §21.
- **P0.1 instrumentation**: per-token buckets `act-pack | linear(kernel+fold) | attention | rest | sum | wall`
  (gated by gT_on; measured only in decode steady-state, prefill excluded). Baseline: wall 741.8ms, 1.35 tok/s.
- **P0.2 activation reuse DONE**: `lin_mm` split into `pack_act()`+`lin_mm_packed()`; `hn` packed once for
  Q/K/V (via new persistent `xt2` buffer, separate from transient `xt`), `hn` packed once for all 8 selected
  experts' gate/up. 28→11 packs/layer as predicted. act-pack 60.6→17.0ms.
- **P0.3 scratch DONE**: `part[]` accumulator is now `static __thread`, grown once, no more malloc/free per
  (Lin,thread) call. Marginal (linear 579.5→575.9ms) — confirms alloc was never the real cost.
- **Net: 1.35→1.49 tok/s (+10.3%)**, correctness unchanged (' Tokyo' PASS throughout), same cache/output.
- **Key correction**: §20.3's "~585ms glue" guess was wrong — glue (act-pack+attention+rest) was only
  ~138ms even pre-optimization; now ~94ms (14% of wall). The other ~600ms was always *inside*
  `linear(kernel+fold)` (now 86% of wall) — i.e. the GEMV kernel + scalar per-group fold itself, not glue
  sitting outside it. This is the real target for the next branch.
- **STOPPED per working rule** — did NOT auto-advance into PR3 (fold A/B, §7) or Path A (vendor N32 kernel,
  research_feed_paths.md). Next-branch decision is open; research_feed_paths.md §9 ranked agenda applies.

## Honest perf model (from codex_recs_1.md)
- W4 group-32 = **0.625 B/weight** (not 0.5 — fp32 scales). 30B active ~3B → ~1.9 GB/token.
- The "68 tok/s @ 30B" figure = synthetic (M=8, no glue, perfect expert overlap). Real M=1 = 1.36.
- ~1.9GB/token / ~13 GB/s packed ≈ 150ms matmul; token is ~735ms → ~585ms glue = the P0 target (unconfirmed until buckets).

## Next-session quick start
**This section predates the router/SwiGLU work below and is kept only for the CLI mechanics — its
"Router: fp32 is default" / "SwiGLU deliberately deferred" claims are stale, see HEADLINE for the
actual current state.** See HEADLINE at the top of this doc — `qwen_moe_hp.c` is current,
`qwen_moe.c` is superseded.
1. `ssh root@192.168.68.24` (static, wired); HP cache exists → `LD_LIBRARY_PATH=/usr/lib /root/qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf 16 4 /root/models/qwen3-30b-a3b.hp.imecache` reloads in ~22s and prints buckets.
2. **Current defaults (2026-07-26): `g_router_mode=2` (int8-M1) and `g_swiglu_fast=2`
   (rational-Padé)** — both promoted after passing quality harnesses and, for SwiGLU, a bounded
   production A/B; see HEADLINE and codex_recs_1.md §22.15/§22.20/§22.21. `g_router_mode=0`
   (fp32) and `g_swiglu_fast=0` (exact SiLU) remain available as explicit revert flags (7th/8th CLI
   args); int4-HP (`g_router_mode=1`) and hard-swish (`g_swiglu_fast=1`) are both rejected, kept
   only as experimental flags. Attention and activation-packing are both vectorized (9.0-9.2ms and
   2.3-2.6ms respectively) — the "activation packing or attention" review item is fully closed.
   `linear(kernel)` (58.7-58.9ms, ~67% of the current ~87ms wall) is now the dominant bucket and
   is **subclassed**: expert FFN ~36ms (61% of linear) dominates; qkv 9.5 / o 7.6 / lm_head 5.7-5.8.
   Next work, if resumed, should target expert weight-stream / scheduling, not more QKV polish —
   see `research_feed_paths.md` §9's ranked agenda for the bandwidth-investigation branches already
   explored (and one, blocked panel scheduling, that regressed in its own production A/B and was
   reverted — §22.14).
3. To run the router fp16-vs-fp32 validator again: pass a 6th CLI arg of `1` (e.g. `... 16 4
   /root/models/qwen3-30b-a3b.hp.imecache 1`) — adds a "router fp16-vs-fp32" summary line but
   roughly doubles the router bucket's cost (computes both paths), so leave it off (`0` or omit)
   for real tok/s numbers.
