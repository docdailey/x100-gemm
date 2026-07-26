# PROGRESS — K3 pure-C IME-2 decode engine (session log, live)

Last update: 2026-07-26. Durable state so work survives a session kill.

## HEADLINE: qwen_moe_hp.c is the current best engine — 8.84-9.25 tok/s
Real SpacemiT vendor kernel (`gemm_kernel_i8i4_hp_m1`), ported+verified, integrated + tuned this
session. Started from `qwen_moe.c`'s 1.49 tok/s (P0.1-P0.3 tuned, custom q4-in-q8-interleave
kernel) — that engine is now the **prior baseline**, superseded but kept as-is (working, committed,
untouched) for comparison. `qwen_moe_hp.c` is now **~6.1x faster**, same correctness bar (`' Tokyo'`
PASS, coherent generation), closing in on the real vendor *binary*'s 11.71-12.89 tok/s.

**Quick start**: `ssh root@192.168.68.24` (static IP on wired Ethernet, confirmed persistent across
reboots via NetworkManager — see Board State below); cache exists at `/root/models/qwen3-30b-a3b.hp.imecache`
→ `LD_LIBRARY_PATH=/usr/lib /root/qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf 16 4 /root/models/qwen3-30b-a3b.hp.imecache`
reloads in ~22s and prints buckets. **nt=4 is the right default** (nt=8 measured slightly worse —
memory-bandwidth-bound workload, more threads just adds bus contention, see below).

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

**One actionable, NOT YET APPLIED candidate**: switch `lin_mm_hp_worker_run`'s panel assignment
from cyclic to blocked — for `lm_head` specifically, or as the new general default (blocked did not
regress any smaller Lin in the probe, and is simpler code). This needs a real production A/B before
adopting — expect `lm_head`'s bucket (currently 5.7ms) dropping toward ~5.1ms with no regression
elsewhere, if the probe result holds. See `codex_recs_1.md` §22.13 for the full five-probe chain.

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
  too simple to need it.
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
See HEADLINE at the top of this doc — `qwen_moe_hp.c` is current, `qwen_moe.c` is superseded.
1. `ssh root@192.168.68.24` (static, wired); HP cache exists → `LD_LIBRARY_PATH=/usr/lib /root/qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf 16 4 /root/models/qwen3-30b-a3b.hp.imecache` reloads in ~22s and prints buckets.
2. Router: fp32 is the production default; int4-HP and int8-M1 are both real, validated, and kept
   behind the `g_router_mode` experimental flag (7th CLI arg) — see HEADLINE and codex_recs_1.md
   §22.7-22.8 for the full quality-vs-speed tradeoff on each. Attention and activation-packing are
   both now vectorized (9.2-14.7ms and 2.3ms respectively, see HEADLINE) — the "activation packing
   or attention" review item is fully closed. SwiGLU (14.0ms) is now the single largest
   fully-untouched bucket, deliberately deferred — needs quality validation beyond the
   single-prompt coherence check before approximating `expf`/sigmoid. `linear(kernel)` (58.6ms,
   ~52% of wall) is now **subclassed**: expert FFN 35.9ms (61% of linear) dominates; qkv 9.4 /
   o 7.6 / lm_head 5.7. Next work should target expert weight-stream / scheduling, not more QKV
   polish. Instrumentation is in `qwen_moe_hp.c` (uncommitted at handoff).
3. To run the router fp16-vs-fp32 validator again: pass a 6th CLI arg of `1` (e.g. `... 16 4
   /root/models/qwen3-30b-a3b.hp.imecache 1`) — adds a "router fp16-vs-fp32" summary line but
   roughly doubles the router bucket's cost (computes both paths), so leave it off (`0` or omit)
   for real tok/s numbers.
