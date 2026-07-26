# PROGRESS — K3 pure-C IME-2 decode engine (session log, live)

Last update: 2026-07-26. Durable state so work survives a session kill.

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

## Headline status
- **Qwen3-30B-A3B MoE runs COHERENT on the K3 IME-2**, pure C: prompt → ' Tokyo' PASS;
  "…Tokyo. The capital of Brazil is Brasília. The capital of Canada is Ottawa." **1.36 tok/s M=1 (nt=4)**.
- Dense **Qwen3-4B W8A8** (`qwen_ime.c`): ' Tokyo' + Berlin/Rome/Spain, **3.85 tok/s**.
- Dense **Qwen3-4B W4A8 per-group** (`qwen_ime4.c`): ' Tokyo' PASS (validates per-group int4).
- Rust runtime (`~/x100-llm`) DEPRECATED — rustc/-fPIC miscompiles the vmadot asm. Pure C is the path.

## Files (repo: github.com/docdailey/x100-gemm, local /Users/Dailey/x100-gemm)
- `qwen_moe.c` — Qwen3-30B-A3B MoE decode, W4A8, mmap GGUF, requant cache. **PRIMARY.**
- `qwen_ime.c` — dense Qwen3 W8A8 (int8 per-channel). `qwen_ime4.c` — dense W4A8 per-group.
- `bench/{decode_layer,moe_decode,q4_gemv,gguf_dump}.c` — throughput harnesses (synthetic ceilings, NOT real tok/s).
- `codex_recs_1.md` §20 = appended findings. `research_feed_paths.md` = feeding research (review before next branch).

## Board state (root@192.168.68.88, static; A100 hart8 VLEN=1024)
- Model: `/root/models/Qwen3-30B-A3B-Q4_0.gguf` (17GB, unsloth base — NOT the Coder variant on NAS).
- Binary: `/root/qwen_moe_cache` (current). Build: `gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe_cache qwen_moe.c -lm`.
- Run: `LD_LIBRARY_PATH=/usr/lib ./qwen_moe_cache /root/models/Qwen3-30B-A3B-Q4_0.gguf <ngen> <nt> [cachepath]`.
- **Local cache**: `/root/models/qwen3-30b-a3b.imecache` (~18-20GB, footer "ENDIMEC" validated). Load ~40s vs 16-min requant.
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
- rustc static-link + `-fPIC` miscompiles the kernel (unbounded loop). gcc non-PIC only. → engines standalone C, Python via ctypes.

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
1. `ssh root@192.168.68.88`; local cache exists → `LD_LIBRARY_PATH=/usr/lib /root/qwen_moe_cache /root/models/Qwen3-30B-A3B-Q4_0.gguf 12 4` reloads in ~40s and prints buckets.
2. Continue P0.2/P0.3, then research_feed_paths.md branch choice.
