# PROGRESS — K3 pure-C IME-2 decode engine (session log, live)

Last update: 2026-07-25. Durable state so work survives a session kill.

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
