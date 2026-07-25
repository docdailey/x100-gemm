# SpaceMIT K3 (Milk-V Jupiter 2) — hardware map for LLM/GEMM, measured

Everything here is measured on a live board (`k3`) unless marked [spec]. Peak IME numbers use the
SpaceMIT 8-accumulator latency-hidden kernel; TOPS uses the spec's exact tile (below), not an estimate.

## Compute topology
| | X100 cluster | A100 cluster |
|---|---|---|
| harts | 0–7 | 8–15 |
| model | `Spacemit(R) X100` | `Spacemit(R) A100` |
| clock | up to **2.0 GHz** | up to **2.4 GHz** |
| RVV VLEN | **256-bit** | **1024-bit** (measured ✓) |
| matrix | IME-1 `vmadot` (per-core) | **IME-2 `vmadotsu`/`vmadotu` — 4 units, each shared by a core PAIR** |
| run code there | normal scheduling | **write `"0"` to `/proc/set_ai_thread`, THEN `sched_setaffinity` to hart 8–15** |

### Reaching the A100 cores — the mechanism (this was the whole unlock)
`taskset`/`sched_setaffinity`/`systemd-run -p AllowedCPUs=0-15` to harts 8–15 all fail **EINVAL**.
The sanctioned path (from SpaceMIT `ime.cpp:1668 bind_ai_thread()`):
1. `fd = open("/proc/set_ai_thread", O_WRONLY); write(fd, "0", 1)` — **UNLOCKS** this thread for AI cores.
2. *Then* `sched_setaffinity()` to a specific hart 8–15 succeeds (it was EINVAL before the write).

After the write: `sched_getcpu()` ≥ 8 and `vlenb` reports **VLEN=1024**. IME-2 is **inline instructions**
(`vmadotsu vd,vs1,vs2,i8`) issued by the migrated thread — NOT a `/dev/ai_dma` submission model
(ai_dma exists, char major 240, but the ggml backend never uses it for weights or matmul).

## IME tile geometry (from the official IME-2 spec)
One `vmadot`/`vmadotsu` = an **M×K×N** int8 tile accumulating into int32, scaling with VLEN:
| VLEN | tile M×K×N | MAC / instruction |
|------|-----------|-------------------|
| 256 (X100) | 4×4×8 | **128** |
| **1024 (A100)** | **8×8×16** | **1024** |

`Copies = (sqrt(VLEN/64) is integer ? 1 : 2)`; VLEN 1024 → Copies=1 → 8×8×16 = 1024 MAC/op.

## Measured IME-2 peak (int8, 8-accumulator kernel)
| config | vmadotsu/s | TOPS int8 | note |
|--------|-----------|-----------|------|
| 1 core (whole unit to itself) | 1.61e9 | **3.29** | 8-acc; 4-acc was 1.84 → latency-bound, not width |
| 4 cores WITHIN pairs (2 units) | 3.57e9 | 7.31 | contended — 2 IME-2 units |
| **4 cores ACROSS pairs (4 units)** | 6.39e9 | **13.09** | **1 core per unit — the efficient point** |
| 8 cores (all, 2 per unit) | 7.12e9 | **14.59** | all 4 units saturated; only +1.5 over 4-across |

**Headline: ~14.6 TOPS int8 sustained, and the board draws only ~12 W** (≈1.2 TOPS/W). The **4 IME-2
matrix units** (dual-core-shared) are the real resource — filling all 4 (via harts 8/10/12/14, or all 8)
is what matters; a lone core nearly saturates its unit. This is **~16× the X100 IME-1 cluster** and the
~60-TOPS spec figure is the int4 / theoretical-issue ceiling above this sustained int8 number.

## ⚠️ Power note
On the **dying battery bank**, ≥4 cores at full tilt brown-out rebooted the board. On a **proper supply
it is rock-solid** through 8-core / 16-thread runs at ~12 W. Not a thermal or software limit.

## Memory (measured, single A100 core unless noted)
- 64-bit **LPDDR5-6400, 51 GB/s peak** [spec], 32 GB. Dependency-free read BW **20.8 GB/s** (1 core),
  write 30.6; ~19-23 aggregate under multi-core load.
- Caches: **L1D 64 KB, L2 1 MB per A100 core** (X100: L2 4 MB). Line 64 B.
- **3 MB TCM** (`/dev/tcm`, direct-mmap) is **UNCACHED device memory** — CPU read **0.41 GB/s** (write 4.8,
  latency 56 ns). Fast SRAM, but only usable at speed via the IME port / ai_dma, NOT CPU `vle8`. **Not a
  CPU-feed lever.** (`/dev/hugetlb_1g`, `/dev/tcm_sync_mem` don't exist here → llama.cpp's `DISABLE_TCM=1`.)

## GEMM ceiling — register-fed peak is NOT reachable by real matmul
The 14.6 TOPS peak loads operands **once into registers** and loops — zero per-op memory traffic. Real GEMM
reloads operands every K-step, and **`vle8`-from-cache is the wall**. Measured microkernel throughput vs
where its working set lives (`bench/ime2_l1_ceiling.c`, 8 threads):

| working set | location | TOPS |
|---|---|---|
| 18 KB | **L1** | **2.3** |
| 72 KB | L1/L2 edge | 2.5 |
| 576 KB | L2 | 0.38 |
| 4.6 MB | DRAM | 0.37 |

So the **realistic int8 GEMM ceiling is ~2.3 TOPS** (L1-resident), with a hard cliff to ~0.38 on L1 spill.
Current `src/gemm_ime2_i8.c`: **0.97 TOPS @2048³** (8-accumulator, `C==CRef`), L1-overflow because a full-K
microkernel tile streams ~144 KB. **The scalar C store is ~36% of runtime** (0.71 vs 1.11 no-store) and is
the current blocker: naive K-blocking to shrink the working set forces C read-modify-write K/KC times, and
that C-store traffic *swamps* the L1 gain (KC=512 halved throughput). **Fix order: (1) cheap vectorized C
store, then (2) MN-blocking with C held in registers across full K.** Above ~2.3 TOPS needs the ai_dma/IME
feed path, not CPU `vle8`.

## Levers for the custom engine (revised, in priority order)
1. **Feed the 4 IME-2 units** — pin 1 thread per unit (harts 8/10/12/14) + 8-accumulator microkernel.
   Register-fed compute is **~14.6 TOPS int8 @ 12W, proven**; L1-fed GEMM realistically **~2.3 TOPS**.
2. **Cheap store + MN cache blocking** — the path from the current 0.97 → ~2.3 TOPS (see GEMM section).
3. **int4 / i2×i8 weights** — 8×8×16 tiles; shipped kernel packs 2-bit weights for 4× density → less
   decode bandwidth pressure (do int8 solid first, then drop the same kernels to int4/Q4).
4. **Decode stays bandwidth-bound** — dense-27B ceiling ≈ 3.4 t/s @ 51 GB/s; big-model speed needs
   MoE + speculative/MTP layered on the now-unlocked compute.
