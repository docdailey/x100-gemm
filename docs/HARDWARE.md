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

## Memory
- 64-bit **LPDDR5-6400, 51 GB/s peak** [spec], 32 GB. Measured CPU-load BW **~19.7 GB/s read / ~23
  prefetched / ~19 copy** (38–45% of peak).
- **3 MB TCM** (`/dev/tcm`, phys 0x0, 8×384KB, direct-mmap) — per-core IME staging via `libspine_tcm.so`.
- Weight pool = **plain DRAM** (transparent hugepage). Fancier backends want `/dev/hugetlb_1g`
  (ioctl `HUGETLB_1G_IOC_ALLOC`) + `/dev/tcm_sync_mem` — **neither exists on this board** → why
  llama.cpp needed `SPACEMIT_DISABLE_TCM=1`.

## Levers for the custom engine (revised, in priority order)
1. **Feed the 4 IME-2 units** — pin 1 thread per unit (harts 8/10/12/14) + 8-accumulator microkernel.
   Compute is unlocked: **~14.6 TOPS int8 @ 12W, proven.**
2. **Exact-tile int8/int4 packing** — 8×8×16 tiles; the shipped kernel is i2×i8 (2-bit weights) for 4×
   weight density → less decode bandwidth pressure. int4 path is the route toward the 60-TOPS ceiling.
3. **TCM staging** (`/dev/tcm` + libspine_tcm) to feed the units without the 19–23 GB/s DRAM wall.
4. **Decode stays bandwidth-bound** — dense-27B ceiling ≈ 3.4 t/s @ 51 GB/s; big-model speed needs
   MoE + speculative/MTP layered on the now-unlocked compute.
