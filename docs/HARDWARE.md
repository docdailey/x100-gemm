# SpaceMIT K3 (Milk-V Jupiter 2) — hardware map for LLM/GEMM, measured

Everything here is measured on a live board (`k3`) unless marked [spec].

## Compute topology — TWO core clusters
| | X100 cluster | A100 cluster |
|---|---|---|
| count | 8 harts (0–7) | 8 harts (8–15) |
| model | `Spacemit(R) X100` | `Spacemit(R) A100` |
| clock | up to **2.0 GHz** | up to **2.4 GHz** |
| RVV VLEN | **256-bit** (measured) | **1024-bit** (measured ✓) |
| IME | IME-1 (`vmadot`) | **IME-2 (`vmadotsu`/`vmadotu`, 1024-bit)** |
| how to run code there | normal scheduling (Cpus_allowed 0–7) | **write `"0"` to `/proc/set_ai_thread`** (kernel migrates the thread; `taskset`/`sched_setaffinity` give EINVAL) |

**The A100 cores ARE reachable from userspace — just not the obvious way.** `taskset -c 8` /
`sched_setaffinity` / `systemd-run -p AllowedCPUs=0-15` all fail EINVAL. The sanctioned path is the
driver hook **`/proc/set_ai_thread`** (write-only, world-writable): open it, `write("0")`, and the kernel
migrates the calling thread onto an A100 core. Confirmed: after the write, `sched_getcpu()` returns a
hart ≥ 8 and `vlenb` reports **VLEN=1024**. This is what SpaceMIT's `bind_ai_thread()` does
(`ime.cpp:1668`), gated on `use_ime2 && !(cpu on AI mask)`.

**IME-2 is INLINE INSTRUCTIONS, not a submit-to-accelerator model.** `ime2_kernels.cpp` issues
`vmadotsu vd,vs1,vs2,i8` / `vmadotu` directly from the migrated thread — same issue style as IME-1's
`vmadot`, just 1024-bit wide on the A100 core. There is **no `/dev/ai_dma` dispatch ring for matmul**
(ai_dma is a separate DMA engine; the ggml backend doesn't use it for weights or compute).

## Measured compute
| path | int8 throughput | note |
|------|-----------------|------|
| scalar | **5.6 GOP/s** | baseline |
| IME-1 `vmadot`, 8× X100, 1-accumulator | ~889 GOP/s | latency-bound (dependent chain) |
| **IME-2 `vmadotsu`, 1× A100, 4-accumulator** | **~918 GOP/s** | one AI core ≈ the whole X100 cluster |
| **IME-2, 2× A100** | **~1633 GOP/s** | **1.78× → near-linear scaling** |
| IME-2, 8× A100 (projected) | ~6–7 TOPS+ | **power-gated on current PSU — see below** |

*Methodology note:* a single-accumulator `vmadot` loop is **latency-bound**; the real kernel interleaves
**4 independent accumulators** (v20/v22/v24/v26) to hide matrix-unit latency — that alone is a **4×**
difference (230 → 918 GOP/s per A100 core). GOP/s assumes 512 MAC/tile (128×VLEN/256, linear estimate);
**exact `vmadotsu` tile geometry still needs confirming from the IME-2 spec** — true TOPS may be higher.

## ⚠️ Power ceiling (current supply)
Running the IME at full tilt on **≥3–4 A100 cores simultaneously reliably reboots the board** (clean
brownout: 1 & 2 cores run fine and stay up; 4 and 8 cores → instant reboot, uptime resets to 0). Almost
certainly a current/PSU limit, not a software fault — the board had been on a battery bank. **A solid
USB-C PD supply is needed to exercise all 8 A100 cores** and get the true 8-core / ~60-TOPS ceiling.

## Memory
- 64-bit **LPDDR5-6400, 51 GB/s peak** [spec], 32 GB.
- **Measured CPU-load bandwidth: ~19.7 GB/s read, ~23 GB/s prefetched, ~19 copy** (38–45% of peak).
- **3 MB TCM** (`/dev/tcm`, phys 0x0, size 0x300000, 8×384KB, direct-mmap) — per-core IME staging via
  `libspine_tcm.so` (`spine_tcm_mem_get(cpu_id)`).
- Weight memory pool = **plain DRAM** (default: transparent hugepage `mmap`+`MADV_HUGEPAGE`). The fancier
  backends want **`/dev/hugetlb_1g`** (ioctl `HUGETLB_1G_IOC_ALLOC`, returns a `dma_addr`) and
  **`/dev/tcm_sync_mem`** — **NEITHER EXISTS on this board**, which is exactly why llama.cpp needed
  `SPACEMIT_DISABLE_TCM=1`. `/dev/ai_dma` (char major 240) is present but unused by the ggml backend.

## Why our earlier llama.cpp run underperformed
Launched from an ssh session, its worker threads *did* call `bind_ai_thread()` — but with TCM disabled and
the mem backend on plain DRAM, plus decode being bandwidth-bound, it never showed the compute ceiling. The
compute path itself is real and now measured directly: **one A100 core = ~918 GOP/s; two = 1633.**

## The real levers for a custom engine (revised)
1. **Fill all 8 A100 cores** (needs the better PSU) — near-linear scaling seen at 2 cores → multi-TOPS.
2. **Nail the exact `vmadotsu` tile** from the IME-2 spec so TOPS is exact, and use the i2×i8 (2-bit
   weight) kernel for 4× weight density → less bandwidth pressure on decode.
3. **TCM staging** (`/dev/tcm` + libspine_tcm) to feed the AI cores without hitting the 19–23 GB/s DRAM wall.
4. **Decode is still bandwidth-bound** — dense-27B ceiling ≈ 3.4 t/s @ 51 GB/s; fast large-model decode
   needs MoE + speculative/MTP on top of the now-unlocked AI-core compute.
