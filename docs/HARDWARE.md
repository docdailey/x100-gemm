# SpaceMIT K3 (Milk-V Jupiter 2) — hardware map for LLM/GEMM, measured

Everything here is measured on a live board (`k3`) unless marked [spec].

## Compute topology — TWO core clusters
| | X100 cluster | A100 cluster |
|---|---|---|
| count | 8 harts (0–7) | 8 harts (8–15) |
| model | `Spacemit(R) X100` | `Spacemit(R) A100` |
| clock | up to **2.0 GHz** | up to **2.4 GHz** |
| RVV VLEN | **256-bit** (measured) | **1024-bit** [spec] |
| IME | IME-1 (`vmadot`, runs as normal insns) | **IME-2 (the ~60 TOPS)** [spec] |
| **Linux-schedulable?** | **YES** (Cpus_allowed 0–7) | **NO** — kernel refuses affinity (EINVAL) even via systemd-run/cgroup |

**The A100 cores are a driver-managed accelerator, not general CPUs.** They are SMP-online
(`Brought up 16 CPUs`) but you cannot run user threads on them — `taskset`, `sched_setaffinity`,
`systemd-run -p AllowedCPUs=0-15`, and cgroup widening all fail with EINVAL for cores 8–15. They are
reached only through **`/dev/ai_dma` + TCM + HUGETLB_1G shared memory** submission (the SpaceMIT
`spine_mem_pool`/`ime2_kernels` machinery), the same model as the RCPU.

## Measured compute
| path | int8 throughput | note |
|------|-----------------|------|
| scalar | **5.6 GOP/s** | baseline |
| **IME-1 `vmadot` (8× X100)** | **889 GOP/s** (111/core) | **160× scalar**, usable now with `-march=rv64gcv_zvfh_xsmtvdotii` |
| IME-2 (A100, 1024-bit) | ~60 TOPS [spec] | ~67× IME-1; **requires the ai_dma/driver path** |

`vmadot` tile = 4×8×4 int8 (128 MAC), B pre-packed (transpose 8×4→4×8). Official demo:
`spacemit-com/riscv-ime-extension-spec/example/vmadot-gemm-demo.c` (builds+runs correct on-board).

## Memory
- 64-bit **LPDDR5-6400, 51 GB/s peak** [spec], 32 GB.
- **Measured CPU-load bandwidth: ~19.7 GB/s read, ~23 GB/s prefetched, ~19 copy** (38–45% of peak).
  Hugepages didn't move it (not TLB-bound). The gap to 51 GB/s needs **`ai_dma`**.
- **3 MB TCM** (`/dev/tcm`, phys 0x0, size 0x300000, 8×384KB, direct-mmap) — fast staging for IME.
- **`/dev/ai_dma`** — DMA engine feeding the A100 cores + TCM.
- Weight staging uses a **`HUGETLB_1G_IOC_ALLOC` ioctl** + `MAP_SHARED` (SpaceMIT `spine_mem_pool.cpp`).

## Why generic runtimes (incl. our llama.cpp run) underperform
Launched from an ssh session, a process is confined to **cores 0–7 (X100)**. SpaceMIT's llama.cpp sets
`use_ime2=1` and `cpu_mask ff00`, but its `pthread_setaffinity_np` to the A100 cores **fails silently in
that context and falls back to X100** → its prefill measured **~616 GOP/s = ~70% of IME-1 (889), but ~1%
of the 60-TOPS IME-2.** **The A100 tensor cores were never engaged.** This is the single biggest reason
for poor numbers — not the model, not the quantization.

## The two frontiers (what a custom engine must crack)
1. **`ai_dma` bandwidth** — reverse its ioctl/mmap ABI; stream DRAM→TCM. Settles 23 vs 51 GB/s → up to
   **2.2× on all memory-bound decode**.
2. **A100 IME-2 submission** — how work is dispatched to the driver-managed A100 cluster (via ai_dma +
   HUGETLB_1G + the barrier/TCM protocol). This is the path to the real **~60 TOPS** (≈67× the IME-1 we
   can already hit) → the huge **prefill** win. Reference: SpaceMIT `ggml-cpu/spacemit/` source.

## Honest performance envelope
- **Prefill:** IME-1 alone (889 GOP/s) already ~1.4× a generic run; A100 IME-2 is ~67× beyond that.
- **Decode:** bandwidth-bound. `ai_dma` → ~2.2×. Dense-27B ceiling ≈ 3.4 t/s @ full 51 GB/s (physics).
  Fast large-model decode needs **MoE + speculative/MTP** on top.
