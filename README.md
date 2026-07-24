# x100-gemm

Custom **GEMM (matrix‑multiply) kernels for the SpaceMIT X100** RISC‑V core
(Milk‑V Jupiter 2 / SpaceMIT K3, RVA23), leveraging the **RVV 1.0 vector** unit and
the **IME "Integrated Matrix Extension"** (`vmadot`).

> Status: **RVV fp32/fp16 kernels build natively and are validated on real X100
> hardware** (bit‑exact vs reference). The IME int8 (`vmadot`) path is scaffolded and
> gated behind the SpaceMIT Clang toolchain — see [Roadmap](#roadmap).

## Hardware (verified live on `Spacemit(R) X100`)
| | |
|---|---|
| Cores | 8× X100, RV64GC + **V** (RVV 1.0) + H |
| Vendor | `mvendorid = 0x710` (SpaceMIT) |
| **VLEN** | **256‑bit** (VLENB=32) |
| fp16 vectors | **Zvfh** ✓ (also Zvbb, vector‑crypto Zvk*) |
| Matrix (IME) | `vmadot` — custom, not in `riscv,isa`; **native gcc `xsmtvdotii`** (assembles + runs ✓) |
| Cache line | 64 B (Zicboz) |

## Backends (one API, `include/x100_gemm.h`)
| backend | dtype | toolchain | notes |
|---------|-------|-----------|-------|
| `gemm_ref_*` | fp32 / int8 | any | scalar correctness oracle |
| `gemm_rvv_f32` | fp32 | gcc/clang `-march=rv64gcv` | RVV, register‑blocked MRx(m4·VL) |
| `gemm_rvv_f16` | fp16 | +`_zvfh` | 2× lanes; fp16 accum (fp32‑accum variant TODO) |
| `gemm_ime_i8` | int8→int32 | **gcc** `+xsmtvdotii` | IME `vmadot` (assembles + runs on X100); scalar fallback until wired |

## Quick start (on the board)
```sh
make                     # native gcc 15, rv64gcv_zvfh
./gemm_bench 256 256 256 # capability print + correctness + GFLOP/s
make IME=1               # enable the IME vmadot path (native gcc, xsmtvdotii — confirmed)
```

## Measured (X100 @ Jupiter 2, gcc 15.2, 256³, this repo's first cut)
```
  rvv_f32   :  16.77 GFLOP/s   maxerr=0.00e+00   OK   (bit-exact vs reference)
  rvv_f16   :  36.98 GFLOP/s   (fp16 accumulate — precision-limited)
  i8->i32   :   4.95 GOP/s     (scalar fallback; IME vmadot pending)
```
These are honest first‑cut numbers from a register‑blocked (not yet cache‑tiled)
kernel; see Roadmap. Large matrices (1024³) fall off (~4 GFLOP/s) until cache
blocking lands — that's the point of the benchmark harness.

## Roadmap
1. **Cache/L2 blocking** (KC×NC panels + A/B packing) — fixes the large‑matrix falloff.
2. **fp16 with fp32 accumulation** (`vfwmacc`) — accuracy without losing throughput.
3. **IME `vmadot` int8 micro‑kernel** — the flagship: 4×8 int8 tiles → int32, the path
   to the X100's matrix throughput. Toolchain **confirmed working** (native gcc
   `-march=…xsmtvdotii`; `vmadot` runs on hardware) — next is wiring/validating the
   micro‑kernel operand packing (see `docs/IME.md`).
4. bf16 (`zvfbfmin`) if present; multi‑threaded (8 cores) outer blocking.

## Docs
- [`docs/ISA.md`](docs/ISA.md) — full X100 capability record (live).
- [`docs/IME.md`](docs/IME.md) — the `vmadot` tile math + toolchain notes.

## References
- SpaceMIT IME / XSTIME: <https://www.remlab.net/op/riscv-xstime.shtml>
- LLVM RISC‑V (XSMTVDot/XSMTVDotII): <https://llvm.org/docs/RISCVUsage.html>
- RVV GEMM background: <https://www.luffca.com/2023/02/gemm-riscv-vector-part1/>

## License
Apache‑2.0.
