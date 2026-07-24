# SpaceMIT X100 — ISA / capability record (live, 2026-07-24)

Read from a running Milk‑V Jupiter 2 (`k3`, `Spacemit(R) X100`), 8 cores.

## Identity
- `mvendorid = 0x710` (SpaceMIT), `marchid`/`mimpid` not exposed (`-1`).
- 8× harts, RVA23‑class.

## `riscv,isa` (per hart)
```
rv64imafdcvh_zicbom_zicbop_zicboz_zicntr_zicond_zicsr_zifencei_zihintntl_zihintpause
_zihpm_zimop_zaamo_zalrsc_zawrs_zfa_zfh_zfhmin_zca_zcb_zcd_zcmop_zba_zbb_zbc_zbs_zkt
_zvbb_zvbc_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkb_zvkg_zvkned_zvknha
_zvknhb_zvksed_zvksh_zvkt_smaia_smstateen_ssaia_sscofpmf_sstc_svinval_svnapot_svpbmt_sdtrig
```

## What matters for GEMM
| feature | value | use |
|---|---|---|
| **V (RVV 1.0)** | yes | vectorized FMA |
| **VLEN** | **256 bit** (VLENB=32; `csrr vlenb`) | 8 fp32 / 16 fp16 lanes per vreg (m1); ×LMUL |
| **Zvfh** | yes | native fp16 vector FMA (`vfmacc.vf` e16) |
| Zve64d / Zve32f | yes | fp64/fp32 vector element support |
| Zvbb, Zvbc | yes | vector bit‑manip / carryless |
| Zvk* (zvkned/zvksh/…) | yes | vector crypto (AES/SHA/SM4) |
| Zba/Zbb/Zbc/Zbs | yes | scalar bitmanip (addressing) |
| Zfa, Zfh, Zfhmin | yes | scalar fp16 |
| Zicboz | 64 B | cache‑line zero (packing) |
| Zicond | yes | branchless select |
| Sstc, Ssaia/Smaia | yes | S‑mode timer, AIA |
| **IME / vmadot** | **present, NOT in isa string** | matrix MAC — see IME.md |

## hwprobe (syscall 258)
- key0 (MVENDORID) = `0x710`
- key3 (BASE_BEHAVIOR) = `0x1` (IMA)
- key4 (IMA_EXT_0 bitmap) = `0x38dbff9ffff00ff`
- key6 (ZICBOZ_SIZE) = `0x40` (64)
- keys 9/10 (misaligned vector/scalar perf) = `0x3` (fast)

## Notes
- The "up to 1024‑bit vector" marketing = LMUL grouping (256×4). Hardware VLEN is 256.
- Core clock is **614.4 MHz** (PLL1 2.4576 GHz ÷ 4), set by firmware; not in Linux
  `clk_summary`. `mcycle` counts at 614.4 MHz (1.63 ns/tick).
