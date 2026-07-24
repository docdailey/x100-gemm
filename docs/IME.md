# SpaceMIT IME (Integrated Matrix Extension) — `vmadot` notes

The X100's matrix engine for the "60 TOPS" figure. It is a **custom SpaceMIT
extension** ("XSTIME" / SpaceMIT Vector‑Dot‑Product), **not** in the kernel
`riscv,isa` string, and needs the SpaceMIT **LLVM/Clang** toolchain (GCC is not
planned). This doc captures what we need to write the int8 micro‑kernel.

## The instructions (CUSTOM_1 opcode)
| mnemonic | op |
|---|---|
| `vmadot`  | signed × signed, int MAC |
| `vmadotu` | unsigned × unsigned |
| `vmadotsu`| signed × unsigned |
| `vmadotus`| unsigned × signed |
| `vfmadot` | floating‑point matrix MAC |
| `vmadotn` / `vfmadotn` | sliding‑window variants |

Operation (widening matrix multiply‑accumulate):

```
MVD += widen(MVS1) * transpose(widen(MVS2))
```

- **Sources:** int8 (practical) or int16 (has reproducible arithmetic error);
  **destination:** always int32.
- **Tile on VLEN=256, e8:** `vl=16` → **4×4** source; `vl=32` → **4×8** source.
  Destination is a **4×4 int32** block, in an **even vreg group** (EMUL=2 → 512 bit).
- Matrices are **row‑major** in the vector registers.

## Minimal sequence
```asm
    vsetivli zero, 16, e8, m1, ta, ma   # 4x4 int8 geometry (vl=32 -> 4x8)
    vle8.v   v7, (a_tile)               # A sub-tile, row-major
    vle8.v   v9, (b_tile)               # B sub-tile, row-major
    smt.vmadot v8, v7, v9               # v8:v9(int32 4x4) += A * B^T   (dest even)
    ...                                 # loop over K in steps of 4 (or 8)
    vse32.v  v8, (c_tile)               # 4x4 int32 out
```
(LLVM prefixes the instructions with `smt.`.)

## Toolchain
- **LLVM feature tokens:** `XSMTVDot` (X60 hw subset, v1.0.0) / `XSMTVDotII` (A100).
  Enable via `-march=rv64gcv..._xsmtvdot`. X100 support token = **to confirm on‑box**
  (try `xsmtvdot`; X100 is newer than X60 and may expose a superset).
- **GCC:** no IME support. RVV paths in this repo build with gcc; IME needs clang.
- Install: SpaceMIT LLVM (their `ai-sdk` / toolchain release) or a mainline Clang new
  enough to carry the SpaceMIT vendor extension.

## Status in this repo
`src/gemm_ime_i8.c` has the 4×4 micro‑kernel structure with the `vmadot` sequence
documented and gated behind `-DX100_HAVE_IME`; the default build uses a scalar
oracle so the library always links. **Next:** install SpaceMIT clang, confirm the
`-march` token, wire the inline‑asm/intrinsic `vmadot`, and validate int8 output
against `gemm_ref_i8` on hardware. Then benchmark GOP/s.

## Open questions (resolve on hardware)
1. Exact `-march` feature string the X100's assembler accepts for `vmadot`.
2. Are intrinsics available (header/name) or is inline asm required?
3. Precise operand packing (which vreg is A vs Bᵀ; the four sign variants).
4. int16 accuracy caveat — is it usable for the accumulation path?
