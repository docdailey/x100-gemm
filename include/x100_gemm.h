/*
 * x100_gemm.h — base C spec for GEMM on the SpaceMIT X100 (RISC-V RVA23).
 *
 * Target: SpaceMIT K3 / Milk-V Jupiter 2, "Spacemit(R) X100" cores.
 *   - RV64GC + V (RVV 1.0), VLEN = 256-bit (VLENB=32), + Zvfh (fp16 vectors)
 *   - IME "Integrated Matrix Extension" (SpaceMIT Vector-Dot-Product / XSTIME):
 *     vmadot family — int8/int16 matrix multiply-accumulate into int32.
 *
 * Convention: all matrices ROW-MAJOR. C[M x N] = alpha*A[M x K]*B[K x N] + beta*C.
 * (Kernels here implement beta=1 accumulate / beta=0 overwrite as noted.)
 *
 * Backends (all share this API; pick at runtime via x100_caps or at link time):
 *   - ref   : portable scalar (correctness oracle)
 *   - rvv   : RVV 1.0 intrinsics, fp32 and fp16 (builds with mainline gcc/clang)
 *   - ime   : SpaceMIT IME int8 (needs clang -march=...+xsmtvdot); int8*int8->int32
 *
 * License: Apache-2.0.
 */
#ifndef X100_GEMM_H
#define X100_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- capability detection ------------------------------------------------ */
typedef struct {
    int  vlen_bits;     /* hardware VLEN in bits (X100 = 256)                 */
    int  vlenb;         /* VLEN in bytes                                       */
    int  has_v;         /* RVV 1.0 present                                     */
    int  has_zvfh;      /* fp16 vector present                                 */
    int  has_zvbb;      /* vector bit-manip (misc)                             */
    int  has_ime;       /* SpaceMIT IME / vmadot present (best-effort probe)   */
    unsigned long mvendorid; /* 0x710 = SpaceMIT                              */
    char isa[1024];     /* raw riscv,isa string from /proc/cpuinfo            */
} x100_caps_t;

/* Fill caps from the running CPU (reads vlenb CSR + parses /proc/cpuinfo). */
void x100_detect(x100_caps_t *caps);
void x100_caps_print(const x100_caps_t *caps);

/* ---- GEMM backends ------------------------------------------------------- *
 * All compute C = A*B + C  (beta=1 accumulate). Zero C first for C = A*B.
 * A: MxK, B: KxN, C: MxN, all row-major, contiguous (lda=K, ldb=N, ldc=N).   */

/* fp32 */
void gemm_ref_f32 (int M, int N, int K, const float *A, const float *B, float *C);
void gemm_rvv_f32 (int M, int N, int K, const float *A, const float *B, float *C);

/* fp16 (needs Zvfh for the rvv path). Uses _Float16 storage. */
#if defined(__riscv) && defined(__riscv_zvfh)
void gemm_rvv_f16 (int M, int N, int K, const _Float16 *A, const _Float16 *B, _Float16 *C);
#endif

/* int8 -> int32 (quantized). C is int32. The IME path uses vmadot; the
 * reference path is a portable scalar oracle so the repo always builds. */
void gemm_ref_i8  (int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C);
void gemm_ime_i8  (int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C);

#ifdef __cplusplus
}
#endif
#endif /* X100_GEMM_H */
