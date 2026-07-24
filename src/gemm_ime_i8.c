/*
 * gemm_ime_i8.c — int8*int8 -> int32 GEMM on the SpaceMIT IME (vmadot).
 *
 * The IME ("Integrated Matrix Extension", SpaceMIT XSTIME / Vector-Dot-Product,
 * CUSTOM_1 opcode) adds MATRIX multiply-accumulate over the vector registers:
 *
 *     vmadot vd, vs1, vs2      ;  MVD += widen(MVS1) * transpose(widen(MVS2))
 *     (variants: vmadot / vmadotu / vmadotsu / vmadotus  for signed/unsigned)
 *
 * On VLEN=256 with e8 sources, one vmadot does a 4x8 (or 4x4) int8 tile:
 *   - vsetivli zero, 16, e8, m1, ta, ma        ; vl=16 -> 4x4 ;  vl=32 -> 4x8
 *   - source tiles are row-major in a vreg; the dest is a 4x4 INT32 block
 *     occupying an even vreg group (EMUL=2, i.e. 512 bits).
 *
 * TOOLCHAIN: this needs Clang with the SpaceMIT feature (LLVM: XSMTVDot for X60 /
 * XSMTVDotII for A100; instructions are prefixed "smt."). GCC does NOT implement
 * IME. Build the IME path only when the assembler understands vmadot:
 *
 *     clang -O3 -march=rv64gcv_zvfh_xsmtvdot -DX100_HAVE_IME ...
 *
 * Until that toolchain is installed AND this micro-kernel is validated on the
 * X100, the default build uses the scalar reference so the library always links.
 * See docs/IME.md for the tile math and status.
 */
#include "x100_gemm.h"

#if defined(X100_HAVE_IME)
#include <riscv_vector.h>

/*
 * 4x4 (MRxNR) int32 accumulator tile over K in steps of KC=8 int8.
 * A tile: 4 rows x 8 cols int8 (row-major in one vreg group)
 * B tile: 8 rows x 4 cols int8 -> presented transposed for vmadot
 * dest  : 4x4 int32 in an even vreg group (EMUL=2).
 *
 * NOTE: operand packing below is the DOCUMENTED intended layout; validate the
 * exact vreg/transpose mapping against the SpaceMIT IME implementation guide on
 * first hardware run (the four extension variants differ only in sign handling).
 */
static inline void ime_micro_4x4(const int8_t *Apk, const int8_t *Bpk,
                                 int32_t *Cblk, int ldc, int K)
{
    /* vl=16 selects the 4x4 int8 source geometry on VLEN=256 (e8,m1). */
    size_t vl = __riscv_vsetvl_e8m1(16);
    (void)vl;
    /* Accumulator: 4x4 int32 = 512 bits = v-group of 2 (EMUL=2), even reg. */
    /* Pseudocode of the inner K loop (KC=4 int8 per step for the 4x4 tile):
     *   for (k = 0; k < K; k += 4) {
     *     va = vle8(Apk + ...);           // 4x4 int8 A sub-tile
     *     vb = vle8(Bpk + ...);           // 4x4 int8 B sub-tile
     *     asm("smt.vmadot v8, %0, %1" :: "vr"(va), "vr"(vb));  // acc in v8..v9
     *   }
     *   vse32(Cblk, v8);                  // store 4x4 int32
     *
     * Inline asm for smt.vmadot requires the SpaceMIT assembler; kept as the
     * validated-on-hardware step (see docs/IME.md). */
    (void)Apk; (void)Bpk; (void)Cblk; (void)ldc; (void)K;
}

void gemm_ime_i8(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C)
{
    const int MR = 4, NR = 4;
    int i = 0;
    for (; i + MR <= M; i += MR) {
        int j = 0;
        for (; j + NR <= N; j += NR)
            ime_micro_4x4(&A[i * K], &B[j] /*packed*/, &C[i * N + j], N, K);
        /* N tail */
        for (; j < N; j++)
            for (int ii = 0; ii < MR; ii++) {
                int32_t s = C[(i + ii) * N + j];
                for (int k = 0; k < K; k++) s += A[(i + ii) * K + k] * (int32_t)B[k * N + j];
                C[(i + ii) * N + j] = s;
            }
    }
    /* M tail: scalar */
    for (; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = C[i * N + j];
            for (int k = 0; k < K; k++) s += A[i * K + k] * (int32_t)B[k * N + j];
            C[i * N + j] = s;
        }
}

#else  /* no IME toolchain: honest scalar fallback so the repo always builds */
void gemm_ime_i8(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C)
{
    gemm_ref_i8(M, N, K, A, B, C);   /* TODO: enable vmadot micro-kernel (clang + xsmtvdot) */
}
#endif
