/*
 * gemm_rvv_f32.c — RVV 1.0 fp32 GEMM (VLEN-agnostic; tuned for X100 VLEN=256).
 *
 * Strategy: register-block M by MR=4 rows, vectorize the free dimension N with
 * LMUL=m4 (so 4*VL/32 = up to 32 fp32 lanes per group on X100), accumulate over
 * K with vfmacc.vf (scalar-times-vector FMA). Keeps 4 accumulator groups live to
 * hide the FMA latency and reuse each B row across the 4 A columns.
 *
 * Builds with mainline gcc/clang: -march=rv64gcv
 */
#include "x100_gemm.h"
#if defined(__riscv_v)
#include <riscv_vector.h>

void gemm_rvv_f32(int M, int N, int K, const float *A, const float *B, float *C)
{
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        const float *a0 = &A[(i + 0) * K], *a1 = &A[(i + 1) * K];
        const float *a2 = &A[(i + 2) * K], *a3 = &A[(i + 3) * K];
        float *c0 = &C[(i + 0) * N], *c1 = &C[(i + 1) * N];
        float *c2 = &C[(i + 2) * N], *c3 = &C[(i + 3) * N];
        int n = 0;
        while (n < N) {
            size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n));
            vfloat32m4_t v0 = __riscv_vle32_v_f32m4(c0 + n, vl);
            vfloat32m4_t v1 = __riscv_vle32_v_f32m4(c1 + n, vl);
            vfloat32m4_t v2 = __riscv_vle32_v_f32m4(c2 + n, vl);
            vfloat32m4_t v3 = __riscv_vle32_v_f32m4(c3 + n, vl);
            for (int k = 0; k < K; k++) {
                vfloat32m4_t b = __riscv_vle32_v_f32m4(&B[k * N + n], vl);
                v0 = __riscv_vfmacc_vf_f32m4(v0, a0[k], b, vl);
                v1 = __riscv_vfmacc_vf_f32m4(v1, a1[k], b, vl);
                v2 = __riscv_vfmacc_vf_f32m4(v2, a2[k], b, vl);
                v3 = __riscv_vfmacc_vf_f32m4(v3, a3[k], b, vl);
            }
            __riscv_vse32_v_f32m4(c0 + n, v0, vl);
            __riscv_vse32_v_f32m4(c1 + n, v1, vl);
            __riscv_vse32_v_f32m4(c2 + n, v2, vl);
            __riscv_vse32_v_f32m4(c3 + n, v3, vl);
            n += (int)vl;
        }
    }
    /* M tail (rows not a multiple of 4) */
    for (; i < M; i++) {
        const float *a = &A[i * K];
        float *c = &C[i * N];
        int n = 0;
        while (n < N) {
            size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n));
            vfloat32m4_t v = __riscv_vle32_v_f32m4(c + n, vl);
            for (int k = 0; k < K; k++)
                v = __riscv_vfmacc_vf_f32m4(v, a[k], __riscv_vle32_v_f32m4(&B[k * N + n], vl), vl);
            __riscv_vse32_v_f32m4(c + n, v, vl);
            n += (int)vl;
        }
    }
}
#else
/* No RVV at compile time: fall back to the scalar reference. */
void gemm_rvv_f32(int M, int N, int K, const float *A, const float *B, float *C)
{ gemm_ref_f32(M, N, K, A, B, C); }
#endif
