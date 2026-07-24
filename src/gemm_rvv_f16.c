/*
 * gemm_rvv_f16.c — RVV 1.0 fp16 GEMM using Zvfh (X100 has zvfh).
 *
 * Same blocking as the fp32 kernel but SEW=16, so twice the lanes per vreg
 * (m4 -> up to 64 fp16 lanes on VLEN=256). Accumulation is in fp16 here for
 * pure-Zvfh throughput; for accuracy-sensitive work, widen to fp32 accumulators
 * (vfwmacc) — left as a documented variant.
 *
 * Build: clang/gcc -march=rv64gcv_zvfh
 */
#include "x100_gemm.h"
#if defined(__riscv_v) && defined(__riscv_zvfh)
#include <riscv_vector.h>

void gemm_rvv_f16(int M, int N, int K, const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        const _Float16 *a0 = &A[(i+0)*K], *a1 = &A[(i+1)*K],
                       *a2 = &A[(i+2)*K], *a3 = &A[(i+3)*K];
        _Float16 *c0 = &C[(i+0)*N], *c1 = &C[(i+1)*N],
                 *c2 = &C[(i+2)*N], *c3 = &C[(i+3)*N];
        int n = 0;
        while (n < N) {
            size_t vl = __riscv_vsetvl_e16m4((size_t)(N - n));
            vfloat16m4_t v0 = __riscv_vle16_v_f16m4(c0 + n, vl);
            vfloat16m4_t v1 = __riscv_vle16_v_f16m4(c1 + n, vl);
            vfloat16m4_t v2 = __riscv_vle16_v_f16m4(c2 + n, vl);
            vfloat16m4_t v3 = __riscv_vle16_v_f16m4(c3 + n, vl);
            for (int k = 0; k < K; k++) {
                vfloat16m4_t b = __riscv_vle16_v_f16m4(&B[k*N + n], vl);
                v0 = __riscv_vfmacc_vf_f16m4(v0, a0[k], b, vl);
                v1 = __riscv_vfmacc_vf_f16m4(v1, a1[k], b, vl);
                v2 = __riscv_vfmacc_vf_f16m4(v2, a2[k], b, vl);
                v3 = __riscv_vfmacc_vf_f16m4(v3, a3[k], b, vl);
            }
            __riscv_vse16_v_f16m4(c0 + n, v0, vl);
            __riscv_vse16_v_f16m4(c1 + n, v1, vl);
            __riscv_vse16_v_f16m4(c2 + n, v2, vl);
            __riscv_vse16_v_f16m4(c3 + n, v3, vl);
            n += (int)vl;
        }
    }
    for (; i < M; i++) {
        const _Float16 *a = &A[i*K];
        _Float16 *c = &C[i*N];
        int n = 0;
        while (n < N) {
            size_t vl = __riscv_vsetvl_e16m4((size_t)(N - n));
            vfloat16m4_t v = __riscv_vle16_v_f16m4(c + n, vl);
            for (int k = 0; k < K; k++)
                v = __riscv_vfmacc_vf_f16m4(v, a[k], __riscv_vle16_v_f16m4(&B[k*N+n], vl), vl);
            __riscv_vse16_v_f16m4(c + n, v, vl);
            n += (int)vl;
        }
    }
}
#endif /* zvfh */
