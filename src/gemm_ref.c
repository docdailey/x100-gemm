/* gemm_ref.c — portable scalar reference GEMM (correctness oracle). */
#include "x100_gemm.h"

/* C[MxN] += A[MxK] * B[KxN], all row-major. */
void gemm_ref_f32(int M, int N, int K, const float *A, const float *B, float *C)
{
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            const float *b = &B[k * N];
            float *c = &C[i * N];
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
}

/* int8 * int8 -> int32 accumulate. */
void gemm_ref_i8(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            int32_t a = A[i * K + k];
            const int8_t *b = &B[k * N];
            int32_t *c = &C[i * N];
            for (int j = 0; j < N; j++) c[j] += a * (int32_t)b[j];
        }
}
