/*
 * bench.c — correctness + throughput harness for the x100-gemm kernels.
 *
 * Usage:  ./bench [M N K] [reps]        (default 256 256 256, reps auto)
 * Verifies each kernel against the scalar reference and reports GFLOP/s
 * (2*M*N*K/time). fp32 always; fp16 if built with Zvfh; int8 always (IME or ref).
 */
#include "x100_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double gflops(int M, int N, int K, double s)
{ return (2.0 * M * N * K) / s / 1e9; }

static float frand(void) { return (float)rand() / RAND_MAX - 0.5f; }

int main(int argc, char **argv)
{
    int M = 256, N = 256, K = 256;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    int reps = (argc >= 5) ? atoi(argv[4]) : 0;
    if (reps <= 0) { long fl = 2L*M*N*K; reps = (int)(2e9/fl); if (reps<3) reps=3; if (reps>200) reps=200; }

    x100_caps_t caps; x100_detect(&caps); x100_caps_print(&caps);
    printf("\nGEMM  M=%d N=%d K=%d  reps=%d\n", M, N, K, reps);

    float *A  = malloc((size_t)M*K*sizeof(float));
    float *B  = malloc((size_t)K*N*sizeof(float));
    float *C  = malloc((size_t)M*N*sizeof(float));
    float *Cr = malloc((size_t)M*N*sizeof(float));
    for (long i=0;i<(long)M*K;i++) A[i]=frand();
    for (long i=0;i<(long)K*N;i++) B[i]=frand();

    /* reference oracle */
    memset(Cr,0,(size_t)M*N*sizeof(float));
    gemm_ref_f32(M,N,K,A,B,Cr);

    /* ---- fp32 RVV ---- */
    {
        memset(C,0,(size_t)M*N*sizeof(float));
        gemm_rvv_f32(M,N,K,A,B,C);           /* correctness */
        double maxerr=0; for(long i=0;i<(long)M*N;i++){double e=fabs(C[i]-Cr[i]);if(e>maxerr)maxerr=e;}
        double t0=now_s();
        for(int r=0;r<reps;r++){memset(C,0,(size_t)M*N*sizeof(float)); gemm_rvv_f32(M,N,K,A,B,C);}
        double dt=(now_s()-t0)/reps;
        printf("  rvv_f32   : %7.2f GFLOP/s   maxerr=%.2e   %s\n",
               gflops(M,N,K,dt), maxerr, maxerr<1e-2?"OK":"FAIL");
    }

    /* ---- fp16 RVV (if built) ---- */
#if defined(__riscv) && defined(__riscv_zvfh)
    {
        _Float16 *A16=malloc((size_t)M*K*2),*B16=malloc((size_t)K*N*2),*C16=malloc((size_t)M*N*2);
        for(long i=0;i<(long)M*K;i++)A16[i]=(_Float16)A[i];
        for(long i=0;i<(long)K*N;i++)B16[i]=(_Float16)B[i];
        memset(C16,0,(size_t)M*N*2);
        gemm_rvv_f16(M,N,K,A16,B16,C16);
        double maxrel=0; for(long i=0;i<(long)M*N;i++){double ref=Cr[i];double e=fabs((double)C16[i]-ref)/(fabs(ref)+1e-3);if(e>maxrel)maxrel=e;}
        double t0=now_s();
        for(int r=0;r<reps;r++){memset(C16,0,(size_t)M*N*2); gemm_rvv_f16(M,N,K,A16,B16,C16);}
        double dt=(now_s()-t0)/reps;
        printf("  rvv_f16   : %7.2f GFLOP/s   maxrel=%.2e   %s (fp16 accum)\n",
               gflops(M,N,K,dt), maxrel, maxrel<0.15?"OK":"CHECK");
        free(A16);free(B16);free(C16);
    }
#else
    printf("  rvv_f16   : (not built — needs Zvfh)\n");
#endif

    /* ---- int8 -> int32 (IME or ref) ---- */
    {
        int8_t *A8=malloc((size_t)M*K),*B8=malloc((size_t)K*N);
        int32_t *C8=malloc((size_t)M*N*4),*C8r=malloc((size_t)M*N*4);
        for(long i=0;i<(long)M*K;i++)A8[i]=(int8_t)(rand()%15-7);
        for(long i=0;i<(long)K*N;i++)B8[i]=(int8_t)(rand()%15-7);
        memset(C8r,0,(size_t)M*N*4); gemm_ref_i8(M,N,K,A8,B8,C8r);
        memset(C8,0,(size_t)M*N*4);  gemm_ime_i8(M,N,K,A8,B8,C8);
        long mism=0; for(long i=0;i<(long)M*N;i++) if(C8[i]!=C8r[i]) mism++;
        double t0=now_s();
        for(int r=0;r<reps;r++){memset(C8,0,(size_t)M*N*4); gemm_ime_i8(M,N,K,A8,B8,C8);}
        double dt=(now_s()-t0)/reps;
        printf("  i8->i32   : %7.2f GOP/s     mismatch=%ld   %s\n",
               gflops(M,N,K,dt), mism, mism==0?"OK":"FAIL");
        free(A8);free(B8);free(C8);free(C8r);
    }

    free(A);free(B);free(C);free(Cr);
    return 0;
}
