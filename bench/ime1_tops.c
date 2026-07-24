/*
 * ime1_tops.c — measure achieved IME-1 (vmadot) int8 throughput vs scalar.
 *
 * IME-1 = the per-core SpaceMIT matrix extension (vmadot), 256-bit, runs as
 * normal instructions on the X100 cores. One vmadot = a 4x8x4 int8 tile (128 MAC).
 * This is the compute peak WHEN fed from registers/L1 (issue-rate bound).
 *
 * Build (on board):  gcc -O3 -march=rv64gcv_xsmtvdotii -fopenmp -o ime1_tops ime1_tops.c
 * Run:               taskset -c 0-7 ./ime1_tops     # X100 cores
 *
 * NOTE: the 8 A100 AI cores (harts 8-15) are NOT schedulable (taskset fails) —
 * their 1024-bit IME-2 (the real 60 TOPS) is reached only via the ai_dma/TCM
 * offload path, not by running this code there. See docs/HARDWARE.md.
 *
 * Measured on Milk-V Jupiter 2 (K3), 8x X100 @ ~2 GHz:
 *   IME-1 vmadot : ~889 GOP/s int8  (~111 GOP/s/core)
 *   scalar int8  : ~5.6 GOP/s   ->  IME-1 is ~160x scalar
 */
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

/* pure vmadot issue rate: v28 += v0(4x8) * v1(4x8 packed), looped n times. */
static void vmadot_loop(int8_t *a, int8_t *b, uint64_t n)
{
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t"
        "vle8.v v0,(%0)\n\t" "vle8.v v1,(%1)\n\t"
        "1:\n\t" "vmadot v28,v0,v1\n\t"
        "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
        : "+r"(a), "+r"(b), "+r"(n)
        :: "t0", "v0", "v1", "v28", "v29", "cc", "memory");
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }

int main(void)
{
    const int nt = 8;
    const uint64_t iters = 200000000ULL;   /* per thread */
    int8_t A[32], B[32];
    for (int i = 0; i < 32; i++){ A[i] = (int8_t)(i-16); B[i] = (int8_t)((i*3)%7-3); }

    double t0 = now();
    #pragma omp parallel num_threads(nt)
    { int8_t a[32], b[32]; for (int i=0;i<32;i++){a[i]=A[i];b[i]=B[i];} vmadot_loop(a,b,iters); }
    double dt = now() - t0;
    double vmadots = (double)iters * nt;
    double ops = vmadots * 128 * 2;        /* 128 MAC * 2 op/MAC per 4x8x4 tile */
    double ime = ops/dt/1e9;
    printf("IME-1 vmadot: %.2f GOP/s int8  (%.1f GOP/s/core, %.2e vmadot/s)\n",
           ime, ime/nt, vmadots/dt);

    /* scalar int8 MAC baseline */
    uint64_t sn = iters*nt/1000; volatile int64_t sink=0;
    t0 = now();
    #pragma omp parallel num_threads(nt)
    { int32_t s=0; for (uint64_t i=0;i<sn;i++) for(int k=0;k<128;k++) s+=A[k&31]*B[(k*7)&31];
      #pragma omp atomic
      sink += s; }
    dt = now()-t0; double sc = ((double)sn*nt*128*2)/dt/1e9;
    printf("scalar int8 : %.2f GOP/s  ->  IME-1 speedup: %.0fx\n", sc, ime/sc);
    (void)sink;
    return 0;
}
