/* peak_ime2.c — PEAK IME-2 throughput on an A100 AI core.
 *
 * Migrates onto an A100 core via /proc/set_ai_thread (VLEN 1024), then runs the
 * SpaceMIT IME-2 inner-kernel pattern: vmadotsu with 4 independent accumulators
 * (v20/v22/v24/v26) to hide the matrix-unit latency, unlike a single-accumulator
 * dependent chain. This is the honest peak-per-core number.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o peak_ime2 peak_ime2.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <omp.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }
static unsigned long vlenb(void){ unsigned long v; __asm__ volatile("csrr %0,vlenb":"=r"(v)); return v; }

static int bind_ai_thread(void){
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0) return -1;
    int r = write(fd, "0", 1); close(fd);
    return r < 0 ? -1 : 0;
}

/* 16 vmadotsu per iter across 4 accumulators (v20,v22,v24,v26), 4 A-operands
 * (v2,v3,v28,v29) x 4 B-operands (v8..v11 group) — mirrors ime2_kernels.cpp. */
static void ime2_peak(int8_t *A, uint8_t *B, uint64_t n){
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t"
        "vle8.v v2,(%0)\n\t"  "add t3,%0,16\n\t vle8.v v3,(t3)\n\t"
        "add t3,%0,32\n\t vle8.v v28,(t3)\n\t add t3,%0,48\n\t vle8.v v29,(t3)\n\t"
        "vle8.v v8,(%1)\n\t"  "add t4,%1,16\n\t vle8.v v9,(t4)\n\t"
        "add t4,%1,32\n\t vle8.v v10,(t4)\n\t add t4,%1,48\n\t vle8.v v11,(t4)\n\t"
        "vxor.vv v20,v20,v20\n\t vxor.vv v22,v22,v22\n\t vxor.vv v24,v24,v24\n\t vxor.vv v26,v26,v26\n\t"
        "1:\n\t"
        "vmadotsu v20,v2,v8,i8\n\t"  "vmadotsu v22,v2,v9,i8\n\t"
        "vmadotsu v24,v2,v10,i8\n\t" "vmadotsu v26,v2,v11,i8\n\t"
        "vmadotsu v20,v3,v8,i8\n\t"  "vmadotsu v22,v3,v9,i8\n\t"
        "vmadotsu v24,v3,v10,i8\n\t" "vmadotsu v26,v3,v11,i8\n\t"
        "vmadotsu v20,v28,v8,i8\n\t" "vmadotsu v22,v28,v9,i8\n\t"
        "vmadotsu v24,v28,v10,i8\n\t""vmadotsu v26,v28,v11,i8\n\t"
        "vmadotsu v20,v29,v8,i8\n\t" "vmadotsu v22,v29,v9,i8\n\t"
        "vmadotsu v24,v29,v10,i8\n\t""vmadotsu v26,v29,v11,i8\n\t"
        "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
        : "+r"(A),"+r"(B),"+r"(n)
        :: "t0","t3","t4","v2","v3","v8","v9","v10","v11","v20","v22","v24","v26","v28","v29","memory");
}

int main(int c, char**v){
    int nt   = (c>1)? atoi(v[1]) : 1;
    uint64_t iters = (c>2)? strtoull(v[2],0,10) : 20000000ULL;  /* per thread; 16 vmadotsu each */
    printf("threads=%d iters=%llu\n", nt, (unsigned long long)iters);
    int harts[64]; unsigned long vls[64]; for(int i=0;i<64;i++){harts[i]=-1;vls[i]=0;}
    double t0=now();
    #pragma omp parallel num_threads(nt)
    {
        int tn = omp_get_thread_num();
        bind_ai_thread();
        /* after the unlock, pin one thread per distinct AI hart (8..15) */
        if (getenv("PIN")){
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(8 + (tn % 8), &s);
            sched_setaffinity(0, sizeof(s), &s);
        }
        for(int i=0;i<5;i++) sched_yield();
        int8_t A[64]; uint8_t B[64];
        for(int i=0;i<64;i++){A[i]=(int8_t)(i-32);B[i]=(uint8_t)(i*7);}
        harts[tn]=sched_getcpu(); vls[tn]=vlenb()*8;
        ime2_peak(A,B,iters);
    }
    double dt=now()-t0;
    int on_ai=0; unsigned long VL=vls[0];
    printf("per-thread landing hart: ");
    for(int i=0;i<nt;i++){ printf("%d(vl%lu) ", harts[i], vls[i]); if(harts[i]>=8) on_ai++; }
    printf("\ncores_on_AI=%d/%d\n", on_ai, nt);
    double vmadotsu = (double)iters * 16 * nt;
    double mac_per_tile = 128.0 * (double)VL/256.0;   /* scales with VLEN (see spec for exact tile) */
    printf("vmadotsu: %.3e op/s | ~%.1f GOP/s int8 (%.0f MAC/tile x2, est)\n",
           vmadotsu/dt, vmadotsu*mac_per_tile*2/dt/1e9, mac_per_tile);
    return 0;
}
