/* peak_ime2_x8.c — is IME-2 issue-rate bound? Try 8 independent accumulators.
 * If per-core throughput rises vs the 4-accumulator kernel, we were latency-bound
 * and there's headroom toward the ~60-TOPS ceiling; if flat, the matrix unit's
 * inverse-throughput (~2.7 cyc/op) is the wall. Pin 1 thread per A100 hart (8..15).
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o peak_ime2_x8 peak_ime2_x8.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <omp.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static unsigned long vlenb(void){ unsigned long v; __asm__ volatile("csrr %0,vlenb":"=r"(v)); return v; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* 8 accumulators v16,v18,v20,v22,v24,v26,v30,v31-ish — use v14/v15 as the two A operands,
 * v8..v11 as B. 8 vmadotsu per iter, each to a distinct accumulator → 8-way ILP. */
static void ime2_x8(int8_t*A,uint8_t*B,uint64_t n){
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t"
        "vle8.v v14,(%0)\n\t" "add t3,%0,16\n\t vle8.v v15,(t3)\n\t"
        "vle8.v v8,(%1)\n\t" "add t4,%1,16\n\t vle8.v v9,(t4)\n\t"
        "add t4,%1,32\n\t vle8.v v10,(t4)\n\t add t4,%1,48\n\t vle8.v v11,(t4)\n\t"
        "vxor.vv v16,v14,v14\n\t vxor.vv v18,v14,v14\n\t vxor.vv v20,v14,v14\n\t vxor.vv v22,v14,v14\n\t"
        "vxor.vv v24,v14,v14\n\t vxor.vv v26,v14,v14\n\t vxor.vv v28,v14,v14\n\t vxor.vv v30,v14,v14\n\t"
        "1:\n\t"
        "vmadotsu v16,v14,v8,i8\n\t"  "vmadotsu v18,v14,v9,i8\n\t"
        "vmadotsu v20,v14,v10,i8\n\t" "vmadotsu v22,v14,v11,i8\n\t"
        "vmadotsu v24,v15,v8,i8\n\t"  "vmadotsu v26,v15,v9,i8\n\t"
        "vmadotsu v28,v15,v10,i8\n\t" "vmadotsu v30,v15,v11,i8\n\t"
        "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
        : "+r"(A),"+r"(B),"+r"(n)
        :: "t0","t3","t4","v8","v9","v10","v11","v14","v15",
           "v16","v18","v20","v22","v24","v26","v28","v30","memory");
}

int main(int c,char**v){
    int nt=(c>1)?atoi(v[1]):8;
    uint64_t iters=(c>2)?strtoull(v[2],0,10):8000000ULL;   /* 8 vmadotsu each */
    int harts[64]; for(int i=0;i<64;i++)harts[i]=-1;
    double t0=now();
    #pragma omp parallel num_threads(nt)
    {
        int tn=omp_get_thread_num();
        bind_ai();
        int stride = getenv("STRIDE")?atoi(getenv("STRIDE")):1;
        int hart = 8 + ((tn*stride)%8);
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(hart,&s); sched_setaffinity(0,sizeof(s),&s);
        for(int i=0;i<5;i++)sched_yield();
        int8_t A[64]; uint8_t B[64]; for(int i=0;i<64;i++){A[i]=(int8_t)(i-32);B[i]=(uint8_t)(i*7);}
        harts[tn]=sched_getcpu();
        ime2_x8(A,B,iters);
    }
    double dt=now()-t0;
    unsigned long VL=vlenb()*8;
    double ops=(double)iters*8*nt;                 /* 8 vmadotsu per iter */
    double mac=1024.0;                             /* VLEN 1024 -> 8x8x16 = 1024 MAC/tile (spec) */
    printf("x8-accum threads=%d VLEN=%lu  harts:", nt, VL);
    for(int i=0;i<nt;i++)printf(" %d",harts[i]);
    printf("\n%.3e vmadotsu/s | %.2f TOPS int8 (1024 MAC/tile, spec 8x8x16)\n", ops/dt, ops*mac*2/dt/1e12);
    return 0;
}
