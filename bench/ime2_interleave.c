/* ime2_interleave.c — does interleaving a CORE PAIR onto its shared IME-2 unit
 * hide memory bubbles? Pure-issue showed 1 core saturates a unit (2nd core +11%).
 * But real GEMM streams operands from memory between vmadotsus. Here the inner loop
 * RELOADS operands every iteration (real load traffic), then compares:
 *   - 1 thread on a unit (hart 8)         -> unit stalls during that core's loads
 *   - 2 threads on the SAME unit (8,9)    -> partner issues while the other loads
 * If 2-on-a-unit >> 1-on-a-unit (toward 2x, vs the 1.11x pure-issue), the shared
 * unit is meant to be fed by BOTH cores interleaved and real GEMM wants 8 threads.
 *
 * pin: hart = 8 + (tn*STRIDE)%8.  STRIDE=1 packs pairs (8,9,10,11); STRIDE=2 spreads (8,10,12,14).
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o ime2_interleave ime2_interleave.c
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

#define BUF 16384   /* 16 KB A + 16 KB B per thread -> L1/L2-resident, real load latency */

/* Inner loop: each iter reloads 2 A + 4 B vector regs from a rotating offset, then
 * 8 vmadotsu into 8 accumulators. VL=128 at VLEN1024, so each vle8.v = 128 B. */
static void stream_kernel(int8_t*A,uint8_t*B,uint64_t n){
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t"
        "vxor.vv v16,v16,v16\n\t vxor.vv v18,v18,v18\n\t vxor.vv v20,v20,v20\n\t vxor.vv v22,v22,v22\n\t"
        "vxor.vv v24,v24,v24\n\t vxor.vv v26,v26,v26\n\t vxor.vv v28,v28,v28\n\t vxor.vv v30,v30,v30\n\t"
        "li t5,0\n\t"                            /* rotating byte offset */
        "1:\n\t"
        "add t3,%0,t5\n\t add t4,%1,t5\n\t"
        "vle8.v v14,(t3)\n\t addi t3,t3,128\n\t vle8.v v15,(t3)\n\t"     /* 2 A tiles */
        "vle8.v v8,(t4)\n\t addi t4,t4,128\n\t vle8.v v9,(t4)\n\t"       /* 4 B tiles */
        "addi t4,t4,128\n\t vle8.v v10,(t4)\n\t addi t4,t4,128\n\t vle8.v v11,(t4)\n\t"
        "vmadotsu v16,v14,v8,i8\n\t"  "vmadotsu v18,v14,v9,i8\n\t"
        "vmadotsu v20,v14,v10,i8\n\t" "vmadotsu v22,v14,v11,i8\n\t"
        "vmadotsu v24,v15,v8,i8\n\t"  "vmadotsu v26,v15,v9,i8\n\t"
        "vmadotsu v28,v15,v10,i8\n\t" "vmadotsu v30,v15,v11,i8\n\t"
        "addi t5,t5,512\n\t"
        "li t6,15360\n\t blt t5,t6,2f\n\t li t5,0\n\t 2:\n\t"           /* wrap offset < BUF-1024 */
        "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
        : "+r"(A),"+r"(B),"+r"(n)
        :: "t0","t3","t4","t5","t6","v8","v9","v10","v11","v14","v15",
           "v16","v18","v20","v22","v24","v26","v28","v30","memory");
}

int main(int c,char**v){
    int nt=(c>1)?atoi(v[1]):1;
    uint64_t iters=(c>2)?strtoull(v[2],0,10):6000000ULL;   /* 8 vmadotsu each */
    int stride=getenv("STRIDE")?atoi(getenv("STRIDE")):1;
    int harts[64]; for(int i=0;i<64;i++)harts[i]=-1;
    static int8_t Abuf[64][BUF]; static uint8_t Bbuf[64][BUF];
    for(int t=0;t<nt&&t<64;t++) for(int i=0;i<BUF;i++){Abuf[t][i]=(int8_t)(i-128);Bbuf[t][i]=(uint8_t)(i*7);}
    double t0=now();
    #pragma omp parallel num_threads(nt)
    {
        int tn=omp_get_thread_num();
        bind_ai();
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(8+((tn*stride)%8),&s); sched_setaffinity(0,sizeof(s),&s);
        for(int i=0;i<5;i++)sched_yield();
        harts[tn]=sched_getcpu();
        stream_kernel(Abuf[tn],Bbuf[tn],iters);
    }
    double dt=now()-t0;
    unsigned long VL=vlenb()*8;
    double ops=(double)iters*8*nt;
    printf("STREAM threads=%d stride=%d VLEN=%lu harts:",nt,stride,VL);
    for(int i=0;i<nt;i++)printf(" %d",harts[i]);
    printf("\n%.3e vmadotsu/s | %.2f TOPS int8 (1024 MAC/tile)\n", ops/dt, ops*1024.0*2/dt/1e12);
    return 0;
}
