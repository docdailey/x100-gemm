/* ime2_l1_ceiling.c — the REAL target for a cache-blocked GEMM.
 *
 * The 14.6 TOPS peak was register-fed (operands loaded once, looped in-register).
 * Real GEMM reloads operands from cache every K-step. This runs the exact 8x64
 * microkernel over an L1-RESIDENT A(8xKc)+packB(64xKc) working set (~18KB < 64KB L1),
 * reused many times -> the steady-state L1-fed throughput = the honest ceiling that
 * perfect Mc/Nc/Kc blocking can approach. If this is ~3 TOPS, blocking is worth it;
 * if ~1 TOPS, vle8-from-L1 bandwidth is the wall and no blocking beats it.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o ime2_l1_ceiling ime2_l1_ceiling.c
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
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* one 8x64 microkernel pass over Kb K-steps (A:128B/step, B:1024B/step), C in registers, no store. */
static void uk(const int8_t*A,const int8_t*B,int Kb){
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t"
        "vxor.vv v16,v16,v16\n\t vxor.vv v18,v18,v18\n\t vxor.vv v20,v20,v20\n\t vxor.vv v22,v22,v22\n\t"
        "vxor.vv v24,v24,v24\n\t vxor.vv v26,v26,v26\n\t vxor.vv v28,v28,v28\n\t vxor.vv v30,v30,v30\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "mv t2,%0\n\t mv t3,%1\n\t mv t4,%2\n\t"
        "1:\n\t"
        "vle8.v v0,(t2)\n\t addi t2,t2,128\n\t"
        "mv t1,t3\n\t"
        "vle8.v v1,(t1)\n\t addi t1,t1,128\n\t vle8.v v2,(t1)\n\t addi t1,t1,128\n\t"
        "vle8.v v3,(t1)\n\t addi t1,t1,128\n\t vle8.v v4,(t1)\n\t addi t1,t1,128\n\t"
        "vle8.v v5,(t1)\n\t addi t1,t1,128\n\t vle8.v v6,(t1)\n\t addi t1,t1,128\n\t"
        "vle8.v v7,(t1)\n\t addi t1,t1,128\n\t vle8.v v8,(t1)\n\t"
        "addi t3,t3,1024\n\t"
        "vmadot v16,v0,v1\n\t vmadot v18,v0,v2\n\t vmadot v20,v0,v3\n\t vmadot v22,v0,v4\n\t"
        "vmadot v24,v0,v5\n\t vmadot v26,v0,v6\n\t vmadot v28,v0,v7\n\t vmadot v30,v0,v8\n\t"
        "addi t4,t4,-1\n\t bnez t4,1b\n\t"
        :: "r"(A),"r"(B),"r"(Kb)
        : "t0","t1","t2","t3","t4","v0","v1","v2","v3","v4","v5","v6","v7","v8",
          "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory");
}

int main(int c,char**v){
    int nt=(c>1)?atoi(v[1]):8;
    int Kc=(c>2)?atoi(v[2]):256;          /* A=8*Kc + B=64*Kc bytes; Kc=256 -> 18KB (L1) */
    int reps=(c>3)?atoi(v[3]):200000;
    int Kb=Kc/16;
    printf("threads=%d Kc=%d (working set %dKB, %s) reps=%d\n", nt, Kc, (8*Kc+64*Kc)/1024,
           (8*Kc+64*Kc)<=64*1024?"L1":((8*Kc+64*Kc)<=1024*1024?"L2":"DRAM"), reps);
    double t0=now();
    #pragma omp parallel num_threads(nt)
    {
        int tn=omp_get_thread_num();
        bind_ai();
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(8+(tn%8),&s); sched_setaffinity(0,sizeof(s),&s);
        for(int i=0;i<5;i++) sched_yield();
        int8_t*A=aligned_alloc(64,8*Kc), *B=aligned_alloc(64,64*Kc);
        for(int i=0;i<8*Kc;i++)A[i]=(int8_t)(i-64); for(int i=0;i<64*Kc;i++)B[i]=(int8_t)(i*3);
        for(int r=0;r<reps;r++) uk(A,B,Kb);
    }
    double dt=now()-t0;
    /* work: reps * nt microkernels, each Kb*8 vmadot, each 1024 MAC */
    double vmadot=(double)reps*nt*Kb*8;
    double tops=vmadot*1024*2/dt/1e12;
    printf("L1-fed 8x64 kernel: %.3e vmadot/s | %.2f TOPS int8\n", vmadot/dt, tops);
    return 0;
}
