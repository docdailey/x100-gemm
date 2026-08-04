/* rvv_f32_peak.c — FP32 vfmacc throughput ceiling for one SpaceMIT A100 hart.
 *
 * Establishes what rosetta_rvv.c's GEMM can possibly reach, so a measured
 * GFLOPS number can be read as a fraction of the machine rather than in a
 * vacuum. Everything is register-resident: no loads in the timed loop, N
 * independent accumulators to cover FMA latency.
 *
 * Build: gcc -O3 -fno-tree-vectorize -march=rv64gcv_zfh_zvfh_xsmtvdotii \
 *            -o rvv_f32_peak bench/rvv_f32_peak.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <riscv_vector.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd>=0){ ssize_t r=write(fd,"0",1); (void)r; close(fd);} }

#define ITERS 2000000

static double peak_m1(void){
    size_t vl=__riscv_vsetvlmax_e32m1();
    double t0=now_s();
    vfloat32m1_t a0=__riscv_vfmv_v_f_f32m1(1.0f,vl),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    vfloat32m1_t b=__riscv_vfmv_v_f_f32m1(1.000001f,vl);
    for(long i=0;i<ITERS;i++){
        a0=__riscv_vfmacc_vf_f32m1(a0,1.0000001f,b,vl);
        a1=__riscv_vfmacc_vf_f32m1(a1,1.0000001f,b,vl);
        a2=__riscv_vfmacc_vf_f32m1(a2,1.0000001f,b,vl);
        a3=__riscv_vfmacc_vf_f32m1(a3,1.0000001f,b,vl);
        a4=__riscv_vfmacc_vf_f32m1(a4,1.0000001f,b,vl);
        a5=__riscv_vfmacc_vf_f32m1(a5,1.0000001f,b,vl);
        a6=__riscv_vfmacc_vf_f32m1(a6,1.0000001f,b,vl);
        a7=__riscv_vfmacc_vf_f32m1(a7,1.0000001f,b,vl);
    }
    volatile float sink=__riscv_vfmv_f_s_f32m1_f32(a0)+__riscv_vfmv_f_s_f32m1_f32(a7); (void)sink;
    double dt=now_s()-t0;
    printf("  m1 vl=%zu : %.2f GFLOP/s\n",vl,8.0*2.0*vl*ITERS/dt/1e9);
    return 8.0*2.0*vl*ITERS/dt/1e9;
}

static double peak_m2(void){
    size_t vl=__riscv_vsetvlmax_e32m2();
    double t0=now_s();
    vfloat32m2_t a0=__riscv_vfmv_v_f_f32m2(1.0f,vl),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    vfloat32m2_t b=__riscv_vfmv_v_f_f32m2(1.000001f,vl);
    for(long i=0;i<ITERS;i++){
        a0=__riscv_vfmacc_vf_f32m2(a0,1.0000001f,b,vl);
        a1=__riscv_vfmacc_vf_f32m2(a1,1.0000001f,b,vl);
        a2=__riscv_vfmacc_vf_f32m2(a2,1.0000001f,b,vl);
        a3=__riscv_vfmacc_vf_f32m2(a3,1.0000001f,b,vl);
        a4=__riscv_vfmacc_vf_f32m2(a4,1.0000001f,b,vl);
        a5=__riscv_vfmacc_vf_f32m2(a5,1.0000001f,b,vl);
        a6=__riscv_vfmacc_vf_f32m2(a6,1.0000001f,b,vl);
        a7=__riscv_vfmacc_vf_f32m2(a7,1.0000001f,b,vl);
    }
    volatile float sink=__riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m2_f32m1(a0,0)); (void)sink;
    double dt=now_s()-t0;
    printf("  m2 vl=%zu : %.2f GFLOP/s\n",vl,8.0*2.0*vl*ITERS/dt/1e9);
    return 8.0*2.0*vl*ITERS/dt/1e9;
}

static double peak_m4(void){
    size_t vl=__riscv_vsetvlmax_e32m4();
    double t0=now_s();
    vfloat32m4_t a0=__riscv_vfmv_v_f_f32m4(1.0f,vl),a1=a0,a2=a0,a3=a0;
    vfloat32m4_t b=__riscv_vfmv_v_f_f32m4(1.000001f,vl);
    for(long i=0;i<ITERS;i++){
        a0=__riscv_vfmacc_vf_f32m4(a0,1.0000001f,b,vl);
        a1=__riscv_vfmacc_vf_f32m4(a1,1.0000001f,b,vl);
        a2=__riscv_vfmacc_vf_f32m4(a2,1.0000001f,b,vl);
        a3=__riscv_vfmacc_vf_f32m4(a3,1.0000001f,b,vl);
    }
    volatile float sink=__riscv_vfmv_f_s_f32m1_f32(__riscv_vget_v_f32m4_f32m1(a0,0)); (void)sink;
    double dt=now_s()-t0;
    printf("  m4 vl=%zu : %.2f GFLOP/s\n",vl,4.0*2.0*vl*ITERS/dt/1e9);
    return 4.0*2.0*vl*ITERS/dt/1e9;
}

int main(int argc,char**argv){
    int hart=argc>1?atoi(argv[1]):8;
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart,&cs);
    if(sched_setaffinity(0,sizeof(cs),&cs)!=0) perror("setaffinity");
    printf("hart %d, vlenb=%d bytes (VLEN=%d bits)\n",hart,(int)__riscv_vlenb(),(int)__riscv_vlenb()*8);
    peak_m1(); peak_m2(); peak_m4();
    return 0;
}
