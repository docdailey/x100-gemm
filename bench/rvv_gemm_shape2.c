/* rvv_gemm_shape2.c — re-measure the GEMM register-blocking shape at the K values
 * ocr_rvv.c actually uses, with the accumulator living in memory the way the
 * engine's Cacc does (phase-1's probe held it in registers across the whole K).
 *
 * PP-OCRv6 Tiny is dominated by 1x1 convs, so K is the input channel count
 * (48..320) rather than the 288 a 32-channel 3x3 block gives. Short K changes
 * the balance: the accumulator load/store and loop overhead are amortised over
 * fewer FMAs, which is exactly the regime phase-1's KLEN=288 probe did not test.
 *
 * Build: gcc -O3 -fno-tree-vectorize -funroll-loops -fno-stack-protector \
 *            -march=rv64gcv_zfh_zvfh_xsmtvdotii -o rvv_gemm_shape2 rvv_gemm_shape2.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <riscv_vector.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static float *A,*B,*C;

#define FM(a,s,b) a=__riscv_vfmacc_vf_f32m2(a,s,b,vl)
#define LD(p) __riscv_vle32_v_f32m2(p,vl)
#define ST(p,v) __riscv_vse32_v_f32m2(p,v,vl)

/* what the engine ships today: 8 output rows x one e32m2 column vector */
static double mr8(int K,int NT,size_t vl,long reps){
    double t0=now_s();
    for(long r=0;r<reps;r++){
        for(int n0=0;n0<NT;n0+=(int)vl){
            const float *ap=A; const float *bt=B+n0;
            vfloat32m2_t a0=LD(C+0*NT+n0),a1=LD(C+1*NT+n0),a2=LD(C+2*NT+n0),a3=LD(C+3*NT+n0);
            vfloat32m2_t a4=LD(C+4*NT+n0),a5=LD(C+5*NT+n0),a6=LD(C+6*NT+n0),a7=LD(C+7*NT+n0);
            for(int k=0;k<K;k++,ap+=8,bt+=NT){
                vfloat32m2_t b0=LD(bt);
                FM(a0,ap[0],b0); FM(a1,ap[1],b0); FM(a2,ap[2],b0); FM(a3,ap[3],b0);
                FM(a4,ap[4],b0); FM(a5,ap[5],b0); FM(a6,ap[6],b0); FM(a7,ap[7],b0);
            }
            ST(C+0*NT+n0,a0); ST(C+1*NT+n0,a1); ST(C+2*NT+n0,a2); ST(C+3*NT+n0,a3);
            ST(C+4*NT+n0,a4); ST(C+5*NT+n0,a5); ST(C+6*NT+n0,a6); ST(C+7*NT+n0,a7);
        }
    }
    return 2.0*8*NT*K*reps/(now_s()-t0)/1e9;
}

/* the phase-1 candidate: 4 output rows x three column vectors.
 * 12 m2 accumulators + 3 m2 operands is 30 of 32 vector registers; letting gcc
 * unroll on top of that makes it spill vectors and miscompile (segfault). */
__attribute__((optimize("no-unroll-loops")))
static double mr4nc3(int K,int NT,size_t vl,long reps){
    int step=3*(int)vl;
    double t0=now_s();
    for(long r=0;r<reps;r++){
        for(int n0=0;n0+step<=NT;n0+=step){
            const float *ap=A; const float *bt=B+n0;
            vfloat32m2_t c00=LD(C+0*NT+n0),c01=LD(C+0*NT+n0+vl),c02=LD(C+0*NT+n0+2*vl);
            vfloat32m2_t c10=LD(C+1*NT+n0),c11=LD(C+1*NT+n0+vl),c12=LD(C+1*NT+n0+2*vl);
            vfloat32m2_t c20=LD(C+2*NT+n0),c21=LD(C+2*NT+n0+vl),c22=LD(C+2*NT+n0+2*vl);
            vfloat32m2_t c30=LD(C+3*NT+n0),c31=LD(C+3*NT+n0+vl),c32=LD(C+3*NT+n0+2*vl);
            for(int k=0;k<K;k++,ap+=4,bt+=NT){
                vfloat32m2_t b0=LD(bt),b1=LD(bt+vl),b2=LD(bt+2*vl);
                float s0=ap[0],s1=ap[1],s2=ap[2],s3=ap[3];
                FM(c00,s0,b0); FM(c01,s0,b1); FM(c02,s0,b2);
                FM(c10,s1,b0); FM(c11,s1,b1); FM(c12,s1,b2);
                FM(c20,s2,b0); FM(c21,s2,b1); FM(c22,s2,b2);
                FM(c30,s3,b0); FM(c31,s3,b1); FM(c32,s3,b2);
            }
            ST(C+0*NT+n0,c00); ST(C+0*NT+n0+vl,c01); ST(C+0*NT+n0+2*vl,c02);
            ST(C+1*NT+n0,c10); ST(C+1*NT+n0+vl,c11); ST(C+1*NT+n0+2*vl,c12);
            ST(C+2*NT+n0,c20); ST(C+2*NT+n0+vl,c21); ST(C+2*NT+n0+2*vl,c22);
            ST(C+3*NT+n0,c30); ST(C+3*NT+n0+vl,c31); ST(C+3*NT+n0+2*vl,c32);
        }
    }
    return 2.0*4*(NT/step)*step*K*reps/(now_s()-t0)/1e9;
}

int main(void){
    int fd=open("/proc/set_ai_thread",O_WRONLY);
    if(fd>=0){ ssize_t r=write(fd,"0",1); (void)r; close(fd); }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs); sched_setaffinity(0,sizeof(cs),&cs);

    size_t vl=__riscv_vsetvlmax_e32m2();
    int NT=192;                       /* 3 * VLMAX, so both shapes tile it exactly */
    if(posix_memalign((void**)&A,128,(size_t)4096*8*sizeof(float))) return 1;
    if(posix_memalign((void**)&B,128,(size_t)4096*NT*sizeof(float))) return 1;
    if(posix_memalign((void**)&C,128,(size_t)8*NT*sizeof(float))) return 1;
    for(int i=0;i<4096*8;i++) A[i]=0.001f*(i%7);
    for(size_t i=0;i<(size_t)4096*NT;i++) B[i]=0.001f*(i%5);
    memset(C,0,(size_t)8*NT*sizeof(float));

    printf("%-6s %-14s %-14s %s\n","K","MR=8/NC=1","MR=4/NC=3","ratio");
    int ks[]={48,96,160,288,512};
    for(int i=0;i<5;i++){
        long reps=(long)(8e8/(8.0*NT*ks[i]));
        double a=mr8(ks[i],NT,vl,reps);
        double b=mr4nc3(ks[i],NT,vl,reps);
        printf("%-6d %-14.2f %-14.2f %.2fx\n",ks[i],a,b,b/a);
    }
    return 0;
}
