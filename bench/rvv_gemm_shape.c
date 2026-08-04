/* rvv_gemm_shape.c — pick the register-blocking shape for rosetta_rvv.c's GEMM.
 *
 * Runs the exact inner loop shape (MR accumulator rows x one e32m2 vector of
 * columns, A broadcast as scalars from a packed panel, B streamed contiguously)
 * over data small enough to stay resident, so the number reported is kernel
 * efficiency alone -- no DRAM, no im2col, no threading.
 *
 * Build: gcc -O3 -fno-tree-vectorize -march=rv64gcv_zfh_zvfh_xsmtvdotii \
 *            -o rvv_gemm_shape bench/rvv_gemm_shape.c -lm
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
static void bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd>=0){ ssize_t r=write(fd,"0",1);(void)r; close(fd);} }

#define KLEN 288          /* matches KB_TARGET: one 32-channel 3x3 k-block */
#define REPS 40000

/* MR=8, one m2 column vector (the shape rosetta_rvv.c ships today). */
static void k_mr8(const float *ap,const float *bt,float *cc,int NT,size_t vl){
    vfloat32m2_t a0=__riscv_vle32_v_f32m2(cc+0*NT,vl),a1=__riscv_vle32_v_f32m2(cc+1*NT,vl);
    vfloat32m2_t a2=__riscv_vle32_v_f32m2(cc+2*NT,vl),a3=__riscv_vle32_v_f32m2(cc+3*NT,vl);
    vfloat32m2_t a4=__riscv_vle32_v_f32m2(cc+4*NT,vl),a5=__riscv_vle32_v_f32m2(cc+5*NT,vl);
    vfloat32m2_t a6=__riscv_vle32_v_f32m2(cc+6*NT,vl),a7=__riscv_vle32_v_f32m2(cc+7*NT,vl);
    for(int k=0;k<KLEN;k++,ap+=8,bt+=NT){
        vfloat32m2_t vb=__riscv_vle32_v_f32m2(bt,vl);
        a0=__riscv_vfmacc_vf_f32m2(a0,ap[0],vb,vl); a1=__riscv_vfmacc_vf_f32m2(a1,ap[1],vb,vl);
        a2=__riscv_vfmacc_vf_f32m2(a2,ap[2],vb,vl); a3=__riscv_vfmacc_vf_f32m2(a3,ap[3],vb,vl);
        a4=__riscv_vfmacc_vf_f32m2(a4,ap[4],vb,vl); a5=__riscv_vfmacc_vf_f32m2(a5,ap[5],vb,vl);
        a6=__riscv_vfmacc_vf_f32m2(a6,ap[6],vb,vl); a7=__riscv_vfmacc_vf_f32m2(a7,ap[7],vb,vl);
    }
    __riscv_vse32_v_f32m2(cc+0*NT,a0,vl); __riscv_vse32_v_f32m2(cc+1*NT,a1,vl);
    __riscv_vse32_v_f32m2(cc+2*NT,a2,vl); __riscv_vse32_v_f32m2(cc+3*NT,a3,vl);
    __riscv_vse32_v_f32m2(cc+4*NT,a4,vl); __riscv_vse32_v_f32m2(cc+5*NT,a5,vl);
    __riscv_vse32_v_f32m2(cc+6*NT,a6,vl); __riscv_vse32_v_f32m2(cc+7*NT,a7,vl);
}

/* MR=14: same vector load feeds 14 FMAs instead of 8 (28 of 32 vregs held). */
static void k_mr14(const float *ap,const float *bt,float *cc,int NT,size_t vl){
    vfloat32m2_t a0=__riscv_vle32_v_f32m2(cc+0*NT,vl),a1=__riscv_vle32_v_f32m2(cc+1*NT,vl);
    vfloat32m2_t a2=__riscv_vle32_v_f32m2(cc+2*NT,vl),a3=__riscv_vle32_v_f32m2(cc+3*NT,vl);
    vfloat32m2_t a4=__riscv_vle32_v_f32m2(cc+4*NT,vl),a5=__riscv_vle32_v_f32m2(cc+5*NT,vl);
    vfloat32m2_t a6=__riscv_vle32_v_f32m2(cc+6*NT,vl),a7=__riscv_vle32_v_f32m2(cc+7*NT,vl);
    vfloat32m2_t a8=__riscv_vle32_v_f32m2(cc+8*NT,vl),a9=__riscv_vle32_v_f32m2(cc+9*NT,vl);
    vfloat32m2_t aa=__riscv_vle32_v_f32m2(cc+10*NT,vl),ab=__riscv_vle32_v_f32m2(cc+11*NT,vl);
    vfloat32m2_t ac=__riscv_vle32_v_f32m2(cc+12*NT,vl),ad=__riscv_vle32_v_f32m2(cc+13*NT,vl);
    for(int k=0;k<KLEN;k++,ap+=14,bt+=NT){
        vfloat32m2_t vb=__riscv_vle32_v_f32m2(bt,vl);
        a0=__riscv_vfmacc_vf_f32m2(a0,ap[0],vb,vl); a1=__riscv_vfmacc_vf_f32m2(a1,ap[1],vb,vl);
        a2=__riscv_vfmacc_vf_f32m2(a2,ap[2],vb,vl); a3=__riscv_vfmacc_vf_f32m2(a3,ap[3],vb,vl);
        a4=__riscv_vfmacc_vf_f32m2(a4,ap[4],vb,vl); a5=__riscv_vfmacc_vf_f32m2(a5,ap[5],vb,vl);
        a6=__riscv_vfmacc_vf_f32m2(a6,ap[6],vb,vl); a7=__riscv_vfmacc_vf_f32m2(a7,ap[7],vb,vl);
        a8=__riscv_vfmacc_vf_f32m2(a8,ap[8],vb,vl); a9=__riscv_vfmacc_vf_f32m2(a9,ap[9],vb,vl);
        aa=__riscv_vfmacc_vf_f32m2(aa,ap[10],vb,vl); ab=__riscv_vfmacc_vf_f32m2(ab,ap[11],vb,vl);
        ac=__riscv_vfmacc_vf_f32m2(ac,ap[12],vb,vl); ad=__riscv_vfmacc_vf_f32m2(ad,ap[13],vb,vl);
    }
    __riscv_vse32_v_f32m2(cc+0*NT,a0,vl); __riscv_vse32_v_f32m2(cc+1*NT,a1,vl);
    __riscv_vse32_v_f32m2(cc+2*NT,a2,vl); __riscv_vse32_v_f32m2(cc+3*NT,a3,vl);
    __riscv_vse32_v_f32m2(cc+4*NT,a4,vl); __riscv_vse32_v_f32m2(cc+5*NT,a5,vl);
    __riscv_vse32_v_f32m2(cc+6*NT,a6,vl); __riscv_vse32_v_f32m2(cc+7*NT,a7,vl);
    __riscv_vse32_v_f32m2(cc+8*NT,a8,vl); __riscv_vse32_v_f32m2(cc+9*NT,a9,vl);
    __riscv_vse32_v_f32m2(cc+10*NT,aa,vl); __riscv_vse32_v_f32m2(cc+11*NT,ab,vl);
    __riscv_vse32_v_f32m2(cc+12*NT,ac,vl); __riscv_vse32_v_f32m2(cc+13*NT,ad,vl);
}

/* MR=8 at LMUL=1: half the bytes per vector load, half the work per FMA. */
static void k_mr8_m1(const float *ap,const float *bt,float *cc,int NT,size_t vl){
    vfloat32m1_t a0=__riscv_vle32_v_f32m1(cc+0*NT,vl),a1=__riscv_vle32_v_f32m1(cc+1*NT,vl);
    vfloat32m1_t a2=__riscv_vle32_v_f32m1(cc+2*NT,vl),a3=__riscv_vle32_v_f32m1(cc+3*NT,vl);
    vfloat32m1_t a4=__riscv_vle32_v_f32m1(cc+4*NT,vl),a5=__riscv_vle32_v_f32m1(cc+5*NT,vl);
    vfloat32m1_t a6=__riscv_vle32_v_f32m1(cc+6*NT,vl),a7=__riscv_vle32_v_f32m1(cc+7*NT,vl);
    for(int k=0;k<KLEN;k++,ap+=8,bt+=NT){
        vfloat32m1_t vb=__riscv_vle32_v_f32m1(bt,vl);
        a0=__riscv_vfmacc_vf_f32m1(a0,ap[0],vb,vl); a1=__riscv_vfmacc_vf_f32m1(a1,ap[1],vb,vl);
        a2=__riscv_vfmacc_vf_f32m1(a2,ap[2],vb,vl); a3=__riscv_vfmacc_vf_f32m1(a3,ap[3],vb,vl);
        a4=__riscv_vfmacc_vf_f32m1(a4,ap[4],vb,vl); a5=__riscv_vfmacc_vf_f32m1(a5,ap[5],vb,vl);
        a6=__riscv_vfmacc_vf_f32m1(a6,ap[6],vb,vl); a7=__riscv_vfmacc_vf_f32m1(a7,ap[7],vb,vl);
    }
    __riscv_vse32_v_f32m1(cc+0*NT,a0,vl); __riscv_vse32_v_f32m1(cc+1*NT,a1,vl);
    __riscv_vse32_v_f32m1(cc+2*NT,a2,vl); __riscv_vse32_v_f32m1(cc+3*NT,a3,vl);
    __riscv_vse32_v_f32m1(cc+4*NT,a4,vl); __riscv_vse32_v_f32m1(cc+5*NT,a5,vl);
    __riscv_vse32_v_f32m1(cc+6*NT,a6,vl); __riscv_vse32_v_f32m1(cc+7*NT,a7,vl);
}

int main(int argc,char**argv){
    int hart=argc>1?atoi(argv[1]):8;
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart,&cs);
    if(sched_setaffinity(0,sizeof(cs),&cs)!=0) perror("setaffinity");

    size_t vl2=__riscv_vsetvlmax_e32m2(), vl1=__riscv_vsetvlmax_e32m1();
    int NT2=(int)vl2, NT1=(int)vl1;
    float *A,*B,*C;
    if(posix_memalign((void**)&A,128,(size_t)KLEN*16*sizeof(float))) return 1;
    if(posix_memalign((void**)&B,128,(size_t)KLEN*NT2*sizeof(float))) return 1;
    if(posix_memalign((void**)&C,128,(size_t)16*NT2*sizeof(float))) return 1;
    for(size_t i=0;i<(size_t)KLEN*16;i++) A[i]=(float)(i%7)*0.01f;
    for(size_t i=0;i<(size_t)KLEN*NT2;i++) B[i]=(float)(i%5)*0.01f;
    memset(C,0,(size_t)16*NT2*sizeof(float));

    printf("hart %d  KLEN=%d  (A %zu KB, B %zu KB, C %zu KB resident)\n",hart,KLEN,
           (size_t)KLEN*16*4/1024,(size_t)KLEN*NT2*4/1024,(size_t)16*NT2*4/1024);

    double t0=now_s();
    for(int r=0;r<REPS;r++) k_mr8(A,B,C,NT2,vl2);
    double dt=now_s()-t0;
    printf("  MR=8  m2 vl=%d : %7.2f GFLOP/s\n",NT2,2.0*8*NT2*KLEN*REPS/dt/1e9);

    t0=now_s();
    for(int r=0;r<REPS;r++) k_mr14(A,B,C,NT2,vl2);
    dt=now_s()-t0;
    printf("  MR=14 m2 vl=%d : %7.2f GFLOP/s\n",NT2,2.0*14*NT2*KLEN*REPS/dt/1e9);

    t0=now_s();
    for(int r=0;r<REPS;r++) k_mr8_m1(A,B,C,NT1,vl1);
    dt=now_s()-t0;
    printf("  MR=8  m1 vl=%d : %7.2f GFLOP/s\n",NT1,2.0*8*NT1*KLEN*REPS/dt/1e9);

    volatile float sink=C[0]+C[NT2]; (void)sink;
    return 0;
}
