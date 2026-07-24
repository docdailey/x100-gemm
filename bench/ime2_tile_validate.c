/* ime2_tile_validate.c — confirm the VLEN-1024 vmadot tile geometry on an A100 core.
 *
 * Hypothesis from the IME-2 spec + the official 4x8x4 demo scaled 4x:
 *   at VLEN 1024, one vmadot computes an M=8, N=8, K=16 int8 tile -> 8x8 int32.
 *   A operand  = A[MxK] row-major, 8x16 = 128 B (one e8,m1 reg)
 *   B operand  = packB[NxK], packB[n*K+k] = B[k*N+n]  (K x N transposed), 128 B
 *   C accum    = 8x8 int32 (e32,m2)
 * Validate against a scalar reference. If mismatch, dump C vs CRef to read the layout.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o ime2_tile_validate ime2_tile_validate.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>

#define M 8
#define N 8
#define K 16

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static unsigned long vlen(void){ unsigned long v; __asm__ volatile("csrr %0,vlenb":"=r"(v)); return v*8; }

static void ref_gemm(const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ int32_t a=0; for(int k=0;k<K;k++) a+=(int32_t)A[m*K+k]*(int32_t)B[k*N+n]; C[m*N+n]=a; }
}
/* transpose B (KxN) -> packB (NxK): packB[n*K+k] = B[k*N+n] */
static void packB(const int8_t*B,int8_t*P){ for(int n=0;n<N;n++)for(int k=0;k<K;k++) P[n*K+k]=B[k*N+n]; }

static void tile_vmadot(const int8_t*A,const int8_t*P,int32_t*C){
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t" "vxor.vv v28,v28,v28\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "vle8.v v0,(%0)\n\t" "vle8.v v1,(%1)\n\t"
        "vmadot v28,v0,v1\n\t"
        "vsetvli t0,zero,e32,m2\n\t" "vse32.v v28,(%2)\n\t"
        :: "r"(A),"r"(P),"r"(C) : "t0","v0","v1","v28","v29","memory");
}

int main(void){
    bind_ai(); for(int i=0;i<5;i++) sched_yield();
    printf("hart=%d VLEN=%lu (need 1024)\n", sched_getcpu(), vlen());
    if (vlen()!=1024){ printf("NOT on a 1024-bit core; abort\n"); return 2; }

    int8_t A[M*K], B[K*N], P[N*K]; int32_t C[M*N]={0}, CRef[M*N]={0};
    srand(12345);
    for(int i=0;i<M*K;i++) A[i]=(int8_t)(rand()%256-128);
    for(int i=0;i<K*N;i++) B[i]=(int8_t)(rand()%256-128);

    ref_gemm(A,B,CRef);
    packB(B,P);
    tile_vmadot(A,P,C);

    int bad=0;
    for(int i=0;i<M*N;i++) if(C[i]!=CRef[i]) bad++;
    if(!bad){ printf(">>> TILE VALID: C == CRef for full %dx%dx%d tile <<<\n",M,N,K); return 0; }

    printf("MISMATCH (%d/%d). C vs CRef:\n", bad, M*N);
    for(int m=0;m<M;m++){ for(int n=0;n<N;n++) printf("%6d",C[m*N+n]); printf("   |");
                          for(int n=0;n<N;n++) printf("%6d",CRef[m*N+n]); printf("\n"); }
    return 1;
}
