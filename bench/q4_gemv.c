/* q4_gemv.c — q4-in-q8 interlaced GEMV on the A100 IME-2 (the decode kernel).
 *
 * y[M,N] = W[N,K] @ x[M,K].  W int4 (2/byte, INTERLACED: within each 128-int8 vmadot B-tile,
 * packed byte j holds tile-lane j in the low nibble and lane j+64 in the high nibble -> unpack is
 * vsll/vsra + one vslideup, no shuffle). x int8, y int32. vmadot tile 8x16x8 (M0=8,K0=16,N0=8);
 * M rows = tokens (1 = decode, up to 8 = speculative/MTP fills the tile). N-block outer / K-block
 * inner: x reused from L1, W streamed once = bandwidth-optimal (decode is bandwidth-bound; int4
 * halves the bytes/token). Tensor-parallel: split N across the 4 IME units (harts 8/10/12/14).
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o q4_gemv q4_gemv.c
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

#define M0 8
#define K0 16
#define N0 8
#define TILE 128            /* int8 per vmadot tile (8x16) */
#define QTILE 64            /* int4 bytes per tile */

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* pack W[N,K] int8 -> Wq int4, K-block-pair interlaced, tiled [nb][kp][128 bytes].
 * Pairs consecutive K-blocks (2kp, 2kp+1): packed byte j holds tile(2kp)[j] in the low
 * nibble and tile(2kp+1)[j] in the high nibble. Unpack is then vsll/vsra (lo) + vsra (hi)
 * -> two full 128-int8 B-tiles from ONE 128-byte load, staying in e8,m1: no vsetvli switch,
 * no vslideup. Requires Kb even (all Qwen dims are K%32==0). Same total bytes as before. */
static void pack_w_int4(int Nt,int Kt,const int8_t*W,uint8_t*Wq){
    int Nb=Nt/N0, Kb=Kt/K0, Kp=Kb/2;
    for(int nb=0;nb<Nb;nb++)for(int kp=0;kp<Kp;kp++){
        int8_t t0[TILE],t1[TILE]; int kb0=2*kp,kb1=2*kp+1;
        for(int n=0;n<N0;n++)for(int k=0;k<K0;k++){
            t0[n*K0+k]=W[(nb*N0+n)*Kt+kb0*K0+k];
            t1[n*K0+k]=W[(nb*N0+n)*Kt+kb1*K0+k];
        }
        uint8_t*d=Wq+((size_t)(nb*Kp+kp))*TILE;
        for(int j=0;j<TILE;j++) d[j]=(uint8_t)((t0[j]&0xf)|((t1[j]&0xf)<<4));
    }
}
/* pack x[M,K] int8 -> xt[kb][128] (M rows, rest 0) */
static void pack_x(int M,int Kt,const int8_t*x,int8_t*xt){
    int Kb=Kt/K0; memset(xt,0,(size_t)Kb*TILE);
    for(int kb=0;kb<Kb;kb++)for(int m=0;m<M;m++)for(int k=0;k<K0;k++) xt[kb*TILE+m*K0+k]=x[m*Kt+kb*K0+k];
}
/* int8 packing of W for the baseline (full 128 int8 per tile) */
static void pack_w_int8(int Nt,int Kt,const int8_t*W,int8_t*Wp){
    int Nb=Nt/N0, Kb=Kt/K0;
    for(int nb=0;nb<Nb;nb++)for(int kb=0;kb<Kb;kb++){
        int8_t*d=Wp+((size_t)(nb*Kb+kb))*TILE;
        for(int n=0;n<N0;n++)for(int k=0;k<K0;k++) d[n*K0+k]=W[(nb*N0+n)*Kt+kb*K0+k];
    }
}

/* one N-block: accumulate over K, C[8,8] -> ctmp. int4 weights, K-block-pair interlaced.
 * Each iteration consumes 2 K-blocks: one 128-byte weight load + vsll/vsra/vsra -> two
 * 128-int8 B-tiles (lo,hi) feeding two vmadots into decoupled accumulators v28/v29 & v30/v31
 * (summed at the end). Stays in e8,m1 -> no per-tile vsetvli switch, no vslideup. Kb must be even. */
static void gemv_nb_int4(const int8_t*xt,const uint8_t*Wq,int Kb,int32_t*ctmp){
    long Kp=Kb/2;
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t vxor.vv v28,v28,v28\n\t vxor.vv v30,v30,v30\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t"
        "vle8.v v0,(%0)\n\t addi %0,%0,128\n\t"                 /* A tile for K-block 2kp   */
        "vle8.v v2,(%0)\n\t addi %0,%0,128\n\t"                 /* A tile for K-block 2kp+1 */
        "vle8.v v4,(%1)\n\t addi %1,%1,128\n\t"                 /* 128 packed bytes = 2 B-tiles */
        "vsll.vi v5,v4,4\n\t vsra.vi v5,v5,4\n\t"               /* v5 = lo nibble = B-tile 2kp   */
        "vsra.vi v6,v4,4\n\t"                                   /* v6 = hi nibble = B-tile 2kp+1 */
        "vmadot v28,v0,v5\n\t"
        "vmadot v30,v2,v6\n\t"
        "addi %2,%2,-1\n\t bnez %2,1b\n\t"
        "vsetvli t0,zero,e32,m2\n\t vadd.vv v28,v28,v30\n\t vse32.v v28,(%3)\n\t"
        : "+r"(xt),"+r"(Wq),"+r"(Kp)
        : "r"(ctmp)
        : "t0","v0","v2","v4","v5","v6","v28","v29","v30","v31","memory");
}
/* baseline int8 weights */
static void gemv_nb_int8(const int8_t*xt,const int8_t*Wp,int Kb,int32_t*ctmp){
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t vxor.vv v28,v28,v28\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t"
        "vle8.v v0,(%0)\n\t addi %0,%0,128\n\t"
        "vle8.v v1,(%1)\n\t addi %1,%1,128\n\t"
        "vmadot v28,v0,v1\n\t"
        "addi %2,%2,-1\n\t bnez %2,1b\n\t"
        "vsetvli t0,zero,e32,m2\n\t vse32.v v28,(%3)\n\t"
        : "+r"(xt),"+r"(Wp),"+r"(Kb) : "r"(ctmp)
        : "t0","v0","v1","v28","v29","memory");
}

static void ref_gemv(int M,int Nt,int Kt,const int8_t*W,const int8_t*x,int32_t*y){
    for(int m=0;m<M;m++)for(int n=0;n<Nt;n++){ int32_t a=0; for(int k=0;k<Kt;k++) a+=(int32_t)W[n*Kt+k]*(int32_t)x[m*Kt+k]; y[m*Nt+n]=a; }
}

int main(int c,char**v){
    int Nt=(c>1)?atoi(v[1]):4864, Kt=(c>2)?atoi(v[2]):896;  /* Qwen0.5B FFN-ish */
    int M=(c>3)?atoi(v[3]):1, nt=(c>4)?atoi(v[4]):1, reps=(c>5)?atoi(v[5]):200;
    if(Nt%8||Kt%32){ printf("need N%%8,K%%32 (int4 pairs 2 K-blocks)\n"); return 2; }
    /* the vmadot tile is 8x16x8 only on A100 (VLEN 1024) — pin the main thread there for correctness */
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    int Nb=Nt/N0, Kb=Kt/K0;
    int8_t*W=malloc((size_t)Nt*Kt); int8_t*x=malloc((size_t)M*Kt);
    srand(7); for(size_t i=0;i<(size_t)Nt*Kt;i++)W[i]=(int8_t)(rand()%15-7);  /* int4 range */
    for(size_t i=0;i<(size_t)M*Kt;i++)x[i]=(int8_t)(rand()%256-128);
    uint8_t*Wq=malloc((size_t)Nb*Kb*QTILE); pack_w_int4(Nt,Kt,W,Wq);
    int8_t*Wp=malloc((size_t)Nb*Kb*TILE);   pack_w_int8(Nt,Kt,W,Wp);
    int8_t*xt=malloc((size_t)Kb*TILE);       pack_x(M,Kt,x,xt);
    int32_t*y4=malloc((size_t)Nt*4),*y8=malloc((size_t)Nt*4),*yr=malloc((size_t)M*Nt*4);
    ref_gemv(M,Nt,Kt,W,x,yr);

    /* correctness (int4, single thread) */
    for(int nb=0;nb<Nb;nb++){ int32_t ct[64]; gemv_nb_int4(xt,Wq+(size_t)nb*Kb*QTILE,Kb,ct);
        for(int n=0;n<N0;n++) y4[nb*N0+n]=ct[0*N0+n]; }
    if(Nb==1 && Kb==1){  /* debug: dump full 8x8 vmadot output vs expected C[m,n]=sum_k A_pad[m,k]*W[n,k] */
        int32_t ct[64]; gemv_nb_int8(xt,Wp,1,ct);
        printf("DEBUG 8x8 grid (ct[m*8+n]) vs expected:\n");
        for(int m=0;m<8;m++){ printf(" m%d ct:",m); for(int n=0;n<8;n++) printf("%7d",ct[m*8+n]);
            printf("   exp:"); for(int n=0;n<8;n++){ int32_t e=0; for(int k=0;k<K0;k++) e+=(int32_t)xt[m*16+k]*(int32_t)W[n*Kt+k]; printf("%7d",e);} printf("\n"); }
    }
    int bad=0; for(int n=0;n<Nt;n++) if(y4[n]!=yr[n]) bad++;
    printf("int4 GEMV %dx%d M=%d: %s (%d/%d mismatch)\n",Nt,Kt,M, bad?"FAIL":"y==ref OK", bad, Nt);
    /* isolate: validate int8 path too */
    for(int nb=0;nb<Nb;nb++){ int32_t ct[64]; gemv_nb_int8(xt,Wp+(size_t)nb*Kb*TILE,Kb,ct);
        for(int n=0;n<N0;n++) y8[nb*N0+n]=ct[0*N0+n]; }
    int bad8=0; for(int n=0;n<Nt;n++) if(y8[n]!=yr[n]) bad8++;
    printf("int8 GEMV: %s (%d/%d mismatch)\n", bad8?"FAIL":"y==ref OK", bad8, Nt);
    printf("samples  ref: %d %d %d %d | int8: %d %d %d %d | int4: %d %d %d %d\n",
        yr[0],yr[1],yr[2],yr[3], y8[0],y8[1],y8[2],y8[3], y4[0],y4[1],y4[2],y4[3]);

    double wq_gb=(double)Nb*Kb*QTILE/1e9, wp_gb=(double)Nb*Kb*TILE/1e9;
    /* bench int4 */
    double t=now();
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); bind_ai(); cpu_set_t s;CPU_ZERO(&s);CPU_SET(8+(tn*2)%8,&s);sched_setaffinity(0,sizeof(s),&s);
      int32_t ct[64];
      for(int r=0;r<reps;r++) for(int nb=tn;nb<Nb;nb+=nt) gemv_nb_int4(xt,Wq+(size_t)nb*Kb*QTILE,Kb,ct); }
    double dt4=(now()-t)/reps;
    /* bench int8 */
    t=now();
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); bind_ai(); cpu_set_t s;CPU_ZERO(&s);CPU_SET(8+(tn*2)%8,&s);sched_setaffinity(0,sizeof(s),&s);
      int32_t ct[64];
      for(int r=0;r<reps;r++) for(int nb=tn;nb<Nb;nb+=nt) gemv_nb_int8(xt,Wp+(size_t)nb*Kb*TILE,Kb,ct); }
    double dt8=(now()-t)/reps;
    printf("int4: %.3f ms/GEMV  %.1f GB/s (W=%.1fMB)   int8: %.3f ms  %.1f GB/s (W=%.1fMB)   threads=%d\n",
           dt4*1e3, wq_gb/dt4, wq_gb*1e3, dt8*1e3, wp_gb/dt8, wp_gb*1e3, nt);
    printf("  int4 is %.2fx faster than int8 (bandwidth halved)\n", dt8/dt4);
    return bad?1:0;
}
