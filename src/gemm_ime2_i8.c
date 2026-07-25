/* gemm_ime2_i8.c — int8 GEMM on the SpaceMIT K3 A100 IME-2 matrix units.
 *
 * C[MxN] += A[MxK] * B[KxN], int8 inputs, int32 accumulate.
 * Built on the validated VLEN-1024 vmadot tile (M0=8, N0=8, K0=16 -> 1024 MAC/op;
 * see bench/ime2_tile_validate.c).
 *
 * v2 — throughput: micro-kernel is 8x64 (1 M-block x 8 N-blocks) with 8 INDEPENDENT
 * int32 accumulators (v16..v30). The 8-accumulator form hides matrix-unit latency
 * (bench/peak_ime2_x8.c: 4-acc 1.84 -> 8-acc 3.29 TOPS/core) and amortizes the C store
 * over 2x the compute. Driver blocks N into L2-resident panels reused across the thread's
 * M-blocks. Threaded across the 4 dual-core-shared IME-2 units (8 threads / 4 pairs).
 *
 * v2 constraints: M%8==0, N%64==0, K%16==0.
 *
 * Test/bench:
 *   gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -DIME2_GEMM_TEST -o g gemm_ime2_i8.c && ./g 2048 2048 2048
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define M0 8
#define N0 8
#define K0 16
#define NRB 8                 /* N-blocks per micro-kernel (8 accumulators) */
#define NR (N0*NRB)           /* 64 */
#define TILE (M0*K0)          /* 128 bytes per packed tile */

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* packA: [mb][kb][8x16] contiguous.  packA[(mb*Kb+kb)*128 + r*16+c] = A[(mb*8+r)*K + kb*16+c] */
void gemm_ime2_pack_a(int Mt,int Kt,const int8_t*A,int8_t*pa){
    int Mb=Mt/M0, Kb=Kt/K0;
    for(int mb=0;mb<Mb;mb++)for(int kb=0;kb<Kb;kb++){
        int8_t*d=pa+(mb*Kb+kb)*TILE;
        for(int r=0;r<M0;r++)for(int c=0;c<K0;c++) d[r*K0+c]=A[(mb*M0+r)*Kt+kb*K0+c];
    }
}
/* packB: grouped NRB N-blocks. layout [ng][kb][g<NRB][8x16-transposed].
 * tile for global N-block nb=ng*NRB+g: t[nin*16+kin] = B[(kb*16+kin)*N + nb*8+nin] */
void gemm_ime2_pack_b(int Kt,int Nt,const int8_t*B,int8_t*pb){
    int Kb=Kt/K0, Ng=Nt/NR;
    for(int ng=0;ng<Ng;ng++)for(int kb=0;kb<Kb;kb++)for(int g=0;g<NRB;g++){
        int8_t*d=pb+((ng*Kb+kb)*NRB+g)*TILE; int nb=ng*NRB+g;
        for(int nin=0;nin<N0;nin++)for(int kin=0;kin<K0;kin++)
            d[nin*K0+kin]=B[(kb*K0+kin)*Nt+nb*N0+nin];
    }
}

/* micro-kernel: A panel (advances 128/kb), B panel (advances NRB*128/kb), Kb steps.
 * 8 accumulators -> 8 output tiles (8x8 int32) to ctmp[NRB*64], row-major within each tile. */
static void ukernel_8x64(const int8_t*A,const int8_t*B,int Kb,int32_t*ctmp){
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t"
        "vxor.vv v16,v16,v16\n\t vxor.vv v18,v18,v18\n\t vxor.vv v20,v20,v20\n\t vxor.vv v22,v22,v22\n\t"
        "vxor.vv v24,v24,v24\n\t vxor.vv v26,v26,v26\n\t vxor.vv v28,v28,v28\n\t vxor.vv v30,v30,v30\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t"
        "vle8.v v0,(%0)\n\t" "addi %0,%0,128\n\t"
        "mv t1,%1\n\t"
        "vle8.v v1,(t1)\n\t addi t1,t1,128\n\t vle8.v v2,(t1)\n\t addi t1,t1,128\n\t"
        "vle8.v v3,(t1)\n\t addi t1,t1,128\n\t vle8.v v4,(t1)\n\t addi t1,t1,128\n\t"
        "vle8.v v5,(t1)\n\t addi t1,t1,128\n\t vle8.v v6,(t1)\n\t addi t1,t1,128\n\t"
        "vle8.v v7,(t1)\n\t addi t1,t1,128\n\t vle8.v v8,(t1)\n\t"
        "addi %1,%1,1024\n\t"
        "vmadot v16,v0,v1\n\t" "vmadot v18,v0,v2\n\t" "vmadot v20,v0,v3\n\t" "vmadot v22,v0,v4\n\t"
        "vmadot v24,v0,v5\n\t" "vmadot v26,v0,v6\n\t" "vmadot v28,v0,v7\n\t" "vmadot v30,v0,v8\n\t"
        "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
        "vsetvli t0,zero,e32,m2\n\t"
        "vse32.v v16,(%3)\n\t addi %3,%3,256\n\t vse32.v v18,(%3)\n\t addi %3,%3,256\n\t"
        "vse32.v v20,(%3)\n\t addi %3,%3,256\n\t vse32.v v22,(%3)\n\t addi %3,%3,256\n\t"
        "vse32.v v24,(%3)\n\t addi %3,%3,256\n\t vse32.v v26,(%3)\n\t addi %3,%3,256\n\t"
        "vse32.v v28,(%3)\n\t addi %3,%3,256\n\t vse32.v v30,(%3)\n\t"
        : "+r"(A),"+r"(B),"+r"(Kb),"+r"(ctmp)
        :: "t0","t1","v0","v1","v2","v3","v4","v5","v6","v7","v8",
           "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory");
}

/* scatter 8x8 int32 tile (row-major contiguous) into C[MxN] at (mb,nb) */
static void store_tile(int32_t*C,int Nt,int mb,int nb,const int32_t*t){
    for(int r=0;r<M0;r++){ int32_t*row=C+(size_t)(mb*M0+r)*Nt+(size_t)nb*N0;
        for(int c=0;c<N0;c++) row[c]=t[r*N0+c]; }
}

void gemm_ime2_i8_st(int Mt,int Nt,int Kt,const int8_t*pa,const int8_t*pb,int32_t*C){
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    int Mb=Mt/M0, Kb=Kt/K0, Ng=Nt/NR;
    int32_t ctmp[NRB*N0*N0];
    for(int ng=0;ng<Ng;ng++)for(int mb=0;mb<Mb;mb++){
        ukernel_8x64(pa+(mb*Kb)*TILE, pb+(ng*Kb)*NRB*TILE, Kb, ctmp);
        for(int g=0;g<NRB;g++) store_tile(C,Nt,mb,ng*NRB+g,ctmp+g*N0*N0);
    }
}

/* multithread: 8 threads bound to A100 harts 8..15 (4 interleaved pairs).
 * ng outer, mb inner: the B N-panel (K*NR int8) stays L1/L2-resident while a thread
 * sweeps all its M-blocks -> B read from DRAM once per (thread,ng). */
void gemm_ime2_i8_mt(int Mt,int Nt,int Kt,const int8_t*pa,const int8_t*pb,int32_t*C,int nthreads){
    int Mb=Mt/M0, Kb=Kt/K0, Ng=Nt/NR;
    #pragma omp parallel num_threads(nthreads)
    {
        int tn=0;
        #ifdef _OPENMP
        tn=omp_get_thread_num();
        #endif
        bind_ai();
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(8+(tn%8),&s); sched_setaffinity(0,sizeof(s),&s);
        for(int i=0;i<5;i++) sched_yield();
        int32_t ctmp[NRB*N0*N0];
        for(int ng=0;ng<Ng;ng++)
            for(int mb=tn; mb<Mb; mb+=nthreads){
                ukernel_8x64(pa+(mb*Kb)*TILE, pb+(ng*Kb)*NRB*TILE, Kb, ctmp);
                for(int g=0;g<NRB;g++) store_tile(C,Nt,mb,ng*NRB+g,ctmp+g*N0*N0);
            }
    }
}

#ifdef IME2_GEMM_TEST
#include <stdio.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void ref(int M,int N,int K,const int8_t*A,const int8_t*B,int32_t*C){
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){int32_t a=0;for(int k=0;k<K;k++)a+=(int32_t)A[m*K+k]*(int32_t)B[k*N+n];C[m*N+n]=a;}
}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):2048, N=argc>2?atoi(argv[2]):2048, K=argc>3?atoi(argv[3]):2048;
    int nt=argc>4?atoi(argv[4]):8, reps=argc>5?atoi(argv[5]):20;
    if(M%8||N%64||K%16){ printf("need M%%8==0,N%%64==0,K%%16==0\n"); return 2; }
    int8_t*A=malloc((size_t)M*K),*B=malloc((size_t)K*N),*pa=malloc((size_t)M*K),*pb=malloc((size_t)K*N);
    int32_t*C=calloc((size_t)M*N,4),*Cr=malloc((size_t)M*N*4);
    srand(2026); for(size_t i=0;i<(size_t)M*K;i++)A[i]=rand()%256-128; for(size_t i=0;i<(size_t)K*N;i++)B[i]=rand()%256-128;
    gemm_ime2_pack_a(M,K,A,pa); gemm_ime2_pack_b(K,N,B,pb);
    if((size_t)M*N<=4096*4096) ref(M,N,K,A,B,Cr); else { printf("(skip ref, too big)\n"); }

    gemm_ime2_i8_mt(M,N,K,pa,pb,C,nt);
    int bad=0; if((size_t)M*N<=4096*4096){ for(size_t i=0;i<(size_t)M*N;i++) if(C[i]!=Cr[i]) bad++; }
    printf("[mt%d] %dx%dx%d  %s (%d mismatches)\n",nt,M,N,K, bad?"FAIL":"C==CRef OK", bad);

    double t=now(); for(int r=0;r<reps;r++) gemm_ime2_i8_mt(M,N,K,pa,pb,C,nt);
    double dt=(now()-t)/reps, gop=2.0*M*N*K/dt/1e9;
    printf("[mt%d] %.3f ms/gemm  %.1f GOP/s int8  (%.2f TOPS)\n",nt,dt*1e3,gop,gop/1000.0);
    return bad?1:0;
}
#endif
