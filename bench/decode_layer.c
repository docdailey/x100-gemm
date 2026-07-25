/* decode_layer.c — end-to-end decode-throughput projection using the real per-layer matmul
 * shapes of a Qwen-style transformer, driven by the q4-in-q8 interlaced IME-2 kernel.
 *
 * Chains the 7 GEMVs of one decoder layer (q,k,v,o,gate,up,down) x L layers and times a full
 * token step, for M tokens/pass (M=1 single-stream, M=8 MTP/speculative-fills-the-tile) and for
 * int4 vs int8 weights, tensor-parallel across the 4 IME units. GEMM-only (attention/softmax/
 * norm/KV are cheap vs weight streaming and omitted) -> an UPPER bound on decode tok/s.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o decode_layer decode_layer.c
 * Usage: ./decode_layer <hidden> <inter> <kv_dim> <n_layers> <M> <nthreads> <reps>
 *   0.5B:  ./decode_layer 896  4864 128 24 8 4 20
 *   4B-ish:./decode_layer 2560 9728 512 36 8 4 5
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

#define N0 8
#define K0 16
#define TILE 128
#define QTILE 64

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* K-block-pair interlaced int4 pack (matches gemv_nb_int4) */
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
static void pack_w_int8(int Nt,int Kt,const int8_t*W,int8_t*Wp){
    int Nb=Nt/N0, Kb=Kt/K0;
    for(int nb=0;nb<Nb;nb++)for(int kb=0;kb<Kb;kb++){
        int8_t*d=Wp+((size_t)(nb*Kb+kb))*TILE;
        for(int n=0;n<N0;n++)for(int k=0;k<K0;k++) d[n*K0+k]=W[(nb*N0+n)*Kt+kb*K0+k];
    }
}
static void gemv_nb_int4(const int8_t*xt,const uint8_t*Wq,int Kb,int32_t*ctmp){
    long Kp=Kb/2;
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t vxor.vv v28,v28,v28\n\t vxor.vv v30,v30,v30\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t"
        "vle8.v v0,(%0)\n\t addi %0,%0,128\n\t"
        "vle8.v v2,(%0)\n\t addi %0,%0,128\n\t"
        "vle8.v v4,(%1)\n\t addi %1,%1,128\n\t"
        "vsll.vi v5,v4,4\n\t vsra.vi v5,v5,4\n\t"
        "vsra.vi v6,v4,4\n\t"
        "vmadot v28,v0,v5\n\t"
        "vmadot v30,v2,v6\n\t"
        "addi %2,%2,-1\n\t bnez %2,1b\n\t"
        "vsetvli t0,zero,e32,m2\n\t vadd.vv v28,v28,v30\n\t vse32.v v28,(%3)\n\t"
        : "+r"(xt),"+r"(Wq),"+r"(Kp) : "r"(ctmp)
        : "t0","v0","v2","v4","v5","v6","v28","v29","v30","v31","memory");
}
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

/* one matmul y[M,N]=W[N,K]@x, tensor-parallel over nt threads (N split). xt pre-packed [Kb][128]. */
typedef struct { int N,K; uint8_t*Wq; int8_t*Wp; } Mat;

static void run_mat_int4(const Mat*m,const int8_t*xt,int32_t*y,int nt){
    int Nb=m->N/N0, Kb=m->K/K0;
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); bind_ai(); cpu_set_t s;CPU_ZERO(&s);CPU_SET(8+(tn*2)%8,&s);sched_setaffinity(0,sizeof(s),&s);
      int32_t ct[64];
      for(int nb=tn;nb<Nb;nb+=nt){ gemv_nb_int4(xt,m->Wq+(size_t)nb*Kb*QTILE,Kb,ct);
          for(int n=0;n<N0;n++) y[nb*N0+n]=ct[n]; } }
}
static void run_mat_int8(const Mat*m,const int8_t*xt,int32_t*y,int nt){
    int Nb=m->N/N0, Kb=m->K/K0;
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); bind_ai(); cpu_set_t s;CPU_ZERO(&s);CPU_SET(8+(tn*2)%8,&s);sched_setaffinity(0,sizeof(s),&s);
      int32_t ct[64];
      for(int nb=tn;nb<Nb;nb+=nt){ gemv_nb_int8(xt,m->Wp+(size_t)nb*Kb*TILE,Kb,ct);
          for(int n=0;n<N0;n++) y[nb*N0+n]=ct[n]; } }
}

static Mat mk(int N,int K){
    Mat m; m.N=N; m.K=K;
    int Nb=N/N0, Kb=K/K0;
    int8_t*W=malloc((size_t)N*K); for(size_t i=0;i<(size_t)N*K;i++)W[i]=(int8_t)((i*2654435761u>>3)%15-7);
    m.Wq=malloc((size_t)Nb*Kb*QTILE); pack_w_int4(N,K,W,m.Wq);
    m.Wp=malloc((size_t)Nb*Kb*TILE);  pack_w_int8(N,K,W,m.Wp);
    free(W); return m;
}

int main(int c,char**v){
    int H =(c>1)?atoi(v[1]):896;   /* hidden */
    int I =(c>2)?atoi(v[2]):4864;  /* intermediate */
    int KV=(c>3)?atoi(v[3]):128;   /* kv_dim = n_kv_heads*head_dim */
    int L =(c>4)?atoi(v[4]):24;    /* layers */
    int M =(c>5)?atoi(v[5]):8;     /* tokens per pass */
    int nt=(c>6)?atoi(v[6]):4;
    int reps=(c>7)?atoi(v[7]):20;
    if(H%8||H%32||I%32||KV%32){ printf("dims must be %%32 (H=%d I=%d KV=%d)\n",H,I,KV); return 2; }
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();

    /* the 7 GEMVs of one layer (shared across layers here — we're measuring throughput, not values) */
    Mat q =mk(H, H);    /* q_proj  H->H  (num_heads*head_dim=H for these configs) */
    Mat k =mk(KV,H);    /* k_proj  H->KV */
    Mat vv=mk(KV,H);    /* v_proj  H->KV */
    Mat o =mk(H, H);    /* o_proj  H->H */
    Mat g =mk(I, H);    /* gate    H->I */
    Mat up=mk(I, H);    /* up      H->I */
    Mat dn=mk(H, I);    /* down    I->H */
    Mat*layer[7]={&q,&k,&vv,&o,&g,&up,&dn};

    int maxK=(I>H)?I:H, maxN=(I>H)?I:H;
    int8_t*xt=malloc((size_t)(maxK/K0)*TILE);      /* packed activation (M rows, rest 0) */
    memset(xt,1,(size_t)(maxK/K0)*TILE);
    int32_t*y=malloc((size_t)maxN*4);

    /* params/token (int8 bytes) = sum over the 7 matmuls, x L layers */
    double pp = (double)(q.N*q.K + k.N*k.K + vv.N*vv.K + o.N*o.K + g.N*g.K + up.N*up.K + dn.N*dn.K)*L;

    for(int q4=1;q4>=0;q4--){
        /* warmup */
        for(int L2=0;L2<2;L2++) for(int li=0;li<L;li++) for(int mi=0;mi<7;mi++)
            if(q4) run_mat_int4(layer[mi],xt,y,nt); else run_mat_int8(layer[mi],xt,y,nt);
        double t=now();
        for(int r=0;r<reps;r++)
            for(int li=0;li<L;li++) for(int mi=0;mi<7;mi++)
                if(q4) run_mat_int4(layer[mi],xt,y,nt); else run_mat_int8(layer[mi],xt,y,nt);
        double dt=(now()-t)/reps;          /* seconds per full forward (all L layers) */
        double tok_s = M / dt;             /* M tokens produced per forward */
        double gbs = pp/(q4?2.0:1.0)/dt/1e9;
        printf("%s  fwd=%.2f ms  M=%d -> %.1f tok/s   (%.0fM params/tok, %.1f GB/s wt)\n",
               q4?"int4":"int8", dt*1e3, M, tok_s, pp/1e6, gbs);
    }
    printf("dims H=%d I=%d KV=%d L=%d  threads=%d  (GEMM-only upper bound; attention/KV extra)\n",H,I,KV,L,nt);
    return 0;
}
