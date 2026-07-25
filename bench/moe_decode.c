/* moe_decode.c — Qwen3-30B-A3B MoE decode throughput on the K3 A100 IME-2, q4/q8 interlaced.
 *
 * The datasheet ">10 tok/s @ 30B" is an MoE claim: 30B total but only ~3B active/token (8 of 128
 * experts). Decode is weight-bandwidth-bound (see decode_layer.c), so what matters is the bytes
 * STREAMED per token = the active experts + attention, not the full 30B. This measures exactly that
 * with the validated q4-in-q8 interlaced vmadot kernel, tensor-parallel over the 4 IME units.
 *
 * Streams the real per-layer matmuls: q/k/v/o (GQA) + router + 8 active experts (gate/up/down at
 * moe_intermediate) x L layers + lm_head. Reports q4 vs q8, M=1 (single-stream) and M=8 (MTP verify).
 * The MoE+MTP subtlety: M draft tokens may route to DIFFERENT experts, so a verify pass streams the
 * UNION of experts (U). U=8 = perfect overlap (~8x); U=64 = disjoint (no MoE reuse). Swept below.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o moe_decode moe_decode.c
 * Usage: ./moe_decode [nthreads]
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
static void gemv_nb_int4(const int8_t*xt,const uint8_t*Wq,int Kb,int32_t*ct){
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
        : "+r"(xt),"+r"(Wq),"+r"(Kp) : "r"(ct)
        : "t0","v0","v2","v4","v5","v6","v28","v29","v30","v31","memory");
}
static void gemv_nb_int8(const int8_t*xt,const int8_t*Wp,int Kb,int32_t*ct){
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t vxor.vv v28,v28,v28\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t"
        "vle8.v v0,(%0)\n\t addi %0,%0,128\n\t"
        "vle8.v v1,(%1)\n\t addi %1,%1,128\n\t"
        "vmadot v28,v0,v1\n\t"
        "addi %2,%2,-1\n\t bnez %2,1b\n\t"
        "vsetvli t0,zero,e32,m2\n\t vse32.v v28,(%3)\n\t"
        : "+r"(xt),"+r"(Wp),"+r"(Kb) : "r"(ct)
        : "t0","v0","v1","v28","v29","memory");
}

typedef struct { int N,K; uint8_t*Wq; int8_t*Wp; } Mat;
static Mat mk(int N,int K){
    Mat m; m.N=N; m.K=K; int Nb=N/N0, Kb=K/K0;
    int8_t*W=malloc((size_t)N*K); for(size_t i=0;i<(size_t)N*K;i++)W[i]=(int8_t)((i*2654435761u>>4)%15-7);
    m.Wq=malloc((size_t)Nb*Kb*QTILE); pack_w_int4(N,K,W,m.Wq);
    m.Wp=malloc((size_t)Nb*Kb*TILE);  pack_w_int8(N,K,W,m.Wp);
    free(W); return m;
}
/* one thread's stripe of a matmul (no fork/join — called inside a persistent pool) */
static inline void mm_stripe(const Mat*m,const int8_t*xt,int32_t*ct,int q4,int tn,int nt){
    int Nb=m->N/N0, Kb=m->K/K0;
    if(q4) for(int nb=tn;nb<Nb;nb+=nt) gemv_nb_int4(xt,m->Wq+(size_t)nb*Kb*QTILE,Kb,ct);
    else   for(int nb=tn;nb<Nb;nb+=nt) gemv_nb_int8(xt,m->Wp+(size_t)nb*Kb*TILE,Kb,ct);
}

/* everything a forward needs; run_fwd is top-level (no nested-fn + OpenMP interaction) */
typedef struct { Mat *q,*k,*vv,*o,*rt,*eg,*eu,*ed,*lm; const int8_t*xt; int nt,L; } Sched;

/* persistent thread pool: pin ONCE, then stream the whole forward with a barrier between matmuls
 * (each matmul's output feeds the next). reps timed. streams U experts/layer. returns ms/forward. */
static double run_fwd(Sched*S,int q4,int U,int reps){
    static double ts, te;   /* written by master, read after the region's implicit barrier */
    #pragma omp parallel num_threads(S->nt)
    {
        int tn=omp_get_thread_num(); bind_ai();
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8+(tn*2)%8,&cs); sched_setaffinity(0,sizeof(cs),&cs);
        int32_t ct[64];
        Mat*am[5]={S->q,S->k,S->vv,S->o,S->rt};
        Mat*em[3]={S->eg,S->eu,S->ed};
        for(int r=-1;r<reps;r++){
            #pragma omp barrier
            if(r==0){
                #pragma omp master
                ts=now();
            }
            for(int li=0;li<S->L;li++){
                for(int a=0;a<5;a++){ mm_stripe(am[a],S->xt,ct,q4,tn,S->nt);
                    #pragma omp barrier
                }
                for(int e=0;e<U;e++) for(int a=0;a<3;a++){ mm_stripe(em[a],S->xt,ct,q4,tn,S->nt);
                    #pragma omp barrier
                }
            }
            mm_stripe(S->lm,S->xt,ct,q4,tn,S->nt);
            #pragma omp barrier
        }
        #pragma omp master
        te=now();
    }
    return (te-ts)/reps*1e3;
}

int main(int c,char**v){
    int nt=(c>1)?atoi(v[1]):4;
    /* ---- Qwen3-30B-A3B ---- */
    int H=2048, nh=32, nkv=4, hd=128, n_exp=128, n_act=8, moe=768, L=48, vocab=151936;
    int qd=nh*hd, kvd=nkv*hd;   /* 4096, 512 */
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    printf("Qwen3-30B-A3B  H=%d heads=%d/%d hd=%d experts=%d/%d moe_inter=%d L=%d vocab=%d  nt=%d\n",
           H,nh,nkv,hd,n_exp,n_act,moe,L,vocab,nt);

    /* distinct weight tensors (shared across layers/experts — throughput = DRAM stream) */
    Mat q=mk(qd,H), k=mk(kvd,H), vv=mk(kvd,H), o=mk(H,qd), rt=mk(n_exp,H);
    Mat eg=mk(moe,H), eu=mk(moe,H), ed=mk(H,moe);      /* one expert's gate/up/down */
    Mat lm=mk(vocab,H);
    int maxK=qd; int8_t*xt=malloc((size_t)(maxK/K0)*TILE); memset(xt,1,(size_t)(maxK/K0)*TILE);
    int32_t*y=malloc((size_t)vocab*4);

    (void)y;
    /* params streamed per token: attn(q+k+v+o) + router + U experts(g+u+d), xL, + lm_head */
    #define P(m) ((double)(m).N*(m).K)
    double attn = P(q)+P(k)+P(vv)+P(o)+P(rt);
    double expert = P(eg)+P(eu)+P(ed);
    double lmp = P(lm);
    Sched S={.q=&q,.k=&k,.vv=&vv,.o=&o,.rt=&rt,.eg=&eg,.eu=&eu,.ed=&ed,.lm=&lm,.xt=xt,.nt=nt,.L=L};

    for(int q4=1;q4>=0;q4--){
        const char*nm=q4?"q4-interlaced":"q8";
        double bytes_tok=(attn*L + expert*n_act*L + lmp)/(q4?2.0:1.0);
        double ms1=run_fwd(&S,q4,n_act,3);         /* M=1: 8 active experts */
        double tot=(attn*L + expert*n_act*L + lmp);
        printf("\n[%s]  active params/tok = %.2fB of 30B  (%.0f MB %s/tok)\n",nm,tot/1e9,bytes_tok/1e6,nm);
        printf("  M=1 single-stream : %.1f ms/tok -> %.1f tok/s   (%.1f GB/s)\n",ms1,1000.0/ms1,bytes_tok/ms1/1e6);
        /* M=8 MTP verify: stream UNION of experts U (attention/router/lm streamed once, M-invariant) */
        printf("  M=8 MTP verify (8 tokens/pass), by expert-union U:\n");
        int Us[]={8,16,24,64};
        for(int ui=0;ui<4;ui++){
            int U=Us[ui]; if(U>n_exp)U=n_exp;
            double msU=run_fwd(&S,q4,U,3);         /* one verify pass streaming U experts */
            /* note: attention+router+lm are M-invariant (streamed once); only experts scale with U */
            double tps=8.0/msU*1000.0;
            const char*tag = U==8?"perfect overlap":U==64?"disjoint (worst)":"partial overlap";
            printf("     U=%-3d (%-16s): %.1f ms/pass -> %.1f tok/s  (%.1fx vs M=1)\n",U,tag,msU,tps,tps/(1000.0/ms1));
        }
    }
    printf("\n(active-expert weight streaming; attention score/softmax + activation quant are extra scalar work,\n");
    printf(" run on the free X100 cores in a full engine. U=distinct experts across the 8 draft tokens = the MoE+MTP knob.)\n");
    return 0;
}
