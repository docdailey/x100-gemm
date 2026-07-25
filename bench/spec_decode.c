/* spec_decode.c — speculative/MTP decode engine on the K3 A100 IME-2, W4A8.
 *
 * A real autoregressive transformer decode step (RMSNorm + q/k/v proj + RoPE + causal
 * KV-cache attention + o_proj + SwiGLU FFN + lm_head), with EVERY matmul run through the
 * validated q4-in-q8 interlaced vmadot kernel (bench/q4_gemv.c), tensor-parallel over the 4
 * IME units. Weights int4, activations int8, int32 accumulate, dequant per tensor.
 *
 * The point it proves ON HARDWARE: a verify forward for M draft tokens costs ~the SAME
 * wall-clock as a 1-token forward (the weight pass is M-invariant, attention is the only
 * M-dependent term and it's small at decode depths). Speculative decode turns that into
 * ~E[accepted]+1 tokens per forward. Draft accept-rate is a MODEL property (draft quality),
 * so it's parameterized here; the hardware contributes the M-invariant verify + int4 bytes.
 *
 * Build: gcc -O3 -fno-stack-protector -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o spec_decode spec_decode.c -lm
 *   (-fno-stack-protector: the vmadot vse32.v store spuriously trips the canary; ASan confirms
 *    the code is memory-safe. NOTE: the attention/rmsnorm/quant GLUE here is naive scalar on one
 *    thread, so it scales with M and hides the M-invariant matmul — see moe_decode.c/decode_layer.c
 *    for the clean matmul-streaming measurement. This file is the full-engine skeleton; the glue
 *    needs RVV-vectorizing + spreading onto the free X100 cores to keep the verify pass M-invariant.)
 * Usage: ./spec_decode <dim> <n_heads> <n_kv> <head_dim> <ffn> <n_layers> <vocab> <ctx> <nt>
 *   0.5B: ./spec_decode 896  14 2 64  4864 24 151936 256 4
 *   4B:   ./spec_decode 2560 20 4 128 9728 36 151936 256 4
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
#include <math.h>
#include <omp.h>

#define N0 8
#define K0 16
#define TILE 128
#define QTILE 64

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static void pin(int hart){ bind_ai(); cpu_set_t s; CPU_ZERO(&s); CPU_SET(hart,&s); sched_setaffinity(0,sizeof(s),&s); }

/* ---- the validated int4 (K-pair interlaced) + int8 vmadot micro-kernels ---- */
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

/* ---- packing ---- */
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

/* ---- a linear W[N,K] (int4-packed), plus its dequant scale ---- */
typedef struct { int N,K; uint8_t*Wq; float ws; } Lin;
static Lin mk_lin(int N,int K){
    Lin l; l.N=N; l.K=K; l.ws=1.0f/16.0f;
    int Nb=N/N0, Kb=K/K0;
    int8_t*W=malloc((size_t)N*K);
    for(size_t i=0;i<(size_t)N*K;i++) W[i]=(int8_t)((i*2654435761u>>4)%15-7);  /* int4 range */
    l.Wq=malloc((size_t)Nb*Kb*QTILE); pack_w_int4(N,K,W,l.Wq); free(W);
    return l;
}
/* y[M,N] fp = dequant(W[N,K] @ quant(x[M,K])), tensor-parallel over nt IME units.
 * xt is scratch [Kb*128]; activations quantized per-call (absmax). */
static void lin_fwd(const Lin*l,const float*x,float*y,int M,int nt,int8_t*xt){
    int Nb=l->N/N0, Kb=l->K/K0;
    float amax=1e-6f; for(int i=0;i<M*l->K;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float sx=amax/127.0f, inv=1.0f/sx;
    memset(xt,0,(size_t)Kb*TILE);
    for(int kb=0;kb<Kb;kb++)for(int m=0;m<M;m++)for(int k=0;k<K0;k++){
        int q=(int)lrintf(x[m*l->K+kb*K0+k]*inv); q=q>127?127:(q<-128?-128:q);
        xt[kb*TILE+m*K0+k]=(int8_t)q;
    }
    float ds=sx*l->ws;
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); pin(8+(tn*2)%8);
      int32_t ct[64];
      for(int nb=tn;nb<Nb;nb+=nt){ gemv_nb_int4(xt,l->Wq+(size_t)nb*Kb*QTILE,Kb,ct);
          for(int m=0;m<M;m++)for(int n=0;n<N0;n++) y[m*l->N+nb*N0+n]=ct[m*N0+n]*ds; } }
}

/* ---- model ---- */
typedef struct {
    int dim,nh,nkv,hd,ffn,nl,vocab,ctx;
    Lin *wq,*wk,*wv,*wo,*wg,*wu,*wd;   /* per layer */
    Lin lm;
    float *Kc,*Vc;                     /* KV cache [nl][ctx][nkv*hd] */
    int clen;
} Model;

static void rmsnorm(float*o,const float*x,int n){ float s=0; for(int i=0;i<n;i++)s+=x[i]*x[i]; s=1.0f/sqrtf(s/n+1e-6f); for(int i=0;i<n;i++)o[i]=x[i]*s; }
static void rope(float*v,int hd,int pos){ for(int i=0;i<hd/2;i++){ float f=powf(1e4f,-2.0f*i/hd); float c=cosf(pos*f),s=sinf(pos*f); float a=v[i],b=v[i+hd/2]; v[i]=a*c-b*s; v[i+hd/2]=a*s+b*c; } }

/* one decode forward for M tokens at positions clen..clen+M-1. Returns logits[M,vocab].
 * Writes M new entries into the KV cache. hid[M,dim] is the (random) input hidden state. */
static void forward(Model*mo,float*hid,int M,int nt,float*logits,
                    float*nrm,float*q,float*k,float*v,float*att,float*tmp,float*g,float*u,int8_t*xt){
    int dim=mo->dim,nh=mo->nh,nkv=mo->nkv,hd=mo->hd,ffn=mo->ffn;
    int qd=nh*hd, kvd=nkv*hd, gpr=nh/nkv;
    for(int L=0;L<mo->nl;L++){
        /* attn norm */
        for(int m=0;m<M;m++) rmsnorm(nrm+m*dim,hid+m*dim,dim);
        lin_fwd(&mo->wq[L],nrm,q,M,nt,xt);
        lin_fwd(&mo->wk[L],nrm,k,M,nt,xt);
        lin_fwd(&mo->wv[L],nrm,v,M,nt,xt);
        /* rope + write KV cache */
        float*Kc=mo->Kc+(size_t)L*mo->ctx*kvd, *Vc=mo->Vc+(size_t)L*mo->ctx*kvd;
        for(int m=0;m<M;m++){
            int pos=mo->clen+m;
            for(int h=0;h<nh;h++)  rope(q+m*qd+h*hd,hd,pos);
            for(int h=0;h<nkv;h++) rope(k+m*kvd+h*hd,hd,pos);
            memcpy(Kc+(size_t)pos*kvd,k+m*kvd,kvd*sizeof(float));
            memcpy(Vc+(size_t)pos*kvd,v+m*kvd,kvd*sizeof(float));
        }
        /* causal attention over cache[0..clen+m] */
        float scale=1.0f/sqrtf(hd);
        for(int m=0;m<M;m++){
            int pm=mo->clen+m;
            for(int h=0;h<nh;h++){
                int kvh=h/gpr; float*qh=q+m*qd+h*hd;
                float mx=-1e30f, *sc=tmp;         /* tmp holds scores up to ctx */
                for(int j=0;j<=pm;j++){ float*kj=Kc+(size_t)j*kvd+kvh*hd; float d=0; for(int t=0;t<hd;t++)d+=qh[t]*kj[t]; d*=scale; sc[j]=d; if(d>mx)mx=d; }
                float den=0; for(int j=0;j<=pm;j++){ sc[j]=expf(sc[j]-mx); den+=sc[j]; }
                float*oh=att+m*qd+h*hd; for(int t=0;t<hd;t++)oh[t]=0;
                float inv=1.0f/den;
                for(int j=0;j<=pm;j++){ float w=sc[j]*inv; float*vj=Vc+(size_t)j*kvd+kvh*hd; for(int t=0;t<hd;t++)oh[t]+=w*vj[t]; }
            }
        }
        lin_fwd(&mo->wo[L],att,tmp,M,nt,xt);
        for(int i=0;i<M*dim;i++) hid[i]+=tmp[i];        /* residual */
        /* ffn */
        for(int m=0;m<M;m++) rmsnorm(nrm+m*dim,hid+m*dim,dim);
        lin_fwd(&mo->wg[L],nrm,g,M,nt,xt);
        lin_fwd(&mo->wu[L],nrm,u,M,nt,xt);
        for(int i=0;i<M*ffn;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }  /* SwiGLU */
        lin_fwd(&mo->wd[L],g,tmp,M,nt,xt);
        for(int i=0;i<M*dim;i++) hid[i]+=tmp[i];
    }
    for(int m=0;m<M;m++) rmsnorm(nrm+m*dim,hid+m*dim,dim);
    lin_fwd(&mo->lm,nrm,logits,M,nt,xt);
}

int main(int c,char**v){
    int dim=(c>1)?atoi(v[1]):896, nh=(c>2)?atoi(v[2]):14, nkv=(c>3)?atoi(v[3]):2,
        hd=(c>4)?atoi(v[4]):64, ffn=(c>5)?atoi(v[5]):4864, nl=(c>6)?atoi(v[6]):24,
        vocab=(c>7)?atoi(v[7]):151936, ctx=(c>8)?atoi(v[8]):256, nt=(c>9)?atoi(v[9]):4;
    int qd=nh*hd, kvd=nkv*hd;
    if(dim%32||ffn%32||qd%32||kvd%32){ printf("dims must be %%32\n"); return 2; }
    pin(8); for(int i=0;i<5;i++) sched_yield();
    printf("model dim=%d heads=%d/%d hd=%d ffn=%d layers=%d vocab=%d ctx=%d nt=%d\n",dim,nh,nkv,hd,ffn,nl,vocab,ctx,nt);

    Model mo={.dim=dim,.nh=nh,.nkv=nkv,.hd=hd,.ffn=ffn,.nl=nl,.vocab=vocab,.ctx=ctx};
    /* share ONE layer's weights across all layers (throughput = DRAM stream; footprint stays small) */
    Lin wq=mk_lin(qd,dim), wk=mk_lin(kvd,dim), wv=mk_lin(kvd,dim), wo=mk_lin(dim,qd),
        wg=mk_lin(ffn,dim), wu=mk_lin(ffn,dim), wd=mk_lin(dim,ffn);
    Lin *Aq=malloc(nl*sizeof(Lin)),*Ak=malloc(nl*sizeof(Lin)),*Av=malloc(nl*sizeof(Lin)),
        *Ao=malloc(nl*sizeof(Lin)),*Ag=malloc(nl*sizeof(Lin)),*Au=malloc(nl*sizeof(Lin)),*Ad=malloc(nl*sizeof(Lin));
    for(int L=0;L<nl;L++){ Aq[L]=wq;Ak[L]=wk;Av[L]=wv;Ao[L]=wo;Ag[L]=wg;Au[L]=wu;Ad[L]=wd; }
    mo.wq=Aq;mo.wk=Ak;mo.wv=Av;mo.wo=Ao;mo.wg=Ag;mo.wu=Au;mo.wd=Ad; mo.lm=mk_lin(vocab,dim);
    mo.Kc=calloc((size_t)nl*ctx*kvd,4); mo.Vc=calloc((size_t)nl*ctx*kvd,4);

    int Mmax=8;
    float *hid=malloc((size_t)Mmax*dim*4),*nrm=malloc((size_t)Mmax*dim*4),
          *q=malloc((size_t)Mmax*qd*4),*k=malloc((size_t)Mmax*kvd*4),*vv=malloc((size_t)Mmax*kvd*4),
          *att=malloc((size_t)Mmax*qd*4),*tmp=malloc((size_t)Mmax*(dim>ffn?dim:ffn)*4),
          *g=malloc((size_t)Mmax*ffn*4),*u=malloc((size_t)Mmax*ffn*4),
          *logits=malloc((size_t)Mmax*vocab*4);
    int maxK=ffn>dim?ffn:dim; int8_t*xt=malloc((size_t)(maxK/K0)*TILE);
    for(int i=0;i<Mmax*dim;i++) hid[i]=0.02f*((i*131+7)%97-48);

    /* fill cache to a decode-representative depth so attention cost is real */
    mo.clen = ctx-Mmax-1;
    for(size_t i=0;i<(size_t)nl*ctx*kvd;i++){ mo.Kc[i]=0.01f*((i*17)%50-25); mo.Vc[i]=0.01f*((i*13)%50-25); }

    /* measure forward-ms vs M (M-invariance of the verify pass, WITH attention) */
    printf("\n-- forward latency vs M (decode depth clen=%d) --\n",mo.clen);
    double ms1=0, msg=0;
    for(int M=1;M<=8;M<<=1){
        for(int w=0;w<2;w++){ for(int i=0;i<M*dim;i++)hid[i]=0.02f*((i*131+7)%97-48); forward(&mo,hid,M,nt,logits,nrm,q,k,vv,att,tmp,g,u,xt); }
        int reps=(dim<=896)?12:4;
        double t=now(); for(int r=0;r<reps;r++){ for(int i=0;i<M*dim;i++)hid[i]=0.02f*((i*131+7)%97-48); forward(&mo,hid,M,nt,logits,nrm,q,k,vv,att,tmp,g,u,xt); }
        double ms=(now()-t)/reps*1e3;
        if(M==1)ms1=ms; if(M==8)msg=ms;
        printf("  M=%d : %.2f ms/forward   (%.2f ms/token if all accepted)\n",M,ms,ms/M);
    }
    printf("  verify(M=8)/forward(M=1) = %.2fx  (ideal 1.0 = fully M-invariant)\n",msg/ms1);

    /* speculative throughput: tokens/forward = E[accepted]+1; wall = verify + gamma*draft */
    printf("\n-- decode throughput --\n");
    printf("  baseline (M=1 autoregressive): %.1f tok/s\n",1000.0/ms1);
    int gamma=7;                          /* draft length (fills the 8-row tile: gamma+1=8) */
    double draft_frac[]={0.0,0.12,0.20};  /* draft cost as fraction of one target forward (EAGLE~0.1-0.2; 0=free/overlapped) */
    const char*dl[]={"self-spec/MTP (draft~free)","small draft 0.12x","small draft 0.20x"};
    double ps[]={0.65,0.75,0.85};
    for(int di=0;di<3;di++){
        printf("  [%s]\n",dl[di]);
        for(int pi=0;pi<3;pi++){
            double p=ps[pi];
            double Ea=p*(1.0-pow(p,gamma))/(1.0-p);   /* expected accepted draft tokens */
            double toks=Ea+1.0;
            double wall_ms=msg + gamma*draft_frac[di]*ms1;   /* one M=8 verify + gamma draft steps */
            double tps=toks/wall_ms*1000.0;
            printf("     accept p=%.2f -> %.1f tok/step, %.2f ms/step -> %.1f tok/s  (%.1fx baseline)\n",
                   p,toks,wall_ms,tps,tps/(1000.0/ms1));
        }
    }
    printf("\n(hardware fact = the M-invariant verify above; accept-rate p is a draft-model property.)\n");
    return 0;
}
