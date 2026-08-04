/* mxbai_rvv.c -- hand-written native RVV/A100-hart inference engine for mxbai-embed-large-v1
 * (standard HF BertModel: 24-layer bidirectional absolute-position-embedding post-norm encoder,
 * exact GELU FFN, CLS-token pooling, no final L2 normalize). Runs directly on the SpaceMIT K3's
 * eight A100 AI harts -- no PyTorch, no ONNX Runtime, no XLA anywhere in the inference path.
 * Second model in this project, after nomic_rvv.c -- deliberately a genuinely different
 * architecture family (classic BERT, not NomicBert) to validate the engine's generality, same
 * rigor. Shares nomic_rvv.c's hardware-access discipline and RVV kernels verbatim (this file is
 * a sibling, not an import -- see nomic_rvv.c's own header for why: this repo's convention is
 * separate files per model variant, e.g. qwen_moe.c/qwen_moe_hp.c/qwen_moe_q3k.c, not one
 * parameterized megafile).
 *
 * Architecture verified against config.json + modules.json + graphify-embedder/server.py (see
 * extract_mxbai.py's docstring for exactly what was confirmed and how):
 *   embeddings = word_emb[id] + position_emb[pos] + token_type_emb[0]   (absolute, learned,
 *     max_position_embeddings=512 -- a real hard limit, unlike nomic's RoPE ceiling)
 *   x = LayerNorm_emb(embeddings)                                        (applied once, before layer 0)
 *   per layer (post-norm, standard BERT):
 *     x = LayerNorm1(x + Attn(x))
 *     x = LayerNorm2(x + MLP(x))
 *   Attn: Q/K/V/out_proj all WITH bias -> non-causal softmax(QK^T/sqrt(hd))V
 *   MLP: fc2(gelu_exact(fc1(x))), WITH bias on both fc1 and fc2 (plain 2-matrix FFN, not gated)
 *   pool: CLS token (position 0 of the final layer's output) -- NOT mean pooling
 *   output: raw CLS vector, NOT L2-normalized (confirmed: no Normalize module in modules.json,
 *     graphify-embedder/server.py's own comment: "no normalization", applies to both models)
 *   NOTE: mxbai's usage convention prepends a query-prompt prefix for asymmetric retrieval
 *   (config_sentence_transformers.json's "prompts.query") -- that's a caller-side convention,
 *   not part of this model's forward pass, and the reference server never applies it either
 *   (plain .encode(text)), so this engine matches by tokenizing exactly what it's given.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <stdatomic.h>
#include <riscv_vector.h>

/* ===================== hart affinity (identical to nomic_rvv.c / qwen_moe_hp.c) ===================== */
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static const int g_hart_order[8]={8,10,12,14,9,11,13,15};
static __thread int g_pinned=0;
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(g_hart_order[tn%8],&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

/* ===================== RVV fp32 primitives (identical to nomic_rvv.c -- e32m2 for plain dot
 * products, measured optimal per OCR project's hardware characterization; see that file's own
 * comment for the GFLOP/s table) ===================== */
static float vdot_f32(const float*a,const float*b,int n){
    vfloat32m2_t vacc=__riscv_vfmv_v_f_f32m2(0.0f,__riscv_vsetvlmax_e32m2());
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m2(n-i);
        vfloat32m2_t va=__riscv_vle32_v_f32m2(a+i,vl), vb=__riscv_vle32_v_f32m2(b+i,vl);
        vacc=__riscv_vfmacc_vv_f32m2(vacc,va,vb,vl); i+=vl; }
    vfloat32m1_t vzero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t vsum=__riscv_vfredusum_vs_f32m2_f32m1(vacc,vzero,__riscv_vsetvlmax_e32m2());
    return __riscv_vfmv_f_s_f32m1_f32(vsum);
}
static float rvv_max_f32(const float*x,int n){
    vfloat32m2_t vacc=__riscv_vfmv_v_f_f32m2(-1e30f,__riscv_vsetvlmax_e32m2());
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m2(n-i);
        vfloat32m2_t vx=__riscv_vle32_v_f32m2(x+i,vl);
        vacc=__riscv_vfmax_vv_f32m2(vacc,vx,vl); i+=vl; }
    vfloat32m1_t vid=__riscv_vfmv_v_f_f32m1(-1e30f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t vred=__riscv_vfredmax_vs_f32m2_f32m1(vacc,vid,__riscv_vsetvlmax_e32m2());
    return __riscv_vfmv_f_s_f32m1_f32(vred);
}
static void rvv_scale_f32(float*x,float s,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m2(n-i);
        vfloat32m2_t vx=__riscv_vle32_v_f32m2(x+i,vl);
        vx=__riscv_vfmul_vf_f32m2(vx,s,vl);
        __riscv_vse32_v_f32m2(x+i,vx,vl); i+=vl; }
}
static void vaxpy_f32(float*y,const float*x,float scale,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m2(n-i);
        vfloat32m2_t vx=__riscv_vle32_v_f32m2(x+i,vl), vy=__riscv_vle32_v_f32m2(y+i,vl);
        vy=__riscv_vfmacc_vf_f32m2(vy,scale,vx,vl);
        __riscv_vse32_v_f32m2(y+i,vy,vl); i+=vl; }
}
static void vadd_f32(float*y,const float*x,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m2(n-i);
        vfloat32m2_t vx=__riscv_vle32_v_f32m2(x+i,vl), vy=__riscv_vle32_v_f32m2(y+i,vl);
        vy=__riscv_vfadd_vv_f32m2(vy,vx,vl);
        __riscv_vse32_v_f32m2(y+i,vy,vl); i+=vl; }
}
static void softmax_rvv(float*x,int n){
    float m=rvv_max_f32(x,n);
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    rvv_scale_f32(x,1.0f/s,n);
}
/* exact GELU (erf-based, torch.nn.functional.gelu's default) -- no tanh approximation, same
 * "don't approximate until quality-tested" standing instruction as nomic_rvv.c's exact SiLU.
 * erff() is libm's single-precision erf; scalar, like softmax's expf -- this project's
 * transcendentals stay exact/scalar throughout, only the surrounding arithmetic is RVV. */
static void gelu_exact(float*y,const float*x,int n){
    for(int i=0;i<n;i++){ float v=x[i]; y[i]=0.5f*v*(1.0f+erff(v*0.70710678118654752440f)); }
}
static void layernorm(float*y,const float*x,const float*w,const float*b,int n,float eps){
    float mean=0; for(int i=0;i<n;i++)mean+=x[i]; mean/=n;
    float var=0; for(int i=0;i<n;i++){ float d=x[i]-mean; var+=d*d; } var/=n;
    float inv=1.0f/sqrtf(var+eps);
    for(int i=0;i<n;i++) y[i]=(x[i]-mean)*inv*w[i]+b[i];
}
/* 4x4 GEMM microkernel, e32m1 -- identical technique and register budget to nomic_rvv.c's
 * gemm_tile_4x4 (measured there: 1-row vdot_f32 baseline 3003.9ms -> 4x4 tile 990.2ms on the
 * S=267 case; output-channel-only blocking at m2 was tried and rejected first, see that file).
 * Reused verbatim since it's the same memory-bandwidth-bound GEMM shape (S positions x Hout
 * channels x Hin reduction), just different dimensions (1024/4096 here vs 768/3072). */
static void gemm_tile_4x4(const float*w0,const float*w1,const float*w2,const float*w3,
                           const float*x0,const float*x1,const float*x2,const float*x3,
                           int n,float out[4][4]){
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t a00=z,a01=z,a02=z,a03=z, a10=z,a11=z,a12=z,a13=z;
    vfloat32m1_t a20=z,a21=z,a22=z,a23=z, a30=z,a31=z,a32=z,a33=z;
    int i=0;
    while(i<n){
        size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vw0=__riscv_vle32_v_f32m1(w0+i,vl), vw1=__riscv_vle32_v_f32m1(w1+i,vl);
        vfloat32m1_t vw2=__riscv_vle32_v_f32m1(w2+i,vl), vw3=__riscv_vle32_v_f32m1(w3+i,vl);
        vfloat32m1_t vx0=__riscv_vle32_v_f32m1(x0+i,vl), vx1=__riscv_vle32_v_f32m1(x1+i,vl);
        vfloat32m1_t vx2=__riscv_vle32_v_f32m1(x2+i,vl), vx3=__riscv_vle32_v_f32m1(x3+i,vl);
        a00=__riscv_vfmacc_vv_f32m1(a00,vw0,vx0,vl); a01=__riscv_vfmacc_vv_f32m1(a01,vw1,vx0,vl);
        a02=__riscv_vfmacc_vv_f32m1(a02,vw2,vx0,vl); a03=__riscv_vfmacc_vv_f32m1(a03,vw3,vx0,vl);
        a10=__riscv_vfmacc_vv_f32m1(a10,vw0,vx1,vl); a11=__riscv_vfmacc_vv_f32m1(a11,vw1,vx1,vl);
        a12=__riscv_vfmacc_vv_f32m1(a12,vw2,vx1,vl); a13=__riscv_vfmacc_vv_f32m1(a13,vw3,vx1,vl);
        a20=__riscv_vfmacc_vv_f32m1(a20,vw0,vx2,vl); a21=__riscv_vfmacc_vv_f32m1(a21,vw1,vx2,vl);
        a22=__riscv_vfmacc_vv_f32m1(a22,vw2,vx2,vl); a23=__riscv_vfmacc_vv_f32m1(a23,vw3,vx2,vl);
        a30=__riscv_vfmacc_vv_f32m1(a30,vw0,vx3,vl); a31=__riscv_vfmacc_vv_f32m1(a31,vw1,vx3,vl);
        a32=__riscv_vfmacc_vv_f32m1(a32,vw2,vx3,vl); a33=__riscv_vfmacc_vv_f32m1(a33,vw3,vx3,vl);
        i+=vl;
    }
    vfloat32m1_t vzero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    size_t vlmax=__riscv_vsetvlmax_e32m1();
#define RED(a) __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a,vzero,vlmax))
    out[0][0]=RED(a00); out[0][1]=RED(a01); out[0][2]=RED(a02); out[0][3]=RED(a03);
    out[1][0]=RED(a10); out[1][1]=RED(a11); out[1][2]=RED(a12); out[1][3]=RED(a13);
    out[2][0]=RED(a20); out[2][1]=RED(a21); out[2][2]=RED(a22); out[2][3]=RED(a23);
    out[3][0]=RED(a30); out[3][1]=RED(a31); out[3][2]=RED(a32); out[3][3]=RED(a33);
#undef RED
}

/* ===================== model ===================== */
typedef struct {
    float *q_w,*q_b,*k_w,*k_b,*v_w,*v_b,*o_w,*o_b;  /* [H,H]/[H] */
    float *norm1_w,*norm1_b;
    float *fc1_w,*fc1_b;   /* [I,H]/[I] */
    float *fc2_w,*fc2_b;   /* [H,I]/[H] */
    float *norm2_w,*norm2_b;
} Layer;

typedef struct {
    int hidden,n_layer,n_head,head_dim,intermediate,vocab,type_vocab,pad_token_id;
    int pooling_mode,normalize,use_rope,max_pos,activation,has_qkv_bias,has_out_bias,has_mlp_bias;
    float eps,rope_base;
    float *word_emb, *pos_emb, *type_emb, *emb_ln_w, *emb_ln_b;
    Layer *layers;
    void *map_base; size_t map_len;
} MxbaiModel;

MxbaiModel* mxbai_load(const char*path){
    int fd=open(path,O_RDONLY);
    if(fd<0){ perror("open"); return NULL; }
    struct stat st; fstat(fd,&st);
    void*base=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if(base==MAP_FAILED){ perror("mmap"); return NULL; }
    const uint8_t*h=(const uint8_t*)base;
    if(memcmp(h,"EMBDMXB1",8)!=0){ fprintf(stderr,"bad magic\n"); return NULL; }
    const int32_t*ints=(const int32_t*)(h+8);
    const float*flts=(const float*)(h+8+16*4);
    MxbaiModel*m=calloc(1,sizeof(MxbaiModel));
    m->hidden=ints[0]; m->n_layer=ints[1]; m->n_head=ints[2]; m->head_dim=ints[3];
    m->intermediate=ints[4]; m->vocab=ints[5]; m->type_vocab=ints[6]; m->pad_token_id=ints[7];
    m->pooling_mode=ints[8]; m->normalize=ints[9]; m->use_rope=ints[10]; m->max_pos=ints[11];
    m->activation=ints[12]; m->has_qkv_bias=ints[13]; m->has_out_bias=ints[14]; m->has_mlp_bias=ints[15];
    m->eps=flts[0];
    m->map_base=base; m->map_len=st.st_size;

    int H=m->hidden, I=m->intermediate, V=m->vocab, TV=m->type_vocab, MP=m->max_pos;
    const float*p=(const float*)(h+128);
    m->word_emb=(float*)p; p+=(size_t)V*H;
    m->pos_emb=(float*)p; p+=(size_t)MP*H;
    m->type_emb=(float*)p; p+=(size_t)TV*H;
    m->emb_ln_w=(float*)p; p+=H;
    m->emb_ln_b=(float*)p; p+=H;
    m->layers=calloc(m->n_layer,sizeof(Layer));
    for(int L=0;L<m->n_layer;L++){
        Layer*ly=&m->layers[L];
        ly->q_w=(float*)p; p+=(size_t)H*H; ly->q_b=(float*)p; p+=H;
        ly->k_w=(float*)p; p+=(size_t)H*H; ly->k_b=(float*)p; p+=H;
        ly->v_w=(float*)p; p+=(size_t)H*H; ly->v_b=(float*)p; p+=H;
        ly->o_w=(float*)p; p+=(size_t)H*H; ly->o_b=(float*)p; p+=H;
        ly->norm1_w=(float*)p; p+=H; ly->norm1_b=(float*)p; p+=H;
        ly->fc1_w=(float*)p; p+=(size_t)I*H; ly->fc1_b=(float*)p; p+=I;
        ly->fc2_w=(float*)p; p+=(size_t)H*I; ly->fc2_b=(float*)p; p+=H;
        ly->norm2_w=(float*)p; p+=H; ly->norm2_b=(float*)p; p+=H;
    }
    size_t used=(const uint8_t*)p-(const uint8_t*)h;
    if(used!=(size_t)st.st_size){ fprintf(stderr,"WARNING: size mismatch, used=%zu file=%lld\n",used,(long long)st.st_size); }
    return m;
}
void mxbai_free(MxbaiModel*m){ if(!m)return; munmap(m->map_base,m->map_len); free(m->layers); free(m); }

/* ===================== generic persistent worker pool (identical to nomic_rvv.c) ===================== */
#define MAXHART 8
static pthread_t g_pool_threads[MAXHART];
static _Atomic int g_pool_gen=0, g_pool_done=0;
static int g_active_nt=1;
static void (*volatile g_work_fn)(int tn,void*ctx)=NULL;
static void *volatile g_work_ctx=NULL;
static int g_pool_started=0;

static void range_split(int total,int tn,int nt,int*lo,int*hi){
    if(tn>=nt){ *lo=*hi=0; return; }
    *lo=(int)((int64_t)total*tn/nt); *hi=(int)((int64_t)total*(tn+1)/nt);
}
static void pool_worker_loop(int tn){
    pin_once(tn);
    int last=0;
    for(;;){
        int gen;
        while((gen=atomic_load_explicit(&g_pool_gen,memory_order_acquire))==last){ /* spin */ }
        last=gen;
        g_work_fn(tn,g_work_ctx);
        atomic_fetch_add_explicit(&g_pool_done,1,memory_order_release);
    }
}
static void* pool_worker_thunk(void*arg){ pool_worker_loop((int)(intptr_t)arg); return NULL; }
static void pool_start(void){
    if(g_pool_started) return;
    g_pool_started=1; pin_once(0);
    for(int i=1;i<MAXHART;i++) pthread_create(&g_pool_threads[i],NULL,pool_worker_thunk,(void*)(intptr_t)i);
}
static void pool_dispatch(void (*fn)(int,void*), void*ctx){
    if(g_active_nt<=1){ fn(0,ctx); return; }
    g_work_fn=fn; g_work_ctx=ctx;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release);
    fn(0,ctx);
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < MAXHART-1){ /* spin */ }
}

/* ===================== forward-pass ops ===================== */
typedef struct { const float*W; const float*Bias; const float*X; float*Y; int Hin,Hout,S; } LinCtx;
static void op_linear(int tn,void*vctx){
    LinCtx*c=vctx; int lo,hi; range_split(c->Hout,tn,g_active_nt,&lo,&hi);
    int Hin=c->Hin,Hout=c->Hout,S=c->S; const float*bias=c->Bias;
    int o=lo;
    for(; o+4<=hi; o+=4){
        const float*w0=c->W+(size_t)o*Hin, *w1=c->W+(size_t)(o+1)*Hin;
        const float*w2=c->W+(size_t)(o+2)*Hin, *w3=c->W+(size_t)(o+3)*Hin;
        float b0=bias?bias[o]:0, b1=bias?bias[o+1]:0, b2=bias?bias[o+2]:0, b3=bias?bias[o+3]:0;
        int s=0;
        for(; s+4<=S; s+=4){
            const float*x0=c->X+(size_t)s*Hin, *x1=c->X+(size_t)(s+1)*Hin;
            const float*x2=c->X+(size_t)(s+2)*Hin, *x3=c->X+(size_t)(s+3)*Hin;
            float outv[4][4];
            gemm_tile_4x4(w0,w1,w2,w3,x0,x1,x2,x3,Hin,outv);
            for(int j=0;j<4;j++){
                c->Y[(size_t)(s+j)*Hout+o]=outv[j][0]+b0; c->Y[(size_t)(s+j)*Hout+o+1]=outv[j][1]+b1;
                c->Y[(size_t)(s+j)*Hout+o+2]=outv[j][2]+b2; c->Y[(size_t)(s+j)*Hout+o+3]=outv[j][3]+b3;
            }
        }
        for(; s<S; s++){
            const float*xs=c->X+(size_t)s*Hin;
            c->Y[(size_t)s*Hout+o]=vdot_f32(w0,xs,Hin)+b0; c->Y[(size_t)s*Hout+o+1]=vdot_f32(w1,xs,Hin)+b1;
            c->Y[(size_t)s*Hout+o+2]=vdot_f32(w2,xs,Hin)+b2; c->Y[(size_t)s*Hout+o+3]=vdot_f32(w3,xs,Hin)+b3;
        }
    }
    for(; o<hi; o++){
        const float*wrow=c->W+(size_t)o*Hin; float b=bias?bias[o]:0;
        for(int s=0;s<S;s++) c->Y[(size_t)s*Hout+o]=vdot_f32(wrow,c->X+(size_t)s*Hin,Hin)+b;
    }
}
static void linear(const float*W,const float*Bias,const float*X,float*Y,int Hin,int Hout,int S){
    LinCtx c={W,Bias,X,Y,Hin,Hout,S}; pool_dispatch(op_linear,&c);
}

typedef struct { const float*Q,*K,*V; float*OUT; int S,n_head,head_dim,H; float scale; float**scratch; } AttnCtx;
static void op_attention(int tn,void*vctx){
    AttnCtx*c=vctx; int lo,hi; range_split(c->n_head,tn,g_active_nt,&lo,&hi);
    float*scores=c->scratch[tn];
    int hd=c->head_dim,H=c->H,S=c->S;
    for(int h=lo;h<hi;h++){
        const float*Qh=c->Q+h*hd, *Kh=c->K+h*hd, *Vh=c->V+h*hd;
        for(int i=0;i<S;i++){
            const float*qi=Qh+(size_t)i*H;
            for(int j=0;j<S;j++) scores[j]=vdot_f32(qi,Kh+(size_t)j*H,hd)*c->scale;
            softmax_rvv(scores,S);
            float*outi=c->OUT+(size_t)i*H+h*hd;
            memset(outi,0,sizeof(float)*hd);
            for(int j=0;j<S;j++) vaxpy_f32(outi,Vh+(size_t)j*H,scores[j],hd);
        }
    }
}

typedef struct { const float*X,*R; float*Y; const float*W,*B; int S,H; float eps; } NormCtx;
static void op_addnorm(int tn,void*vctx){
    NormCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    float tmp[2048];
    for(int s=lo;s<hi;s++){
        const float*x=c->X+(size_t)s*c->H;
        if(c->R){ const float*r=c->R+(size_t)s*c->H; for(int i=0;i<c->H;i++) tmp[i]=x[i]+r[i]; x=tmp; }
        layernorm(c->Y+(size_t)s*c->H,x,c->W,c->B,c->H,c->eps);
    }
}

typedef struct { const int32_t*ids; float*Y; const float*word,*pos,*type_; int S,H; } EmbCtx;
static void op_embed(int tn,void*vctx){
    EmbCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    for(int s=lo;s<hi;s++){
        const float*we=c->word+(size_t)c->ids[s]*c->H;
        const float*pe=c->pos+(size_t)s*c->H;
        const float*te=c->type_; /* token_type_id always 0 for single-segment encoding */
        float*y=c->Y+(size_t)s*c->H;
        for(int i=0;i<c->H;i++) y[i]=we[i]+pe[i]+te[i];
    }
}

typedef struct { float*Y; const float*X; int S,I; } GeluCtx;
static void op_gelu(int tn,void*vctx){
    GeluCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    for(int s=lo;s<hi;s++) gelu_exact(c->Y+(size_t)s*c->I,c->X+(size_t)s*c->I,c->I);
}

/* ===================== profiling buckets ===================== */
static double gT_lin=0,gT_attn=0,gT_norm=0,gT_gelu=0,gT_embed=0;
static int g_profile=0;
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
#define TIMED(bucket,stmt) do{ if(g_profile){ double _t0=now_s(); stmt; bucket+=now_s()-_t0; } else { stmt; } }while(0)

/* ===================== full forward pass for one text ===================== */
/* ids: token ids INCLUDING [CLS]/[SEP]. nthreads: 1/2/4/8, clamped. out: caller-allocated
 * float[hidden]. Returns wall time (seconds). Pooling is CLS (position 0), not mean -- no
 * cross-hart reduction needed, just copy the final layer's row 0. */
double mxbai_embed(MxbaiModel*m,const int32_t*ids,int S,int nthreads,float*out){
    pool_start();
    if(nthreads<1)nthreads=1; if(nthreads>MAXHART)nthreads=MAXHART;
    g_active_nt=nthreads;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);

    int H=m->hidden,I=m->intermediate,nh=m->n_head,hd=m->head_dim;
    float*x=malloc((size_t)S*H*4), *xn=malloc((size_t)S*H*4);
    float*q=malloc((size_t)S*H*4), *k=malloc((size_t)S*H*4), *v=malloc((size_t)S*H*4);
    float*attn_out=malloc((size_t)S*H*4), *proj=malloc((size_t)S*H*4);
    float*ffn_hidden=malloc((size_t)S*I*4), *ffn_gelu=malloc((size_t)S*I*4), *ffn_out=malloc((size_t)S*H*4);
    float*attn_scratch[MAXHART]; for(int t=0;t<MAXHART;t++) attn_scratch[t]=malloc((size_t)S*4);

    if(S>m->max_pos){ fprintf(stderr,"S=%d exceeds max_position_embeddings=%d\n",S,m->max_pos); S=m->max_pos; }

    { EmbCtx c={ids,x,m->word_emb,m->pos_emb,m->type_emb,S,H}; TIMED(gT_embed,pool_dispatch(op_embed,&c)); }
    { NormCtx c={x,NULL,xn,m->emb_ln_w,m->emb_ln_b,S,H,m->eps}; TIMED(gT_norm,pool_dispatch(op_addnorm,&c)); }
    float*cur=xn;

    for(int L=0;L<m->n_layer;L++){
        Layer*ly=&m->layers[L];
        TIMED(gT_lin,linear(ly->q_w,ly->q_b,cur,q,H,H,S));
        TIMED(gT_lin,linear(ly->k_w,ly->k_b,cur,k,H,H,S));
        TIMED(gT_lin,linear(ly->v_w,ly->v_b,cur,v,H,H,S));
        { AttnCtx c={q,k,v,attn_out,S,nh,hd,H,1.0f/sqrtf((float)hd),attn_scratch}; TIMED(gT_attn,pool_dispatch(op_attention,&c)); }
        TIMED(gT_lin,linear(ly->o_w,ly->o_b,attn_out,proj,H,H,S));
        { NormCtx c={proj,cur,x,ly->norm1_w,ly->norm1_b,S,H,m->eps}; TIMED(gT_norm,pool_dispatch(op_addnorm,&c)); }
        cur=x;

        TIMED(gT_lin,linear(ly->fc1_w,ly->fc1_b,cur,ffn_hidden,H,I,S));
        { GeluCtx c={ffn_gelu,ffn_hidden,S,I}; TIMED(gT_gelu,pool_dispatch(op_gelu,&c)); }
        TIMED(gT_lin,linear(ly->fc2_w,ly->fc2_b,ffn_gelu,ffn_out,I,H,S));
        { NormCtx c={ffn_out,cur,xn,ly->norm2_w,ly->norm2_b,S,H,m->eps}; TIMED(gT_norm,pool_dispatch(op_addnorm,&c)); }
        cur=xn;
    }

    memcpy(out,cur,sizeof(float)*H); /* CLS token = position 0 */
    if(m->normalize){
        float nrm=sqrtf(vdot_f32(out,out,H));
        if(nrm>1e-12f) rvv_scale_f32(out,1.0f/nrm,H);
    }

    free(x);free(xn);free(q);free(k);free(v);free(attn_out);free(proj);
    free(ffn_hidden);free(ffn_gelu);free(ffn_out);
    for(int t=0;t<MAXHART;t++) free(attn_scratch[t]);

    clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
}

void mxbai_set_profile(int on){ g_profile=on; }
void mxbai_print_profile(void){
    fprintf(stderr,"buckets (sum over all calls since load, ms): lin=%.1f attn=%.1f norm=%.1f gelu=%.1f embed=%.1f\n",
        gT_lin*1000,gT_attn*1000,gT_norm*1000,gT_gelu*1000,gT_embed*1000);
}

#ifdef MXBAI_STANDALONE
int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.bin [-bench N] [-nt K] < ids.txt\n",argv[0]); return 1; }
    MxbaiModel*m=mxbai_load(argv[1]);
    if(!m){ fprintf(stderr,"load failed\n"); return 1; }
    int bench_n=0, nt=8;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"-bench") && i+1<argc) bench_n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-nt") && i+1<argc) nt=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-profile")) g_profile=1;
    }
    char line[1<<20];
    float out[2048];
    while(fgets(line,sizeof(line),stdin)){
        int32_t ids[4096]; int n=0;
        char*p=line;
        while(*p){
            while(*p==' '||*p=='\n'||*p=='\t') p++;
            if(!*p) break;
            ids[n++]=(int32_t)strtol(p,&p,10);
        }
        if(n==0) continue;
        if(bench_n>0){
            double total=0;
            for(int r=0;r<bench_n;r++){ double dt=mxbai_embed(m,ids,n,nt,out); total+=dt;
                fprintf(stderr,"  rep %d/%d: %.1fms\n",r+1,bench_n,dt*1000.0); }
            printf("S=%d nt=%d reps=%d avg_ms=%.3f\n",n,nt,bench_n,1000.0*total/bench_n);
        } else {
            double dt=mxbai_embed(m,ids,n,nt,out);
            for(int i=0;i<m->hidden;i++) printf("%.8g%c",out[i],i+1<m->hidden?' ':'\n');
            fprintf(stderr,"S=%d nt=%d time_ms=%.3f\n",n,nt,dt*1000.0);
        }
    }
    if(g_profile) mxbai_print_profile();
    mxbai_free(m);
    return 0;
}
#endif
