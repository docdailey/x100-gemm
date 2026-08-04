/* nomic_rvv.c -- hand-written native RVV/A100-hart inference engine for nomic-embed-text-v1.5
 * (NomicBertModel: 12-layer bidirectional RoPE+SwiGLU post-norm encoder, mean pooling, no final
 * L2 normalize). Runs directly on the SpaceMIT K3's eight A100 AI harts -- no PyTorch, no ONNX
 * Runtime, no XLA anywhere in the inference path. New project, separate from qwen_moe_hp.c and
 * vision_api/ocr_rvv.c, but reuses their proven hardware-access discipline (bind_ai/pin_once/
 * hart order, persistent spin-dispatch worker pool) and RVV technique (vdot_f32, RVV-vectorized
 * softmax max/normalize, GPT-NeoX-style RoPE) verbatim where the underlying operation is the
 * same. FP32 throughout, same rationale as ocr_rvv.c: makes correctness falsifiable (near-bit-
 * exact vs the real oracle, not "close enough"), and this hardware's plain FP32 RVV FMA is fast
 * enough not to need int8/IME-2.
 *
 * Architecture verified against the actual forward-pass code (modeling_hf_nomic_bert.py, read
 * on ryzen's HF cache, NOT assumed from config.json alone -- see extract_nomic.py's docstring
 * and embed_native/PROGRESS.md for the specific facts and where they were confirmed):
 *   embeddings = word_emb[id] + token_type_emb[0]           (no absolute position embeddings)
 *   x = LayerNorm_emb(embeddings)                             (applied once, before layer 0)
 *   per layer (post-norm, prenorm=false):
 *     x = LayerNorm1(x + Attn(x))
 *     x = LayerNorm2(x + MLP(x))
 *   Attn: Wqkv (no bias) -> RoPE (full head_dim, base=1000, GPT-NeoX rotate-half, non-
 *     interleaved) on Q and K -> non-causal softmax(QK^T/sqrt(hd))V -> out_proj (no bias)
 *   MLP: fc2( fc11(x) * silu(fc12(x)) ), no biases anywhere (fc11=value/up, fc12=gate)
 *   pool: mean over all token positions (single non-padded sequence per call -- no attention-
 *     mask weighting needed, every position is real)
 *   output: raw pooled 768-vector, NOT L2-normalized (confirmed: no Normalize module in
 *     modules.json, and graphify-embedder/server.py's own comment: "no normalization")
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

/* ===================== hart affinity (identical pattern to qwen_moe_hp.c / ocr_rvv.c) =====================
 * Primary harts 8/10/12/14 are never contended; 9/11/13/15 share IME-2 hardware pairwise with
 * their primary partner for IME-2 dot-product instructions specifically -- NOT for plain RVV
 * vector FMA, which both prior native-engine projects on this board measured to scale cleanly
 * across all 8 harts. This engine uses only plain RVV FMA (no IME-2), so all 8 harts are
 * expected to scale cleanly -- verified in the hart-scaling benchmark, not assumed. */
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static const int g_hart_order[8]={8,10,12,14,9,11,13,15};
static __thread int g_pinned=0;
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(g_hart_order[tn%8],&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

/* ===================== RVV fp32 primitives =====================
 * LMUL=2 (e32m2): OCR project (vision_api/BARE_IME_OCR_PROGRESS.md sec.3) measured this
 * hardware's plain-FP32-FMA peak on one AI hart (VLEN=1024, vlenb=128): m1=182.9 GFLOP/s
 * (vl=32), m2=363.0 GFLOP/s (vl=64), m4=230.0 GFLOP/s (vl=128) -- the machine retires a fixed
 * ~2.85G vector-FMA instructions/s regardless of LMUL, so m2 (2x the FLOPs/instruction of m1,
 * without m4's per-instruction regression) is strictly optimal. Reused directly here rather
 * than re-measured, since it's the same silicon; the *shape* question specific to this engine
 * (8-way output-channel blocking below) was re-measured, per this repo's own standing
 * discipline of not assuming a prior finding transfers to a new kernel shape unverified. */
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
/* 8-way and 4-way output-channel-blocked GEMM kernels (qk8_dot's "load one operand's chunk
 * once, update N independent accumulators" technique from qwen_moe_hp.c, applied to output
 * channels sharing one activation row instead of qk8_dot's query heads sharing one K row) were
 * TRIED and REJECTED here, measured not assumed: on the S=267 test case, unblocked e32m2
 * vdot_f32 in a plain o-outer/s-inner loop = 2966ms; the identical kernel blocked 8-wide =
 * 3727ms; blocked 4-wide = 3917ms -- both regressions, not improvements. Root cause (measured
 * via register-pressure reasoning, not re-verified with a disassembly): at LMUL=2 each
 * accumulator group occupies 2 of the 32 physical vector registers, so 8 live accumulators
 * alone consume all 16 available m2 register groups, leaving no headroom for the shared
 * activation-chunk register and almost certainly forcing spills qk8_dot never hits at LMUL=1
 * (8 accumulators = 8 of 32 registers there). The plain unblocked loop is what ships. */
/* exact softmax: RVV max+normalize (order-independent, bit-exact to scalar), scalar expf in
 * between -- identical division of labor to qwen_moe_hp.c's Phase 5, KEPT there after A/B. */
static void softmax_rvv(float*x,int n){
    float m=rvv_max_f32(x,n);
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    rvv_scale_f32(x,1.0f/s,n);
}
/* exact SiLU (scalar expf, no approximation -- same "don't approximate until quality-tested"
 * standing instruction as qwen's swiglu_exact). val[i] <- val[i] * silu(gate[i]) in place. */
static void silu_gate_mul(float*val,const float*gate,int n){
    for(int i=0;i<n;i++){ float g=gate[i]; float s=g/(1.0f+expf(-g)); val[i]*=s; }
}
/* RoPE cos/sin table + apply -- formula and rotate-half convention verified bit-identical to
 * NomicBertRotaryEmbedding/apply_rotary_emb(interleaved=False) in modeling_hf_nomic_bert.py:
 * GPT-NeoX style (rotate first-half/second-half, not interleaved pairs), full head_dim. */
static void rope_table(float*cosb,float*sinb,int hd,int pos,float base){
    for(int i=0;i<hd/2;i++){ float fr=powf(base,-2.0f*i/hd),a=pos*fr; cosb[i]=cosf(a); sinb[i]=sinf(a); }
}
static void rope_apply(float*v,int hd,const float*cosb,const float*sinb){
    for(int i=0;i<hd/2;i++){ float c=cosb[i],s=sinb[i],x=v[i],y=v[i+hd/2]; v[i]=x*c-y*s; v[i+hd/2]=x*s+y*c; }
}
static void layernorm(float*y,const float*x,const float*w,const float*b,int n,float eps){
    float mean=0; for(int i=0;i<n;i++)mean+=x[i]; mean/=n;
    float var=0; for(int i=0;i<n;i++){ float d=x[i]-mean; var+=d*d; } var/=n;
    float inv=1.0f/sqrtf(var+eps);
    for(int i=0;i<n;i++) y[i]=(x[i]-mean)*inv*w[i]+b[i];
}

/* ===================== model ===================== */
typedef struct {
    float *q_w,*k_w,*v_w,*o_w;          /* [H,H] row-major (out,in) */
    float *fc11_w,*fc12_w;               /* [I,H] */
    float *fc2_w;                        /* [H,I] */
    float *norm1_w,*norm1_b,*norm2_w,*norm2_b; /* [H] */
} Layer;

typedef struct {
    int hidden,n_layer,n_head,head_dim,intermediate,vocab,type_vocab,pad_token_id;
    int pooling_mode,normalize,use_rope,max_pos,activation,has_qkv_bias,has_out_bias,has_mlp_bias;
    float eps,rope_base;
    float *word_emb, *type_emb, *emb_ln_w, *emb_ln_b;
    Layer *layers;
    void *map_base; size_t map_len;
} NomicModel;

NomicModel* nomic_load(const char*path){
    int fd=open(path,O_RDONLY);
    if(fd<0){ perror("open"); return NULL; }
    struct stat st; fstat(fd,&st);
    void*base=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if(base==MAP_FAILED){ perror("mmap"); return NULL; }
    const uint8_t*h=(const uint8_t*)base;
    if(memcmp(h,"EMBDNOM1",8)!=0){ fprintf(stderr,"bad magic\n"); return NULL; }
    const int32_t*ints=(const int32_t*)(h+8);
    const float*flts=(const float*)(h+8+16*4);
    NomicModel*m=calloc(1,sizeof(NomicModel));
    m->hidden=ints[0]; m->n_layer=ints[1]; m->n_head=ints[2]; m->head_dim=ints[3];
    m->intermediate=ints[4]; m->vocab=ints[5]; m->type_vocab=ints[6]; m->pad_token_id=ints[7];
    m->pooling_mode=ints[8]; m->normalize=ints[9]; m->use_rope=ints[10]; m->max_pos=ints[11];
    m->activation=ints[12]; m->has_qkv_bias=ints[13]; m->has_out_bias=ints[14]; m->has_mlp_bias=ints[15];
    m->eps=flts[0]; m->rope_base=flts[1];
    m->map_base=base; m->map_len=st.st_size;

    int H=m->hidden, I=m->intermediate, V=m->vocab, TV=m->type_vocab;
    const float*p=(const float*)(h+128);
    m->word_emb=(float*)p; p+=(size_t)V*H;
    m->type_emb=(float*)p; p+=(size_t)TV*H;
    m->emb_ln_w=(float*)p; p+=H;
    m->emb_ln_b=(float*)p; p+=H;
    m->layers=calloc(m->n_layer,sizeof(Layer));
    for(int L=0;L<m->n_layer;L++){
        Layer*ly=&m->layers[L];
        ly->q_w=(float*)p; p+=(size_t)H*H;
        ly->k_w=(float*)p; p+=(size_t)H*H;
        ly->v_w=(float*)p; p+=(size_t)H*H;
        ly->o_w=(float*)p; p+=(size_t)H*H;
        ly->fc11_w=(float*)p; p+=(size_t)I*H;
        ly->fc12_w=(float*)p; p+=(size_t)I*H;
        ly->fc2_w=(float*)p; p+=(size_t)H*I;
        ly->norm1_w=(float*)p; p+=H;
        ly->norm1_b=(float*)p; p+=H;
        ly->norm2_w=(float*)p; p+=H;
        ly->norm2_b=(float*)p; p+=H;
    }
    size_t used=(const uint8_t*)p-(const uint8_t*)h;
    if(used!=(size_t)st.st_size){ fprintf(stderr,"WARNING: size mismatch, used=%zu file=%lld\n",used,(long long)st.st_size); }
    return m;
}
void nomic_free(NomicModel*m){ if(!m)return; munmap(m->map_base,m->map_len); free(m->layers); free(m); }

/* ===================== generic persistent worker pool ===================== */
#define MAXHART 8
static pthread_t g_pool_threads[MAXHART];
static _Atomic int g_pool_gen=0, g_pool_done=0;
static int g_active_nt=1;         /* how many harts participate in the CURRENT dispatch */
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
/* dispatch fn(tn,ctx) across g_active_nt harts (hart 0 runs inline, 1..nt-1 via the pool);
 * threads at index >= g_active_nt still wake and no-op inside fn (range_split gives them an
 * empty range) but still post to g_pool_done, which is why the wait target is MAXHART-1
 * always -- avoids reasoning about a variable number of live workers per call. */
static void pool_dispatch(void (*fn)(int,void*), void*ctx){
    if(g_active_nt<=1){ fn(0,ctx); return; }
    g_work_fn=fn; g_work_ctx=ctx;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release);
    fn(0,ctx);
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < MAXHART-1){ /* spin */ }
}

/* ===================== forward-pass ops (dispatched across harts) ===================== */
/* 4x4 GEMM microkernel (MR=4 positions x NR=4 output channels), e32m1: profiling
 * (embed_native/PROGRESS.md) found linear layers memory-traffic-bound, not compute-bound (m1
 * -> m2 alone only gained ~11%, far short of m2's ~2x compute throughput, and 8-way
 * output-channel-only blocking at m2 REGRESSED from register pressure -- see the rejected-
 * blocking note above). A 1-row-at-a-time dot product re-reads all of X once per output
 * channel (X doesn't fit the K3's 1MB L2 at realistic S): total traffic ~Hout*S*Hin*4 bytes.
 * Tiling BOTH dimensions at once (unlike the earlier N-only attempt) lets each loaded Hin-chunk
 * of X and W serve MR*NR accumulators before eviction, cutting traffic by ~MR (for X) and ~NR
 * (for W). Measured (S=267, nt=8): 1-row vdot_f32 baseline (m2) lin=3003.9ms; 4x2 tile (8
 * accumulators, 14 live regs) lin=2059.6ms; 4x4 tile (16 accumulators + 8 operand regs = 24 of
 * 32 physical m1 registers, still under the m2-blocking regression's 16-just-for-accumulators
 * ceiling) is what ships -- see PROGRESS.md for the head-to-head number. */
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

typedef struct { const float*W; const float*X; float*Y; int Hin,Hout,S; } LinCtx;
static void op_linear(int tn,void*vctx){
    LinCtx*c=vctx; int lo,hi; range_split(c->Hout,tn,g_active_nt,&lo,&hi);
    int Hin=c->Hin,Hout=c->Hout,S=c->S;
    int o=lo;
    for(; o+4<=hi; o+=4){
        const float*w0=c->W+(size_t)o*Hin, *w1=c->W+(size_t)(o+1)*Hin;
        const float*w2=c->W+(size_t)(o+2)*Hin, *w3=c->W+(size_t)(o+3)*Hin;
        int s=0;
        for(; s+4<=S; s+=4){
            const float*x0=c->X+(size_t)s*Hin, *x1=c->X+(size_t)(s+1)*Hin;
            const float*x2=c->X+(size_t)(s+2)*Hin, *x3=c->X+(size_t)(s+3)*Hin;
            float outv[4][4];
            gemm_tile_4x4(w0,w1,w2,w3,x0,x1,x2,x3,Hin,outv);
            for(int j=0;j<4;j++) for(int k=0;k<4;k++) c->Y[(size_t)(s+j)*Hout+o+k]=outv[j][k];
        }
        for(; s<S; s++){
            const float*xs=c->X+(size_t)s*Hin;
            c->Y[(size_t)s*Hout+o]=vdot_f32(w0,xs,Hin); c->Y[(size_t)s*Hout+o+1]=vdot_f32(w1,xs,Hin);
            c->Y[(size_t)s*Hout+o+2]=vdot_f32(w2,xs,Hin); c->Y[(size_t)s*Hout+o+3]=vdot_f32(w3,xs,Hin);
        }
    }
    for(; o<hi; o++){
        const float*wrow=c->W+(size_t)o*Hin;
        for(int s=0;s<S;s++) c->Y[(size_t)s*Hout+o]=vdot_f32(wrow,c->X+(size_t)s*Hin,Hin);
    }
}
static void linear(const float*W,const float*X,float*Y,int Hin,int Hout,int S){
    LinCtx c={W,X,Y,Hin,Hout,S}; pool_dispatch(op_linear,&c);
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
    float tmp[2048]; /* H<=2048 for both target models (768/1024) */
    for(int s=lo;s<hi;s++){
        const float*x=c->X+(size_t)s*c->H;
        if(c->R){ const float*r=c->R+(size_t)s*c->H; for(int i=0;i<c->H;i++) tmp[i]=x[i]+r[i]; x=tmp; }
        layernorm(c->Y+(size_t)s*c->H,x,c->W,c->B,c->H,c->eps);
    }
}

typedef struct { const int32_t*ids; float*Y; const float*word,*type_; int S,H; } EmbCtx;
static void op_embed(int tn,void*vctx){
    EmbCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    for(int s=lo;s<hi;s++){
        const float*we=c->word+(size_t)c->ids[s]*c->H;
        const float*te=c->type_; /* token_type_id always 0 for single-segment encoding */
        float*y=c->Y+(size_t)s*c->H;
        for(int i=0;i<c->H;i++) y[i]=we[i]+te[i];
    }
}

typedef struct { float*val,*gate; int S,I; } SwigluCtx;
static void op_swiglu(int tn,void*vctx){
    SwigluCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    for(int s=lo;s<hi;s++) silu_gate_mul(c->val+(size_t)s*c->I,c->gate+(size_t)s*c->I,c->I);
}

typedef struct { float*Q,*K; int S,n_head,head_dim,H; float base; } RopeCtx;
static void op_rope(int tn,void*vctx){
    RopeCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    float cosb[128],sinb[128]; /* head_dim<=128 for both target models */
    for(int s=lo;s<hi;s++){
        rope_table(cosb,sinb,c->head_dim,s,c->base);
        for(int hh=0;hh<c->n_head;hh++){
            rope_apply(c->Q+(size_t)s*c->H+hh*c->head_dim,c->head_dim,cosb,sinb);
            rope_apply(c->K+(size_t)s*c->H+hh*c->head_dim,c->head_dim,cosb,sinb);
        }
    }
}

typedef struct { const float*X; float*partial; int S,H; } PoolCtx;
static void op_pool_partial(int tn,void*vctx){
    PoolCtx*c=vctx; int lo,hi; range_split(c->S,tn,g_active_nt,&lo,&hi);
    float*acc=c->partial+(size_t)tn*c->H;
    memset(acc,0,sizeof(float)*c->H);
    for(int s=lo;s<hi;s++) vadd_f32(acc,c->X+(size_t)s*c->H,c->H);
}

/* ===================== profiling buckets (env NOMIC_PROFILE=1) ===================== */
static double gT_lin=0,gT_attn=0,gT_rope=0,gT_norm=0,gT_swiglu=0,gT_embed=0,gT_pool=0;
static int g_profile=0;
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
#define TIMED(bucket,stmt) do{ if(g_profile){ double _t0=now_s(); stmt; bucket+=now_s()-_t0; } else { stmt; } }while(0)

/* ===================== full forward pass for one text ===================== */
/* ids: token ids INCLUDING [CLS]/[SEP] (tokenization stays in Python, per project convention --
 * lightweight CPU-side preprocessing, same division of labor as ocr_rvv.c's image pipeline).
 * nthreads: 1/2/4/8, clamped. out: caller-allocated float[hidden]. Returns wall time (seconds). */
double nomic_embed(NomicModel*m,const int32_t*ids,int S,int nthreads,float*out){
    pool_start();
    if(nthreads<1)nthreads=1; if(nthreads>MAXHART)nthreads=MAXHART;
    g_active_nt=nthreads;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);

    int H=m->hidden,I=m->intermediate,nh=m->n_head,hd=m->head_dim;
    float*x=malloc((size_t)S*H*4), *xn=malloc((size_t)S*H*4);
    float*q=malloc((size_t)S*H*4), *k=malloc((size_t)S*H*4), *v=malloc((size_t)S*H*4);
    float*attn_out=malloc((size_t)S*H*4), *proj=malloc((size_t)S*H*4);
    float*gate=malloc((size_t)S*I*4), *val=malloc((size_t)S*I*4), *mlp_out=malloc((size_t)S*H*4);
    float*attn_scratch[MAXHART]; for(int t=0;t<MAXHART;t++) attn_scratch[t]=malloc((size_t)S*4);
    float*pool_partial=malloc((size_t)MAXHART*H*4);

    { EmbCtx c={ids,x,m->word_emb,m->type_emb,S,H}; TIMED(gT_embed,pool_dispatch(op_embed,&c)); }
    { NormCtx c={x,NULL,xn,m->emb_ln_w,m->emb_ln_b,S,H,m->eps}; TIMED(gT_norm,pool_dispatch(op_addnorm,&c)); }
    float*cur=xn;

    for(int L=0;L<m->n_layer;L++){
        Layer*ly=&m->layers[L];
        TIMED(gT_lin,linear(ly->q_w,cur,q,H,H,S));
        TIMED(gT_lin,linear(ly->k_w,cur,k,H,H,S));
        TIMED(gT_lin,linear(ly->v_w,cur,v,H,H,S));
        { RopeCtx c={q,k,S,nh,hd,H,m->rope_base}; TIMED(gT_rope,pool_dispatch(op_rope,&c)); }
        { AttnCtx c={q,k,v,attn_out,S,nh,hd,H,1.0f/sqrtf((float)hd),attn_scratch}; TIMED(gT_attn,pool_dispatch(op_attention,&c)); }
        TIMED(gT_lin,linear(ly->o_w,attn_out,proj,H,H,S));
        { NormCtx c={proj,cur,x,ly->norm1_w,ly->norm1_b,S,H,m->eps}; TIMED(gT_norm,pool_dispatch(op_addnorm,&c)); }
        cur=x; /* x now holds post-attn-block hidden states */

        TIMED(gT_lin,linear(ly->fc11_w,cur,val,H,I,S));
        TIMED(gT_lin,linear(ly->fc12_w,cur,gate,H,I,S));
        { SwigluCtx c={val,gate,S,I}; TIMED(gT_swiglu,pool_dispatch(op_swiglu,&c)); }
        TIMED(gT_lin,linear(ly->fc2_w,val,mlp_out,I,H,S));
        { NormCtx c={mlp_out,cur,xn,ly->norm2_w,ly->norm2_b,S,H,m->eps}; TIMED(gT_norm,pool_dispatch(op_addnorm,&c)); }
        cur=xn; /* xn now holds this layer's output; x is free scratch again -- this restores
                    the exact same (cur==xn, x==free) invariant the loop started with, so next
                    iteration's norm1 write into `x` never aliases its own R=cur read */
    }

    /* mean pool over all S positions (single non-padded sequence -- no mask weighting needed) */
    { PoolCtx c={cur,pool_partial,S,H}; TIMED(gT_pool,pool_dispatch(op_pool_partial,&c)); }
    memset(out,0,sizeof(float)*H);
    for(int t=0;t<g_active_nt;t++) vadd_f32(out,pool_partial+(size_t)t*H,H);
    rvv_scale_f32(out,1.0f/(float)S,H);
    if(m->normalize){
        float nrm=sqrtf(vdot_f32(out,out,H));
        if(nrm>1e-12f) rvv_scale_f32(out,1.0f/nrm,H);
    }

    free(x);free(xn);free(q);free(k);free(v);free(attn_out);free(proj);
    free(gate);free(val);free(mlp_out);free(pool_partial);
    for(int t=0;t<MAXHART;t++) free(attn_scratch[t]);

    clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
}

void nomic_set_profile(int on){ g_profile=on; }
void nomic_print_profile(void){
    fprintf(stderr,"buckets (sum over all calls since load, ms): lin=%.1f attn=%.1f rope=%.1f norm=%.1f swiglu=%.1f embed=%.1f pool=%.1f\n",
        gT_lin*1000,gT_attn*1000,gT_rope*1000,gT_norm*1000,gT_swiglu*1000,gT_embed*1000,gT_pool*1000);
}

#ifdef NOMIC_STANDALONE
/* CLI test/bench harness: reads one line per call from stdin, each line a whitespace-separated
 * list of token ids (already tokenized in Python -- see nomic_native.py). Prints the 768-dim
 * embedding as space-separated floats on one line, or with -bench times N reps and reports
 * throughput instead of printing vectors. */
int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.bin [-bench N] [-nt K] < ids.txt\n",argv[0]); return 1; }
    NomicModel*m=nomic_load(argv[1]);
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
            for(int r=0;r<bench_n;r++){ double dt=nomic_embed(m,ids,n,nt,out); total+=dt;
                fprintf(stderr,"  rep %d/%d: %.1fms\n",r+1,bench_n,dt*1000.0); }
            printf("S=%d nt=%d reps=%d avg_ms=%.3f\n",n,nt,bench_n,1000.0*total/bench_n);
        } else {
            double dt=nomic_embed(m,ids,n,nt,out);
            for(int i=0;i<m->hidden;i++) printf("%.8g%c",out[i],i+1<m->hidden?' ':'\n');
            fprintf(stderr,"S=%d nt=%d time_ms=%.3f\n",n,nt,dt*1000.0);
        }
    }
    if(g_profile) nomic_print_profile();
    nomic_free(m);
    return 0;
}
#endif
