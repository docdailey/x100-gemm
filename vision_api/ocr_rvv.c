/* ocr_rvv.c — native FP32 RVV forward pass for PaddleOCR text recognizers.
 *
 * Runs on the SpaceMIT K3 ("Jupiter 2") A100 AI harts. No ONNX Runtime, no
 * SpaceMIT EP, no XQuant, no Paddle runtime, no CPU fallback: every FLOP goes
 * through the RVV vector kernels below.
 *
 * One engine, two models, selected entirely by the blob it is given:
 *   PP-OCRv6 Tiny (PP-LCNet MBConv + SE + GELU + HardSwish, 6906 classes)
 *   Rosetta       (ResNet34-vd + CTC, 37 classes)
 * Both blobs come from extract_ocr.py, which folds BatchNorm, fuses residual
 * adds and activations onto the producing conv, collapses Squeeze-and-Excite to
 * one op, and drops the reshape/transpose no-ops. The runtime therefore only
 * implements Conv (dense + depthwise), MaxPool, AveragePool and an FC head.
 *
 * Build (ON the board — RISC-V codegen with these extensions must be native):
 *   shared lib:
 *     gcc -O3 -fno-tree-vectorize -funroll-loops -march=rv64gcv_zfh_zvfh_xsmtvdotii \
 *         -fPIC -shared -o libocr_rvv.so ocr_rvv.c -lm -lpthread
 *   test/bench binary:
 *     gcc -O3 -fno-tree-vectorize -funroll-loops -march=rv64gcv_zfh_zvfh_xsmtvdotii \
 *         -DOCR_MAIN -o ocr_test ocr_rvv.c -lm -lpthread
 *
 * Both flags were measured on this board, not assumed: dropping
 * -fno-tree-vectorize costs ~11% even though all vectorization here is explicit
 * intrinsics, and -funroll-loops is worth ~1.4x. Neither changes an output bit.
 *
 * Hardware access follows qwen_moe_hp.c: bind_ai() grants the calling thread
 * AI-hart access, g_hart_order pins one thread per hart, and a persistent
 * spin-dispatch pool avoids per-layer thread spawn. This engine issues no IME-2
 * dot-product instructions, so the per-hart-pair IME-2 contention that caps that
 * engine's linear layers at four harts does not apply — all eight are used.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <riscv_vector.h>

#define OP_CONV 1
#define OP_MAXPOOL 2
#define OP_AVGPOOL 3
#define OP_FC 4
#define OP_FIELDS 32
#define ACT_NONE 0
#define ACT_RELU 1
#define ACT_GELU 2
#define ACT_HARDSWISH 3
#define ACT_HARDSIGMOID 4
#define MR 8            /* GEMM register-block rows; must match extract_ocr.py */
#define NT_MAX 256      /* max columns per output tile */
#define CACC_MAX 16384  /* floats in a work unit's output accumulator (64 KB) */
#define KB_TARGET 288   /* K-block depth for im2col */
#define MAXNT 16

/* op record field indices, mirroring extract_ocr.py's struct.pack order */
enum { F_OP, F_IN, F_OUT, F_ADD, F_ACT, F_OC, F_IC, F_KH, F_KW, F_SH, F_SW,
       F_PH, F_PW, F_CEIL, F_CIP, F_WOFF, F_BOFF, F_GROUP, F_SEOFF, F_SESQ,
       F_FCLAYOUT, F_FCSOFTMAX };

/* dispatch phases within one op */
enum { PH_MAIN, PH_SE_APPLY, PH_SOFTMAX };

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ===================== hart binding ===================== */
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=(int)write(fd,"0",1); close(fd); return r<0?-1:0; }
static const int g_hart_order[8]={8,10,12,14,9,11,13,15};
static __thread int g_pinned=0;
static int g_use_ai_harts=1, g_debug=0, g_prof=0;
static void pin_once(int tn){
    if(g_pinned) return;
    g_pinned=1;
    if(!g_use_ai_harts) return;
    if(bind_ai()!=0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(g_hart_order[tn%8],&cs);
    sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
}

/* ===================== vector transcendentals =====================
 * exp: Cephes single-precision expf, range-reduced to 2^k * exp(r).
 * erf: Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7), mirrored for x<0.
 * Both are accurate to well under the ~1e-5 bar the correctness harness holds
 * the whole engine to; erf is used only inside GELU, where the residual error
 * is scaled by |x| and stays comparable to float32 rounding. */
static inline vfloat32m2_t vexp_m2(vfloat32m2_t x,size_t vl){
    x=__riscv_vfmin_vf_f32m2(x,88.3762626f,vl);
    x=__riscv_vfmax_vf_f32m2(x,-87.3365479f,vl);
    vint32m2_t k=__riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(x,1.44269504088896341f,vl),vl);
    vfloat32m2_t kf=__riscv_vfcvt_f_x_v_f32m2(k,vl);
    vfloat32m2_t r=__riscv_vfnmsac_vf_f32m2(x,0.693145751953125f,kf,vl);
    r=__riscv_vfnmsac_vf_f32m2(r,1.42860682030941723e-6f,kf,vl);
    vfloat32m2_t p=__riscv_vfmv_v_f_f32m2(1.9875691500e-4f,vl);
    p=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p,r,vl),1.3981999507e-3f,vl);
    p=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p,r,vl),8.3334519073e-3f,vl);
    p=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p,r,vl),4.1665795894e-2f,vl);
    p=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p,r,vl),1.6666665459e-1f,vl);
    p=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p,r,vl),5.0000001201e-1f,vl);
    vfloat32m2_t r2=__riscv_vfmul_vv_f32m2(r,r,vl);
    vfloat32m2_t e=__riscv_vfadd_vf_f32m2(__riscv_vfmacc_vv_f32m2(r,r2,p,vl),1.0f,vl);
    vint32m2_t bits=__riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(k,127,vl),23,vl);
    return __riscv_vfmul_vv_f32m2(e,__riscv_vreinterpret_v_i32m2_f32m2(bits),vl);
}
static inline vfloat32m2_t verf_m2(vfloat32m2_t x,size_t vl){
    vfloat32m2_t ax=__riscv_vfabs_v_f32m2(x,vl);
    vfloat32m2_t t=__riscv_vfrdiv_vf_f32m2(
        __riscv_vfadd_vf_f32m2(__riscv_vfmul_vf_f32m2(ax,0.3275911f,vl),1.0f,vl),1.0f,vl);
    vfloat32m2_t y=__riscv_vfmv_v_f_f32m2(1.061405429f,vl);
    y=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(y,t,vl),-1.453152027f,vl);
    y=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(y,t,vl), 1.421413741f,vl);
    y=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(y,t,vl),-0.284496736f,vl);
    y=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(y,t,vl), 0.254829592f,vl);
    y=__riscv_vfmul_vv_f32m2(y,t,vl);
    vfloat32m2_t ex=vexp_m2(__riscv_vfneg_v_f32m2(__riscv_vfmul_vv_f32m2(ax,ax,vl),vl),vl);
    vfloat32m2_t m=__riscv_vfnmsac_vv_f32m2(__riscv_vfmv_v_f_f32m2(1.0f,vl),y,ex,vl);
    return __riscv_vfsgnj_vv_f32m2(m,x,vl);   /* copy the sign of x onto |erf| */
}
static inline vfloat32m2_t vhardsigmoid_m2(vfloat32m2_t x,size_t vl){
    vfloat32m2_t h=__riscv_vfadd_vf_f32m2(__riscv_vfmul_vf_f32m2(x,0.16666667f,vl),0.5f,vl);
    return __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(h,0.0f,vl),1.0f,vl);
}
static inline vfloat32m2_t act_apply(vfloat32m2_t v,int act,size_t vl){
    switch(act){
    case ACT_RELU:   return __riscv_vfmax_vf_f32m2(v,0.0f,vl);
    case ACT_GELU: {
        vfloat32m2_t e=verf_m2(__riscv_vfmul_vf_f32m2(v,0.70710678118654752f,vl),vl);
        return __riscv_vfmul_vv_f32m2(__riscv_vfmul_vf_f32m2(v,0.5f,vl),
                                      __riscv_vfadd_vf_f32m2(e,1.0f,vl),vl); }
    case ACT_HARDSWISH:   return __riscv_vfmul_vv_f32m2(v,vhardsigmoid_m2(v,vl),vl);
    case ACT_HARDSIGMOID: return vhardsigmoid_m2(v,vl);
    default: return v;
    }
}

/* ===================== blob ===================== */
typedef struct { int c,h,w; } Shape;

static struct {
    int loaded;
    unsigned char *map; size_t map_bytes;
    const int32_t *ops; int n_ops, n_bufs, final_buf, n_classes, mr, input_h;
    const float *data;
    float **bufs; size_t *buf_elems, *buf_alloc;
    Shape *shape, *op_in, *op_out;
    int prepared_w, out_steps;
    int nw;
    size_t pad_floats;          /* per-worker depthwise padded-plane scratch */
    float *se_sum, *se_gate;    /* max_channels each */
    pthread_mutex_t lock;
} G;

static int op_i(int idx,int f){ return G.ops[(size_t)idx*OP_FIELDS+f]; }

static int pool_dim(int in,int k,int s,int p,int ceil_mode){
    int num=in+2*p-k,o;
    if(ceil_mode){ o=(num+s-1)/s+1; if((o-1)*s>=in+p) o--; }
    else o=num/s+1;
    return o;
}

static int shape_pass(int W){
    for(int i=0;i<G.n_bufs;i++) G.shape[i]=(Shape){0,0,0};
    int in0=op_i(0,F_IN);
    G.shape[in0]=(Shape){3,G.input_h,W};
    if((size_t)3*G.input_h*W>G.buf_elems[in0]) G.buf_elems[in0]=(size_t)3*G.input_h*W;
    G.pad_floats=0;
    for(int i=0;i<G.n_ops;i++){
        Shape s=G.shape[op_i(i,F_IN)],d;
        int kind=op_i(i,F_OP);
        if(kind==OP_CONV){
            if(s.c!=op_i(i,F_IC)) return -1;
            d.c=op_i(i,F_OC);
            d.h=(s.h+2*op_i(i,F_PH)-op_i(i,F_KH))/op_i(i,F_SH)+1;
            d.w=(s.w+2*op_i(i,F_PW)-op_i(i,F_KW))/op_i(i,F_SW)+1;
            int a=op_i(i,F_ADD);
            if(a>=0&&(G.shape[a].c!=d.c||G.shape[a].h!=d.h||G.shape[a].w!=d.w)) return -2;
            if(op_i(i,F_GROUP)>1){
                size_t need=(size_t)s.h*(s.w+2*op_i(i,F_PW));
                if(need>G.pad_floats) G.pad_floats=need;
            }
        } else if(kind==OP_MAXPOOL||kind==OP_AVGPOOL){
            d.c=s.c;
            d.h=pool_dim(s.h,op_i(i,F_KH),op_i(i,F_SH),op_i(i,F_PH),op_i(i,F_CEIL));
            d.w=pool_dim(s.w,op_i(i,F_KW),op_i(i,F_SW),op_i(i,F_PW),op_i(i,F_CEIL));
        } else if(kind==OP_FC){
            /* An FC is a 1x1 conv over a (IC, 1, T) tensor. Height must already
             * have collapsed; extract_ocr.py drops the squeeze that says so. */
            if(s.h!=1||s.c!=op_i(i,F_IC)) return -3;
            if(op_i(i,F_FCLAYOUT)) { d.c=1; d.h=s.w; d.w=op_i(i,F_OC); }
            else                   { d.c=op_i(i,F_OC); d.h=1; d.w=s.w; }
        } else return -4;
        if(d.h<1||d.w<1) return -5;
        G.op_in[i]=s; G.op_out[i]=d;
        G.shape[op_i(i,F_OUT)]=d;
        size_t need=(size_t)d.c*d.h*d.w;
        if(need>G.buf_elems[op_i(i,F_OUT)]) G.buf_elems[op_i(i,F_OUT)]=need;
    }
    Shape f=G.shape[G.final_buf];
    G.out_steps=op_i(G.n_ops-1,F_FCLAYOUT)?f.h:f.w;
    return 0;
}

/* ===================== worker pool ===================== */
typedef struct {
    int kind;
    const float *src,*add,*wt,*bias,*se;
    float *dst;
    int IC,IH,IW,OC,OH,OW,K,cols;
    int KH,KW,SH,SW,PH,PW,act,ceil_mode,cip,group;
    int se_squeeze,fc_layout,fc_softmax,nclass;
    int NT,MB,n_tiles,m_blocks,units,ic_chunk,KB,direct_b;
} Work;

static Work g_work;
/* The dispatch phase is a real atomic, not a plain field of g_work. As a plain
 * field this gcc kept a stale value in the worker loop -- every dispatch after
 * the first silently re-ran the previous phase, which turned Squeeze-and-Excite
 * into a no-op gate and was invisible except as wrong numbers. An atomic load
 * cannot be hoisted out of the loop; the rest of g_work is still published by
 * the release/acquire pair on g_gen. */
static _Atomic int g_phase=0;
static _Atomic int g_gen=0,g_done=0;
static pthread_t g_threads[MAXNT];
static float *g_scratch[MAXNT];
static int *g_colidx[MAXNT];

#define GEMM_SCRATCH ((size_t)KB_TARGET*NT_MAX+(size_t)CACC_MAX+64)

/* im2col for one (channel-block, column-tile) slice: Btile[kk*NT + nn]. */
static void im2col_block(const Work *w,const int *ohw,int ic0,int ic1,
                         int ncol,float *Btile,int NT)
{
    int kk=0;
    for(int ic=ic0;ic<ic1;ic++){
        const float *s=w->src+(size_t)ic*w->IH*w->IW;
        for(int i=0;i<w->KH;i++){
            for(int j=0;j<w->KW;j++,kk++){
                float *d=Btile+(size_t)kk*NT;
                if(w->SW==1){
                    int nn=0;
                    while(nn<ncol){
                        int oh=ohw[2*nn],ow=ohw[2*nn+1];
                        int run=w->OW-ow; if(run>ncol-nn) run=ncol-nn;
                        int ih=oh*w->SH-w->PH+i;
                        if(ih<0||ih>=w->IH){ memset(d+nn,0,(size_t)run*sizeof(float)); nn+=run; continue; }
                        const float *sr=s+(size_t)ih*w->IW;
                        int iw0=ow-w->PW+j;
                        int lo=iw0<0?-iw0:0;
                        int hi=(iw0+run>w->IW)?(w->IW-iw0):run;
                        if(lo>run) lo=run;
                        if(hi>run) hi=run;
                        if(hi<lo) hi=lo;
                        if(lo>0) memset(d+nn,0,(size_t)lo*sizeof(float));
                        if(hi>lo) memcpy(d+nn+lo,sr+iw0+lo,(size_t)(hi-lo)*sizeof(float));
                        if(run>hi) memset(d+nn+hi,0,(size_t)(run-hi)*sizeof(float));
                        nn+=run;
                    }
                } else {
                    for(int nn=0;nn<ncol;nn++){
                        int ih=ohw[2*nn]*w->SH-w->PH+i;
                        int iw=ohw[2*nn+1]*w->SW-w->PW+j;
                        d[nn]=(ih>=0&&ih<w->IH&&iw>=0&&iw<w->IW)?s[(size_t)ih*w->IW+iw]:0.0f;
                    }
                }
            }
        }
    }
}

/* MR=8 output rows x vl columns, accumulating one K-block into Cacc.
 * `bstride` is the B row stride, which is NOT the tile width when B points
 * straight at a source tensor (1x1 convs and the FC head skip im2col). */
static void micro_mr8(const float *ap,int kblen,const float *bt,int bstride,
                      float *cc,int NT,int nact)
{
    for(int n0=0;n0<nact;){
        size_t vl=__riscv_vsetvl_e32m2((size_t)(nact-n0));
        vfloat32m2_t a0=__riscv_vle32_v_f32m2(cc+0*NT+n0,vl);
        vfloat32m2_t a1=__riscv_vle32_v_f32m2(cc+1*NT+n0,vl);
        vfloat32m2_t a2=__riscv_vle32_v_f32m2(cc+2*NT+n0,vl);
        vfloat32m2_t a3=__riscv_vle32_v_f32m2(cc+3*NT+n0,vl);
        vfloat32m2_t a4=__riscv_vle32_v_f32m2(cc+4*NT+n0,vl);
        vfloat32m2_t a5=__riscv_vle32_v_f32m2(cc+5*NT+n0,vl);
        vfloat32m2_t a6=__riscv_vle32_v_f32m2(cc+6*NT+n0,vl);
        vfloat32m2_t a7=__riscv_vle32_v_f32m2(cc+7*NT+n0,vl);
        const float *app=ap; const float *bp=bt+n0;
        for(int k=0;k<kblen;k++,app+=MR,bp+=bstride){
            vfloat32m2_t vb=__riscv_vle32_v_f32m2(bp,vl);
            a0=__riscv_vfmacc_vf_f32m2(a0,app[0],vb,vl);
            a1=__riscv_vfmacc_vf_f32m2(a1,app[1],vb,vl);
            a2=__riscv_vfmacc_vf_f32m2(a2,app[2],vb,vl);
            a3=__riscv_vfmacc_vf_f32m2(a3,app[3],vb,vl);
            a4=__riscv_vfmacc_vf_f32m2(a4,app[4],vb,vl);
            a5=__riscv_vfmacc_vf_f32m2(a5,app[5],vb,vl);
            a6=__riscv_vfmacc_vf_f32m2(a6,app[6],vb,vl);
            a7=__riscv_vfmacc_vf_f32m2(a7,app[7],vb,vl);
        }
        __riscv_vse32_v_f32m2(cc+0*NT+n0,a0,vl);
        __riscv_vse32_v_f32m2(cc+1*NT+n0,a1,vl);
        __riscv_vse32_v_f32m2(cc+2*NT+n0,a2,vl);
        __riscv_vse32_v_f32m2(cc+3*NT+n0,a3,vl);
        __riscv_vse32_v_f32m2(cc+4*NT+n0,a4,vl);
        __riscv_vse32_v_f32m2(cc+5*NT+n0,a5,vl);
        __riscv_vse32_v_f32m2(cc+6*NT+n0,a6,vl);
        __riscv_vse32_v_f32m2(cc+7*NT+n0,a7,vl);
        n0+=(int)vl;
    }
}

/* Total vector iterations the microkernel will run over all column tiles. */
static long tile_iters(int cols,int NT,int vlmax)
{
    long it=0;
    for(int c=0;c<cols;c+=NT){ int n=cols-c; if(n>NT) n=NT; it+=(n+vlmax-1)/vlmax; }
    return it;
}

static void conv_unit(const Work *w,int unit,float *scratch,int *ohw)
{
    int NT=w->NT,MB=w->MB;
    int mb=unit%w->m_blocks,nt=unit/w->m_blocks;
    int col0=nt*NT; int ncol=w->cols-col0; if(ncol>NT) ncol=NT;
    int oc0=mb*MB; int nrow=w->OC-oc0; if(nrow>MB) nrow=MB;
    int npanel=(nrow+MR-1)/MR;

    float *Btile=scratch;
    float *Cacc=scratch+(size_t)w->KB*NT;

    if(!w->direct_b)
        for(int nn=0;nn<ncol;nn++){ int col=col0+nn; ohw[2*nn]=col/w->OW; ohw[2*nn+1]=col-ohw[2*nn]*w->OW; }

    for(int p=0;p<npanel;p++)
        for(int r=0;r<MR;r++){
            int oc=oc0+p*MR+r;
            float bv=(oc<w->OC)?w->bias[oc]:0.0f;
            float *row=Cacc+(size_t)(p*MR+r)*NT;
            for(int n0=0;n0<ncol;){ size_t vl=__riscv_vsetvl_e32m2((size_t)(ncol-n0));
                __riscv_vse32_v_f32m2(row+n0,__riscv_vfmv_v_f_f32m2(bv,vl),vl); n0+=(int)vl; }
        }

    for(int ic0=0;ic0<w->IC;ic0+=w->ic_chunk){
        int ic1=ic0+w->ic_chunk; if(ic1>w->IC) ic1=w->IC;
        int kblen=(ic1-ic0)*w->KH*w->KW;
        const float *B; int bstride;
        if(w->direct_b){
            /* 1x1 stride-1 conv (and the FC head): the im2col of a column tile
             * is literally a window of the source, so skip the copy entirely. */
            B=w->src+(size_t)ic0*w->cols+col0; bstride=w->cols;
        } else {
            im2col_block(w,ohw,ic0,ic1,ncol,Btile,NT);
            B=Btile; bstride=NT;
        }
        int kb=ic0*w->KH*w->KW;
        for(int p=0;p<npanel;p++){
            const float *ap=w->wt+((size_t)(oc0/MR+p)*w->K+kb)*MR;
            micro_mr8(ap,kblen,B,bstride,Cacc+(size_t)p*MR*NT,NT,ncol);
        }
    }

    int fuse=(w->se_squeeze==0);   /* SE needs a global reduction first */
    if(w->kind==OP_FC&&w->fc_layout){
        /* Transposed store into out[t][class]. Timestep-outer keeps the writes
         * contiguous (nrow consecutive classes per t); class-outer would touch a
         * fresh cache line on every single store, which at 6906 classes x 400
         * timesteps is millions of scattered lines. */
        for(int nn=0;nn<ncol;nn++){
            float *dr=w->dst+(size_t)(col0+nn)*w->nclass+oc0;
            const float *cc=Cacc+nn;
            for(int r=0;r<nrow;r++) dr[r]=cc[(size_t)r*NT];
        }
        return;
    }
    for(int r=0;r<nrow;r++){
        int oc=oc0+r;
        const float *cr=Cacc+(size_t)r*NT;
        float *dr=w->dst+(size_t)oc*w->cols+col0;
        const float *ar=(fuse&&w->add)?w->add+(size_t)oc*w->cols+col0:NULL;
        int n0=0;
        while(n0<ncol){
            size_t vl=__riscv_vsetvl_e32m2((size_t)(ncol-n0));
            vfloat32m2_t v=__riscv_vle32_v_f32m2(cr+n0,vl);
            if(ar) v=__riscv_vfadd_vv_f32m2(v,__riscv_vle32_v_f32m2(ar+n0,vl),vl);
            if(fuse&&w->act) v=act_apply(v,w->act,vl);
            __riscv_vse32_v_f32m2(dr+n0,v,vl);
            n0+=(int)vl;
        }
    }
}

/* Depthwise conv: each output channel touches only its own input channel, so
 * im2col+GEMM would be pure overhead. Direct convolution instead, with the
 * kernel taps held as loop-invariant scalars (which sidesteps the scalar-load /
 * vector-load interaction that limits the dense GEMM on this core). */
static void depthwise_channel(const Work *w,int c,float *pad)
{
    int PW=w->PW,IW=w->IW,IH=w->IH,PS=IW+2*PW;
    const float *s=w->src+(size_t)c*IH*IW;
    for(int ih=0;ih<IH;ih++){
        float *d=pad+(size_t)ih*PS;
        if(PW) memset(d,0,(size_t)PW*sizeof(float));
        memcpy(d+PW,s+(size_t)ih*IW,(size_t)IW*sizeof(float));
        if(PW) memset(d+PW+IW,0,(size_t)PW*sizeof(float));
    }
    const float *kern=w->wt+(size_t)c*w->KH*w->KW;
    float bv=w->bias[c];
    int fuse=(w->se_squeeze==0);
    float *dst=w->dst+(size_t)c*w->OH*w->OW;
    const float *add=(fuse&&w->add)?w->add+(size_t)c*w->OH*w->OW:NULL;
    for(int oh=0;oh<w->OH;oh++){
        float *dr=dst+(size_t)oh*w->OW;
        const float *ar=add?add+(size_t)oh*w->OW:NULL;
        int n0=0;
        while(n0<w->OW){
            size_t vl=__riscv_vsetvl_e32m2((size_t)(w->OW-n0));
            vfloat32m2_t acc=__riscv_vfmv_v_f_f32m2(bv,vl);
            for(int i=0;i<w->KH;i++){
                int ih=oh*w->SH-w->PH+i;
                if(ih<0||ih>=IH) continue;
                const float *pr=pad+(size_t)ih*PS;
                for(int j=0;j<w->KW;j++){
                    vfloat32m2_t vb=(w->SW==1)
                        ? __riscv_vle32_v_f32m2(pr+n0+j,vl)
                        : __riscv_vlse32_v_f32m2(pr+(size_t)n0*w->SW+j,(ptrdiff_t)w->SW*4,vl);
                    acc=__riscv_vfmacc_vf_f32m2(acc,kern[i*w->KW+j],vb,vl);
                }
            }
            if(ar) acc=__riscv_vfadd_vv_f32m2(acc,__riscv_vle32_v_f32m2(ar+n0,vl),vl);
            if(fuse&&w->act) acc=act_apply(acc,w->act,vl);
            __riscv_vse32_v_f32m2(dr+n0,acc,vl);
            n0+=(int)vl;
        }
    }
}

static void pool_channel(const Work *w,int c)
{
    const float *s=w->src+(size_t)c*w->IH*w->IW;
    float *d=w->dst+(size_t)c*w->OH*w->OW;
    for(int oh=0;oh<w->OH;oh++){
        int hs=oh*w->SH-w->PH,h0=hs<0?0:hs,h1=hs+w->KH; if(h1>w->IH) h1=w->IH;
        for(int ow=0;ow<w->OW;ow++){
            int ws=ow*w->SW-w->PW,w0=ws<0?0:ws,w1=ws+w->KW; if(w1>w->IW) w1=w->IW;
            if(w->kind==OP_MAXPOOL){
                float m=-INFINITY;
                for(int i=h0;i<h1;i++){ const float *r=s+(size_t)i*w->IW;
                    for(int j=w0;j<w1;j++) if(r[j]>m) m=r[j]; }
                d[(size_t)oh*w->OW+ow]=m;
            } else {
                float acc=0.0f;
                for(int i=h0;i<h1;i++){ const float *r=s+(size_t)i*w->IW;
                    for(int j=w0;j<w1;j++) acc+=r[j]; }
                int denom=w->cip?(w->KH*w->KW):((h1-h0)*(w1-w0));
                d[(size_t)oh*w->OW+ow]=acc/(float)denom;
            }
        }
    }
}

/* Phase 3: scale by the gate, then the residual add and activation that were
 * deferred because they must happen after the gate. */
static void se_apply_channel(const Work *w,int c)
{
    float g=G.se_gate[c];
    float *d=w->dst+(size_t)c*w->cols;
    const float *ar=w->add?w->add+(size_t)c*w->cols:NULL;
    int n=0;
    while(n<w->cols){
        size_t vl=__riscv_vsetvl_e32m2((size_t)(w->cols-n));
        vfloat32m2_t v=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(d+n,vl),g,vl);
        if(ar) v=__riscv_vfadd_vv_f32m2(v,__riscv_vle32_v_f32m2(ar+n,vl),vl);
        if(w->act) v=act_apply(v,w->act,vl);
        __riscv_vse32_v_f32m2(d+n,v,vl);
        n+=(int)vl;
    }
}

static void softmax_step(const Work *w,int t)
{
    float *row=w->dst+(size_t)t*w->nclass;
    int C=w->nclass;
    size_t vlmax=__riscv_vsetvlmax_e32m2();
    vfloat32m1_t one=__riscv_vfmv_v_f_f32m1(-INFINITY,1);
    vfloat32m2_t mx=__riscv_vfmv_v_f_f32m2(-INFINITY,vlmax);
    int n=0;
    while(n<C){ size_t vl=__riscv_vsetvl_e32m2((size_t)(C-n));
        mx=__riscv_vfmax_vv_f32m2(mx,__riscv_vle32_v_f32m2(row+n,vl),vl); n+=(int)vl; }
    float m=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(mx,one,vlmax));
    vfloat32m2_t sum=__riscv_vfmv_v_f_f32m2(0.0f,vlmax);
    n=0;
    while(n<C){ size_t vl=__riscv_vsetvl_e32m2((size_t)(C-n));
        vfloat32m2_t e=vexp_m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(row+n,vl),m,vl),vl);
        __riscv_vse32_v_f32m2(row+n,e,vl);
        sum=__riscv_vfadd_vv_f32m2(sum,e,vl); n+=(int)vl; }
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,1);
    float s=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(sum,z,vlmax));
    float inv=1.0f/s;
    n=0;
    while(n<C){ size_t vl=__riscv_vsetvl_e32m2((size_t)(C-n));
        __riscv_vse32_v_f32m2(row+n,__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(row+n,vl),inv,vl),vl);
        n+=(int)vl; }
}

static void worker_run(int tn)
{
    const Work *w=&g_work;
    switch(atomic_load_explicit(&g_phase,memory_order_acquire)){
    case PH_SE_APPLY:  for(int c=tn;c<w->OC;c+=G.nw) se_apply_channel(w,c);  return;
    case PH_SOFTMAX:   for(int t=tn;t<w->cols;t+=G.nw) softmax_step(w,t);    return;
    default: break;
    }
    if(w->kind==OP_MAXPOOL||w->kind==OP_AVGPOOL){
        for(int c=tn;c<w->IC;c+=G.nw) pool_channel(w,c);
    } else if(w->group>1){
        float *pad=g_scratch[tn]+GEMM_SCRATCH;
        for(int c=tn;c<w->OC;c+=G.nw) depthwise_channel(w,c,pad);
    } else {
        for(int u=tn;u<w->units;u+=G.nw) conv_unit(w,u,g_scratch[tn],g_colidx[tn]);
    }
}

/* Workers spin briefly and then sleep. A pure spin pool is right for a batch
 * process but wrong for a long-lived HTTP service: it would hold all eight AI
 * harts at 100% forever between requests. The spin window covers the sub-
 * microsecond gaps between the ~40 dispatches inside one inference, so the fast
 * path never touches the mutex; only the gap between requests reaches the wait.
 *
 * The spin also has to be cheap while it lasts: eight harts polling one cache
 * line saturate the coherence fabric and measurably starve whatever else is
 * running (the caller's own post-processing, in this service). Hence the PAUSE
 * hint and a spin window just long enough to bridge one inference's dispatches
 * rather than to outlast a whole request. */
#define SPIN_LIMIT 4000
/* Zihintpause PAUSE, spelled as its raw encoding because this assembler does
 * not accept the `fence w, 0` mnemonic it is defined as. */
static inline void cpu_relax(void){ __asm__ volatile(".insn i 0x0f, 0, x0, x0, 0x010" ::: "memory"); }
static pthread_mutex_t g_wake_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_wake_cv=PTHREAD_COND_INITIALIZER;

static void *worker_main(void *arg)
{
    int tn=(int)(intptr_t)arg; pin_once(tn);
    int last=0;
    for(;;){
        int gen;
        long spins=0;
        for(;;){
            gen=atomic_load_explicit(&g_gen,memory_order_acquire);
            if(gen!=last) break;
            if(++spins<SPIN_LIMIT){ cpu_relax(); continue; }
            pthread_mutex_lock(&g_wake_lock);
            while((gen=atomic_load_explicit(&g_gen,memory_order_acquire))==last)
                pthread_cond_wait(&g_wake_cv,&g_wake_lock);
            pthread_mutex_unlock(&g_wake_lock);
            break;
        }
        last=gen;
        if(gen<0) return NULL;
        __asm__ volatile("" ::: "memory");   /* reload everything under g_work */
        worker_run(tn);
        atomic_fetch_add_explicit(&g_done,1,memory_order_release);
    }
}

static void wake_workers(void)
{
    pthread_mutex_lock(&g_wake_lock);
    atomic_fetch_add_explicit(&g_gen,1,memory_order_release);
    pthread_cond_broadcast(&g_wake_cv);
    pthread_mutex_unlock(&g_wake_lock);
}

/* The calling thread MUST participate as worker 0, and therefore must be
 * AI-bound and pinned. This is not a performance choice: a thread that is not
 * AI-bound does not reliably observe stores made by AI-hart threads. Measured
 * directly -- with the caller demoted to a pure signaller, its read of a buffer
 * the workers had just filled returned stale zeros, and calling bind_ai() on it
 * without pinning did not help either; the same build with the workers on the
 * ordinary harts read back correct values. Keep the caller inside the pool.
 *
 * The consequence is that the caller's thread is pinned for life, so callers
 * that also do heavy pre/post-processing should drive this engine from a
 * dedicated thread -- see ocr_native.py, which does exactly that. */
static void dispatch(int phase)
{
    atomic_store_explicit(&g_phase,phase,memory_order_relaxed);
    atomic_store_explicit(&g_done,0,memory_order_relaxed);
    wake_workers();
    worker_run(0);
    while(atomic_load_explicit(&g_done,memory_order_acquire)<G.nw-1) cpu_relax();
    __asm__ volatile("" ::: "memory");   /* results the workers just wrote */
}

/* ===================== program execution ===================== */
static void se_gate_compute(const Work *w)
{
    int C=w->OC,S=w->se_squeeze;
    /* The global average pool is O(C*cols) -- 23k floats for the widest SE layer
     * at W=320 -- so the caller just does it. Handing it to the worker pool
     * bought nothing and added a cross-thread publication of se_sum. */
    for(int c=0;c<C;c++){
        const float *p=w->dst+(size_t)c*w->cols;
        double acc=0.0;
        for(int n=0;n<w->cols;n++) acc+=p[n];
        G.se_sum[c]=(float)acc;
    }
    const float *w1=w->se, *b1=w1+(size_t)S*C, *w2=b1+S, *b2=w2+(size_t)C*S;
    float inv=1.0f/(float)w->cols;
    float hidden[64];
    for(int s=0;s<S;s++){
        float acc=b1[s];
        const float *row=w1+(size_t)s*C;
        for(int c=0;c<C;c++) acc+=row[c]*(G.se_sum[c]*inv);
        hidden[s]=acc>0.0f?acc:0.0f;
    }
    for(int c=0;c<C;c++){
        float acc=b2[c];
        const float *row=w2+(size_t)c*S;
        for(int s=0;s<S;s++) acc+=row[s]*hidden[s];
        float g=acc*0.16666667f+0.5f;
        G.se_gate[c]=g<0.0f?0.0f:(g>1.0f?1.0f:g);
    }
}

static void run_program(void)
{
    for(int i=0;i<G.n_ops;i++){
        Work *w=&g_work;
        memset(w,0,sizeof(*w));
        Shape s=G.op_in[i],d=G.op_out[i];
        w->kind=op_i(i,F_OP);
        w->src=G.bufs[op_i(i,F_IN)];
        w->dst=G.bufs[op_i(i,F_OUT)];
        w->IC=s.c; w->IH=s.h; w->IW=s.w;
        w->KH=op_i(i,F_KH); w->KW=op_i(i,F_KW);
        w->SH=op_i(i,F_SH); w->SW=op_i(i,F_SW);
        w->PH=op_i(i,F_PH); w->PW=op_i(i,F_PW);
        w->act=op_i(i,F_ACT); w->ceil_mode=op_i(i,F_CEIL); w->cip=op_i(i,F_CIP);
        w->group=op_i(i,F_GROUP);
        w->wt=G.data+op_i(i,F_WOFF); w->bias=G.data+op_i(i,F_BOFF);
        w->se_squeeze=op_i(i,F_SESQ);
        w->se=w->se_squeeze?G.data+op_i(i,F_SEOFF):NULL;
        w->fc_layout=op_i(i,F_FCLAYOUT); w->fc_softmax=op_i(i,F_FCSOFTMAX);
        int a=op_i(i,F_ADD);
        w->add=a>=0?G.bufs[a]:NULL;

        double t0=g_prof?now_s():0;
        if(w->kind==OP_MAXPOOL||w->kind==OP_AVGPOOL){
            w->OC=d.c; w->OH=d.h; w->OW=d.w;
            dispatch(PH_MAIN);
        } else {
            if(w->kind==OP_FC){
                w->OC=op_i(i,F_OC); w->OH=1; w->OW=s.w;
                w->nclass=op_i(i,F_OC);
            } else {
                w->OC=d.c; w->OH=d.h; w->OW=d.w;
            }
            w->cols=w->OH*w->OW;
            w->K=w->group>1?w->KH*w->KW:w->IC*w->KH*w->KW;
            if(w->group>1){
                dispatch(PH_MAIN);
            } else {
                w->direct_b=(w->KH==1&&w->KW==1&&w->SH==1&&w->SW==1&&w->PH==0&&w->PW==0);
                w->ic_chunk=KB_TARGET/(w->KH*w->KW); if(w->ic_chunk<1) w->ic_chunk=1;
                if(w->ic_chunk>w->IC) w->ic_chunk=w->IC;
                w->KB=w->ic_chunk*w->KH*w->KW;
                /* Splitting the output both ways costs redundant work: each of
                 * the m row-blocks rebuilds im2col over every column it touches,
                 * and each of the n column tiles re-reads the whole weight
                 * matrix. Minimise m*cols + n*OC subject to m*n >= nharts. */
                int max_m=w->OC/MR; if(max_m<1) max_m=1;
                long best=-1; int bm=1,bn=1;
                for(int m=1;m<=G.nw&&m<=max_m;m++){
                    int n=(G.nw+m-1)/m;
                    if(n>w->cols) n=w->cols;
                    long cost=(long)m*w->cols+(long)n*w->OC;
                    if(best<0||cost<best){ best=cost; bm=m; bn=n; }
                }
                w->MB=((w->OC+bm-1)/bm+MR-1)/MR*MR;
                w->NT=(w->cols+bn-1)/bn;
                if(w->NT>NT_MAX) w->NT=NT_MAX;
                /* The accumulator must fit in cache, but give up output ROWS
                 * before giving up COLUMNS: columns are the vector dimension, so
                 * a narrow tile wastes vector lanes outright. The 6906-class FC
                 * head is the case that makes this matter -- one row block is
                 * huge and cols is only the timestep count. */
                while((long)w->MB*w->NT>CACC_MAX&&w->MB>MR)
                    w->MB=(w->MB/2+MR-1)/MR*MR;
                while((long)w->MB*w->NT>CACC_MAX&&w->NT>1) w->NT=(w->NT+1)/2;
                w->m_blocks=(w->OC+w->MB-1)/w->MB;
                /* A tile whose width is not a multiple of the vector length pays
                 * for a whole extra vector on its tail. Widening the tile to a
                 * VLMAX multiple only helps when it actually removes vector
                 * iterations -- otherwise it just inflates the im2col row stride
                 * and costs cache. At 180 columns it turns 4 three-quarter-full
                 * vectors into 3 full ones; at 100 it changes nothing, so the
                 * stride is left alone. */
                int vlmax=(int)__riscv_vsetvlmax_e32m2();
                if(w->NT>vlmax){
                    int rounded=(w->NT+vlmax-1)/vlmax*vlmax;
                    long tiles=(w->cols+rounded-1)/rounded;
                    if((long)w->MB*rounded<=CACC_MAX&&(long)w->m_blocks*tiles>=G.nw&&
                       tile_iters(w->cols,rounded,vlmax)<tile_iters(w->cols,w->NT,vlmax))
                        w->NT=rounded;
                }
                w->n_tiles=(w->cols+w->NT-1)/w->NT;
                w->units=w->m_blocks*w->n_tiles;
                dispatch(PH_MAIN);
            }
            if(w->se_squeeze){
                se_gate_compute(w);
                dispatch(PH_SE_APPLY);
            }
            if(w->fc_softmax){
                w->cols=d.h;            /* timesteps in the row-major output */
                dispatch(PH_SOFTMAX);
            }
        }
        if(g_prof) fprintf(stderr,"op%02d kind=%d grp=%d act=%d se=%d out=(%d,%d,%d) %.3f ms\n",
                           i,w->kind,w->group,w->act,w->se_squeeze,d.c,d.h,d.w,(now_s()-t0)*1e3);
        if(g_debug){
            const float *p=G.bufs[op_i(i,F_OUT)];
            size_t n=(size_t)d.c*d.h*d.w;
            double sum=0,asum=0; int bad=0; float mn=p[0],mx=p[0];
            for(size_t k=0;k<n;k++){ float v=p[k];
                if(isnan(v)||isinf(v)) bad++;
                else { sum+=v; asum+=fabs(v); if(v<mn)mn=v; if(v>mx)mx=v; } }
            fprintf(stderr,"op%02d kind=%d out=%d shape=(%d,%d,%d) sum=%.6f asum=%.6f min=%.6f max=%.6f bad=%d\n",
                    i,op_i(i,F_OP),op_i(i,F_OUT),d.c,d.h,d.w,sum,asum,mn,mx,bad);
        }
    }
}

/* ===================== public API ===================== */
int ocr_load(const char *path,int nthreads)
{
    if(G.loaded) return 0;
    const char *env=getenv("OCR_HARTS");
    if(env) nthreads=atoi(env);
    if(nthreads<1) nthreads=8;
    if(nthreads>MAXNT) nthreads=MAXNT;
    if(getenv("OCR_NO_AI_HARTS")&&atoi(getenv("OCR_NO_AI_HARTS"))) g_use_ai_harts=0;
    if(getenv("OCR_DEBUG")) g_debug=atoi(getenv("OCR_DEBUG"));
    if(getenv("OCR_PROF")) g_prof=atoi(getenv("OCR_PROF"));

    int fd=open(path,O_RDONLY);
    if(fd<0) return -1;
    struct stat st;
    if(fstat(fd,&st)!=0){ close(fd); return -2; }
    unsigned char *m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if(m==MAP_FAILED) return -3;
    if(memcmp(m,"OCR1",4)!=0){ munmap(m,(size_t)st.st_size); return -4; }
    const int32_t *h=(const int32_t*)(m+4);
    if(h[0]!=3){ munmap(m,(size_t)st.st_size); return -5; }
    G.map=m; G.map_bytes=(size_t)st.st_size;
    G.n_ops=h[1]; G.n_bufs=h[2]; G.final_buf=h[3];
    G.ops=(const int32_t*)(m+h[4]); G.data=(const float*)(m+h[5]);
    G.n_classes=h[7]; G.mr=h[8]; G.input_h=h[9];
    if(G.mr!=MR){ munmap(m,G.map_bytes); return -6; }

    G.bufs=calloc((size_t)G.n_bufs,sizeof(float*));
    G.buf_elems=calloc((size_t)G.n_bufs,sizeof(size_t));
    G.buf_alloc=calloc((size_t)G.n_bufs,sizeof(size_t));
    G.shape=calloc((size_t)G.n_bufs,sizeof(Shape));
    G.op_in=calloc((size_t)G.n_ops,sizeof(Shape));
    G.op_out=calloc((size_t)G.n_ops,sizeof(Shape));
    int maxc=1;
    for(int i=0;i<G.n_ops;i++) if(op_i(i,F_OC)>maxc) maxc=op_i(i,F_OC);
    G.se_sum=calloc((size_t)maxc,sizeof(float));
    G.se_gate=calloc((size_t)maxc,sizeof(float));
    G.prepared_w=-1; G.nw=nthreads;
    pthread_mutex_init(&G.lock,NULL);
    for(int i=0;i<G.nw;i++) g_colidx[i]=malloc(2*NT_MAX*sizeof(int));
    pin_once(0);
    for(int i=1;i<G.nw;i++) pthread_create(&g_threads[i],NULL,worker_main,(void*)(intptr_t)i);
    G.loaded=1;
    return 0;
}

static size_t g_scratch_floats=0;

static int prepare(int W)
{
    if(G.prepared_w==W) return 0;
    int rc=shape_pass(W);
    if(rc!=0) return rc;
    for(int i=0;i<G.n_bufs;i++){
        if(G.buf_alloc[i]>=G.buf_elems[i]&&G.bufs[i]) continue;
        free(G.bufs[i]);
        if(posix_memalign((void**)&G.bufs[i],128,G.buf_elems[i]*sizeof(float))!=0) return -8;
        G.buf_alloc[i]=G.buf_elems[i];
    }
    size_t need=GEMM_SCRATCH+G.pad_floats+64;
    if(need>g_scratch_floats){
        for(int i=0;i<G.nw;i++){
            free(g_scratch[i]);
            if(posix_memalign((void**)&g_scratch[i],128,need*sizeof(float))!=0) return -9;
            memset(g_scratch[i],0,need*sizeof(float));
        }
        g_scratch_floats=need;
    }
    G.prepared_w=W;
    return 0;
}

int ocr_out_steps(int W)
{
    if(!G.loaded||W<8) return -1;
    pthread_mutex_lock(&G.lock);
    pin_once(0);   /* caller is worker 0; see dispatch() for why this is required */
    int rc=prepare(W);
    int steps=rc==0?G.out_steps:rc;
    pthread_mutex_unlock(&G.lock);
    return steps;
}

int ocr_input_height(void){ return G.loaded?G.input_h:-1; }
int ocr_n_classes(void){ return G.loaded?G.n_classes:-1; }
int ocr_harts(void){ return G.nw; }

int ocr_infer(const float *input,int W,float *out)
{
    if(!G.loaded) return -1;
    pthread_mutex_lock(&G.lock);
    pin_once(0);   /* caller is worker 0; see dispatch() for why this is required */
    int rc=prepare(W);
    if(rc!=0){ pthread_mutex_unlock(&G.lock); return rc; }
    memcpy(G.bufs[op_i(0,F_IN)],input,(size_t)3*G.input_h*W*sizeof(float));
    run_program();
    memcpy(out,G.bufs[G.final_buf],(size_t)G.out_steps*G.n_classes*sizeof(float));
    int steps=G.out_steps;
    pthread_mutex_unlock(&G.lock);
    return steps;
}

void ocr_shutdown(void)
{
    if(!G.loaded) return;
    pthread_mutex_lock(&g_wake_lock);
    atomic_store_explicit(&g_gen,-1,memory_order_release);
    pthread_cond_broadcast(&g_wake_cv);
    pthread_mutex_unlock(&g_wake_lock);
    for(int i=1;i<G.nw;i++) pthread_join(g_threads[i],NULL);
    for(int i=0;i<G.n_bufs;i++) free(G.bufs[i]);
    free(G.bufs); free(G.buf_elems); free(G.buf_alloc);
    free(G.shape); free(G.op_in); free(G.op_out);
    free(G.se_sum); free(G.se_gate);
    for(int i=0;i<G.nw;i++){ free(g_scratch[i]); free(g_colidx[i]); }
    munmap(G.map,G.map_bytes);
    memset(&G,0,sizeof(G));
    g_scratch_floats=0;
}

#ifdef OCR_MAIN
int main(int argc,char **argv)
{
    if(argc<4){ fprintf(stderr,"usage: %s <blob> <in.f32> <W> [out.f32] [iters]\n",argv[0]); return 2; }
    int W=atoi(argv[3]);
    const char *outpath=argc>4?argv[4]:NULL;
    int iters=argc>5?atoi(argv[5]):1;

    int rc=ocr_load(argv[1],0);
    if(rc!=0){ fprintf(stderr,"load failed rc=%d\n",rc); return 1; }
    int H=ocr_input_height(), NC=ocr_n_classes();

    FILE *f=fopen(argv[2],"rb");
    if(!f){ perror("open input"); return 1; }
    fseek(f,0,SEEK_END); long bytes=ftell(f); fseek(f,0,SEEK_SET);
    size_t per=(size_t)3*H*W;
    int n=(int)((size_t)bytes/(per*sizeof(float)));
    if(n<1){ fprintf(stderr,"input holds no complete (3,%d,%d) crop\n",H,W); return 1; }
    float *in=malloc((size_t)n*per*sizeof(float));
    if(fread(in,sizeof(float),(size_t)n*per,f)!=(size_t)n*per){ fprintf(stderr,"short read\n"); return 1; }
    fclose(f);

    int steps=ocr_out_steps(W);
    fprintf(stderr,"harts=%d crops=%d H=%d W=%d steps=%d classes=%d\n",ocr_harts(),n,H,W,steps,NC);
    float *out=malloc((size_t)n*steps*NC*sizeof(float));
    for(int i=0;i<n;i++) ocr_infer(in+(size_t)i*per,W,out+(size_t)i*steps*NC);

    double best=1e30,total=0;
    for(int it=0;it<iters;it++){
        double t0=now_s();
        for(int i=0;i<n;i++) ocr_infer(in+(size_t)i*per,W,out+(size_t)i*steps*NC);
        double dt=now_s()-t0;
        total+=dt; if(dt<best) best=dt;
    }
    fprintf(stderr,"iters=%d best=%.1f ms/batch %.3f ms/crop | mean=%.1f ms/batch %.3f ms/crop\n",
            iters,best*1e3,best*1e3/n,total/iters*1e3,total/iters*1e3/n);
    if(outpath){ FILE *o=fopen(outpath,"wb"); fwrite(out,sizeof(float),(size_t)n*steps*NC,o); fclose(o); }
    for(int i=0;i<n&&i<4;i++){
        const float *p=out+(size_t)i*steps*NC;
        printf("crop %d argmax:",i);
        for(int t=0;t<steps;t++){
            int bj=0; float bv=-1;
            for(int j=0;j<NC;j++) if(p[(size_t)t*NC+j]>bv){ bv=p[(size_t)t*NC+j]; bj=j; }
            printf(" %d",bj);
        }
        printf("\n");
    }
    ocr_shutdown();
    return 0;
}
#endif
