/* qwen_moe_hp.c — Qwen3-30B-A3B MoE decode using the REAL vendor-shaped IME-2 kernel
 * (gemm_kernel_i8i4_hp_m1, ported+validated in bench/vendor_ime_a2_full.c — max rel diff 2.3%
 * vs an independent dequant oracle, consistent with expected fp16-accumulation noise).
 *
 * This is qwen_moe.c with ONLY the low-level GEMV layer swapped: same GGUF reader, same model
 * struct/forward()/attention/MoE routing, same P0.2 activation-reuse structure. The weight
 * format changes (vendor N32-panel/K256-superblock int4 + fp16 two-level scale, not our old
 * q4-in-q8 interleave) so the requant cache is a new, incompatible format (bumped IMEC ver=2,
 * separate cache path) -- this is the "behind a feature flag" A/B: a separate binary/cache next
 * to the original qwen_moe, not a runtime toggle, matching the research_feed_paths.md Path A
 * probe plan (A5: full-token substitution).
 *
 * All of this model's Lin shapes happen to be exact multiples of 256 (K) / 32 (N) -- checked
 * against qwen3moe d=2048 qd=4096 kvd=512 moe_ffn=768 vocab=151936 -- so no remainder/padding
 * handling is needed anywhere.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe_hp qwen_moe_hp.c -lm
 * Run  : LD_LIBRARY_PATH=/usr/lib ./qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf [ngen] [nt]
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
#include <omp.h>
#include <pthread.h>
#include <stdatomic.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static __thread int g_pinned=0;
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(8+(tn*2)%8,&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

/* ===================== fp16 helpers ===================== */
static uint16_t f32_to_f16(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000;
    int32_t  exp=((x>>23)&0xff)-127+15;
    uint32_t mant=x&0x7fffff;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; mant|=0x800000; uint32_t shift=14-exp; return (uint16_t)(sign|(mant>>shift)); }
    if(exp>=31) return (uint16_t)(sign|0x7c00);
    return (uint16_t)(sign|((uint32_t)exp<<10)|(mant>>13));
}
static float f16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1f, mant=h&0x3ff, bits;
    if(exp==0){ if(mant==0) bits=sign; else { int e=-1; do{e++;mant<<=1;}while(!(mant&0x400)); mant&=0x3ff; bits=sign|((uint32_t)(127-15-e)<<23)|(mant<<13);} }
    else if(exp==31) bits=sign|0x7f800000|(mant<<13);
    else bits=sign|((exp-15+127)<<23)|(mant<<13);
    float f; memcpy(&f,&bits,4); return f;
}

/* ===================== vendor IME-2 HP int4 GEMV — verified (bench/vendor_ime_a2_full.c) =====================
 * A record per 256-wide block (290B): 8x[2B fp16 subblk-scale][32B int8 data](272B) +
 * 8x fp16 asum PRE-SCALED as -true_asum*8.0 (16B) + 1x fp16 block-avg scale (2B).
 * B record (block_q4_0x32, 576B/K32-group/N32-panel): 64B fp16 scale + 512B int4 data, adjacent-
 * pair nibbling {2j,2j+1}, nibble=signed+8. Ground truth: reference/spacemit-backend/
 * rvv_kernels.cpp:1989 (quantize_a_row_i8_hp) + repack.cpp:292 (make_block_q4_0x32) +
 * ime2_kernels.cpp:2883 (gemm_kernel_i8i4_hp_m1), traced via ime.cpp's dispatch. */
#define NSUB 8      /* 256/32 subblocks per A/B superblock */
#define AREC 290    /* bytes per 256-wide A record */
#define BREC 576    /* bytes per 32-wide-K x 32-wide-N B record */
#define BSUPER (NSUB*BREC) /* 4608: bytes per 256-wide-K x 32-wide-N B superblock */

typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_native;

static void quantize_q4_0_native(const float*w /*32 elems*/, q4_0_native*out){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(w[i]); if(v>amax)amax=v; }
    float d=amax/8.0f; float inv = d? 1.0f/d : 0.0f;
    int8_t q[32];
    for(int i=0;i<32;i++){ int v=(int)lrintf(w[i]*inv); if(v>7)v=7; if(v<-8)v=-8; q[i]=(int8_t)v; }
    out->d=f32_to_f16(d);
    for(int l=0;l<16;l++){ uint8_t lo=(uint8_t)(q[l]+8), hi=(uint8_t)(q[l+16]+8); out->qs[l]=(lo&0xf)|((hi&0xf)<<4); }
}
static void pack_B_q4_0x32(q4_0_native rows[32], uint8_t*out /* 576 bytes */){
    for(int i=0;i<32;i++) memcpy(out+i*2,&rows[i].d,2);
    uint8_t*qs=out+64;
    for(int i=0;i<32;i++){
        for(int j=0;j<8;j++) qs[i*16+j]     = (rows[i].qs[j*2]&0x0F) | ((rows[i].qs[j*2+1]&0x0F)<<4);
        for(int j=0;j<8;j++) qs[i*16+8+j]   = ((rows[i].qs[j*2]&0xF0)>>4) | (rows[i].qs[j*2+1]&0xF0);
    }
}
/* activation pack: portable scalar port of quantize_a_row_i8_hp (rvv_kernels.cpp:1989, vlenb==128) */
static void pack_A_hp(const float*a, uint8_t*out /* 290 bytes */){
    float scale_temp[NSUB]; int8_t qa[NSUB][32]; int16_t asum[NSUB];
    float scale_avg=0;
    for(int kk=0;kk<NSUB;kk++){
        float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(a[kk*32+i]); if(v>amax)amax=v; }
        scale_temp[kk]=amax/127.0f; scale_avg+=scale_temp[kk];
    }
    scale_avg/=NSUB;
    float scale_factor = scale_avg? 1.0f/scale_avg : 0.0f;
    for(int kk=0;kk<NSUB;kk++){
        float rep=scale_temp[kk]? 1.0f/scale_temp[kk] : 0.0f;
        int32_t s=0;
        for(int i=0;i<32;i++){ int q=(int)lrintf(a[kk*32+i]*rep); if(q>127)q=127; if(q<-128)q=-128; qa[kk][i]=(int8_t)q; s+=q; }
        asum[kk]=(int16_t)s;
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor);
        memcpy(out+kk*34,&ssub,2); memcpy(out+kk*34+2,qa[kk],32);
    }
    for(int kk=0;kk<NSUB;kk++){ uint16_t as=f32_to_f16(-(float)asum[kk]*8.0f); memcpy(out+272+kk*2,&as,2); }
    uint16_t blkscale=f32_to_f16(scale_avg);
    memcpy(out+288,&blkscale,2);
}
/* verbatim asm port of gemm_kernel_i8i4_hp_m1, generalized to k_blks superblocks */
static void run_hp_m1(const uint8_t*a_data, const uint8_t*b_data, float*dst_c, long k_blks){
    __asm__ volatile(
        "vsetvli        t0, x0, e16, m1         \n\t"
        "vxor.vv        v31, v31, v31           \n\t"
        "mv             t4, %[BK]               \n\t"
        "li             t0, 0x4c00              \n\t"
        "fmv.h.x        fa0, t0                 \n\t"
        ".align 4                               \n\t"
        "BLK_LOOP%=:                            \n\t"
        "li             t5, 8                   \n\t"
        "addi           t6, %[A], 288           \n\t"
        "flh            ft1, (t6)               \n\t"
        "addi           t6, %[A], 272           \n\t"
        "vsetvli        t0, x0, e16, m1         \n\t"
        "vxor.vv        v16, v18, v18           \n\t"
        "vxor.vv        v17, v18, v18           \n\t"
        "vxor.vv        v18, v18, v18           \n\t"
        "vxor.vv        v19, v18, v18           \n\t"
        "INNER_BLK_LOOP%=:                      \n\t"
        "flh            fa1, (t6)               \n\t"
        "addi           t6, t6, 2               \n\t"
        "flh            ft0, (%[A])             \n\t"
        "addi           %[A], %[A], 2           \n\t"
        "vsetvli        t0, x0, e8, mf4         \n\t"
        "vle8.v         v3, (%[A])              \n\t"
        "addi           %[A], %[A], 32          \n\t"
        "vsetvli        t0, x0, e16, mf2        \n\t"
        "vle16.v        v8, (%[B])              \n\t"
        "addi           %[B], %[B], 64          \n\t"
        "vl4r.v         v4, (%[B])              \n\t"
        "addi           %[B], %[B], 512         \n\t"
        "vfmul.vf       v8, v8, ft0             \n\t"
        "vfmul.vf       v9, v8, fa0             \n\t"
        "vfmul.vf       v10, v8, fa1            \n\t"
        "vfwmacc.vf     v31, ft1, v10           \n\t"
        "vsetvli        t0, x0, e8, m1          \n\t"
        "vpack.vv       v0, v8, v9, 3           \n\t"
        "vsrl.vi        v28, v3, 4              \n\t"
        "vsetvli        t0, x0, e16, m1         \n\t"
        "vnpack4.vv     v2, v3, v3, 3           \n\t"
        "vnpack4.vv     v3, v28, v28, 3         \n\t"
        "vsetvli        t0, x0, e16, m1         \n\t"
        "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"
        "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
        "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
        "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
        "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"
        "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
        "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
        "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"
        "addi           t5, t5, -1              \n\t"
        "bgtz           t5, INNER_BLK_LOOP%=    \n\t"
        "vpack.vv       v8, v16, v17, 1         \n\t"
        "vpack.vv       v12, v18, v19, 1        \n\t"
        "vpack.vv       v20, v8, v12, 2         \n\t"
        "vsetvli        t0, x0, e16, mf2        \n\t"
        "addi           t4, t4, -1              \n\t"
        "vfwmacc.vf     v31, ft1, v20           \n\t"
        "addi           %[A], t6, 2             \n\t"
        "bgtz           t4, BLK_LOOP%=          \n\t"
        "vsetvli        t0, x0, e32, m1         \n\t"
        "vse32.v        v31, (%[DST])           \n\t"
        : [A] "+r"(a_data), [B] "+r"(b_data)
        : [DST] "r"(dst_c), [BK] "r"(k_blks)
        : "t0","t1","t2","t3","t4","t5","t6","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
          "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23",
          "v24","v25","v26","v27","v28","v29","v30","v31","fa0","fa1","ft0","ft1");
}

typedef struct { int N,K; uint8_t*B; } Lin; /* B: (N/32)*(K/256)*BSUPER bytes */

static Lin lin_new_hp(const float*wf32,int N,int K){
    Lin l; l.N=N; l.K=K; int Np=N/32, Sb=K/256;
    l.B=malloc((size_t)Np*Sb*BSUPER);
    for(int np=0;np<Np;np++){
        for(int sb=0;sb<Sb;sb++){
            for(int kk=0;kk<NSUB;kk++){
                q4_0_native rows[32];
                for(int r=0;r<32;r++){
                    int row=np*32+r;
                    quantize_q4_0_native(wf32+(size_t)row*K + sb*256+kk*32, &rows[r]);
                }
                pack_B_q4_0x32(rows, l.B + ((size_t)np*Sb+sb)*BSUPER + (size_t)kk*BREC);
            }
        }
    }
    return l;
}
static void pack_act_hp(const float*x,int K,uint8_t*Abuf){
    int Sb=K/256; for(int sb=0;sb<Sb;sb++) pack_A_hp(x+sb*256, Abuf+(size_t)sb*AREC);
}
static double gT_actpack=0, gT_lin=0, gT_attn=0, gT_rest=0; static long gT_tok=0; static int gT_on=0;

/* PR8 (codex_recs_1.md §17/§22.3): with the vendor kernel at ~446ns/call, the ~1392 fresh
 * #pragma omp parallel spawns/token that lin_mm_hp used to do dominated wall-clock (100.3ms,
 * 62%). Replace with a persistent spin-dispatch pool: threads created once, wait on a generation
 * counter instead of libgomp fork/join. Same round-robin panel partitioning as before (np=tn;
 * np<Np; np+=nt), so the actual math is unchanged -- only the dispatch mechanism differs. */
#define MAXNT 16
typedef struct { const Lin*l; const uint8_t*Abuf; float*y; int kb; } HpWork;
static _Atomic int g_pool_gen=0, g_pool_done=0;
static HpWork g_pool_work;
static int g_pool_nt=0;
static pthread_t g_pool_threads[MAXNT];

static void lin_mm_hp_worker_run(int tn){
    int Np=g_pool_work.l->N/32;
    for(int np=tn; np<Np; np+=g_pool_nt)
        run_hp_m1(g_pool_work.Abuf, g_pool_work.l->B+(size_t)np*g_pool_work.kb*BSUPER, g_pool_work.y+np*32, g_pool_work.kb);
}
static void* lin_mm_hp_worker(void*arg){
    int tn=(int)(intptr_t)arg; pin_once(tn);
    int last=0;
    for(;;){
        int gen;
        while((gen=atomic_load_explicit(&g_pool_gen,memory_order_acquire))==last) { /* spin */ }
        last=gen;
        if(gen<0) return NULL; /* shutdown sentinel */
        lin_mm_hp_worker_run(tn);
        atomic_fetch_add_explicit(&g_pool_done,1,memory_order_release);
    }
}
static void lin_mm_pool_init(int nt){
    g_pool_nt=nt; pin_once(0);
    for(int i=1;i<nt;i++) pthread_create(&g_pool_threads[i],NULL,lin_mm_hp_worker,(void*)(intptr_t)i);
}
static void lin_mm_hp(const Lin*l,const uint8_t*Abuf,float*y,int nt){
    g_pool_work.l=l; g_pool_work.Abuf=Abuf; g_pool_work.y=y; g_pool_work.kb=l->K/256;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release); /* wake workers 1..nt-1 */
    lin_mm_hp_worker_run(0); /* main thread does its own share (tn=0) */
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < nt-1) { /* spin */ }
}
static void lin_mm(const Lin*l,const float*x,float*y,int nt,uint8_t*Abuf){
    double _ta=gT_on?now():0;
    pack_act_hp(x,l->K,Abuf);
    double _tb=gT_on?now():0; if(gT_on) gT_actpack+=_tb-_ta;
    lin_mm_hp(l,Abuf,y,nt);
    if(gT_on) gT_lin+=now()-_tb;
}

/* ===================== GGUF reader (mmap; Q4_0/Q4_1/Q8_0/F32/F16) — unchanged from qwen_moe.c ===================== */
typedef struct { uint32_t typ; int nd; uint64_t dims[4]; uint64_t off; char name[64]; } TInfo;
typedef struct { unsigned char*p; size_t fsz; uint64_t data_start; TInfo*t; int nt; char**tok; int ntok;
    int block_count, embd, ffn, nh, nkv, hd, vocab, n_exp, n_act, moe_ffn; float rope_base; } Gguf;
static uint64_t U64(unsigned char*b){ uint64_t x; memcpy(&x,b,8); return x; }
static uint32_t U32(unsigned char*b){ uint32_t x; memcpy(&x,b,4); return x; }
static float f16f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,b;
    if(e==0){ if(m==0)b=s<<31; else{ int ee=-14; while(!(m&0x400)){m<<=1;ee--;} m&=0x3ff; b=(s<<31)|((uint32_t)(ee+127)<<23)|(m<<13);} }
    else if(e==0x1f) b=(s<<31)|(0xff<<23)|(m<<13); else b=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&b,4); return f; }
static size_t skipval(unsigned char*p,uint32_t t){
    switch(t){ case 0:case 1:case 7: return 1; case 2:case 3: return 2; case 4:case 5:case 6: return 4;
        case 10:case 11:case 12: return 8; case 8: return 8+U64(p);
        case 9:{ uint32_t et=U32(p); uint64_t n=U64(p+4); size_t o=12; for(uint64_t i=0;i<n;i++) o+=skipval(p+o,et); return o; }
        default: return 4; } }
static void gguf_open(Gguf*g,const char*path){
    int fd=open(path,O_RDONLY); if(fd<0){perror("open");exit(1);} struct stat st; fstat(fd,&st); g->fsz=st.st_size;
    g->p=mmap(NULL,g->fsz,PROT_READ,MAP_PRIVATE,fd,0); if(g->p==MAP_FAILED){perror("mmap");exit(1);} close(fd);
    madvise(g->p,g->fsz,MADV_SEQUENTIAL);
    unsigned char*p=g->p; size_t o=8; uint64_t ntensor=U64(p+o); o+=8; uint64_t nkv=U64(p+o); o+=8;
    g->rope_base=1e6f; g->tok=NULL; g->ntok=0; g->n_act=8; g->n_exp=128; g->moe_ffn=768;
    for(uint64_t i=0;i<nkv;i++){ uint64_t kl=U64(p+o); char key[128]; int c=kl<127?kl:127; memcpy(key,p+o+8,c); key[c]=0; o+=8+kl;
        uint32_t t=U32(p+o); o+=4;
        #define KEYIS(s) (strcmp(key,s)==0)
        if(t==9 && KEYIS("tokenizer.ggml.tokens")){ uint32_t et=U32(p+o); uint64_t n=U64(p+o+4); size_t oo=o+12;
            g->ntok=n; g->tok=malloc(n*sizeof(char*));
            for(uint64_t j=0;j<n;j++){ uint64_t l=U64(p+oo); char*s=malloc(l+1); memcpy(s,p+oo+8,l); s[l]=0; g->tok[j]=s; oo+=8+l; }
            o=oo; continue; }
        if(t==4||t==5){ uint32_t v=U32(p+o);
            if(KEYIS("qwen3moe.block_count"))g->block_count=v; else if(KEYIS("qwen3moe.embedding_length"))g->embd=v;
            else if(KEYIS("qwen3moe.feed_forward_length"))g->ffn=v; else if(KEYIS("qwen3moe.attention.head_count"))g->nh=v;
            else if(KEYIS("qwen3moe.attention.head_count_kv"))g->nkv=v; else if(KEYIS("qwen3moe.attention.key_length"))g->hd=v;
            else if(KEYIS("qwen3moe.expert_count"))g->n_exp=v; else if(KEYIS("qwen3moe.expert_used_count"))g->n_act=v;
            else if(KEYIS("qwen3moe.expert_feed_forward_length"))g->moe_ffn=v; }
        else if(t==6){ float v; uint32_t x=U32(p+o); memcpy(&v,&x,4); if(KEYIS("qwen3moe.rope.freq_base"))g->rope_base=v; }
        o+=skipval(p+o,t);
    }
    g->t=malloc(ntensor*sizeof(TInfo)); g->nt=ntensor;
    for(uint64_t i=0;i<ntensor;i++){ uint64_t nl=U64(p+o); TInfo*ti=&g->t[i]; int c=nl<63?nl:63; memcpy(ti->name,p+o+8,c); ti->name[c]=0; o+=8+nl;
        ti->nd=U32(p+o); o+=4; for(int d=0;d<ti->nd&&d<4;d++){ ti->dims[d]=U64(p+o); o+=8; }
        ti->typ=U32(p+o); o+=4; ti->off=U64(p+o); o+=8; }
    uint64_t align=32; g->data_start=(o+align-1)/align*align;
    g->vocab=0; for(int i=0;i<g->nt;i++) if(strcmp(g->t[i].name,"token_embd.weight")==0) g->vocab=g->t[i].dims[1];
}
static TInfo* gguf_find(Gguf*g,const char*name){ for(int i=0;i<g->nt;i++) if(strcmp(g->t[i].name,name)==0) return &g->t[i]; printf("missing %s\n",name); exit(1); }
static int gguf_has(Gguf*g,const char*name){ for(int i=0;i<g->nt;i++) if(strcmp(g->t[i].name,name)==0) return 1; return 0; }
static void gguf_dequant_into(Gguf*g,TInfo*ti,size_t elem0,size_t n,float*out){
    unsigned char*base=g->p+g->data_start+ti->off;
    if(ti->typ==0){ float*b=(float*)base+elem0; for(size_t i=0;i<n;i++)out[i]=b[i]; return; }
    if(ti->typ==1){ uint16_t*b=(uint16_t*)base+elem0; for(size_t i=0;i<n;i++)out[i]=f16f(b[i]); return; }
    size_t bl0=elem0/32, nb=(n+31)/32;
    if(ti->typ==8){ for(size_t bl=0;bl<nb;bl++){ unsigned char*q=base+(bl0+bl)*34; float sc=f16f(*(uint16_t*)q);
        for(int j=0;j<32 && bl*32+j<n;j++) out[bl*32+j]=(float)(int8_t)q[2+j]*sc; } return; }
    if(ti->typ==2){ for(size_t bl=0;bl<nb;bl++){ unsigned char*q=base+(bl0+bl)*18; float d=f16f(*(uint16_t*)q); unsigned char*qs=q+2;
        for(int j=0;j<32 && bl*32+j<n;j++){ int nib=(j<16)?(qs[j]&0xf):(qs[j-16]>>4); out[bl*32+j]=(nib-8)*d; } } return; }
    if(ti->typ==3){ for(size_t bl=0;bl<nb;bl++){ unsigned char*q=base+(bl0+bl)*20; float d=f16f(*(uint16_t*)q),m=f16f(*(uint16_t*)(q+2)); unsigned char*qs=q+4;
        for(int j=0;j<32 && bl*32+j<n;j++){ int nib=(j<16)?(qs[j]&0xf):(qs[j-16]>>4); out[bl*32+j]=nib*d+m; } } return; }
    if(ti->typ==14){ size_t nbk=(n+255)/256;
        for(size_t sb=0;sb<nbk;sb++){ unsigned char*blk=base+sb*210; unsigned char*ql=blk,*qh=blk+128; int8_t*sc=(int8_t*)(blk+192);
            float d=f16f(*(uint16_t*)(blk+208)); float*y=out+sb*256;
            for(int h=0;h<2;h++){ unsigned char*qlh=ql+h*64,*qhh=qh+h*32; int8_t*sch=sc+h*8; float*yh=y+h*128;
                for(int l=0;l<32;l++){ int is=l/16;
                    int q1=((qlh[l]&0xF)|(((qhh[l]>>0)&3)<<4))-32, q2=((qlh[l+32]&0xF)|(((qhh[l]>>2)&3)<<4))-32;
                    int q3=((qlh[l]>>4)|(((qhh[l]>>4)&3)<<4))-32, q4=((qlh[l+32]>>4)|(((qhh[l]>>6)&3)<<4))-32;
                    yh[l]=d*sch[is]*q1; yh[l+32]=d*sch[is+2]*q2; yh[l+64]=d*sch[is+4]*q3; yh[l+96]=d*sch[is+6]*q4; } } }
        return; }
    printf("dequant type %u unsupported (%s)\n",ti->typ,ti->name); exit(1);
}
static float* gguf_dequant(Gguf*g,const char*name){ TInfo*ti=gguf_find(g,name); size_t total=1; for(int d=0;d<ti->nd;d++)total*=ti->dims[d];
    float*out=malloc(total*4); gguf_dequant_into(g,ti,0,total,out); return out; }

/* ===================== model ===================== */
typedef struct { float*attn_norm,*ffn_norm,*q_norm,*k_norm,*router; Lin q,k,v,o; Lin*eg,*eu,*ed; } Layer;
typedef struct { int d,nl,nh,nkv,hd,vocab,nt,n_exp,n_act,moe; float rope_base,eps; float*tok_embd,*out_norm; Layer*L; Lin lm; } Model;

static void rmsnorm(float*o,const float*x,const float*w,int n,float eps){ float s=0; for(int i=0;i<n;i++)s+=x[i]*x[i]; s=1.0f/sqrtf(s/n+eps); for(int i=0;i<n;i++)o[i]=x[i]*s*w[i]; }
static void rope(float*v,int hd,int pos,float base){ for(int i=0;i<hd/2;i++){ float fr=powf(base,-2.0f*i/hd),a=pos*fr,c=cosf(a),s=sinf(a),x=v[i],y=v[i+hd/2]; v[i]=x*c-y*s; v[i+hd/2]=x*s+y*c; } }
static void softmax(float*x,int n){ float m=-1e30f; for(int i=0;i<n;i++)if(x[i]>m)m=x[i]; float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} float inv=1.0f/s; for(int i=0;i<n;i++)x[i]*=inv; }

static void model_load(Model*m,Gguf*g,int nt){
    m->d=g->embd; m->nl=g->block_count; m->nh=g->nh; m->nkv=g->nkv; m->hd=g->hd; m->vocab=g->vocab;
    m->rope_base=g->rope_base; m->eps=1e-6f; m->nt=nt; m->n_exp=g->n_exp; m->n_act=g->n_act; m->moe=g->moe_ffn;
    int qd=m->nh*m->hd, kvd=m->nkv*m->hd, d=m->d, moe=m->moe, ne=m->n_exp;
    if(d%256||qd%256||kvd%256||moe%256){ fprintf(stderr,"model dims not multiples of 256 -- HP kernel needs remainder handling not implemented here\n"); exit(1); }
    fprintf(stderr,"packing token_embd + lm_head (vendor int4) ... "); fflush(stderr);
    m->tok_embd=gguf_dequant(g,"token_embd.weight"); m->out_norm=gguf_dequant(g,"output_norm.weight");
    if(gguf_has(g,"output.weight")){ float*ow=gguf_dequant(g,"output.weight"); m->lm=lin_new_hp(ow,m->vocab,d); free(ow);
        fprintf(stderr,"(untied lm_head from output.weight) "); }
    else m->lm=lin_new_hp(m->tok_embd,m->vocab,d);
    fprintf(stderr,"done\n");
    m->L=malloc(m->nl*sizeof(Layer));
    { cpu_set_t s; CPU_ZERO(&s); for(int i=0;i<8;i++)CPU_SET(i,&s); sched_setaffinity(0,sizeof(s),&s); }
    volatile int _done=0;
    #pragma omp parallel for schedule(dynamic) num_threads(8)
    for(int l=0;l<m->nl;l++){ char nm[64]; Layer*ly=&m->L[l];
        #define DQ(suf) ({ snprintf(nm,64,"blk.%d.%s",l,suf); gguf_dequant(g,nm); })
        #define LN(suf,N,K) ({ snprintf(nm,64,"blk.%d.%s",l,suf); float*w=gguf_dequant(g,nm); Lin lin=lin_new_hp(w,N,K); free(w); lin; })
        ly->attn_norm=DQ("attn_norm.weight"); ly->ffn_norm=DQ("ffn_norm.weight");
        ly->q_norm=DQ("attn_q_norm.weight"); ly->k_norm=DQ("attn_k_norm.weight");
        ly->q=LN("attn_q.weight",qd,d); ly->k=LN("attn_k.weight",kvd,d); ly->v=LN("attn_v.weight",kvd,d); ly->o=LN("attn_output.weight",d,qd);
        ly->router=DQ("ffn_gate_inp.weight");
        ly->eg=malloc(ne*sizeof(Lin)); ly->eu=malloc(ne*sizeof(Lin)); ly->ed=malloc(ne*sizeof(Lin));
        snprintf(nm,64,"blk.%d.ffn_gate_exps.weight",l); TInfo*tg=gguf_find(g,nm);
        snprintf(nm,64,"blk.%d.ffn_up_exps.weight",l);   TInfo*tu=gguf_find(g,nm);
        snprintf(nm,64,"blk.%d.ffn_down_exps.weight",l); TInfo*td=gguf_find(g,nm);
        float*eg=malloc((size_t)moe*d*4),*eu=malloc((size_t)moe*d*4),*ed=malloc((size_t)d*moe*4);
        for(int e=0;e<ne;e++){
            gguf_dequant_into(g,tg,(size_t)e*moe*d,(size_t)moe*d,eg); ly->eg[e]=lin_new_hp(eg,moe,d);
            gguf_dequant_into(g,tu,(size_t)e*moe*d,(size_t)moe*d,eu); ly->eu[e]=lin_new_hp(eu,moe,d);
            gguf_dequant_into(g,td,(size_t)e*d*moe,(size_t)d*moe,ed); ly->ed[e]=lin_new_hp(ed,d,moe);
        }
        free(eg);free(eu);free(ed);
        #pragma omp atomic
        _done++;
        #pragma omp critical
        { fprintf(stderr,"\rrequant %d/%d layers",_done,m->nl); }
    }
    fprintf(stderr,"\n");
}

/* ===================== requant cache (new format, ver=2, incompatible with qwen_moe.c's) ===================== */
static void wlin(FILE*f,Lin*l){ int Np=l->N/32,Sb=l->K/256; fwrite(&l->N,4,1,f); fwrite(&l->K,4,1,f);
    fwrite(l->B,1,(size_t)Np*Sb*BSUPER,f); }
static Lin rlin(FILE*f){ Lin l; fread(&l.N,4,1,f); fread(&l.K,4,1,f); int Np=l.N/32,Sb=l.K/256;
    l.B=malloc((size_t)Np*Sb*BSUPER); fread(l.B,1,(size_t)Np*Sb*BSUPER,f); return l; }
static void cache_save(Model*m,const char*path){
    FILE*f=fopen(path,"wb"); if(!f){fprintf(stderr,"cache write fail\n");return;}
    int hdr[9]={m->d,m->nl,m->nh,m->nkv,m->hd,m->vocab,m->n_exp,m->n_act,m->moe}; fwrite("IMEC",1,4,f); int ver=2; fwrite(&ver,4,1,f);
    fwrite(hdr,4,9,f); fwrite(&m->rope_base,4,1,f); fwrite(&m->eps,4,1,f);
    fwrite(m->tok_embd,4,(size_t)m->vocab*m->d,f); fwrite(m->out_norm,4,m->d,f); wlin(f,&m->lm);
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l]; fwrite(ly->attn_norm,4,m->d,f); fwrite(ly->ffn_norm,4,m->d,f);
        fwrite(ly->q_norm,4,m->hd,f); fwrite(ly->k_norm,4,m->hd,f); fwrite(ly->router,4,(size_t)m->n_exp*m->d,f);
        wlin(f,&ly->q); wlin(f,&ly->k); wlin(f,&ly->v); wlin(f,&ly->o);
        for(int e=0;e<m->n_exp;e++){ wlin(f,&ly->eg[e]); wlin(f,&ly->eu[e]); wlin(f,&ly->ed[e]); } }
    fwrite("ENDIMEC",1,8,f);
    if(fflush(f)||fclose(f)) fprintf(stderr,"cache write error\n");
}
static int cache_load(Model*m,const char*path,int nt){
    FILE*f=fopen(path,"rb"); if(!f) return 0;
    char mg[4]; if(fread(mg,1,4,f)!=4 || memcmp(mg,"IMEC",4)){fclose(f);return 0;}
    char foot[8]; if(fseek(f,-8,SEEK_END)||fread(foot,1,8,f)!=8||memcmp(foot,"ENDIMEC",8)){ fprintf(stderr,"cache incomplete/corrupt -> requant\n"); fclose(f); return 0; }
    fseek(f,4,SEEK_SET);
    int ver; fread(&ver,4,1,f); if(ver!=2){ fprintf(stderr,"cache is v%d, this binary needs v2 (vendor HP format) -> requant\n",ver); fclose(f); return 0; }
    int hdr[9]; fread(hdr,4,9,f); m->d=hdr[0];m->nl=hdr[1];m->nh=hdr[2];m->nkv=hdr[3];m->hd=hdr[4];m->vocab=hdr[5];m->n_exp=hdr[6];m->n_act=hdr[7];m->moe=hdr[8];
    fread(&m->rope_base,4,1,f); fread(&m->eps,4,1,f); m->nt=nt;
    m->tok_embd=malloc((size_t)m->vocab*m->d*4); fread(m->tok_embd,4,(size_t)m->vocab*m->d,f);
    m->out_norm=malloc((size_t)m->d*4); fread(m->out_norm,4,m->d,f); m->lm=rlin(f);
    m->L=malloc(m->nl*sizeof(Layer));
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        ly->attn_norm=malloc(m->d*4); fread(ly->attn_norm,4,m->d,f); ly->ffn_norm=malloc(m->d*4); fread(ly->ffn_norm,4,m->d,f);
        ly->q_norm=malloc(m->hd*4); fread(ly->q_norm,4,m->hd,f); ly->k_norm=malloc(m->hd*4); fread(ly->k_norm,4,m->hd,f);
        ly->router=malloc((size_t)m->n_exp*m->d*4); fread(ly->router,4,(size_t)m->n_exp*m->d,f);
        ly->q=rlin(f); ly->k=rlin(f); ly->v=rlin(f); ly->o=rlin(f);
        ly->eg=malloc(m->n_exp*sizeof(Lin)); ly->eu=malloc(m->n_exp*sizeof(Lin)); ly->ed=malloc(m->n_exp*sizeof(Lin));
        for(int e=0;e<m->n_exp;e++){ ly->eg[e]=rlin(f); ly->eu[e]=rlin(f); ly->ed[e]=rlin(f); } }
    fclose(f); return 1;
}

typedef struct { float*Kc,*Vc; int kvd,ctx; } Kv;
static void forward(Model*m,int tok,int pos,Kv*kv,float*logits,
                    float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*g,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2){
    int d=m->d,nh=m->nh,nkv=m->nkv,hd=m->hd,nt=m->nt,qd=nh*hd,kvd=nkv*hd,gpr=nh/nkv,moe=m->moe,ne=m->n_exp,na=m->n_act;
    static float*h=NULL; if(!h)h=malloc(d*4); memcpy(h,m->tok_embd+(size_t)tok*d,d*4);
    double _f0=gT_on?now():0, _a0=gT_actpack,_l0=gT_lin,_at0=gT_attn;
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        rmsnorm(hn,h,ly->attn_norm,d,m->eps);
        { double _ta=gT_on?now():0; pack_act_hp(hn,d,Abuf2);
          double _tb=gT_on?now():0; if(gT_on) gT_actpack+=_tb-_ta;
          lin_mm_hp(&ly->q,Abuf2,q,nt); lin_mm_hp(&ly->k,Abuf2,k,nt); lin_mm_hp(&ly->v,Abuf2,vv,nt);
          if(gT_on) gT_lin+=now()-_tb; }
        for(int hh=0;hh<nh;hh++){ rmsnorm(q+hh*hd,q+hh*hd,ly->q_norm,hd,m->eps); rope(q+hh*hd,hd,pos,m->rope_base); }
        for(int hh=0;hh<nkv;hh++){ rmsnorm(k+hh*hd,k+hh*hd,ly->k_norm,hd,m->eps); rope(k+hh*hd,hd,pos,m->rope_base); }
        float*Kc=kv->Kc+(size_t)l*kv->ctx*kvd,*Vc=kv->Vc+(size_t)l*kv->ctx*kvd;
        memcpy(Kc+(size_t)pos*kvd,k,kvd*4); memcpy(Vc+(size_t)pos*kvd,vv,kvd*4);
        float scale=1.0f/sqrtf(hd); double _at=gT_on?now():0;
        for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q+hh*hd,*sc=tmp;
            for(int j=0;j<=pos;j++){ float*kj=Kc+(size_t)j*kvd+kvh*hd,dd=0; for(int t=0;t<hd;t++)dd+=qh[t]*kj[t]; sc[j]=dd*scale; }
            softmax(sc,pos+1); float*oh=att+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
            for(int j=0;j<=pos;j++){ float w=sc[j],*vj=Vc+(size_t)j*kvd+kvh*hd; for(int t=0;t<hd;t++)oh[t]+=w*vj[t]; } }
        if(gT_on) gT_attn += now()-_at;
        lin_mm(&ly->o,att,tmp,nt,Abuf); for(int i=0;i<d;i++)h[i]+=tmp[i];
        rmsnorm(hn,h,ly->ffn_norm,d,m->eps);
        float rl[256]; for(int e=0;e<ne;e++){ float*rw=ly->router+(size_t)e*d,s=0; for(int i=0;i<d;i++)s+=rw[i]*hn[i]; rl[e]=s; }
        softmax(rl,ne);
        int sel[32]; float sw[32];
        for(int a=0;a<na;a++){ int bi=-1; float bv=-1; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(sel[b]==e)used=1; if(!used&&rl[e]>bv){bv=rl[e];bi=e;} } sel[a]=bi; sw[a]=bv; }
        float ssum=0; for(int a=0;a<na;a++)ssum+=sw[a]; for(int a=0;a<na;a++)sw[a]/=ssum;
        for(int i=0;i<d;i++)eout[i]=0;
        { double _ta=gT_on?now():0; pack_act_hp(hn,d,Abuf2);
          if(gT_on) gT_actpack+=now()-_ta;
          for(int a=0;a<na;a++){ int e=sel[a]; float w=sw[a];
              double _tb=gT_on?now():0;
              lin_mm_hp(&ly->eg[e],Abuf2,g,nt); lin_mm_hp(&ly->eu[e],Abuf2,u,nt);
              if(gT_on) gT_lin+=now()-_tb;
              for(int i=0;i<moe;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }
              lin_mm(&ly->ed[e],g,tmp,nt,Abuf); for(int i=0;i<d;i++)eout[i]+=w*tmp[i]; } }
        for(int i=0;i<d;i++)h[i]+=eout[i];
    }
    rmsnorm(hn,h,m->out_norm,d,m->eps); lin_mm(&m->lm,hn,logits,nt,Abuf);
    if(gT_on){ double ft=now()-_f0; gT_rest += ft-(gT_actpack-_a0)-(gT_lin-_l0)-(gT_attn-_at0); gT_tok++; }
}
static int argmax(const float*l,int n){ int b=0; float bv=l[0]; for(int i=1;i<n;i++)if(l[i]>bv){bv=l[i];b=i;} return b; }

static uint8_t g_bdec[0x200]; static int g_bi=0;
static void bdec_init(void){ int cs[256],bs[256],nn=0; for(int i=0;i<256;i++)bs[i]=-1; int idx=0;
    for(int b=0x21;b<=0x7e;b++){cs[idx]=b;bs[b]=b;idx++;} for(int b=0xa1;b<=0xac;b++){cs[idx]=b;bs[b]=b;idx++;} for(int b=0xae;b<=0xff;b++){cs[idx]=b;bs[b]=b;idx++;}
    for(int b=0;b<256;b++)if(bs[b]<0){cs[idx]=256+nn;bs[b]=256+nn;nn++;idx++;}
    memset(g_bdec,0,sizeof(g_bdec)); for(int b=0;b<256;b++){int cp=bs[b]; if(cp<0x200)g_bdec[cp]=(uint8_t)b;} g_bi=1; }
static void tok_print(Gguf*g,int id){ if(!g_bi)bdec_init(); if(id<0||id>=g->ntok)return; char*s=g->tok[id];
    for(int i=0;s[i];){ unsigned c=(unsigned char)s[i],cp; if(c<0x80){cp=c;i+=1;} else if((c>>5)==6){cp=((c&0x1f)<<6)|((unsigned char)s[i+1]&0x3f);i+=2;}
        else if((c>>4)==14){cp=((c&0xf)<<12)|(((unsigned char)s[i+1]&0x3f)<<6)|((unsigned char)s[i+2]&0x3f);i+=3;} else {cp=c;i+=1;}
        if(cp<0x200)putchar(g_bdec[cp]); } }

int main(int c,char**v){
    if(c<2){ printf("usage: %s model.gguf [ngen] [nt]\n",v[0]); return 1; }
    int ngen=(c>2)?atoi(v[2]):16, nt=(c>3)?atoi(v[3]):4;
    bind_ai(); { cpu_set_t s;CPU_ZERO(&s);CPU_SET(8,&s);sched_setaffinity(0,sizeof(s),&s);} for(int i=0;i<5;i++)sched_yield();
    Gguf g; double t0=now(); gguf_open(&g,v[1]);
    fprintf(stderr,"qwen3moe (HP kernel): %d layers d=%d experts=%d/%d moe_ffn=%d heads=%d/%d hd=%d vocab=%d (parse %.1fs)\n",
        g.block_count,g.embd,g.n_exp,g.n_act,g.moe_ffn,g.nh,g.nkv,g.hd,g.vocab,now()-t0);
    const char*cpath=(c>4)?v[4]:"/root/models/qwen3-30b-a3b.hp.imecache";
    Model m; double tl=now(); int cached=0;
    if(cache_load(&m,cpath,nt)){ cached=1; fprintf(stderr,"loaded from cache in %.1fs  (%s)\n",now()-tl,cpath); }
    else { model_load(&m,&g,nt); fprintf(stderr,"requant loaded in %.1fs\n",now()-tl); }

    int ctx=64; Kv kv; kv.kvd=m.nkv*m.hd; kv.ctx=ctx; kv.Kc=calloc((size_t)m.nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m.nl*ctx*kv.kvd,4);
    int d=m.d,qd=m.nh*m.hd,moe=m.moe,maxk=qd>moe?(qd>d?qd:d):(moe>d?moe:d); if(d>maxk)maxk=d;
    float*hn=malloc(d*4),*q=malloc(qd*4),*k=malloc(kv.kvd*4),*vv=malloc(kv.kvd*4),*att=malloc(qd*4),
         *tmp=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4),*gg=malloc(moe*4),*u=malloc(moe*4),*eout=malloc(d*4),*logits=malloc((size_t)m.vocab*4);
    uint8_t*Abuf=malloc((size_t)(maxk/256)*AREC),*Abuf2=malloc((size_t)(maxk/256)*AREC);
    lin_mm_pool_init(nt); /* PR8: persistent workers, spawned once, spin-dispatched per lin_mm_hp call */

    int prompt[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374}; int np=12;
    double tp=now(); int first=0;
    for(int p=0;p<np;p++){ forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2); if(p==np-1)first=argmax(logits,m.vocab); }
    printf("\nprompt      : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]);
    printf("\nfirst argmax: %d ('",first); tok_print(&g,first); printf("')  expect 26194 (' Tokyo') -> %s\n", first==26194?"PASS":"FAIL");
    printf("prefill %.2fs (%d tok)\n",now()-tp,np);
    printf("generation  : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]); tok_print(&g,first);
    int cur=first; double tg=now(); gT_on=1;
    for(int s=0;s<ngen;s++){ int pos=np+s; forward(&m,cur,pos,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2); cur=argmax(logits,m.vocab); tok_print(&g,cur); }
    double dt=now()-tg; gT_on=0;
    printf("\ndecode: %.2f tok/s (Qwen3-30B-A3B, vendor IME-2-HP int4 W4A8, nt=%d)\n", ngen/dt, nt);
    if(gT_tok) printf("  per-token buckets (avg/%ld tok, ms): act-pack %.1f | linear(kernel) %.1f | attention %.1f | rest %.1f | sum %.1f | wall %.1f\n",
        gT_tok, gT_actpack/gT_tok*1e3, gT_lin/gT_tok*1e3, gT_attn/gT_tok*1e3, gT_rest/gT_tok*1e3,
        (gT_actpack+gT_lin+gT_attn+gT_rest)/gT_tok*1e3, dt/ngen*1e3);
    if(!cached){ fprintf(stderr,"saving requant cache -> %s ...\n",cpath); double ts=now(); cache_save(&m,cpath); fprintf(stderr,"cache saved in %.1fs\n",now()-ts); }
    return 0;
}
