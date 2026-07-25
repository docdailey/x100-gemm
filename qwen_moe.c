/* qwen_moe.c — pure-C Qwen3-30B-A3B MoE decode on the K3 A100 IME-2 (W4A8).
 *
 * Extends qwen_ime.c to MoE: 128 experts / 8 active, expert FFN 768. Only ~3B of 30B streams per
 * token. Weights int4 (q4-in-q8 interlaced kernel — 30B fits 32GB at int4, not int8). GGUF is
 * mmap'd (17GB) and each tensor is dequantized (Q4_0/Q4_1/F32) -> requantized per-output-channel to
 * int4 + packed for the IME-2 vmadot. Router (fp32) -> softmax -> top-8 -> renormalize -> weighted
 * sum of the 8 experts' SwiGLU. Same attention/norm/RoPE as the dense engine.
 *
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe qwen_moe.c -lm
 * Run  : LD_LIBRARY_PATH=/usr/lib ./qwen_moe /root/models/Qwen3-30B-A3B-Q4_0.gguf [ngen] [nt]
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

#define N0 8
#define K0 16
#define TILE 128
#define QTILE 64

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static __thread int g_pinned=0;
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(8+(tn*2)%8,&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

/* ===================== IME-2 int4 (q4-in-q8 interlaced) GEMV — validated ===================== */
static void pack_w_int4(int Nt,int Kt,const int8_t*W,uint8_t*Wq){
    int Nb=Nt/N0, Kb=Kt/K0, Kp=Kb/2;
    for(int nb=0;nb<Nb;nb++)for(int kp=0;kp<Kp;kp++){
        int8_t t0[TILE],t1[TILE]; int kb0=2*kp,kb1=2*kp+1;
        for(int n=0;n<N0;n++)for(int k=0;k<K0;k++){ t0[n*K0+k]=W[(nb*N0+n)*Kt+kb0*K0+k]; t1[n*K0+k]=W[(nb*N0+n)*Kt+kb1*K0+k]; }
        uint8_t*d=Wq+((size_t)(nb*Kp+kp))*TILE;
        for(int j=0;j<TILE;j++) d[j]=(uint8_t)((t0[j]&0xf)|((t1[j]&0xf)<<4));
    }
}
/* per-GROUP int4 (group=32=one interleaved iteration, matches Q4_0): each group writes its 8x8
 * int32 partial so a per-(output,group) scale can be applied — per-channel int4 is too lossy. */
__attribute__((noinline,optimize("no-tree-vectorize","no-stack-protector")))
static void gemv_nb_int4_grouped(const int8_t*xt,const uint8_t*Wq,int Kb,int32_t*part){
    long Kp=Kb/2;
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t"
        "vsetvli t0,zero,e32,m2\n\t vxor.vv v28,v28,v28\n\t vxor.vv v30,v30,v30\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "vle8.v v0,(%0)\n\t addi %0,%0,128\n\t vle8.v v2,(%0)\n\t addi %0,%0,128\n\t"
        "vle8.v v4,(%1)\n\t addi %1,%1,128\n\t"
        "vsll.vi v5,v4,4\n\t vsra.vi v5,v5,4\n\t vsra.vi v6,v4,4\n\t"
        "vmadot v28,v0,v5\n\t vmadot v30,v2,v6\n\t"
        "vsetvli t0,zero,e32,m2\n\t vadd.vv v28,v28,v30\n\t vse32.v v28,(%3)\n\t addi %3,%3,256\n\t"
        "addi %2,%2,-1\n\t bnez %2,1b\n\t"
        : "+r"(xt),"+r"(Wq),"+r"(Kp),"+r"(part) : : "t0","v0","v2","v4","v5","v6","v28","v29","v30","v31","memory");
}
#define GRP 32
typedef struct { int N,K,Kp; uint8_t*Wq; float*gs; } Lin;  /* gs[N*Kp] per-(out,group) scale */
static Lin lin_new(const float*wf32,int N,int K){
    Lin l; l.N=N; l.K=K; int Nb=N/N0,Kb=K/K0,Kp=Kb/2; l.Kp=Kp;
    int8_t*wi=malloc((size_t)N*K); l.gs=malloc((size_t)N*Kp*4);
    for(int r=0;r<N;r++){ const float*row=wf32+(size_t)r*K;
        for(int gp=0;gp<Kp;gp++){ float amax=1e-6f; int c0=gp*GRP;
            for(int c=c0;c<c0+GRP;c++){ float a=fabsf(row[c]); if(a>amax)amax=a; }
            float s=amax/7.0f; l.gs[(size_t)r*Kp+gp]=s; float inv=1.0f/s;
            for(int c=c0;c<c0+GRP;c++){ int q=(int)lrintf(row[c]*inv); q=q>7?7:(q<-7?-7:q); wi[(size_t)r*K+c]=(int8_t)q; } } }
    l.Wq=malloc((size_t)Nb*Kb*QTILE); pack_w_int4(N,K,wi,l.Wq); free(wi); return l;
}
static void lin_mm(const Lin*l,const float*x,float*y,int nt,int8_t*xt){
    int Nb=l->N/N0, Kb=l->K/K0, Kp=l->Kp, kb=Kb;
    float amax=1e-6f; for(int i=0;i<l->K;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float xs=amax/127.0f, inv=1.0f/xs;
    memset(xt,0,(size_t)kb*TILE);
    for(int c=0;c<l->K;c++){ int q=(int)lrintf(x[c]*inv); q=q>127?127:(q<-128?-128:q); xt[(c/K0)*TILE+(c%K0)]=(int8_t)q; }
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); pin_once(tn); int32_t*part=malloc((size_t)Kp*64*4);
      for(int nb=tn;nb<Nb;nb+=nt){ gemv_nb_int4_grouped(xt, l->Wq+(size_t)nb*Kb*QTILE, Kb, part);
          for(int n=0;n<N0;n++){ int out=nb*N0+n; const float*gsr=l->gs+(size_t)out*Kp; float acc=0;
              for(int gp=0;gp<Kp;gp++) acc+=(float)part[gp*64+n]*gsr[gp]; y[out]=acc*xs; } }
      free(part); }
}

/* ===================== GGUF reader (mmap; Q4_0/Q4_1/Q8_0/F32/F16) ===================== */
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
/* dequant `n` elems starting at element `elem0` of a tensor into out (caller-alloc'd n floats) */
static void gguf_dequant_into(Gguf*g,TInfo*ti,size_t elem0,size_t n,float*out){
    unsigned char*base=g->p+g->data_start+ti->off;
    if(ti->typ==0){ float*b=(float*)base+elem0; for(size_t i=0;i<n;i++)out[i]=b[i]; return; }
    if(ti->typ==1){ uint16_t*b=(uint16_t*)base+elem0; for(size_t i=0;i<n;i++)out[i]=f16f(b[i]); return; }
    /* block-quantized: elem0 must be a multiple of 32 (it always is for our slices) */
    size_t bl0=elem0/32, nb=(n+31)/32;
    if(ti->typ==8){ for(size_t bl=0;bl<nb;bl++){ unsigned char*q=base+(bl0+bl)*34; float sc=f16f(*(uint16_t*)q);
        for(int j=0;j<32 && bl*32+j<n;j++) out[bl*32+j]=(float)(int8_t)q[2+j]*sc; } return; }
    if(ti->typ==2){ for(size_t bl=0;bl<nb;bl++){ unsigned char*q=base+(bl0+bl)*18; float d=f16f(*(uint16_t*)q); unsigned char*qs=q+2;
        for(int j=0;j<32 && bl*32+j<n;j++){ int nib=(j<16)?(qs[j]&0xf):(qs[j-16]>>4); out[bl*32+j]=(nib-8)*d; } } return; }
    if(ti->typ==3){ for(size_t bl=0;bl<nb;bl++){ unsigned char*q=base+(bl0+bl)*20; float d=f16f(*(uint16_t*)q),m=f16f(*(uint16_t*)(q+2)); unsigned char*qs=q+4;
        for(int j=0;j<32 && bl*32+j<n;j++){ int nib=(j<16)?(qs[j]&0xf):(qs[j-16]>>4); out[bl*32+j]=nib*d+m; } } return; }
    if(ti->typ==14){ /* Q6_K, super-block 256 (210 bytes). Only whole-tensor (elem0==0). */
        size_t nbk=(n+255)/256;
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
    fprintf(stderr,"packing token_embd + lm_head (int4) ... "); fflush(stderr);
    m->tok_embd=gguf_dequant(g,"token_embd.weight"); m->out_norm=gguf_dequant(g,"output_norm.weight");
    if(gguf_has(g,"output.weight")){ float*ow=gguf_dequant(g,"output.weight"); m->lm=lin_new(ow,m->vocab,d); free(ow);
        fprintf(stderr,"(untied lm_head from output.weight) "); }
    else m->lm=lin_new(m->tok_embd,m->vocab,d); /* tied */
    fprintf(stderr,"done\n");
    m->L=malloc(m->nl*sizeof(Layer));
    for(int l=0;l<m->nl;l++){ char nm[64]; Layer*ly=&m->L[l]; double lt=now();
        #define DQ(suf) ({ snprintf(nm,64,"blk.%d.%s",l,suf); gguf_dequant(g,nm); })
        #define LN(suf,N,K) ({ snprintf(nm,64,"blk.%d.%s",l,suf); float*w=gguf_dequant(g,nm); Lin lin=lin_new(w,N,K); free(w); lin; })
        ly->attn_norm=DQ("attn_norm.weight"); ly->ffn_norm=DQ("ffn_norm.weight");
        ly->q_norm=DQ("attn_q_norm.weight"); ly->k_norm=DQ("attn_k_norm.weight");
        ly->q=LN("attn_q.weight",qd,d); ly->k=LN("attn_k.weight",kvd,d); ly->v=LN("attn_v.weight",kvd,d); ly->o=LN("attn_output.weight",d,qd);
        ly->router=DQ("ffn_gate_inp.weight"); /* [ne, d] fp32 */
        /* experts: stacked 3D [in, moe, ne] (gate/up), [moe(in), d(out), ne] (down). Extract per-expert. */
        ly->eg=malloc(ne*sizeof(Lin)); ly->eu=malloc(ne*sizeof(Lin)); ly->ed=malloc(ne*sizeof(Lin));
        snprintf(nm,64,"blk.%d.ffn_gate_exps.weight",l); TInfo*tg=gguf_find(g,nm);
        snprintf(nm,64,"blk.%d.ffn_up_exps.weight",l);   TInfo*tu=gguf_find(g,nm);
        snprintf(nm,64,"blk.%d.ffn_down_exps.weight",l); TInfo*td=gguf_find(g,nm);
        float*eg=malloc((size_t)moe*d*4),*eu=malloc((size_t)moe*d*4),*ed=malloc((size_t)d*moe*4);
        for(int e=0;e<ne;e++){
            gguf_dequant_into(g,tg,(size_t)e*moe*d,(size_t)moe*d,eg); ly->eg[e]=lin_new(eg,moe,d);
            gguf_dequant_into(g,tu,(size_t)e*moe*d,(size_t)moe*d,eu); ly->eu[e]=lin_new(eu,moe,d);
            gguf_dequant_into(g,td,(size_t)e*d*moe,(size_t)d*moe,ed); ly->ed[e]=lin_new(ed,d,moe);
        }
        free(eg);free(eu);free(ed);
        fprintf(stderr,"\rpacked layer %d/%d (%.1fs)   ",l+1,m->nl,now()-lt);
    }
    fprintf(stderr,"\n");
}

/* ===================== requant cache (skip the ~15-min requant on reruns) ===================== */
static void wlin(FILE*f,Lin*l){ int Nb=l->N/N0,Kb=l->K/K0; fwrite(&l->N,4,1,f); fwrite(&l->K,4,1,f); fwrite(&l->Kp,4,1,f);
    fwrite(l->Wq,1,(size_t)Nb*Kb*QTILE,f); fwrite(l->gs,4,(size_t)l->N*l->Kp,f); }
static Lin rlin(FILE*f){ Lin l; fread(&l.N,4,1,f); fread(&l.K,4,1,f); fread(&l.Kp,4,1,f); int Nb=l.N/N0,Kb=l.K/K0;
    l.Wq=malloc((size_t)Nb*Kb*QTILE); fread(l.Wq,1,(size_t)Nb*Kb*QTILE,f); l.gs=malloc((size_t)l.N*l.Kp*4); fread(l.gs,4,(size_t)l.N*l.Kp,f); return l; }
static void cache_save(Model*m,const char*path){
    FILE*f=fopen(path,"wb"); if(!f){fprintf(stderr,"cache write fail\n");return;}
    int hdr[9]={m->d,m->nl,m->nh,m->nkv,m->hd,m->vocab,m->n_exp,m->n_act,m->moe}; fwrite("IMEC",1,4,f); int ver=1; fwrite(&ver,4,1,f);
    fwrite(hdr,4,9,f); fwrite(&m->rope_base,4,1,f); fwrite(&m->eps,4,1,f);
    fwrite(m->tok_embd,4,(size_t)m->vocab*m->d,f); fwrite(m->out_norm,4,m->d,f); wlin(f,&m->lm);
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l]; fwrite(ly->attn_norm,4,m->d,f); fwrite(ly->ffn_norm,4,m->d,f);
        fwrite(ly->q_norm,4,m->hd,f); fwrite(ly->k_norm,4,m->hd,f); fwrite(ly->router,4,(size_t)m->n_exp*m->d,f);
        wlin(f,&ly->q); wlin(f,&ly->k); wlin(f,&ly->v); wlin(f,&ly->o);
        for(int e=0;e<m->n_exp;e++){ wlin(f,&ly->eg[e]); wlin(f,&ly->eu[e]); wlin(f,&ly->ed[e]); } }
    fclose(f);
}
static int cache_load(Model*m,const char*path,int nt){
    FILE*f=fopen(path,"rb"); if(!f) return 0; char mg[4]; fread(mg,1,4,f); if(memcmp(mg,"IMEC",4)){fclose(f);return 0;}
    int ver; fread(&ver,4,1,f); int hdr[9]; fread(hdr,4,9,f); m->d=hdr[0];m->nl=hdr[1];m->nh=hdr[2];m->nkv=hdr[3];m->hd=hdr[4];m->vocab=hdr[5];m->n_exp=hdr[6];m->n_act=hdr[7];m->moe=hdr[8];
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
                    float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*g,float*u,float*eout,int8_t*xt){
    int d=m->d,nh=m->nh,nkv=m->nkv,hd=m->hd,nt=m->nt,qd=nh*hd,kvd=nkv*hd,gpr=nh/nkv,moe=m->moe,ne=m->n_exp,na=m->n_act;
    static float*h=NULL; if(!h)h=malloc(d*4); memcpy(h,m->tok_embd+(size_t)tok*d,d*4);
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        rmsnorm(hn,h,ly->attn_norm,d,m->eps);
        lin_mm(&ly->q,hn,q,nt,xt); lin_mm(&ly->k,hn,k,nt,xt); lin_mm(&ly->v,hn,vv,nt,xt);
        for(int hh=0;hh<nh;hh++){ rmsnorm(q+hh*hd,q+hh*hd,ly->q_norm,hd,m->eps); rope(q+hh*hd,hd,pos,m->rope_base); }
        for(int hh=0;hh<nkv;hh++){ rmsnorm(k+hh*hd,k+hh*hd,ly->k_norm,hd,m->eps); rope(k+hh*hd,hd,pos,m->rope_base); }
        float*Kc=kv->Kc+(size_t)l*kv->ctx*kvd,*Vc=kv->Vc+(size_t)l*kv->ctx*kvd;
        memcpy(Kc+(size_t)pos*kvd,k,kvd*4); memcpy(Vc+(size_t)pos*kvd,vv,kvd*4);
        float scale=1.0f/sqrtf(hd);
        for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q+hh*hd,*sc=tmp;
            for(int j=0;j<=pos;j++){ float*kj=Kc+(size_t)j*kvd+kvh*hd,dd=0; for(int t=0;t<hd;t++)dd+=qh[t]*kj[t]; sc[j]=dd*scale; }
            softmax(sc,pos+1); float*oh=att+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
            for(int j=0;j<=pos;j++){ float w=sc[j],*vj=Vc+(size_t)j*kvd+kvh*hd; for(int t=0;t<hd;t++)oh[t]+=w*vj[t]; } }
        lin_mm(&ly->o,att,tmp,nt,xt); for(int i=0;i<d;i++)h[i]+=tmp[i];
        /* ---- MoE FFN ---- */
        rmsnorm(hn,h,ly->ffn_norm,d,m->eps);
        float rl[256]; for(int e=0;e<ne;e++){ float*rw=ly->router+(size_t)e*d,s=0; for(int i=0;i<d;i++)s+=rw[i]*hn[i]; rl[e]=s; }
        /* softmax over all experts, then top-na, renormalize the selected */
        softmax(rl,ne);
        int sel[32]; float sw[32];
        for(int a=0;a<na;a++){ int bi=-1; float bv=-1; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(sel[b]==e)used=1; if(!used&&rl[e]>bv){bv=rl[e];bi=e;} } sel[a]=bi; sw[a]=bv; }
        float ssum=0; for(int a=0;a<na;a++)ssum+=sw[a]; for(int a=0;a<na;a++)sw[a]/=ssum;
        for(int i=0;i<d;i++)eout[i]=0;
        for(int a=0;a<na;a++){ int e=sel[a]; float w=sw[a];
            lin_mm(&ly->eg[e],hn,g,nt,xt); lin_mm(&ly->eu[e],hn,u,nt,xt);
            for(int i=0;i<moe;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }
            lin_mm(&ly->ed[e],g,tmp,nt,xt); for(int i=0;i<d;i++)eout[i]+=w*tmp[i]; }
        for(int i=0;i<d;i++)h[i]+=eout[i];
    }
    rmsnorm(hn,h,m->out_norm,d,m->eps); lin_mm(&m->lm,hn,logits,nt,xt);
}
static int argmax(const float*l,int n){ int b=0; float bv=l[0]; for(int i=1;i<n;i++)if(l[i]>bv){bv=l[i];b=i;} return b; }

/* GPT-2 byte-level decode */
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
    fprintf(stderr,"qwen3moe: %d layers d=%d experts=%d/%d moe_ffn=%d heads=%d/%d hd=%d vocab=%d (parse %.1fs)\n",
        g.block_count,g.embd,g.n_exp,g.n_act,g.moe_ffn,g.nh,g.nkv,g.hd,g.vocab,now()-t0);
    const char*cpath=(c>4)?v[4]:"/mnt/jupiter2/qwen3-30b-a3b.imecache";
    Model m; double tl=now(); int cached=0;
    if(cache_load(&m,cpath,nt)){ cached=1; fprintf(stderr,"loaded from cache in %.1fs  (%s)\n",now()-tl,cpath); }
    else {
      /* early dequant sanity: Q4_0 gate + Q4_1 down + Q6_K output of blk.0 (catches dequant bugs in <1s) */
      { TInfo*tg=gguf_find(&g,"blk.0.ffn_gate_exps.weight"); size_t N=(size_t)g.moe_ffn*g.embd; float*t=malloc(N*4);
        gguf_dequant_into(&g,tg,0,N,t); double mn=0,sd=0; for(size_t i=0;i<N;i++)mn+=t[i]; mn/=N;
        for(size_t i=0;i<N;i++)sd+=(t[i]-mn)*(t[i]-mn); sd=sqrt(sd/N);
        fprintf(stderr,"[dequant] gate_exps e0 (Q4_0 t=%u): mean=%.4f std=%.4f\n",tg->typ,mn,sd);
        TInfo*td=gguf_find(&g,"blk.0.ffn_down_exps.weight"); size_t Md=(size_t)g.embd*g.moe_ffn;
        gguf_dequant_into(&g,td,0,Md,t); mn=0; for(size_t i=0;i<Md;i++)mn+=t[i]; mn/=Md; sd=0; for(size_t i=0;i<Md;i++)sd+=(t[i]-mn)*(t[i]-mn); sd=sqrt(sd/Md);
        fprintf(stderr,"[dequant] down_exps e0 (Q4_1 t=%u): mean=%.4f std=%.4f\n",td->typ,mn,sd); free(t);
        if(gguf_has(&g,"output.weight")){ TInfo*to=gguf_find(&g,"output.weight"); float*t2=malloc(25600*4); gguf_dequant_into(&g,to,0,25600,t2);
            double m2=0,s2=0; for(int i=0;i<25600;i++)m2+=t2[i]; m2/=25600; for(int i=0;i<25600;i++)s2+=(t2[i]-m2)*(t2[i]-m2); s2=sqrt(s2/25600);
            fprintf(stderr,"[dequant] output.weight (Q6_K t=%u): mean=%.4f std=%.4f  (UNTIED lm_head)\n",to->typ,m2,s2); free(t2); } }
      model_load(&m,&g,nt); fprintf(stderr,"requant loaded in %.1fs\n",now()-tl);
    }

    int ctx=64; Kv kv; kv.kvd=m.nkv*m.hd; kv.ctx=ctx; kv.Kc=calloc((size_t)m.nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m.nl*ctx*kv.kvd,4);
    int d=m.d,qd=m.nh*m.hd,moe=m.moe,maxk=qd>moe?(qd>d?qd:d):(moe>d?moe:d); if(d>maxk)maxk=d;
    float*hn=malloc(d*4),*q=malloc(qd*4),*k=malloc(kv.kvd*4),*vv=malloc(kv.kvd*4),*att=malloc(qd*4),
         *tmp=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4),*gg=malloc(moe*4),*u=malloc(moe*4),*eout=malloc(d*4),*logits=malloc((size_t)m.vocab*4);
    int8_t*xt=malloc((size_t)(maxk/K0)*TILE);

    int prompt[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374}; int np=12;
    double tp=now(); int first=0;
    for(int p=0;p<np;p++){ forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,xt); if(p==np-1)first=argmax(logits,m.vocab); }
    printf("\nprompt      : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]);
    printf("\nfirst argmax: %d ('",first); tok_print(&g,first); printf("')  expect 26194 (' Tokyo') -> %s\n", first==26194?"PASS":"FAIL");
    printf("prefill %.2fs (%d tok)\n",now()-tp,np);
    printf("generation  : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]); tok_print(&g,first);
    int cur=first; double tg=now();
    for(int s=0;s<ngen;s++){ int pos=np+s; forward(&m,cur,pos,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,xt); cur=argmax(logits,m.vocab); tok_print(&g,cur); }
    double dt=now()-tg; printf("\ndecode: %.2f tok/s (Qwen3-30B-A3B, IME-2 int4 W4A8, nt=%d)\n", ngen/dt, nt);
    if(!cached){ fprintf(stderr,"saving requant cache -> %s ...\n",cpath); double ts=now(); cache_save(&m,cpath); fprintf(stderr,"cache saved in %.1fs\n",now()-ts); }
    return 0;
}
