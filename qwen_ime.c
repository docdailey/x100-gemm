/* qwen_ime.c — pure-C Qwen3 decode engine on the K3 A100 IME-2 (W8A8).
 *
 * Everything the Rust runtime did, in C (which the board's gcc compiles correctly — the vmadot
 * heap-corruption was rustc-static-link-only). Reads a real Qwen3 GGUF, dequantizes Q8_0 ->
 * requantizes per-output-channel int8 + packs for the IME-2 kernel, runs the full Qwen3 forward
 * (RMSNorm + QK-norm + NEOX RoPE + GQA + SwiGLU + tied lm_head) with matmuls on the 4 IME-2 units,
 * greedy-generates and decodes to text.
 *
 * Build: gcc -O3 -fno-tree-vectorize -fno-stack-protector -march=rv64gcv_zvfh_xsmtvdotii -fopenmp \
 *            -o qwen_ime qwen_ime.c -lm
 * Run  : LD_LIBRARY_PATH=/usr/lib ./qwen_ime /root/models/Qwen3-4B-Q8_0.gguf [ngen] [nt]
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

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static __thread int g_pinned=0;
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(8+(tn*2)%8,&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

/* ===================== IME-2 int8 GEMV (validated) ===================== */
static void pack_w_int8(int Nt,int Kt,const int8_t*W,int8_t*Wp){
    int Nb=Nt/N0, Kb=Kt/K0;
    for(int nb=0;nb<Nb;nb++)for(int kb=0;kb<Kb;kb++){ int8_t*d=Wp+((size_t)(nb*Kb+kb))*TILE;
        for(int n=0;n<N0;n++)for(int k=0;k<K0;k++) d[n*K0+k]=W[(nb*N0+n)*Kt+kb*K0+k]; }
}
static void gemv_nb_int8(const int8_t*xt,const int8_t*Wp,int Kb,int32_t*ct){
    __asm__ volatile(
        "vsetvli t0,zero,e32,m2\n\t vxor.vv v28,v28,v28\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "1:\n\t vle8.v v0,(%0)\n\t addi %0,%0,128\n\t vle8.v v1,(%1)\n\t addi %1,%1,128\n\t"
        "vmadot v28,v0,v1\n\t addi %2,%2,-1\n\t bnez %2,1b\n\t"
        "vsetvli t0,zero,e32,m2\n\t vse32.v v28,(%3)\n\t"
        : "+r"(xt),"+r"(Wp),"+r"(Kb) : "r"(ct) : "t0","v0","v1","v28","v29","memory");
}

/* a linear: per-output-channel int8 weights (packed) + scale */
typedef struct { int N,K; int8_t*Wp; float*ws; } Lin;
static Lin lin_new(const float*wf32,int N,int K){
    Lin l; l.N=N; l.K=K; int Nb=N/N0,Kb=K/K0;
    int8_t*wi=malloc((size_t)N*K); l.ws=malloc((size_t)N*4);
    for(int r=0;r<N;r++){ const float*row=wf32+(size_t)r*K; float amax=1e-6f;
        for(int c=0;c<K;c++){ float a=fabsf(row[c]); if(a>amax)amax=a; }
        float s=amax/127.0f; l.ws[r]=s; float inv=1.0f/s;
        for(int c=0;c<K;c++){ int q=(int)lrintf(row[c]*inv); q=q>127?127:(q<-128?-128:q); wi[(size_t)r*K+c]=(int8_t)q; } }
    l.Wp=malloc((size_t)Nb*Kb*TILE); pack_w_int8(N,K,wi,l.Wp); free(wi); return l;
}
/* y[N] = dequant( W @ quant(x) ), tensor-parallel over nt IME units */
static void lin_mm(const Lin*l,const float*x,float*y,int nt,int8_t*xt){
    int Nb=l->N/N0, Kb=l->K/K0, kb=Kb;
    float amax=1e-6f; for(int i=0;i<l->K;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float xs=amax/127.0f, inv=1.0f/xs;
    memset(xt,0,(size_t)kb*TILE);
    for(int c=0;c<l->K;c++){ int q=(int)lrintf(x[c]*inv); q=q>127?127:(q<-128?-128:q); xt[(c/K0)*TILE+(c%K0)]=(int8_t)q; }
    #pragma omp parallel num_threads(nt)
    { int tn=omp_get_thread_num(); pin_once(tn); int32_t ct[64];
      for(int nb=tn;nb<Nb;nb+=nt){ gemv_nb_int8(xt, l->Wp+(size_t)nb*Kb*TILE, Kb, ct);
          for(int n=0;n<N0;n++) y[nb*N0+n]=(float)ct[n]*l->ws[nb*N0+n]*xs; } }
}

/* ===================== GGUF reader ===================== */
typedef struct { uint32_t typ; int nd; uint64_t dims[4]; uint64_t off; char name[64]; } TInfo;
typedef struct { unsigned char*p; uint64_t data_start; TInfo*t; int nt;
    uint32_t mu_get; char**tok; int ntok;
    int block_count, embd, ffn, nh, nkv, hd, vocab; float rope_base; } Gguf;
static uint64_t U64(unsigned char*b){ uint64_t x; memcpy(&x,b,8); return x; }
static uint32_t U32(unsigned char*b){ uint32_t x; memcpy(&x,b,4); return x; }
static float f16f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,b;
    if(e==0){ if(m==0)b=s<<31; else{ int ee=-14; while(!(m&0x400)){m<<=1;ee--;} m&=0x3ff; b=(s<<31)|((uint32_t)(ee+127)<<23)|(m<<13);} }
    else if(e==0x1f) b=(s<<31)|(0xff<<23)|(m<<13);
    else b=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&b,4); return f; }

static size_t skipval(unsigned char*p,uint32_t t){ /* returns bytes consumed for a metadata value */
    switch(t){ case 0:case 1:case 7: return 1; case 2:case 3: return 2; case 4:case 5:case 6: return 4;
        case 10:case 11:case 12: return 8; case 8: return 8+U64(p);
        case 9:{ uint32_t et=U32(p); uint64_t n=U64(p+4); size_t o=12; for(uint64_t i=0;i<n;i++) o+=skipval(p+o,et); return o; }
        default: return 4; } }

static void gguf_open(Gguf*g,const char*path){
    FILE*f=fopen(path,"rb"); if(!f){perror("open");exit(1);} fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    g->p=malloc(sz); if(fread(g->p,1,sz,f)!=(size_t)sz){printf("read fail\n");exit(1);} fclose(f);
    unsigned char*p=g->p; size_t o=4; /* magic */
    o+=4; uint64_t ntensor=U64(p+o); o+=8; uint64_t nkv=U64(p+o); o+=8;
    g->rope_base=1e6f; g->tok=NULL; g->ntok=0;
    for(uint64_t i=0;i<nkv;i++){
        uint64_t kl=U64(p+o); char key[128]; int c=kl<127?kl:127; memcpy(key,p+o+8,c); key[c]=0; o+=8+kl;
        uint32_t t=U32(p+o); o+=4;
        #define KEYIS(s) (strcmp(key,s)==0)
        if(t==9 && KEYIS("tokenizer.ggml.tokens")){ uint32_t et=U32(p+o); uint64_t n=U64(p+o+4); size_t oo=o+12;
            g->ntok=n; g->tok=malloc(n*sizeof(char*));
            for(uint64_t j=0;j<n;j++){ uint64_t l=U64(p+oo); char*s=malloc(l+1); memcpy(s,p+oo+8,l); s[l]=0; g->tok[j]=s; oo+=8+l; }
            o=oo; continue; }
        if(t==4||t==5){ uint32_t v=U32(p+o);
            if(KEYIS("qwen3.block_count"))g->block_count=v; else if(KEYIS("qwen3.embedding_length"))g->embd=v;
            else if(KEYIS("qwen3.feed_forward_length"))g->ffn=v; else if(KEYIS("qwen3.attention.head_count"))g->nh=v;
            else if(KEYIS("qwen3.attention.head_count_kv"))g->nkv=v; else if(KEYIS("qwen3.attention.key_length"))g->hd=v; }
        else if(t==6){ float v; uint32_t x=U32(p+o); memcpy(&v,&x,4); if(KEYIS("qwen3.rope.freq_base"))g->rope_base=v; }
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
/* dequant first `max` elems of a tensor to a fresh float* (caller frees) */
static float* gguf_dequant(Gguf*g,const char*name,size_t max){
    TInfo*ti=gguf_find(g,name); size_t total=1; for(int d=0;d<ti->nd;d++) total*=ti->dims[d]; size_t n=total<max?total:max;
    unsigned char*b=g->p+g->data_start+ti->off; float*out=malloc(n*4);
    if(ti->typ==0){ for(size_t i=0;i<n;i++) out[i]=*(float*)(b+i*4); }
    else if(ti->typ==1){ for(size_t i=0;i<n;i++) out[i]=f16f(*(uint16_t*)(b+i*2)); }
    else if(ti->typ==8){ size_t nb=(n+31)/32; for(size_t bl=0;bl<nb;bl++){ float sc=f16f(*(uint16_t*)(b+bl*34));
        for(int j=0;j<32 && bl*32+j<n;j++) out[bl*32+j]=(float)(int8_t)b[bl*34+2+j]*sc; } }
    else { printf("dequant type %u unsupported\n",ti->typ); exit(1); }
    return out;
}

/* ===================== model ===================== */
typedef struct { float*attn_norm,*ffn_norm,*q_norm,*k_norm; Lin q,k,v,o,gate,up,down; } Layer;
typedef struct { int d,nl,nh,nkv,hd,ffn,vocab,nt; float rope_base,eps; float*tok_embd,*out_norm; Layer*L; Lin lm; } Model;

static void rmsnorm(float*o,const float*x,const float*w,int n,float eps){ float s=0; for(int i=0;i<n;i++)s+=x[i]*x[i]; s=1.0f/sqrtf(s/n+eps); for(int i=0;i<n;i++)o[i]=x[i]*s*w[i]; }
static void rope(float*v,int hd,int pos,float base){ for(int i=0;i<hd/2;i++){ float fr=powf(base,-2.0f*i/hd),a=pos*fr,c=cosf(a),s=sinf(a),x=v[i],y=v[i+hd/2]; v[i]=x*c-y*s; v[i+hd/2]=x*s+y*c; } }
static void softmax(float*x,int n){ float m=-1e30f; for(int i=0;i<n;i++)if(x[i]>m)m=x[i]; float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} float inv=1.0f/s; for(int i=0;i<n;i++)x[i]*=inv; }

static void model_load(Model*m,Gguf*g,int nt){
    m->d=g->embd; m->nl=g->block_count; m->nh=g->nh; m->nkv=g->nkv; m->hd=g->hd; m->ffn=g->ffn; m->vocab=g->vocab;
    m->rope_base=g->rope_base; m->eps=1e-6f; m->nt=nt;
    int qd=m->nh*m->hd, kvd=m->nkv*m->hd;
    fprintf(stderr,"packing token_embd + lm_head ... "); fflush(stderr);
    m->tok_embd=gguf_dequant(g,"token_embd.weight",(size_t)-1);
    m->out_norm=gguf_dequant(g,"output_norm.weight",(size_t)-1);
    m->lm=lin_new(m->tok_embd, m->vocab, m->d); /* tied */
    fprintf(stderr,"done\n");
    m->L=malloc(m->nl*sizeof(Layer));
    for(int l=0;l<m->nl;l++){ char nm[64]; Layer*ly=&m->L[l];
        #define DQ(suf) ({ snprintf(nm,64,"blk.%d.%s",l,suf); gguf_dequant(g,nm,(size_t)-1); })
        #define LN(suf,N,K) ({ snprintf(nm,64,"blk.%d.%s",l,suf); float*w=gguf_dequant(g,nm,(size_t)-1); Lin lin=lin_new(w,N,K); free(w); lin; })
        ly->attn_norm=DQ("attn_norm.weight"); ly->ffn_norm=DQ("ffn_norm.weight");
        ly->q_norm=DQ("attn_q_norm.weight"); ly->k_norm=DQ("attn_k_norm.weight");
        ly->q=LN("attn_q.weight",qd,m->d); ly->k=LN("attn_k.weight",kvd,m->d); ly->v=LN("attn_v.weight",kvd,m->d);
        ly->o=LN("attn_output.weight",m->d,qd);
        ly->gate=LN("ffn_gate.weight",m->ffn,m->d); ly->up=LN("ffn_up.weight",m->ffn,m->d); ly->down=LN("ffn_down.weight",m->d,m->ffn);
        fprintf(stderr,"\rpacked layer %d/%d",l+1,m->nl);
    }
    fprintf(stderr,"\n");
}

/* KV cache: per layer grows by kvd each pos */
typedef struct { float*Kc,*Vc; int kvd,ctx; } Kv;
static void forward(Model*m,int tok,int pos,Kv*kv,float*logits,
                    float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*g,float*u,int8_t*xt){
    int d=m->d,nh=m->nh,nkv=m->nkv,hd=m->hd,ffn=m->ffn,nt=m->nt,qd=nh*hd,kvd=nkv*hd,gpr=nh/nkv;
    static float*h=NULL; if(!h) h=malloc(d*4);
    memcpy(h, m->tok_embd+(size_t)tok*d, d*4);
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        rmsnorm(hn,h,ly->attn_norm,d,m->eps);
        lin_mm(&ly->q,hn,q,nt,xt); lin_mm(&ly->k,hn,k,nt,xt); lin_mm(&ly->v,hn,vv,nt,xt);
        for(int hh=0;hh<nh;hh++){ rmsnorm(q+hh*hd,q+hh*hd,ly->q_norm,hd,m->eps); rope(q+hh*hd,hd,pos,m->rope_base); }
        for(int hh=0;hh<nkv;hh++){ rmsnorm(k+hh*hd,k+hh*hd,ly->k_norm,hd,m->eps); rope(k+hh*hd,hd,pos,m->rope_base); }
        float*Kc=kv->Kc+(size_t)l*kv->ctx*kvd, *Vc=kv->Vc+(size_t)l*kv->ctx*kvd;
        memcpy(Kc+(size_t)pos*kvd,k,kvd*4); memcpy(Vc+(size_t)pos*kvd,vv,kvd*4);
        float scale=1.0f/sqrtf(hd);
        for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q+hh*hd; float*sc=tmp;
            for(int j=0;j<=pos;j++){ float*kj=Kc+(size_t)j*kvd+kvh*hd,dd=0; for(int t=0;t<hd;t++)dd+=qh[t]*kj[t]; sc[j]=dd*scale; }
            softmax(sc,pos+1);
            float*oh=att+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
            for(int j=0;j<=pos;j++){ float w=sc[j],*vj=Vc+(size_t)j*kvd+kvh*hd; for(int t=0;t<hd;t++)oh[t]+=w*vj[t]; } }
        lin_mm(&ly->o,att,tmp,nt,xt); for(int i=0;i<d;i++)h[i]+=tmp[i];
        rmsnorm(hn,h,ly->ffn_norm,d,m->eps);
        lin_mm(&ly->gate,hn,g,nt,xt); lin_mm(&ly->up,hn,u,nt,xt);
        for(int i=0;i<ffn;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }
        lin_mm(&ly->down,g,tmp,nt,xt); for(int i=0;i<d;i++)h[i]+=tmp[i];
    }
    rmsnorm(hn,h,m->out_norm,d,m->eps); lin_mm(&m->lm,hn,logits,nt,xt);
}
static int argmax(const float*l,int n){ int b=0; float bv=l[0]; for(int i=1;i<n;i++)if(l[i]>bv){bv=l[i];b=i;} return b; }

/* GPT-2 byte-level decode of a token id -> append raw bytes */
static uint8_t g_bdec[0x200]; static int g_bdec_init=0;
static void bdec_init(void){ int cs[256]; int bs[256]; int nn=0; for(int i=0;i<256;i++)bs[i]=-1;
    int idx=0; for(int b=0x21;b<=0x7e;b++){cs[idx]=b;bs[b]=b;idx++;} for(int b=0xa1;b<=0xac;b++){cs[idx]=b;bs[b]=b;idx++;} for(int b=0xae;b<=0xff;b++){cs[idx]=b;bs[b]=b;idx++;}
    for(int b=0;b<256;b++) if(bs[b]<0){ cs[idx]=256+nn; bs[b]=256+nn; nn++; idx++; }
    /* map unicode codepoint -> byte */ memset(g_bdec,0,sizeof(g_bdec));
    for(int b=0;b<256;b++){ int cp=bs[b]; if(cp<0x200) g_bdec[cp]=(uint8_t)b; }
    g_bdec_init=1; }
static void tok_print(Gguf*g,int id){ if(!g_bdec_init)bdec_init(); if(id<0||id>=g->ntok)return; char*s=g->tok[id];
    for(int i=0;s[i];){ /* decode one UTF-8 codepoint */ unsigned c=(unsigned char)s[i]; unsigned cp;
        if(c<0x80){cp=c;i+=1;} else if((c>>5)==6){cp=((c&0x1f)<<6)|((unsigned char)s[i+1]&0x3f);i+=2;}
        else if((c>>4)==14){cp=((c&0xf)<<12)|(((unsigned char)s[i+1]&0x3f)<<6)|((unsigned char)s[i+2]&0x3f);i+=3;} else {cp=c;i+=1;}
        if(cp<0x200) putchar(g_bdec[cp]); } }

int main(int c,char**v){
    if(c<2){ printf("usage: %s model.gguf [ngen] [nt]\n",v[0]); return 1; }
    int ngen=(c>2)?atoi(v[2]):16, nt=(c>3)?atoi(v[3]):4;
    bind_ai(); { cpu_set_t s;CPU_ZERO(&s);CPU_SET(8,&s);sched_setaffinity(0,sizeof(s),&s);} for(int i=0;i<5;i++)sched_yield();
    Gguf g; double t0=now(); gguf_open(&g,v[1]);
    fprintf(stderr,"GGUF: %d layers embd=%d ffn=%d heads=%d/%d hd=%d vocab=%d rope=%g  (parse %.1fs)\n",
        g.block_count,g.embd,g.ffn,g.nh,g.nkv,g.hd,g.vocab,g.rope_base,now()-t0);
    Model m; double tl=now(); model_load(&m,&g,nt); fprintf(stderr,"loaded in %.1fs\n",now()-tl);

    int ctx=64; Kv kv; kv.kvd=m.nkv*m.hd; kv.ctx=ctx;
    kv.Kc=calloc((size_t)m.nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m.nl*ctx*kv.kvd,4);
    int d=m.d,qd=m.nh*m.hd,ffn=m.ffn,maxk=ffn>qd?ffn:qd;
    float*hn=malloc(d*4),*q=malloc(qd*4),*k=malloc(kv.kvd*4),*vv=malloc(kv.kvd*4),*att=malloc(qd*4),
         *tmp=malloc((size_t)(ffn>ctx?ffn:ctx)*4),*gg=malloc(ffn*4),*u=malloc(ffn*4),*logits=malloc((size_t)m.vocab*4);
    int8_t*xt=malloc((size_t)(maxk/K0)*TILE);

    /* "The capital of France is Paris. The capital of Japan is" (llama-tokenize) */
    int prompt[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374}; int np=12;
    double tp=now(); int first=0;
    for(int p=0;p<np;p++){ forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,xt); if(p==np-1) first=argmax(logits,m.vocab); }
    printf("\nprompt      : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]);
    printf("\nfirst argmax: %d ('",first); tok_print(&g,first); printf("')  expect 26194 (' Tokyo') -> %s\n", first==26194?"PASS":"FAIL");
    printf("prefill %.2fs (%d tok)\n", now()-tp, np);
    printf("generation  : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]); tok_print(&g,first);
    int cur=first; double tg=now();
    for(int s=0;s<ngen;s++){ int pos=np+s; forward(&m,cur,pos,&kv,logits,hn,q,k,vv,att,tmp,gg,u,xt); cur=argmax(logits,m.vocab); tok_print(&g,cur); }
    double dt=now()-tg; printf("\ndecode: %.2f tok/s (IME-2 int8 W8A8, nt=%d)\n", ngen/dt, nt);
    return 0;
}
