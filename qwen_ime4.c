/* qwen_ime4.c — DENSE Qwen3 decode, int4 weights (W4A8) — isolation test for per-channel int4.
 * Same as qwen_ime.c but int4 kernel + Q4_0/Q4_1 dequant. Run on Qwen3-4B-Q8_0 (int4 requant) to
 * test whether per-output-channel int4 alone breaks the model (fast 4B load, not the 15-min 30B).
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_ime4 qwen_ime4.c -lm
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
static __thread int g_pinned=0;
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(8+(tn*2)%8,&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

static void pack_w_int4(int Nt,int Kt,const int8_t*W,uint8_t*Wq){
    int Nb=Nt/N0, Kb=Kt/K0, Kp=Kb/2;
    for(int nb=0;nb<Nb;nb++)for(int kp=0;kp<Kp;kp++){ int8_t t0[TILE],t1[TILE]; int kb0=2*kp,kb1=2*kp+1;
        for(int n=0;n<N0;n++)for(int k=0;k<K0;k++){ t0[n*K0+k]=W[(nb*N0+n)*Kt+kb0*K0+k]; t1[n*K0+k]=W[(nb*N0+n)*Kt+kb1*K0+k]; }
        uint8_t*d=Wq+((size_t)(nb*Kp+kp))*TILE; for(int j=0;j<TILE;j++) d[j]=(uint8_t)((t0[j]&0xf)|((t1[j]&0xf)<<4)); }
}
/* per-GROUP int4: each interleaved iteration (2 K-blocks = 32 K = one group) writes its 8x8 int32
 * partial to `part`, so a per-(output,group) scale can be applied. Group=32 matches Q4_0. */
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

typedef struct { uint32_t typ; int nd; uint64_t dims[4]; uint64_t off; char name[64]; } TInfo;
typedef struct { unsigned char*p; uint64_t data_start; TInfo*t; int nt; char**tok; int ntok;
    int block_count, embd, ffn, nh, nkv, hd, vocab; float rope_base; } Gguf;
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
    FILE*f=fopen(path,"rb"); if(!f){perror("open");exit(1);} fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    g->p=malloc(sz); if(fread(g->p,1,sz,f)!=(size_t)sz){printf("read fail\n");exit(1);} fclose(f);
    unsigned char*p=g->p; size_t o=8; uint64_t ntensor=U64(p+o); o+=8; uint64_t nkv=U64(p+o); o+=8;
    g->rope_base=1e6f; g->tok=NULL; g->ntok=0;
    for(uint64_t i=0;i<nkv;i++){ uint64_t kl=U64(p+o); char key[128]; int c=kl<127?kl:127; memcpy(key,p+o+8,c); key[c]=0; o+=8+kl;
        uint32_t t=U32(p+o); o+=4;
        #define KEYIS(s) (strcmp(key,s)==0)
        if(t==9 && KEYIS("tokenizer.ggml.tokens")){ uint32_t et=U32(p+o); uint64_t n=U64(p+o+4); size_t oo=o+12;
            g->ntok=n; g->tok=malloc(n*sizeof(char*));
            for(uint64_t j=0;j<n;j++){ uint64_t l=U64(p+oo); char*s=malloc(l+1); memcpy(s,p+oo+8,l); s[l]=0; g->tok[j]=s; oo+=8+l; } o=oo; continue; }
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
static float* gguf_dequant(Gguf*g,const char*name){
    TInfo*ti=gguf_find(g,name); size_t n=1; for(int d=0;d<ti->nd;d++) n*=ti->dims[d];
    unsigned char*b=g->p+g->data_start+ti->off; float*out=malloc(n*4);
    if(ti->typ==0){ for(size_t i=0;i<n;i++) out[i]=*(float*)(b+i*4); }
    else if(ti->typ==1){ for(size_t i=0;i<n;i++) out[i]=f16f(*(uint16_t*)(b+i*2)); }
    else if(ti->typ==8){ size_t nb=(n+31)/32; for(size_t bl=0;bl<nb;bl++){ float sc=f16f(*(uint16_t*)(b+bl*34));
        for(int j=0;j<32 && bl*32+j<n;j++) out[bl*32+j]=(float)(int8_t)b[bl*34+2+j]*sc; } }
    else if(ti->typ==2){ size_t nb=(n+31)/32; for(size_t bl=0;bl<nb;bl++){ unsigned char*q=b+bl*18; float d=f16f(*(uint16_t*)q); unsigned char*qs=q+2;
        for(int j=0;j<32 && bl*32+j<n;j++){ int nib=(j<16)?(qs[j]&0xf):(qs[j-16]>>4); out[bl*32+j]=(nib-8)*d; } } }
    else if(ti->typ==3){ size_t nb=(n+31)/32; for(size_t bl=0;bl<nb;bl++){ unsigned char*q=b+bl*20; float d=f16f(*(uint16_t*)q),m=f16f(*(uint16_t*)(q+2)); unsigned char*qs=q+4;
        for(int j=0;j<32 && bl*32+j<n;j++){ int nib=(j<16)?(qs[j]&0xf):(qs[j-16]>>4); out[bl*32+j]=nib*d+m; } } }
    else { printf("dequant type %u unsupported\n",ti->typ); exit(1); }
    return out;
}

typedef struct { float*attn_norm,*ffn_norm,*q_norm,*k_norm; Lin q,k,v,o,gate,up,down; } Layer;
typedef struct { int d,nl,nh,nkv,hd,ffn,vocab,nt; float rope_base,eps; float*tok_embd,*out_norm; Layer*L; Lin lm; } Model;
static void rmsnorm(float*o,const float*x,const float*w,int n,float eps){ float s=0; for(int i=0;i<n;i++)s+=x[i]*x[i]; s=1.0f/sqrtf(s/n+eps); for(int i=0;i<n;i++)o[i]=x[i]*s*w[i]; }
static void rope(float*v,int hd,int pos,float base){ for(int i=0;i<hd/2;i++){ float fr=powf(base,-2.0f*i/hd),a=pos*fr,c=cosf(a),s=sinf(a),x=v[i],y=v[i+hd/2]; v[i]=x*c-y*s; v[i+hd/2]=x*s+y*c; } }
static void softmax(float*x,int n){ float m=-1e30f; for(int i=0;i<n;i++)if(x[i]>m)m=x[i]; float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} float inv=1.0f/s; for(int i=0;i<n;i++)x[i]*=inv; }
static void model_load(Model*m,Gguf*g,int nt){
    m->d=g->embd; m->nl=g->block_count; m->nh=g->nh; m->nkv=g->nkv; m->hd=g->hd; m->ffn=g->ffn; m->vocab=g->vocab;
    m->rope_base=g->rope_base; m->eps=1e-6f; m->nt=nt; int qd=m->nh*m->hd, kvd=m->nkv*m->hd;
    fprintf(stderr,"packing (int4) ... "); fflush(stderr);
    m->tok_embd=gguf_dequant(g,"token_embd.weight"); m->out_norm=gguf_dequant(g,"output_norm.weight"); m->lm=lin_new(m->tok_embd,m->vocab,m->d);
    fprintf(stderr,"done\n"); m->L=malloc(m->nl*sizeof(Layer));
    for(int l=0;l<m->nl;l++){ char nm[64]; Layer*ly=&m->L[l];
        #define DQ(suf) ({ snprintf(nm,64,"blk.%d.%s",l,suf); gguf_dequant(g,nm); })
        #define LN(suf,N,K) ({ snprintf(nm,64,"blk.%d.%s",l,suf); float*w=gguf_dequant(g,nm); Lin lin=lin_new(w,N,K); free(w); lin; })
        ly->attn_norm=DQ("attn_norm.weight"); ly->ffn_norm=DQ("ffn_norm.weight"); ly->q_norm=DQ("attn_q_norm.weight"); ly->k_norm=DQ("attn_k_norm.weight");
        ly->q=LN("attn_q.weight",qd,m->d); ly->k=LN("attn_k.weight",kvd,m->d); ly->v=LN("attn_v.weight",kvd,m->d); ly->o=LN("attn_output.weight",m->d,qd);
        ly->gate=LN("ffn_gate.weight",m->ffn,m->d); ly->up=LN("ffn_up.weight",m->ffn,m->d); ly->down=LN("ffn_down.weight",m->d,m->ffn);
        fprintf(stderr,"\rpacked layer %d/%d",l+1,m->nl); }
    fprintf(stderr,"\n");
}
typedef struct { float*Kc,*Vc; int kvd,ctx; } Kv;
static void forward(Model*m,int tok,int pos,Kv*kv,float*logits,float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*g,float*u,int8_t*xt){
    int d=m->d,nh=m->nh,nkv=m->nkv,hd=m->hd,ffn=m->ffn,nt=m->nt,qd=nh*hd,kvd=nkv*hd,gpr=nh/nkv;
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
        rmsnorm(hn,h,ly->ffn_norm,d,m->eps);
        lin_mm(&ly->gate,hn,g,nt,xt); lin_mm(&ly->up,hn,u,nt,xt);
        for(int i=0;i<ffn;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }
        lin_mm(&ly->down,g,tmp,nt,xt); for(int i=0;i<d;i++)h[i]+=tmp[i]; }
    rmsnorm(hn,h,m->out_norm,d,m->eps); lin_mm(&m->lm,hn,logits,nt,xt);
}
static int argmax(const float*l,int n){ int b=0; float bv=l[0]; for(int i=1;i<n;i++)if(l[i]>bv){bv=l[i];b=i;} return b; }
static uint8_t g_bdec[0x200]; static int g_bi=0;
static void bdec_init(void){ int cs[256],bs[256],nn=0; for(int i=0;i<256;i++)bs[i]=-1; int idx=0;
    for(int b=0x21;b<=0x7e;b++){cs[idx]=b;bs[b]=b;idx++;} for(int b=0xa1;b<=0xac;b++){cs[idx]=b;bs[b]=b;idx++;} for(int b=0xae;b<=0xff;b++){cs[idx]=b;bs[b]=b;idx++;}
    for(int b=0;b<256;b++)if(bs[b]<0){cs[idx]=256+nn;bs[b]=256+nn;nn++;idx++;} memset(g_bdec,0,sizeof(g_bdec));
    for(int b=0;b<256;b++){int cp=bs[b]; if(cp<0x200)g_bdec[cp]=(uint8_t)b;} g_bi=1; }
static void tok_print(Gguf*g,int id){ if(!g_bi)bdec_init(); if(id<0||id>=g->ntok)return; char*s=g->tok[id];
    for(int i=0;s[i];){ unsigned c=(unsigned char)s[i],cp; if(c<0x80){cp=c;i+=1;} else if((c>>5)==6){cp=((c&0x1f)<<6)|((unsigned char)s[i+1]&0x3f);i+=2;}
        else if((c>>4)==14){cp=((c&0xf)<<12)|(((unsigned char)s[i+1]&0x3f)<<6)|((unsigned char)s[i+2]&0x3f);i+=3;} else {cp=c;i+=1;} if(cp<0x200)putchar(g_bdec[cp]); } }
int main(int c,char**v){
    if(c<2){ printf("usage: %s model.gguf [ngen] [nt]\n",v[0]); return 1; }
    int ngen=(c>2)?atoi(v[2]):12, nt=(c>3)?atoi(v[3]):4;
    bind_ai(); { cpu_set_t s;CPU_ZERO(&s);CPU_SET(8,&s);sched_setaffinity(0,sizeof(s),&s);} for(int i=0;i<5;i++)sched_yield();
    Gguf g; gguf_open(&g,v[1]);
    fprintf(stderr,"GGUF: %d layers embd=%d ffn=%d heads=%d/%d hd=%d vocab=%d\n",g.block_count,g.embd,g.ffn,g.nh,g.nkv,g.hd,g.vocab);
    Model m; double tl=now(); model_load(&m,&g,nt); fprintf(stderr,"loaded %.1fs\n",now()-tl);
    int ctx=64; Kv kv; kv.kvd=m.nkv*m.hd; kv.ctx=ctx; kv.Kc=calloc((size_t)m.nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m.nl*ctx*kv.kvd,4);
    int d=m.d,qd=m.nh*m.hd,ffn=m.ffn,maxk=ffn>qd?ffn:qd;
    float*hn=malloc(d*4),*q=malloc(qd*4),*k=malloc(kv.kvd*4),*vv=malloc(kv.kvd*4),*att=malloc(qd*4),*tmp=malloc((size_t)(ffn>ctx?ffn:ctx)*4),*gg=malloc(ffn*4),*u=malloc(ffn*4),*logits=malloc((size_t)m.vocab*4);
    int8_t*xt=malloc((size_t)(maxk/K0)*TILE);
    int prompt[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374}; int np=12; int first=0;
    for(int p=0;p<np;p++){ forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,xt); if(p==np-1)first=argmax(logits,m.vocab); }
    printf("\nfirst argmax: %d ('",first); tok_print(&g,first); printf("')  expect 26194 (' Tokyo') -> %s\n", first==26194?"PASS":"FAIL");
    printf("generation  : "); for(int i=0;i<np;i++)tok_print(&g,prompt[i]); tok_print(&g,first);
    int cur=first; for(int s=0;s<ngen;s++){ int pos=np+s; forward(&m,cur,pos,&kv,logits,hn,q,k,vv,att,tmp,gg,u,xt); cur=argmax(logits,m.vocab); tok_print(&g,cur); }
    printf("\n"); return 0;
}
