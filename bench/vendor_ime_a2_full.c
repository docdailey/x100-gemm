/* vendor_ime_a2_full.c — A2: full validation of the REAL M=1 decode kernel.
 *
 * Ground truth (not guessed): dispatch chain in reference/spacemit-backend/ime.cpp confirms
 * for Q4_0/Q4_1, INTER_SIZE==256, count_m<4: gemm_kernel_i8i4_hp -> gemm_kernel_i8i4_hp_m1
 * (ime2_kernels.cpp:2883), paired with quantize_a_row_i8_hp (rvv_kernels.cpp:1989) for A and
 * make_block_q4_0x32 (repack.cpp:292) for B. This supersedes the earlier gemm_kernel_i8i4_m1
 * port (wrong function, wrong A-pack, wrong B nibble-pairing -- see research_feed_paths.md §12
 * A2 row for the postmortem).
 *
 * A record per 256-wide block (290B, confirmed against the asm's own "+272"/"+288" offsets):
 *   8x [2B fp16 subblk-scale][32B int8 data]  (272B)
 *   8x fp16 asum, PRE-SCALED as -true_asum*8.0                                (16B, at +272)
 *   1x fp16 block-average scale (= mean of the 8 subblk scale_temp values)     (2B, at +288)
 * True per-subblock dequant scale = stored_subblk_scale[kk] * block_avg_scale.
 *
 * B record (block_q4_0x32, 576B): 32x fp16 native-Q4_0 scale (64B) + 512B int4 data, where a
 * row's 16 native Q4_0 bytes are RE-PAIRED (not the native ggml j/j+16 pairing) into adjacent
 * pairs {2j,2j+1} for j=0..7 (elements 0-15) then {16+2j,17+2j} for j=0..7 (elements 16-31).
 * Nibble value = signed_int4 + 8 (ggml Q4_0 native convention, unchanged by repack).
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o vendor_ime_a2_full vendor_ime_a2_full.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* our own current N8K16-tile int4 kernel (qwen_moe.c gemv_nb_int4_grouped/pack_w_int4), copied
 * verbatim for an apples-to-apples A3 hot-loop timing comparison at identical N=32,K=256. */
#define OUR_N0 8
#define OUR_K0 16
#define OUR_TILE 128
static void our_pack_w_int4(int Nt,int Kt,const int8_t*W,uint8_t*Wq){
    int Nb=Nt/OUR_N0, Kb=Kt/OUR_K0, Kp=Kb/2;
    for(int nb=0;nb<Nb;nb++)for(int kp=0;kp<Kp;kp++){
        int8_t t0[OUR_TILE],t1[OUR_TILE]; int kb0=2*kp,kb1=2*kp+1;
        for(int n=0;n<OUR_N0;n++)for(int k=0;k<OUR_K0;k++){ t0[n*OUR_K0+k]=W[(nb*OUR_N0+n)*Kt+kb0*OUR_K0+k]; t1[n*OUR_K0+k]=W[(nb*OUR_N0+n)*Kt+kb1*OUR_K0+k]; }
        uint8_t*d=Wq+((size_t)(nb*Kp+kp))*OUR_TILE;
        for(int j=0;j<OUR_TILE;j++) d[j]=(uint8_t)((t0[j]&0xf)|((t1[j]&0xf)<<4));
    }
}
__attribute__((noinline,optimize("no-tree-vectorize","no-stack-protector")))
static void our_gemv_nb_int4_grouped(const int8_t*xt,const uint8_t*Wq,int Kb,int32_t*part){
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

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

#define N 32
#define K 256
#define NSUB 8   /* 256/32 */

/* ---- fp16 helpers (portable, no _Float16 arithmetic assumptions) ---- */
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

/* ---- A pack: portable scalar port of quantize_a_row_i8_hp (rvv_kernels.cpp:1989, vlenb==128 path) ---- */
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
        /* stored subblock scale = scale_temp[kk]*scale_factor, in fp16 */
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor);
        memcpy(out+kk*34,&ssub,2); memcpy(out+kk*34+2,qa[kk],32);
    }
    for(int kk=0;kk<NSUB;kk++){
        float trueasum = (float)asum[kk];
        uint16_t as = f32_to_f16(-trueasum*8.0f);
        memcpy(out+272+kk*2,&as,2);
    }
    uint16_t blkscale=f32_to_f16(scale_avg);
    memcpy(out+288,&blkscale,2);
}

/* ---- ref Q4_0 row block: native ggml layout (d fp16 scale; qs[16], nibble=val+8, byte l holds
 * elements {l, l+16} as {lo,hi}) ---- */
typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_native;

static void quantize_q4_0_native(const float*w /*32 elems*/, q4_0_native*out){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(w[i]); if(v>amax)amax=v; }
    float d=amax/8.0f; float inv = d? 1.0f/d : 0.0f;
    int8_t q[32];
    for(int i=0;i<32;i++){ int v=(int)lrintf(w[i]*inv); if(v>7)v=7; if(v<-8)v=-8; q[i]=(int8_t)v; }
    out->d=f32_to_f16(d);
    for(int l=0;l<16;l++){ uint8_t lo=(uint8_t)(q[l]+8), hi=(uint8_t)(q[l+16]+8); out->qs[l]=(lo&0xf)|((hi&0xf)<<4); }
}

/* ---- B pack: port of make_block_q4_0x32 (repack.cpp:292) — repacks 32 rows' native Q4_0
 * blocks (one K32 group) into the N32-panel layout the .hp kernel expects. ---- */
static void pack_B_q4_0x32(q4_0_native rows[32], uint8_t*out /* 576 bytes: 64B scale + 512B data */){
    for(int i=0;i<32;i++) memcpy(out+i*2,&rows[i].d,2);
    uint8_t*qs=out+64;
    for(int i=0;i<32;i++){
        for(int j=0;j<8;j++) qs[i*16+j]     = (rows[i].qs[j*2]&0x0F) | ((rows[i].qs[j*2+1]&0x0F)<<4);
        for(int j=0;j<8;j++) qs[i*16+8+j]   = ((rows[i].qs[j*2]&0xF0)>>4) | (rows[i].qs[j*2+1]&0xF0);
    }
}

/* ---- verbatim asm port of gemm_kernel_i8i4_hp_m1 (ime2_kernels.cpp:2883), k_blks=1, count_n=32 ---- */
static void run_hp_m1(const uint8_t*a_data /*290B*/, const uint8_t*b_data /*8*576B, one per K32-subblk*/, float*dst_c){
    long k_blks=1;
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

int main(void){
    bind_ai(); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs);
    sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield();

    srand(12345);
    static float W[N][K], x[K];
    for(int i=0;i<N;i++)for(int k=0;k<K;k++) W[i][k]=((rand()%2001)-1000)/1000.0f;
    for(int k=0;k<K;k++) x[k]=((rand()%2001)-1000)/1000.0f;

    /* quantize W into native Q4_0 (8 subblocks/row), and A into the hp record */
    static q4_0_native qW[NSUB][N]; /* [k32-group][row] */
    for(int kk=0;kk<NSUB;kk++) for(int i=0;i<N;i++) quantize_q4_0_native(&W[i][kk*32], &qW[kk][i]);
    uint8_t Arec[290]; pack_A_hp(x,Arec);
    static uint8_t Brec[NSUB][576];
    for(int kk=0;kk<NSUB;kk++) pack_B_q4_0x32(qW[kk], Brec[kk]);

    /* dequant reference: reconstruct the QUANTIZED W/x (not the original random floats) and do
     * a plain scalar dot -- isolates kernel correctness from quantization noise. */
    float yref[N]={0};
    /* recover A's true per-subblock scale + quantized values from Arec (round-trip our own pack) */
    for(int kk=0;kk<NSUB;kk++){
        uint16_t ssub; memcpy(&ssub,Arec+kk*34,2); float sub_ratio=f16_to_f32(ssub);
        uint16_t blks; memcpy(&blks,Arec+288,2); float blk_avg=f16_to_f32(blks);
        float a_scale = sub_ratio*blk_avg;
        int8_t qa[32]; memcpy(qa,Arec+kk*34+2,32);
        for(int i=0;i<N;i++){
            float d = f16_to_f32(qW[kk][i].d);
            float acc=0;
            for(int c=0;c<32;c++){
                int lo = qW[kk][i].qs[c%16]&0xf, hi=(qW[kk][i].qs[c%16]>>4)&0xf;
                int nib = (c<16)? lo : hi;
                int wsigned = nib-8;
                acc += (float)qa[c]*a_scale * (float)wsigned*d;
            }
            yref[i]+=acc;
        }
    }

    float ykernel[N]={0};
    const uint8_t*bptrs[NSUB]; for(int kk=0;kk<NSUB;kk++) bptrs[kk]=Brec[kk];
    /* the real kernel expects B as one contiguous k_blks*576B stream; NSUB=8 subblocks share
     * one outer 256-wide block (k_blks=1 covers all 8 inner iterations) so concat them. */
    static uint8_t Bcat[NSUB*576];
    for(int kk=0;kk<NSUB;kk++) memcpy(Bcat+kk*576,Brec[kk],576);
    run_hp_m1(Arec,Bcat,ykernel);

    printf("row  yref        ykernel     absdiff   reldiff\n");
    double maxrel=0, maxabs=0;
    for(int i=0;i<N;i++){
        float d=ykernel[i]-yref[i]; float ad=fabsf(d); float rel=ad/(fabsf(yref[i])+1e-6f);
        if(ad>maxabs)maxabs=ad; if(rel>maxrel)maxrel=rel;
        printf("%3d  %10.4f  %10.4f  %8.4f  %7.4f\n", i, yref[i], ykernel[i], ad, rel);
    }
    printf("max abs diff=%.5f  max rel diff=%.5f  -> %s\n", maxabs, maxrel, maxrel<0.05?"PASS (kernel matches dequant, within quant/fp16 noise)":"FAIL (mismatch beyond expected quant/fp16 noise)");

    /* ---- A3: hot A/B, identical N=32,K=256, in-cache repeated calls, kernel-only (no packing) ---- */
    int8_t ourW[N][K]; for(int i=0;i<N;i++)for(int k=0;k<K;k++) ourW[i][k]=(int8_t)((rand()%15)-7);
    int Nb=N/OUR_N0, Kb=K/OUR_K0, Kp=Kb/2;
    uint8_t*ourWq=malloc((size_t)Nb*Kb*OUR_TILE); our_pack_w_int4(N,K,(int8_t*)ourW,ourWq);
    int8_t ourXt[K/OUR_K0*OUR_TILE]; memset(ourXt,0,sizeof ourXt);
    for(int c=0;c<K;c++) ourXt[(c/OUR_K0)*OUR_TILE+(c%OUR_K0)]=(int8_t)((rand()%255)-128);
    int32_t ourPart[64];

    const int REPS=200000;
    double t0=now();
    for(int r=0;r<REPS;r++) for(int nb=0;nb<Nb;nb++) our_gemv_nb_int4_grouped(ourXt, ourWq+(size_t)nb*Kb*OUR_TILE, Kb, ourPart);
    double t_ours=now()-t0;

    double t1=now();
    for(int r=0;r<REPS;r++) run_hp_m1(Arec,Bcat,ykernel);
    double t_vendor=now()-t1;

    printf("\n=== A3: hot kernel-only timing, N=32 K=256, %d reps ===\n",REPS);
    printf("ours (4x N8K256 gemv_nb_int4_grouped): %.3f ns/call  (%.2f Mcalls/s)\n", t_ours/REPS*1e9, REPS/t_ours/1e6);
    printf("vendor (1x hp_m1, N32K256):             %.3f ns/call  (%.2f Mcalls/s)\n", t_vendor/REPS*1e9, REPS/t_vendor/1e6);
    printf("speedup (vendor faster by): %.2fx\n", t_ours/t_vendor);
    return 0;
}
