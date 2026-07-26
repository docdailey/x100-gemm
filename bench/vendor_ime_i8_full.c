/* vendor_ime_i8_full.c — A2-equivalent for the vendor int8 M1 kernel (gemm_kernel_i8i8_m1,
 * ime2_kernels.cpp:4773), proposed as a router precision option between int4 (fast, 58.9% of
 * routing decisions perturbed) and fp32 (exact, slow). Ground truth, not guessed:
 *   - B pack: make_block_q8_0x32 (repack.cpp:357) is a PLAIN row-major memcpy per row -- no
 *     nibble interleaving needed (unlike int4), so byte[i*32+j] = row i, K-index j, straight.
 *   - A pack: the SAME simple format the first (wrong) int4 attempt assumed -- fp32 scale +
 *     int16 asum + 32B int8 data, 38B/group -- this kernel genuinely uses that simple format
 *     (no two-level fp16 scale scheme; that was specific to the int4 .hp kernel).
 *   - Kernel: plain typed `vmadot ...,i8` (signed x signed) -- no zero-point trickery at all,
 *     unlike int4's unsigned-nibble+implicit-zp=8 convention.
 * Validated the same way as the int4 kernel: independent scalar dequant oracle built from the
 * same quantized values, isolating kernel correctness from quantization noise.
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o vendor_ime_i8_full vendor_ime_i8_full.c -lm
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
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

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

#define N 32
#define K 256
#define NSUB 8

/* ---- B pack: Q8_0-native quantize (fp16 scale + signed int8, standard, no zp) then plain
 * row-major pack into an N32-panel (make_block_q8_0x32: straight memcpy per row). ---- */
typedef struct { uint16_t d; int8_t qs[32]; } q8_0_native;
static void quantize_q8_0_native(const float*w, q8_0_native*out){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(w[i]); if(v>amax)amax=v; }
    float d=amax/127.0f; float inv=d?1.0f/d:0.0f;
    out->d=f32_to_f16(d);
    for(int i=0;i<32;i++){ int q=(int)lrintf(w[i]*inv); if(q>127)q=127; if(q<-128)q=-128; out->qs[i]=(int8_t)q; }
}
static void pack_B_q8_0x32(q8_0_native rows[32], uint8_t*out /* 64B scale + 1024B data = 1088 */){
    for(int i=0;i<32;i++) memcpy(out+i*2,&rows[i].d,2);
    uint8_t*qs=out+64;
    for(int i=0;i<32;i++) memcpy(qs+i*32, rows[i].qs, 32);
}

/* ---- A pack: simple format -- fp32 scale, int16 asum, 32B int8 data, per K32-group. ---- */
static void pack_A_i8(const float*x, uint8_t*out /* 38 bytes */){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(x[i]); if(v>amax)amax=v; }
    float scale=amax/127.0f; float inv=scale?1.0f/scale:0.0f;
    int8_t q[32]; int32_t sum=0;
    for(int i=0;i<32;i++){ int v=(int)lrintf(x[i]*inv); if(v>127)v=127; if(v<-128)v=-128; q[i]=(int8_t)v; sum+=v; }
    int16_t asum=(int16_t)sum;
    memcpy(out,&scale,4); memcpy(out+4,&asum,2); memcpy(out+6,q,32);
}

/* ---- verbatim asm port of gemm_kernel_i8i8_m1, generalized to k_blks superblocks ---- */
static void run_i8_m1(const uint8_t*a_data, const uint8_t*b_data, float*dst_c, long k_blks){
    __asm__ volatile(
        "mv           t3, %[BCK]              \n\t"
        "mv           t4, %[NBLKS]            \n\t"
        "mv           s2, %[pA]               \n\t"
        "addi         s3, %[pA], 4+2          \n\t"
        "mv           s4, %[pB]               \n\t"
        "addi         s5, %[pB], 32*2         \n\t"
        "mv           s6, %[pC]               \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vxor.vv      v2, v0, v0              \n\t"

        ".align 4                             \n\t"
        "_K_LPST%=:                           \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vl4r.v       v4, (s5)                \n\t"
        "addi         s5, s5, 128*4           \n\t"
        "vl4r.v       v8, (s5)                \n\t"
        "addi         s5, s5, 128*4+64        \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v0, (s4)                \n\t"
        "addi         s4, s4, 64+128*8        \n\t"

        "vsetvli      t0, x0, e8, mf4         \n\t"
        "vle8.v       v3, (s3)                \n\t"
        "addi         s3, s3, 32+6            \n\t"

        "flw          f0, (s2)                \n\t"
        "addi         s2, s2, 6+32            \n\t"

        "vsetvli      t0, zero, e32, m1       \n\t"
        "vupack.vv    v24, v4, v5, 1          \n\t"
        "vupack.vv    v26, v6, v7, 1          \n\t"
        "vupack.vv    v28, v8, v9, 1          \n\t"
        "vupack.vv    v30, v10, v11, 1        \n\t"

        "vslidedown.vi  v4, v3, 4             \n\t"

        "vxor.vv      v16, v16, v16           \n\t"
        "vxor.vv      v18, v16, v16           \n\t"
        "vxor.vv      v20, v16, v16           \n\t"
        "vxor.vv      v22, v16, v16           \n\t"

        "vmadot       v16, v3, v24, i8         \n\t"
        "vmadot       v18, v3, v26, i8         \n\t"
        "vmadot       v20, v3, v28, i8         \n\t"
        "vmadot       v22, v3, v30, i8         \n\t"

        "vmadot       v16, v4, v25, i8         \n\t"
        "vmadot       v18, v4, v27, i8         \n\t"
        "vmadot       v20, v4, v29, i8         \n\t"
        "vmadot       v22, v4, v31, i8         \n\t"

        "vpack.vv     v24, v16, v18, 2        \n\t"
        "vpack.vv     v26, v20, v22, 2        \n\t"
        "vpack.vv     v16, v24, v26, 3        \n\t"

        "vsetvli      t0, x0, e16, mf2        \n\t"
        "vfwcvt.f.f.v v24, v0                 \n\t"
        "vsetvli      t0, x0, e32, m1         \n\t"
        "vfcvt.f.x.v  v26, v16                \n\t"
        "vfmul.vf     v1, v24, f0             \n\t"
        "vfmacc.vv    v2, v1, v26             \n\t"

        "addi         t3, t3, -1              \n\t"
        "bgtz         t3, _K_LPST%=           \n\t"
        "_K_LPND%=:                           \n\t"

        "_ST32%=:                             \n\t"
        "vsetvli      t0, t4, e32, m1         \n\t"
        "vse32.v      v2, (s6)                \n\t"
        "_FUNC_END%=:                         \n\t"

        :
        : [BCK] "r"(k_blks), [NBLKS] "r"(32L), [pA] "r"(a_data), [pB] "r"(b_data), [pC] "r"(dst_c)
        : "cc","t0","t3","t4","f0","s2","s3","s4","s5","s6",
          "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
          "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory");
}

int main(void){
    bind_ai(); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs);
    sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield();

    srand(12345);
    static float W[N][K], x[K];
    for(int i=0;i<N;i++)for(int k=0;k<K;k++) W[i][k]=((rand()%2001)-1000)/1000.0f;
    for(int k=0;k<K;k++) x[k]=((rand()%2001)-1000)/1000.0f;

    static q8_0_native qW[NSUB][N];
    for(int kk=0;kk<NSUB;kk++) for(int i=0;i<N;i++) quantize_q8_0_native(&W[i][kk*32], &qW[kk][i]);
    static uint8_t Brec[NSUB][1088];
    for(int kk=0;kk<NSUB;kk++) pack_B_q8_0x32(qW[kk], Brec[kk]);
    static uint8_t Bcat[NSUB*1088];
    for(int kk=0;kk<NSUB;kk++) memcpy(Bcat+kk*1088, Brec[kk], 1088);

    static uint8_t Arec[NSUB][38];
    for(int kk=0;kk<NSUB;kk++) pack_A_i8(x+kk*32, Arec[kk]);
    static uint8_t Acat[NSUB*38];
    for(int kk=0;kk<NSUB;kk++) memcpy(Acat+kk*38, Arec[kk], 38);

    /* dequant reference from the SAME quantized values (isolates kernel correctness from
     * quantization noise) */
    float yref[N]={0};
    for(int kk=0;kk<NSUB;kk++){
        uint8_t*ar=Acat+kk*38; float a_scale; memcpy(&a_scale,ar,4); int8_t*qa=(int8_t*)(ar+6);
        for(int i=0;i<N;i++){
            float d=f16_to_f32(qW[kk][i].d); float acc=0;
            for(int c=0;c<32;c++) acc += (float)qa[c]*a_scale * (float)qW[kk][i].qs[c]*d;
            yref[i]+=acc;
        }
    }

    float ykernel[N]={0};
    run_i8_m1(Acat,Bcat,ykernel,NSUB);

    printf("row  yref        ykernel     absdiff   reldiff\n");
    double maxrel=0,maxabs=0;
    for(int i=0;i<N;i++){
        float d=ykernel[i]-yref[i]; float ad=fabsf(d); float rel=ad/(fabsf(yref[i])+1e-6f);
        if(ad>maxabs)maxabs=ad; if(rel>maxrel)maxrel=rel;
        printf("%3d  %10.4f  %10.4f  %8.4f  %7.4f\n", i, yref[i], ykernel[i], ad, rel);
    }
    printf("max abs diff=%.5f  max rel diff=%.5f  -> %s\n", maxabs, maxrel,
        maxrel<0.05?"PASS (kernel matches dequant, within quant noise)":"FAIL (mismatch beyond expected quant noise)");

    /* hot kernel-only timing vs the int4 kernel's already-measured 446.4ns/call (A3) */
    const int REPS=200000;
    double t0=now();
    for(int r=0;r<REPS;r++) run_i8_m1(Acat,Bcat,ykernel,NSUB);
    double dt=now()-t0;
    printf("hot: %.1f ns/call (N=32,K=256,%d reps) vs int4 kernel's 446.4ns/call (A3, same N,K)\n", dt/REPS*1e9, REPS);
    return 0;
}
