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
 * Build: gcc -O3 -fno-tree-vectorize -march=rv64gcv_zvfh_xsmtvdotii -fopenmp -o qwen_moe_hp qwen_moe_hp.c -lm -lpthread
 * Run  : LD_LIBRARY_PATH=/usr/lib ./qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf [ngen] [nt]
 *
 * -fno-tree-vectorize is REQUIRED, not optional (codex_recs_1.md §22.16): this file hand-schedules
 * RVV vector-register state across custom vmadot asm blocks and RVV intrinsics; gcc's -O3
 * auto-vectorizer repeatedly collides with that state when applied to ordinary nearby scalar code
 * (three confirmed incidents this session: a SIGSEGV from an unrelated f32->f16 loop, a baseline
 * decode crash from new harness functions, and a ~4x slowdown -- not a crash -- on the int8-M1
 * router path from unguarded pack_A_i8/lin_mm_hp_worker_run, only found by comparing against this
 * flag). Per-function `__attribute__((noinline,optimize("no-tree-vectorize")))` on individual hot
 * functions was the original, narrower mitigation and is kept in place where already applied, but
 * proved incomplete -- disabling the pass for the whole translation unit is the systematic fix.
 * All hot-path vectorization in this file is explicit (RVV intrinsics or inline asm), so gcc's
 * auto-vectorizer was never buying real performance here; disabling it is close to free and,
 * empirically, was a net ~8% *speedup* (9.2->9.9 tok/s) by removing whatever pathological
 * interaction was slowing unrelated scalar code down.
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
#include <riscv_vector.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static __thread int g_pinned=0;
/* docs/HARDWARE.md: 4 IME-2 units, each shared by a core PAIR (8,9)(10,11)(12,13)(14,15); using
 * both cores of one pair is measured CONTENDED (7.31 TOPS/2 units) vs one-per-unit (13.09/4 units).
 * Old formula (8+(tn*2)%8) only worked by luck up to nt=4 (gives 8,10,12,14) and collides for
 * nt>4 (tn=4 maps back to hart 8). This table fills one-per-unit first, then the paired partners,
 * so it's collision-free and contention-aware for any nt in 1..8. */
static const int g_hart_order[8]={8,10,12,14,9,11,13,15};
static void pin_once(int tn){ if(g_pinned)return; bind_ai(); cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(g_hart_order[tn%8],&cs);sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

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
/* activation pack: RVV port of the real vendor quantize_a_row_i8_hp (rvv_kernels.cpp:1989,
 * vlenb==128 branch -- this board's A100 cores, VLEN=1024, one 32-wide subblock per e32m1 vsetvl).
 * Validated byte-identical against the prior scalar port across 200k random trials spanning 5
 * distributions (generic/tiny-magnitude/outlier/all-zero/alternating-sign), 3.90x faster hot
 * (bench/vendor_ime_actpack_probe.c). The all-zero-row case needed an explicit 1e-6f amax floor
 * to match the scalar port exactly -- the vendor code omits it (numerically inconsequential, an
 * all-zero block contributes 0 to the matmul regardless of the stored scale, but matching it here
 * keeps this a true drop-in rather than merely equivalent). MUST run on a hart that has called
 * bind_ai()+pinned to harts 8-15 (A100/VLEN=1024) -- on an unpinned/X100 hart (VLEN=256) the
 * vsetvl_e32m1(32) calls below silently clamp to vl=8, only packing 8 of each 32-wide subblock's
 * elements. All call sites (lin_mm, forward()'s router block) run on the main thread, which main()
 * already bind_ai()+pins to hart 8 before the decode loop starts. */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void pack_A_hp(const float*a, uint8_t*out /* 290 bytes */){
    float scale_temp[NSUB];
    float scale_avg=0.0f;
    for(int kk=0;kk<NSUB;kk++){
        size_t vl=__riscv_vsetvl_e32m1(32);
        vfloat32m1_t v_a=__riscv_vle32_v_f32m1(a+kk*32,vl);
        vfloat32m1_t v_a_abs=__riscv_vfabs_v_f32m1(v_a,vl);
        vfloat32m1_t tmp=__riscv_vfmv_v_f_f32m1(0.0f,vl);
        vfloat32m1_t v_a_max=__riscv_vfredmax_vs_f32m1_f32m1(v_a_abs,tmp,vl);
        float max_abs_a=__riscv_vfmv_f_s_f32m1_f32(v_a_max);
        if(max_abs_a<1e-6f) max_abs_a=1e-6f;
        scale_temp[kk]=max_abs_a/127.0f;
        scale_avg+=scale_temp[kk];
    }
    scale_avg/=NSUB;
    const float scale_factor = scale_avg? 1.0f/scale_avg : 0.0f;
    uint16_t blkscale=f32_to_f16(scale_avg);
    memcpy(out+288,&blkscale,2);

    for(int kk=0;kk<NSUB;kk++){
        uint8_t*base=out+kk*34;
        size_t vl=__riscv_vsetvl_e32m1(32);
        vfloat32m1_t v_a=__riscv_vle32_v_f32m1(a+kk*32,vl);
        float rep_scale_a = scale_temp[kk]? 1.0f/scale_temp[kk] : 0.0f;
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor);
        memcpy(base,&ssub,2);

        vfloat32m1_t v_a_scale = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
        vint16mf2_t  v_a_quant = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
        vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

        vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
        int16_t a_sum = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
        uint16_t as = f32_to_f16(-(float)a_sum*8.0f);
        memcpy(out+272+kk*2,&as,2);

        __riscv_vse8_v_i8mf4(base+2, v_a_quant_i8, vl);
    }
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

/* ===== vendor IME-2 int8 M1 kernel (gemm_kernel_i8i8_m1) -- validated in bench/vendor_ime_i8_full.c:
 * max abs diff 0.00000, max rel diff 0.00001 vs an independent dequant oracle. Genuinely simpler
 * than the int4 .hp kernel: plain signed vmadot (no zero-point trickery -- int8 has enough range
 * to store signed values directly), and the SIMPLE fp32-scale/int16-asum/32B-data A-format (38B
 * per K32-group, no two-level fp16 scheme -- that was specific to the int4 .hp kernel). Structural
 * difference from the int4 kernel: this loop is FLAT over K32-groups (k_blks = K/32), not nested
 * 256-wide-superblock-of-8-subblocks (int4's k_blks = K/256). B ground truth: make_block_q8_0x32
 * (repack.cpp:357) is a plain row-major memcpy per row, no nibble interleaving needed. */
#define BREC_I8 1088 /* bytes per 32-wide-K x 32-wide-N B record: 64B fp16 scale + 1024B int8 data */
typedef struct { uint16_t d; int8_t qs[32]; } q8_0_native;
static void quantize_q8_0_native(const float*w /*32 elems*/, q8_0_native*out){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(w[i]); if(v>amax)amax=v; }
    float d=amax/127.0f; float inv=d?1.0f/d:0.0f;
    out->d=f32_to_f16(d);
    for(int i=0;i<32;i++){ int q=(int)lrintf(w[i]*inv); if(q>127)q=127; if(q<-128)q=-128; out->qs[i]=(int8_t)q; }
}
static void pack_B_q8_0x32(q8_0_native rows[32], uint8_t*out /* 1088 bytes */){
    for(int i=0;i<32;i++) memcpy(out+i*2,&rows[i].d,2);
    uint8_t*qs=out+64;
    for(int i=0;i<32;i++) memcpy(qs+i*32, rows[i].qs, 32);
}
static void pack_A_i8(const float*x, uint8_t*out /* 38 bytes: 4B fp32 scale + 2B int16 asum + 32B int8 */){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(x[i]); if(v>amax)amax=v; }
    float scale=amax/127.0f; float inv=scale?1.0f/scale:0.0f;
    int8_t q[32]; int32_t sum=0;
    for(int i=0;i<32;i++){ int v=(int)lrintf(x[i]*inv); if(v>127)v=127; if(v<-128)v=-128; q[i]=(int8_t)v; sum+=v; }
    int16_t asum=(int16_t)sum;
    memcpy(out,&scale,4); memcpy(out+4,&asum,2); memcpy(out+6,q,32);
}
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

typedef struct { int N,K; uint8_t*B; } Lin; /* B: (N/32)*(K/256)*BSUPER bytes for int4-HP Lins; int8 Lins use BREC_I8*(K/32) per panel instead -- see lin_new_i8 */

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
/* int8 Lin: same (N,K,B) shape as the int4 Lin, but B is packed flat -- (N/32) panels, each a
 * contiguous run of (K/32) BREC_I8 (1088B) records, no 256-wide superblock grouping (that was
 * specific to int4's nested BLK_LOOP/INNER_BLK_LOOP structure; int8's kernel loop is flat over
 * K32-groups, k_blks = K/32). */
static Lin lin_new_i8(const float*wf32,int N,int K){
    Lin l; l.N=N; l.K=K; int Np=N/32, Kg=K/32;
    l.B=malloc((size_t)Np*Kg*BREC_I8);
    for(int np=0;np<Np;np++){
        for(int kg=0;kg<Kg;kg++){
            q8_0_native rows[32];
            for(int r=0;r<32;r++){
                int row=np*32+r;
                quantize_q8_0_native(wf32+(size_t)row*K + kg*32, &rows[r]);
            }
            pack_B_q8_0x32(rows, l.B + ((size_t)np*Kg+kg)*BREC_I8);
        }
    }
    return l;
}
static void pack_act_i8(const float*x,int K,uint8_t*Abuf){
    int Kg=K/32; for(int kg=0;kg<Kg;kg++) pack_A_i8(x+kg*32, Abuf+(size_t)kg*38);
}
static double gT_actpack=0, gT_lin=0, gT_attn=0, gT_rest=0, gT_rope=0, gT_router=0, gT_swiglu=0; static long gT_tok=0; static int gT_on=0;
/* linear(kernel) subclass breakdown (codex synthesis + PROGRESS next step): total gT_lin stays
 * the rest-accounting sink; these four partition it by consumer so we can see whether residual
 * vendor gap lives in QKV, O, expert FFN (gate/up/down), or lm_head. */
static double gT_lin_qkv=0, gT_lin_o=0, gT_lin_exp=0, gT_lin_lm=0;
enum { LIN_NONE=0, LIN_QKV, LIN_O, LIN_EXP, LIN_LM };
static int g_lin_class=LIN_NONE;
static void lin_add(double d){
    if(!gT_on) return;
    gT_lin+=d;
    if(g_lin_class==LIN_QKV) gT_lin_qkv+=d;
    else if(g_lin_class==LIN_O) gT_lin_o+=d;
    else if(g_lin_class==LIN_EXP) gT_lin_exp+=d;
    else if(g_lin_class==LIN_LM) gT_lin_lm+=d;
}
/* router precision validation counters (research_feed_paths.md router-quantization experiments) */
/* separate counters per experimental mode so one validate run reports both int4-vs-fp32 and
 * int8-vs-fp32 quality independently -- directly answers "which tradeoff is better" */
static long g_rtr_cmp=0;
static long g_rtr_hp_mismatch=0, g_rtr_hp_diffcount=0; static float g_rtr_hp_maxabs=0, g_rtr_hp_maxrel=0;
static long g_rtr_i8_mismatch=0, g_rtr_i8_diffcount=0; static float g_rtr_i8_maxabs=0, g_rtr_i8_maxrel=0;
static int g_router_validate=0; /* set from main() via env/arg */
/* router precision mode, EXPERIMENTAL, fp32 is the default: 0=fp32 (exact, matches the original
 * engine), 1=int4 HP-Lin (33% faster router bucket but 58.9% of routing decisions perturbed vs
 * fp32, avg 1.38/8 experts -- codex_recs_1.md §22.7), 2=int8 M1 (validated near-bit-exact
 * standalone, bench/vendor_ime_i8_full.c max rel diff 1e-5, but not yet measured for real
 * routing-quality impact on this model -- codex_recs_1.md §22.8). Set via 7th CLI arg. */
static int g_router_mode=0;

/* PR8 (codex_recs_1.md §17/§22.3): with the vendor kernel at ~446ns/call, the ~1392 fresh
 * #pragma omp parallel spawns/token that lin_mm_hp used to do dominated wall-clock (100.3ms,
 * 62%). Replace with a persistent spin-dispatch pool: threads created once, wait on a generation
 * counter instead of libgomp fork/join. Same round-robin panel partitioning as before (np=tn;
 * np<Np; np+=nt), so the actual math is unchanged -- only the dispatch mechanism differs.
 * Generalized (kind field) to also dispatch the int8 M1 kernel through the same pool. */
#define MAXNT 16
typedef struct { int kind; const Lin*l; const uint8_t*Abuf; float*y; int kb; } HpWork; /* kind: 0=int4 HP, 1=int8 M1 */
static _Atomic int g_pool_gen=0, g_pool_done=0;
static HpWork g_pool_work;
static int g_pool_nt=0;
static pthread_t g_pool_threads[MAXNT];

static void lin_mm_hp_worker_run(int tn){
    int Np=g_pool_work.l->N/32;
    if(g_pool_work.kind==0){
        for(int np=tn; np<Np; np+=g_pool_nt)
            run_hp_m1(g_pool_work.Abuf, g_pool_work.l->B+(size_t)np*g_pool_work.kb*BSUPER, g_pool_work.y+np*32, g_pool_work.kb);
    } else {
        for(int np=tn; np<Np; np+=g_pool_nt)
            run_i8_m1(g_pool_work.Abuf, g_pool_work.l->B+(size_t)np*g_pool_work.kb*BREC_I8, g_pool_work.y+np*32, g_pool_work.kb);
    }
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
    g_pool_work.kind=0; g_pool_work.l=l; g_pool_work.Abuf=Abuf; g_pool_work.y=y; g_pool_work.kb=l->K/256;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release); /* wake workers 1..nt-1 */
    lin_mm_hp_worker_run(0); /* main thread does its own share (tn=0) */
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < nt-1) { /* spin */ }
}
static void lin_mm_i8(const Lin*l,const uint8_t*Abuf,float*y,int nt){
    g_pool_work.kind=1; g_pool_work.l=l; g_pool_work.Abuf=Abuf; g_pool_work.y=y; g_pool_work.kb=l->K/32;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release);
    lin_mm_hp_worker_run(0);
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < nt-1) { /* spin */ }
}
static void lin_mm(const Lin*l,const float*x,float*y,int nt,uint8_t*Abuf){
    double _ta=gT_on?now():0;
    pack_act_hp(x,l->K,Abuf);
    double _tb=gT_on?now():0; if(gT_on) gT_actpack+=_tb-_ta;
    lin_mm_hp(l,Abuf,y,nt);
    if(gT_on) lin_add(now()-_tb);
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
typedef struct { float*attn_norm,*ffn_norm,*q_norm,*k_norm,*router; Lin router_hp,router_i8; Lin q,k,v,o; Lin*eg,*eu,*ed; } Layer;
typedef struct { int d,nl,nh,nkv,hd,vocab,nt,n_exp,n_act,moe; float rope_base,eps; float*tok_embd,*out_norm; Layer*L; Lin lm; } Model;

static void rmsnorm(float*o,const float*x,const float*w,int n,float eps){ float s=0; for(int i=0;i<n;i++)s+=x[i]*x[i]; s=1.0f/sqrtf(s/n+eps); for(int i=0;i<n;i++)o[i]=x[i]*s*w[i]; }
/* RoPE cos/sin depend only on (hd,pos,base) -- identical across every head AND every layer for a
 * given token (48 layers x up to 36 heads = up to 1728 redundant powf/sinf/cosf table builds per
 * token, all producing the same numbers). Build once per forward() call, apply per head. */
static void rope_table(float*cosb,float*sinb,int hd,int pos,float base){
    for(int i=0;i<hd/2;i++){ float fr=powf(base,-2.0f*i/hd),a=pos*fr; cosb[i]=cosf(a); sinb[i]=sinf(a); }
}
static void rope_apply(float*v,int hd,const float*cosb,const float*sinb){
    for(int i=0;i<hd/2;i++){ float c=cosb[i],s=sinb[i],x=v[i],y=v[i+hd/2]; v[i]=x*c-y*s; v[i+hd/2]=x*s+y*c; }
}
static void softmax(float*x,int n){ float m=-1e30f; for(int i=0;i<n;i++)if(x[i]>m)m=x[i]; float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} float inv=1.0f/s; for(int i=0;i<n;i++)x[i]*=inv; }
/* router matvec was the 2nd-biggest scalar bucket (29.4ms, measured): 128 experts x d=2048
 * unvectorized mults/layer, ~1MB/layer, cache-miss-heavy. RVV fp32 dot, vector-length-agnostic. */
static float vdot_f32(const float*a,const float*b,int n){
    vfloat32m1_t vacc=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t va=__riscv_vle32_v_f32m1(a+i,vl), vb=__riscv_vle32_v_f32m1(b+i,vl);
        vacc=__riscv_vfmacc_vv_f32m1(vacc,va,vb,vl); i+=vl; }
    vfloat32m1_t vzero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t vsum=__riscv_vfredusum_vs_f32m1_f32m1(vacc,vzero,__riscv_vsetvlmax_e32m1());
    return __riscv_vfmv_f_s_f32m1_f32(vsum);
}
/* y[i] += scale*x[i], vector-length-agnostic. Attention's QK-dot (vdot_f32, reused) and AV
 * weighted-accumulate (this) are the same unvectorized-dot/axpy patterns already vectorized for
 * the router matvec -- same technique, applied to the other scalar-C hot loop (item 5 of the
 * router review: "otherwise move to activation packing or attention"). */
static void vaxpy_f32(float*y,const float*x,float scale,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl), vy=__riscv_vle32_v_f32m1(y+i,vl);
        vy=__riscv_vfmacc_vf_f32m1(vy,scale,vx,vl);
        __riscv_vse32_v_f32m1(y+i,vy,vl); i+=vl; }
}
/* Router-as-HP-Lin experiment (research_feed_paths.md/codex_recs_1.md): router is itself a 128xd
 * Lin, same shape family as q/k/v/eg/eu/ed/lm. First attempt (retracted) hand-wrote a standalone
 * fp16-weight dot product run single-threaded on the main thread -- NOT representative of how
 * every other Lin in this engine actually runs. This version packs the router through lin_new_hp
 * (real int4 vendor format) and runs it through lin_mm_hp (pooled, multi-threaded, the proven
 * ~446ns/call kernel from A3) -- a true apples-to-apples test. Router logits feed a *discrete*
 * top-8 selection, so quantization noise could flip which experts get chosen, not just perturb a
 * smooth activation -- validated against the fp32 reference (g_router_* counters below/forward()),
 * not just eyeballed via ' Tokyo' coherence. ly->router (fp32) is kept only as that reference. */
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
        ly->router_hp=lin_new_hp(ly->router,ne,d);
        ly->router_i8=lin_new_i8(ly->router,ne,d);
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
        ly->router_hp=lin_new_hp(ly->router,m->n_exp,m->d);
        ly->router_i8=lin_new_i8(ly->router,m->n_exp,m->d);
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
    double _f0=gT_on?now():0, _a0=gT_actpack,_l0=gT_lin,_at0=gT_attn,_r0=gT_rope,_ro0=gT_router,_sw0=gT_swiglu;
    { double _tr0=gT_on?now():0; float cosb[256],sinb[256]; rope_table(cosb,sinb,hd,pos,m->rope_base);
    if(gT_on) gT_rope += now()-_tr0;
    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        rmsnorm(hn,h,ly->attn_norm,d,m->eps);
        { double _ta=gT_on?now():0; pack_act_hp(hn,d,Abuf2);
          double _tb=gT_on?now():0; if(gT_on) gT_actpack+=_tb-_ta;
          g_lin_class=LIN_QKV;
          lin_mm_hp(&ly->q,Abuf2,q,nt); lin_mm_hp(&ly->k,Abuf2,k,nt); lin_mm_hp(&ly->v,Abuf2,vv,nt);
          if(gT_on) lin_add(now()-_tb); g_lin_class=LIN_NONE; }
        { double _tr=gT_on?now():0;
          for(int hh=0;hh<nh;hh++){ rmsnorm(q+hh*hd,q+hh*hd,ly->q_norm,hd,m->eps); rope_apply(q+hh*hd,hd,cosb,sinb); }
          for(int hh=0;hh<nkv;hh++){ rmsnorm(k+hh*hd,k+hh*hd,ly->k_norm,hd,m->eps); rope_apply(k+hh*hd,hd,cosb,sinb); }
          if(gT_on) gT_rope += now()-_tr; }
        float*Kc=kv->Kc+(size_t)l*kv->ctx*kvd,*Vc=kv->Vc+(size_t)l*kv->ctx*kvd;
        memcpy(Kc+(size_t)pos*kvd,k,kvd*4); memcpy(Vc+(size_t)pos*kvd,vv,kvd*4);
        float scale=1.0f/sqrtf(hd); double _at=gT_on?now():0;
        for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q+hh*hd,*sc=tmp;
            for(int j=0;j<=pos;j++){ float*kj=Kc+(size_t)j*kvd+kvh*hd; sc[j]=vdot_f32(qh,kj,hd)*scale; }
            softmax(sc,pos+1); float*oh=att+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
            for(int j=0;j<=pos;j++){ float*vj=Vc+(size_t)j*kvd+kvh*hd; vaxpy_f32(oh,vj,sc[j],hd); } }
        if(gT_on) gT_attn += now()-_at;
        g_lin_class=LIN_O; lin_mm(&ly->o,att,tmp,nt,Abuf); g_lin_class=LIN_NONE;
        for(int i=0;i<d;i++)h[i]+=tmp[i];
        rmsnorm(hn,h,ly->ffn_norm,d,m->eps);
        /* router precision mode (codex_recs_1.md §22.7-22.8): fp32 is the DEFAULT (exact, matches
         * the original engine). int4 HP-Lin: 33% faster router-bucket but 58.9% of routing
         * decisions get >=1/8 expert swapped vs fp32 (avg 1.38/8, usually a near-tie swap). int8
         * M1: validated near-bit-exact standalone but real routing-quality impact not yet measured.
         * Neither clears "keep only if quality holds" on its own yet -- both gated behind
         * g_router_mode (0=fp32 default) until broader eval settles it. router shares its int4-HP
         * activation pack with eg/eu below regardless of mode (same hn, same K=d) -- pack once,
         * matching the P0.2 pattern used for q/k/v; int8 mode needs its own differently-shaped
         * activation pack, computed separately only when needed. */
        { double _ta=gT_on?now():0; pack_act_hp(hn,d,Abuf2);
          double _tb=gT_on?now():0; if(gT_on) gT_actpack+=_tb-_ta;
        int need_hp=(g_router_mode==1)||g_router_validate, need_i8=(g_router_mode==2)||g_router_validate, need_fp32=(g_router_mode==0)||g_router_validate;
        float rl_hp[256]; if(need_hp) lin_mm_hp(&ly->router_hp,Abuf2,rl_hp,nt);
        float rl_i8[256]; if(need_i8){ uint8_t Abuf_i8[3000]; pack_act_i8(hn,d,Abuf_i8); lin_mm_i8(&ly->router_i8,Abuf_i8,rl_i8,nt); }
        float rl_fp32[256]; if(need_fp32) for(int e=0;e<ne;e++) rl_fp32[e]=vdot_f32(ly->router+(size_t)e*d,hn,d);
        float rl[256]; memcpy(rl, g_router_mode==1?rl_hp:(g_router_mode==2?rl_i8:rl_fp32), (size_t)ne*4);
        softmax(rl,ne);
        int sel[32]; float sw[32];
        for(int a=0;a<na;a++){ int bi=-1; float bv=-1e30f; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(sel[b]==e)used=1; if(!used&&rl[e]>bv){bv=rl[e];bi=e;} } sel[a]=bi; sw[a]=bv; }
        float ssum=0; for(int a=0;a<na;a++)ssum+=sw[a]; for(int a=0;a<na;a++)sw[a]/=ssum;
        /* stop the router-bucket clock here -- matches exactly what the pre-HP-Lin baseline
         * measured (kernel/dot-product + softmax + top-8 select + renormalize bundled together),
         * so before/after numbers are a true apples-to-apples comparison. Validation (below) is
         * extra reference-only compute and must NOT land inside this bucket. */
        if(gT_on) gT_router += now()-_tb;
        if(g_router_validate){
            /* argmax sentinel must be -inf-ish, not -1: these are raw logits (can run well below
             * -1, observed to ~-5), not post-softmax probabilities -- this exact bug caused a
             * 100% false-mismatch rate before it was found (codex_recs_1.md §22.7). */
            int selr[32]; for(int a=0;a<na;a++){ int bi=-1; float bv=-1e30f; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(selr[b]==e)used=1; if(!used&&rl_fp32[e]>bv){bv=rl_fp32[e];bi=e;} } selr[a]=bi; }
            g_rtr_cmp++;
            for(int e=0;e<ne;e++){ float ad=fabsf(rl_hp[e]-rl_fp32[e]),rel=ad/(fabsf(rl_fp32[e])+1e-6f);
                if(ad>g_rtr_hp_maxabs)g_rtr_hp_maxabs=ad; if(rel>g_rtr_hp_maxrel)g_rtr_hp_maxrel=rel; }
            int selhp[32]; for(int a=0;a<na;a++){ int bi=-1; float bv=-1e30f; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(selhp[b]==e)used=1; if(!used&&rl_hp[e]>bv){bv=rl_hp[e];bi=e;} } selhp[a]=bi; }
            int diffhp=0,ndiffhp=0; for(int a=0;a<na;a++){ int found=0; for(int b=0;b<na;b++) if(selr[a]==selhp[b]) found=1; if(!found){ diffhp=1; ndiffhp++; } }
            if(diffhp) g_rtr_hp_mismatch++; g_rtr_hp_diffcount+=ndiffhp;

            for(int e=0;e<ne;e++){ float ad=fabsf(rl_i8[e]-rl_fp32[e]),rel=ad/(fabsf(rl_fp32[e])+1e-6f);
                if(ad>g_rtr_i8_maxabs)g_rtr_i8_maxabs=ad; if(rel>g_rtr_i8_maxrel)g_rtr_i8_maxrel=rel; }
            int seli8[32]; for(int a=0;a<na;a++){ int bi=-1; float bv=-1e30f; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(seli8[b]==e)used=1; if(!used&&rl_i8[e]>bv){bv=rl_i8[e];bi=e;} } seli8[a]=bi; }
            int diffi8=0,ndiffi8=0; for(int a=0;a<na;a++){ int found=0; for(int b=0;b<na;b++) if(selr[a]==seli8[b]) found=1; if(!found){ diffi8=1; ndiffi8++; } }
            if(diffi8) g_rtr_i8_mismatch++; g_rtr_i8_diffcount+=ndiffi8;
        }
        for(int i=0;i<d;i++)eout[i]=0;
          for(int a=0;a<na;a++){ int e=sel[a]; float w=sw[a];
              g_lin_class=LIN_EXP;
              double _tb=gT_on?now():0;
              lin_mm_hp(&ly->eg[e],Abuf2,g,nt); lin_mm_hp(&ly->eu[e],Abuf2,u,nt);
              if(gT_on) lin_add(now()-_tb);
              double _ts=gT_on?now():0;
              for(int i=0;i<moe;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }
              if(gT_on) gT_swiglu+=now()-_ts;
              lin_mm(&ly->ed[e],g,tmp,nt,Abuf); /* attributes to gT_lin_exp via g_lin_class */
              g_lin_class=LIN_NONE;
              for(int i=0;i<d;i++)eout[i]+=w*tmp[i]; } }
        for(int i=0;i<d;i++)h[i]+=eout[i];
    } }
    rmsnorm(hn,h,m->out_norm,d,m->eps);
    g_lin_class=LIN_LM; lin_mm(&m->lm,hn,logits,nt,Abuf); g_lin_class=LIN_NONE;
    if(gT_on){ double ft=now()-_f0; gT_rest += ft-(gT_actpack-_a0)-(gT_lin-_l0)-(gT_attn-_at0)-(gT_rope-_r0)-(gT_router-_ro0)-(gT_swiglu-_sw0); gT_tok++; }
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

/* ===== Multi-prompt quality harness (codex_recs_1.md §22.15) =====
 * Replaces single-prompt ' Tokyo' coherence with teacher-forced evaluation across a fixed prompt
 * set spanning factual/reasoning/code/multilingual/long-context. "First use it to decide the int8
 * router" (int4-HP already decided against via the single-prompt test, not retested here).
 *
 * Promotion thresholds are fixed HERE, before any harness run -- not tuned after seeing results:
 *   1. router expert-set mismatch < 10% of layer-token decisions
 *   2. avg NLL delta (int8 teacher-forced on fp32's tokens, minus fp32's own self-NLL) < 0.5 nats/token
 *   3. token-argmax divergence (does int8 predict the same next token fp32 chose?) < 15%
 *   4. router bucket must be >=10% faster than fp32 -- no quality risk is worth taking for free
 * All four must PASS for a promotion verdict; any single FAIL blocks it.
 *
 * Methodology: for each prompt, (1) generate a reference continuation greedily under fp32
 * (g_router_mode=0), recording each chosen token's own self-NLL (a confidence baseline, not a
 * comparison metric by itself); (2) replay the SAME prompt+reference-token sequence under int8
 * (g_router_mode=2) via TEACHER FORCING -- at each step, feed the REFERENCE token regardless of
 * what int8 itself would have picked, so divergence never compounds into a different context for
 * later positions. At each step, before feeding, compare argmax(int8_logits) against the
 * reference token (divergence) and compute NLL of the reference token under int8's distribution.
 * Router expert-set mismatch stats come from the SAME int8 pass (g_router_validate=1), so they
 * reflect genuine in-context behavior (hidden states already perturbed by int8's own earlier
 * routing), not a counterfactual-only comparison against fp32 hidden states. */
#define HARNESS_NP 10
#define HARNESS_MAXCTX 160
#define HARNESS_GEN_SHORT 20
#define HARNESS_GEN_LONG 16

static const int hp1[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374};
static const int hp2[]={28253,374,23415,315,34684,323};
static const int hp3[]={2679,678,19423,525,55569,11,323,678,55569,525,9898,11,1221,678,19423,525};
static const int hp4[]={785,8500,374,220,17,11,220,19,11,220,23,11,220,16,21,11,220,18,17,13,576,1790,1372,374};
static const int hp5[]={750,75698,1445,982,262,421,308,2651,220,16,510,286,470,308,198,262,470,75698,1445,12,16,8,488};
static const int hp6[]={2,5712,311,1779,421,264,1372,374,10250,198,750,374,38217,1445,982,262,421,308,366,220,17,510,286,470};
static const int hp7[]={8747,60410,1574,409,1187,9625,1788};
static const int hp8[]={107513,20412};
static const int hp9[]={641,264,2613,14126,88677,1948,1378,23501,11,1052,12163,458,2310,8866,25766,6941,85656,13,7209,6556,11,566,1035,1787,806,8061,518,6896,8094,297,62410,11,44029,279,36038,53160,3080,807,557,603,1075,6623,11,323,40786,1817,6002,448,264,8205,15289,13,3776,1899,11,264,25382,33958,10636,806,8061,15331,264,10865,17822,3736,429,1030,10497,6896,518,32333,2326,1635,4134,13,576,33958,11247,429,279,3736,1030,45859,311,806,37850,11,323,902,1008,8866,25766,304,279,5537,1030,1012,2952,311,12733,432,13,85656,24109,279,3736,15516,1212,806,8455,7766,8991,11,323,1283,264,1293,21162,11,566,5499,1053,11};
static const int hp10[]={785,3840,315,24231,44295,3807,1376,2714,300,13,758,279,220,16,24,19,15,82,11,279,1156,14346,4586,58238,18495,1075,5190,40,1706,1033,5798,1667,28202,32983,11,892,1033,3460,11,11392,11,323,36997,311,7901,13,576,27130,315,279,97941,304,220,16,24,19,22,13791,1506,279,2070,11,6388,311,9155,323,803,14720,12645,6814,279,220,16,24,20,15,82,323,220,16,24,21,15,82,13,576,220,16,24,22,15,82,5485,279,10000,315,8003,81748,11,892,18250,458,4453,13940,8630,264,3175,16392,11,81468,279,1616,369,4345,18495,304,279,2701,13212,13,3216,279,220,16,24,24,15,82,11,279,7602,1030,23507,24231,504,24203,12645,1119,264,30450,8433,3922,13,21131,1182,518,419,32724,11,279,3175,1429,2989,27130,572};

typedef struct { const int*toks; int n; const char*name; int gen; } HarnessPrompt;
static const HarnessPrompt g_hprompts[HARNESS_NP] = {
    {hp1,12,"factual/capitals",HARNESS_GEN_SHORT}, {hp2,6,"factual/chemistry",HARNESS_GEN_SHORT},
    {hp3,15,"reasoning/syllogism",HARNESS_GEN_SHORT}, {hp4,22,"reasoning/sequence",HARNESS_GEN_SHORT},
    {hp5,23,"code/fibonacci",HARNESS_GEN_SHORT}, {hp6,24,"code/is_prime",HARNESS_GEN_SHORT},
    {hp7,7,"multilingual/french",HARNESS_GEN_SHORT}, {hp8,2,"multilingual/chinese",HARNESS_GEN_SHORT},
    {hp9,113,"long-context/narrative",HARNESS_GEN_LONG}, {hp10,113,"long-context/technical",HARNESS_GEN_LONG},
};

__attribute__((noinline,optimize("no-tree-vectorize")))
static double harness_logsumexp(const float*l,int n){ float mx=l[0]; for(int i=1;i<n;i++) if(l[i]>mx) mx=l[i];
    double s=0; for(int i=0;i<n;i++) s+=exp((double)(l[i]-mx)); return (double)mx+log(s); }
__attribute__((noinline,optimize("no-tree-vectorize")))
static float harness_nll(const float*logits,int n,int target){ return (float)(harness_logsumexp(logits,n)-(double)logits[target]); }

__attribute__((noinline,optimize("no-tree-vectorize")))
static void harness_run_fp32(Model*m,const HarnessPrompt*hp,int*out_toks,float*out_selfnll,Kv*kv,
        float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*gg,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2,float*logits){
    memset(kv->Kc,0,(size_t)m->nl*kv->ctx*kv->kvd*4); memset(kv->Vc,0,(size_t)m->nl*kv->ctx*kv->kvd*4);
    g_router_mode=0; g_router_validate=0;
    for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    int cur=argmax(logits,m->vocab); out_toks[0]=cur; out_selfnll[0]=harness_nll(logits,m->vocab,cur);
    for(int s=1;s<hp->gen;s++){
        forward(m,cur,hp->n+s-1,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
        cur=argmax(logits,m->vocab); out_toks[s]=cur; out_selfnll[s]=harness_nll(logits,m->vocab,cur);
    }
}
/* do_validate=1 also computes router expert-set mismatch stats (g_router_validate=1), which
 * makes EVERY forward() call additionally compute the other router variants plus O(experts)
 * sentinel-argmax/mismatch-counting overhead -- real cost, but not part of what int8's own
 * decode buckets cost in actual production use. Call with do_validate=0 for a clean, apples-to-
 * apples SPEED comparison against the fp32 pass; call again with do_validate=1 (timing discarded)
 * to harvest router-mismatch stats separately. Mixing the two in one pass is exactly the kind of
 * "measure the wrong thing" bug this session has hit before (research_feed_paths.md §12) -- the
 * first version of this harness did that and produced a bogus "int8 router is 29% slower" result
 * (all in the untimed 'rest' bucket, from validation overhead, not real work). */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void harness_run_teacherforced(Model*m,const HarnessPrompt*hp,const int*ref,int mode,int do_validate,int*out_div,float*out_nll,
        int*out_rtr_cmp,int*out_rtr_mm,int*out_rtr_dc,float*out_rtr_ma,Kv*kv,
        float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*gg,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2,float*logits){
    memset(kv->Kc,0,(size_t)m->nl*kv->ctx*kv->kvd*4); memset(kv->Vc,0,(size_t)m->nl*kv->ctx*kv->kvd*4);
    g_router_mode=mode; g_router_validate=do_validate;
    long cmp0=g_rtr_cmp, mm0=(mode==1?g_rtr_hp_mismatch:g_rtr_i8_mismatch), dc0=(mode==1?g_rtr_hp_diffcount:g_rtr_i8_diffcount);
    for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    for(int s=0;s<hp->gen;s++){
        out_div[s]=(argmax(logits,m->vocab)!=ref[s]); out_nll[s]=harness_nll(logits,m->vocab,ref[s]);
        forward(m,ref[s],hp->n+s,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    }
    *out_rtr_cmp=(int)(g_rtr_cmp-cmp0); *out_rtr_mm=(int)((mode==1?g_rtr_hp_mismatch:g_rtr_i8_mismatch)-mm0);
    *out_rtr_dc=(int)((mode==1?g_rtr_hp_diffcount:g_rtr_i8_diffcount)-dc0); *out_rtr_ma=(mode==1?g_rtr_hp_maxabs:g_rtr_i8_maxabs);
    g_router_validate=0;
}

__attribute__((noinline,optimize("no-tree-vectorize")))
static void run_quality_harness(Model*m){
    int ctx=HARNESS_MAXCTX; Kv kv; kv.kvd=m->nkv*m->hd; kv.ctx=ctx;
    kv.Kc=calloc((size_t)m->nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m->nl*ctx*kv.kvd,4);
    int d=m->d,qd=m->nh*m->hd,moe=m->moe,maxk=qd>moe?(qd>d?qd:d):(moe>d?moe:d); if(d>maxk)maxk=d;
    float*hn=malloc(d*4),*q=malloc(qd*4),*k=malloc(kv.kvd*4),*vv=malloc(kv.kvd*4),*att=malloc(qd*4),
         *tmp=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4),*gg=malloc(moe*4),*u=malloc(moe*4),*eout=malloc(d*4),*logits=malloc((size_t)m->vocab*4);
    uint8_t*Abuf=malloc((size_t)(maxk/256)*AREC),*Abuf2=malloc((size_t)(maxk/256)*AREC);

    static int ref_toks[HARNESS_NP][HARNESS_GEN_SHORT];
    static float self_nll[HARNESS_NP][HARNESS_GEN_SHORT];

    printf("\n=== quality harness: phase 1/2 -- fp32 reference generation (%d prompts) ===\n",HARNESS_NP); fflush(stdout);
    gT_on=1; gT_tok=0; gT_actpack=gT_lin=gT_attn=gT_rope=gT_router=gT_swiglu=gT_rest=0;
    double t0=now();
    for(int i=0;i<HARNESS_NP;i++){
        harness_run_fp32(m,&g_hprompts[i],ref_toks[i],self_nll[i],&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
        printf("  [%2d/%d] %-24s prefill=%3d gen=%2d done\n",i+1,HARNESS_NP,g_hprompts[i].name,g_hprompts[i].n,g_hprompts[i].gen);
    }
    long tok_fp32=gT_tok; double lin_fp32=gT_lin,attn_fp32=gT_attn,rope_fp32=gT_rope,router_fp32=gT_router,swiglu_fp32=gT_swiglu,actpack_fp32=gT_actpack,rest_fp32=gT_rest;
    fprintf(stderr,"  phase 1 wall: %.1fs\n",now()-t0);

    printf("\n=== quality harness: phase 2a/2 -- int8-M1 router expert-set mismatch (validate on, timing discarded) ===\n");
    typedef struct { const char*name; int gen,divergent,rtr_cmp,rtr_mm,rtr_dc; float nll_i8,nll_fp32,rtr_ma; } HRes;
    HRes res[HARNESS_NP];
    for(int i=0;i<HARNESS_NP;i++){
        int div[HARNESS_GEN_SHORT]; float nll[HARNESS_GEN_SHORT]; int rc,rm,rd; float rma;
        int save_gTon=gT_on; gT_on=0; /* this pass's cost must not leak into any speed bucket */
        harness_run_teacherforced(m,&g_hprompts[i],ref_toks[i],2,1,div,nll,&rc,&rm,&rd,&rma,&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
        gT_on=save_gTon;
        HRes*r=&res[i]; r->name=g_hprompts[i].name; r->gen=g_hprompts[i].gen; r->divergent=0; r->nll_i8=0; r->nll_fp32=0;
        for(int s=0;s<r->gen;s++){ r->divergent+=div[s]; r->nll_i8+=nll[s]; r->nll_fp32+=self_nll[i][s]; }
        r->rtr_cmp=rc; r->rtr_mm=rm; r->rtr_dc=rd; r->rtr_ma=rma;
        printf("  [%2d/%d] %-24s divergent=%2d/%-2d  nll_delta=%+.4f  rtr_mismatch=%3d/%-4d\n",
            i+1,HARNESS_NP,r->name,r->divergent,r->gen,(r->nll_i8-r->nll_fp32)/r->gen,r->rtr_mm,r->rtr_cmp);
    }

    printf("\n=== quality harness: phase 2b/2 -- int8-M1 router speed (validate off, clean apples-to-apples timing) ===\n");
    t0=now();
    for(int i=0;i<HARNESS_NP;i++){
        int div[HARNESS_GEN_SHORT]; float nll[HARNESS_GEN_SHORT]; int rc,rm,rd; float rma;
        harness_run_teacherforced(m,&g_hprompts[i],ref_toks[i],2,0,div,nll,&rc,&rm,&rd,&rma,&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
        printf("  [%2d/%d] %-24s done\n",i+1,HARNESS_NP,g_hprompts[i].name);
    }
    long tok_i8=gT_tok-tok_fp32; double lin_i8=gT_lin-lin_fp32,attn_i8=gT_attn-attn_fp32,rope_i8=gT_rope-rope_fp32,
        router_i8=gT_router-router_fp32,swiglu_i8=gT_swiglu-swiglu_fp32,actpack_i8=gT_actpack-actpack_fp32,rest_i8=gT_rest-rest_fp32;
    fprintf(stderr,"  phase 2 wall: %.1fs\n",now()-t0);
    gT_on=0;

    int total_gen=0,total_div=0,total_rc=0,total_rm=0,total_rd=0; double total_nll_i8=0,total_nll_fp32=0; float max_rma=0;
    for(int i=0;i<HARNESS_NP;i++){ total_gen+=res[i].gen; total_div+=res[i].divergent; total_nll_i8+=res[i].nll_i8; total_nll_fp32+=res[i].nll_fp32;
        total_rc+=res[i].rtr_cmp; total_rm+=res[i].rtr_mm; total_rd+=res[i].rtr_dc; if(res[i].rtr_ma>max_rma)max_rma=res[i].rtr_ma; }
    double div_rate=100.0*total_div/total_gen, nll_delta=(total_nll_i8-total_nll_fp32)/total_gen, rtr_rate=100.0*total_rm/total_rc;
    double fp32_router_ms=tok_fp32?router_fp32/tok_fp32*1e3:0, i8_router_ms=tok_i8?router_i8/tok_i8*1e3:0;
    double speedup_pct=fp32_router_ms?100.0*(fp32_router_ms-i8_router_ms)/fp32_router_ms:0;

    printf("\n=== QUALITY HARNESS REPORT (int8-M1 router vs fp32, %d prompts, %d total generated tokens, teacher-forced) ===\n",HARNESS_NP,total_gen);
    printf("token-argmax divergence:  %d/%d (%.1f%%)\n",total_div,total_gen,div_rate);
    printf("avg NLL delta (int8 tf - fp32 self): %+.4f nats/token\n",nll_delta);
    printf("router expert-set mismatch: %d/%d (%.1f%%), max abs logit delta %e\n",total_rm,total_rc,rtr_rate,max_rma);
    printf("speed: fp32 router %.1fms/tok, int8 router %.1fms/tok (%.1f%% %s)\n",
        fp32_router_ms,i8_router_ms,speedup_pct>=0?speedup_pct:-speedup_pct,speedup_pct>=0?"faster":"slower");
    printf("fp32 buckets (avg/%ld tok, ms): act-pack %.1f | linear %.1f | attention %.1f | rope %.1f | router %.1f | swiglu %.1f | rest %.1f\n",
        tok_fp32,actpack_fp32/tok_fp32*1e3,lin_fp32/tok_fp32*1e3,attn_fp32/tok_fp32*1e3,rope_fp32/tok_fp32*1e3,router_fp32/tok_fp32*1e3,swiglu_fp32/tok_fp32*1e3,rest_fp32/tok_fp32*1e3);
    printf("int8 buckets (avg/%ld tok, ms): act-pack %.1f | linear %.1f | attention %.1f | rope %.1f | router %.1f | swiglu %.1f | rest %.1f\n",
        tok_i8,actpack_i8/tok_i8*1e3,lin_i8/tok_i8*1e3,attn_i8/tok_i8*1e3,rope_i8/tok_i8*1e3,router_i8/tok_i8*1e3,swiglu_i8/tok_i8*1e3,rest_i8/tok_i8*1e3);

    printf("\n--- promotion thresholds (fixed before this run, codex_recs_1.md §22.15) ---\n");
    int p1=rtr_rate<10.0, p2=nll_delta<0.5, p3=div_rate<15.0, p4=speedup_pct>=10.0;
    printf("  [%s] router expert-set mismatch < 10%%      (got %.1f%%)\n",p1?"PASS":"FAIL",rtr_rate);
    printf("  [%s] avg NLL delta < 0.5 nats/token         (got %+.4f)\n",p2?"PASS":"FAIL",nll_delta);
    printf("  [%s] token divergence < 15%%                 (got %.1f%%)\n",p3?"PASS":"FAIL",div_rate);
    printf("  [%s] router bucket >=10%% faster than fp32   (got %.1f%%)\n",p4?"PASS":"FAIL",speedup_pct);
    printf("VERDICT: int8-M1 router %s promotion to default.\n",(p1&&p2&&p3&&p4)?"PASSES all thresholds -- ELIGIBLE FOR":"FAILS at least one threshold -- NOT ELIGIBLE FOR");

    free(hn);free(q);free(k);free(vv);free(att);free(tmp);free(gg);free(u);free(eout);free(logits);free(Abuf);free(Abuf2);
    free(kv.Kc);free(kv.Vc);
}

int main(int c,char**v){
    if(c<2){ printf("usage: %s model.gguf [ngen] [nt]\n",v[0]); return 1; }
    int ngen=(c>2)?atoi(v[2]):16, nt=(c>3)?atoi(v[3]):4;
    bind_ai(); { cpu_set_t s;CPU_ZERO(&s);CPU_SET(8,&s);sched_setaffinity(0,sizeof(s),&s);} for(int i=0;i<5;i++)sched_yield();
    Gguf g; double t0=now(); gguf_open(&g,v[1]);
    fprintf(stderr,"qwen3moe (HP kernel): %d layers d=%d experts=%d/%d moe_ffn=%d heads=%d/%d hd=%d vocab=%d (parse %.1fs)\n",
        g.block_count,g.embd,g.n_exp,g.n_act,g.moe_ffn,g.nh,g.nkv,g.hd,g.vocab,now()-t0);
    fflush(stderr);
    const char*cpath=(c>4)?v[4]:"/root/models/qwen3-30b-a3b.hp.imecache";
    g_router_validate=(c>5)?atoi(v[5]):0; /* router HP-Lin/int8-vs-fp32 expert-selection validation, off by default (extra compute) */
    g_router_mode=(c>6)?atoi(v[6]):2; /* 0=fp32(exact, revert flag) 1=int4-HP(rejected, see §22.7) 2=int8-M1(DEFAULT since 2026-07-26 -- passed the multi-prompt quality harness, codex_recs_1.md §22.15: 6.1% router mismatch, -0.0034 nats/tok NLL delta, 2.1% token divergence, 13.7% faster, all four pre-registered thresholds PASS) */
    Model m; double tl=now(); int cached=0;
    if(cache_load(&m,cpath,nt)){ cached=1; fprintf(stderr,"loaded from cache in %.1fs  (%s)\n",now()-tl,cpath); }
    else { model_load(&m,&g,nt); fprintf(stderr,"requant loaded in %.1fs\n",now()-tl); }

    int ctx=64; Kv kv; kv.kvd=m.nkv*m.hd; kv.ctx=ctx; kv.Kc=calloc((size_t)m.nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m.nl*ctx*kv.kvd,4);
    int d=m.d,qd=m.nh*m.hd,moe=m.moe,maxk=qd>moe?(qd>d?qd:d):(moe>d?moe:d); if(d>maxk)maxk=d;
    float*hn=malloc(d*4),*q=malloc(qd*4),*k=malloc(kv.kvd*4),*vv=malloc(kv.kvd*4),*att=malloc(qd*4),
         *tmp=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4),*gg=malloc(moe*4),*u=malloc(moe*4),*eout=malloc(d*4),*logits=malloc((size_t)m.vocab*4);
    uint8_t*Abuf=malloc((size_t)(maxk/256)*AREC),*Abuf2=malloc((size_t)(maxk/256)*AREC);
    lin_mm_pool_init(nt); /* PR8: persistent workers, spawned once, spin-dispatched per lin_mm_hp call */

    /* QWEN_HARNESS=1 (env var, not an 8th CLI arg) triggers the multi-prompt quality harness --
     * see codex_recs_1.md §22.15. NOTE: an 8th positional CLI arg was tried first and reproducibly
     * read back as a corrupted/wild pointer by the time execution reached here (valid at main()
     * entry, clobbered somewhere during cache_load/model setup) -- a real, pre-existing memory
     * bug this exposed, not a bug in the harness itself; never manifested before because nothing
     * previously read past argv[6]. Root cause not yet found -- flagged in
     * research_feed_paths.md/codex_recs_1.md for follow-up, worked around here via getenv instead
     * of argv so the harness isn't blocked on debugging it. */
    { const char*hv=getenv("QWEN_HARNESS"); if(hv && atoi(hv)){ run_quality_harness(&m); return 0; } }

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
    const char*router_names[]={"fp32(exact,revert-flag)","int4-HP(rejected)","int8-M1(default)"};
    printf("\ndecode: %.2f tok/s (Qwen3-30B-A3B, vendor IME-2-HP int4 W4A8, nt=%d, router=%s)\n", ngen/dt, nt, router_names[g_router_mode]);
    if(gT_tok){
        printf("  per-token buckets (avg/%ld tok, ms): act-pack %.1f | linear(kernel) %.1f | attention %.1f | rope+qknorm %.1f | router %.1f | swiglu %.1f | rest(other) %.1f | sum %.1f | wall %.1f\n",
            gT_tok, gT_actpack/gT_tok*1e3, gT_lin/gT_tok*1e3, gT_attn/gT_tok*1e3, gT_rope/gT_tok*1e3, gT_router/gT_tok*1e3, gT_swiglu/gT_tok*1e3, gT_rest/gT_tok*1e3,
            (gT_actpack+gT_lin+gT_attn+gT_rope+gT_router+gT_swiglu+gT_rest)/gT_tok*1e3, dt/ngen*1e3);
        printf("  linear breakdown (avg/%ld tok, ms): qkv %.1f | o %.1f | expert(gate/up/down) %.1f | lm_head %.1f | sum %.1f\n",
            gT_tok, gT_lin_qkv/gT_tok*1e3, gT_lin_o/gT_tok*1e3, gT_lin_exp/gT_tok*1e3, gT_lin_lm/gT_tok*1e3,
            (gT_lin_qkv+gT_lin_o+gT_lin_exp+gT_lin_lm)/gT_tok*1e3);
    }
    if(g_router_validate){
        printf("  router int4-HP-vs-fp32: %ld comparisons, %ld expert-set mismatches (avg %.2f/%d experts differ per mismatch), max abs logit delta %e, max rel %e\n",
            g_rtr_cmp, g_rtr_hp_mismatch, g_rtr_hp_mismatch?(double)g_rtr_hp_diffcount/g_rtr_hp_mismatch:0.0, m.n_act, g_rtr_hp_maxabs, g_rtr_hp_maxrel);
        printf("  router int8-M1-vs-fp32: %ld comparisons, %ld expert-set mismatches (avg %.2f/%d experts differ per mismatch), max abs logit delta %e, max rel %e\n",
            g_rtr_cmp, g_rtr_i8_mismatch, g_rtr_i8_mismatch?(double)g_rtr_i8_diffcount/g_rtr_i8_mismatch:0.0, m.n_act, g_rtr_i8_maxabs, g_rtr_i8_maxrel);
    }
    if(!cached){ fprintf(stderr,"saving requant cache -> %s ...\n",cpath); double ts=now(); cache_save(&m,cpath); fprintf(stderr,"cache saved in %.1fs\n",now()-ts); }
    return 0;
}
