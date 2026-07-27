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
/* attention_optimization_plan.md Phase 1: split gT_attn (already the sum of these three) into
 * QK dot-product, softmax, and weighted-V accumulate, per head, to establish which operation
 * dominates before changing the KV layout or adding threading -- instrumentation only, no math
 * changed. */
static double gT_attn_qk=0, gT_attn_sm=0, gT_attn_av=0;
/* Phase 3 (codex_recs_1.md §22.29): dispatch/sync overhead when attention is parallelized across
 * the persistent pool -- wall time of the whole attention step minus the slowest single worker's
 * own QK+softmax+AV time this round, i.e. cost not explained by any one worker's real compute.
 * Zero when g_attn_nt<=1 (serial path, exact same code as Phase 2, no dispatch involved). */
static double gT_attn_dispatch=0;
/* g_attn_nt: how many pool threads participate in attention this run (independent of g_pool_nt,
 * which still fully participates in the GEMM kernels regardless). DEFAULT (set in main() once
 * m.nkv is known) is min(nt,nkv) -- PROMOTED 2026-07-26 after passing every Phase 3 keep criterion,
 * codex_recs_1.md §22.29. 1=serial, byte-identical to Phase 2, kept as an explicit revert flag via
 * QWEN_ATTN_NT, same "avoid another positional production arg" convention as QWEN_CTXLEN/
 * QWEN_HARNESS. The initializer below (1) only matters before main() sets the real default. */
static int g_attn_nt=1;
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
/* router precision mode: 0=fp32 (exact, revert flag, matches the original engine), 1=int4 HP-Lin
 * (33% faster router bucket but 58.9% of routing decisions perturbed vs fp32, avg 1.38/8 experts
 * -- REJECTED, codex_recs_1.md §22.7), 2=int8 M1 (DEFAULT since 2026-07-26 -- passed the
 * multi-prompt quality harness: 6.1% router mismatch, -0.0034 nats/tok NLL delta, 2.1% token
 * divergence, 13.7% faster, all four pre-registered thresholds PASS -- codex_recs_1.md §22.15).
 * Set via 7th CLI arg. */
static int g_router_mode=0;

/* PR8 (codex_recs_1.md §17/§22.3): with the vendor kernel at ~446ns/call, the ~1392 fresh
 * #pragma omp parallel spawns/token that lin_mm_hp used to do dominated wall-clock (100.3ms,
 * 62%). Replace with a persistent spin-dispatch pool: threads created once, wait on a generation
 * counter instead of libgomp fork/join. Same round-robin panel partitioning as before (np=tn;
 * np<Np; np+=nt), so the actual math is unchanged -- only the dispatch mechanism differs.
 * Generalized (kind field) to also dispatch the int8 M1 kernel through the same pool. */
#define MAXNT 16
/* kind: 0=int4 HP GEMM, 1=int8 M1 GEMM, 2=attention (Phase 3, codex_recs_1.md §22.29). The
 * attention-only fields are unused by kind 0/1 and vice versa; kept as plain extra fields (not a
 * union) since HpWork is small and this file already treats struct simplicity as more valuable
 * than the few bytes a union would save. attn_worker_run/attn_dispatch (defined later, after the
 * vdot_f32/softmax/vaxpy_f32 primitives they call) fill and read these. */
typedef struct {
    int kind; const Lin*l; const uint8_t*Abuf; float*y; int kb;
    const float*aq; const float*aKc; const float*aVc; float*aatt;
    int apos; float ascale; int ahd; int ankv; int agpr; int actx; int aattn_nt;
} HpWork;
static _Atomic int g_pool_gen=0, g_pool_done=0;
static HpWork g_pool_work;
static int g_pool_nt=0;
static pthread_t g_pool_threads[MAXNT];
static void attn_worker_run(int tn); /* defined after vdot_f32/softmax/vaxpy_f32, called below */

static void lin_mm_hp_worker_run(int tn){
    if(g_pool_work.kind==2){ attn_worker_run(tn); return; }
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
/* attention_optimization_plan.md Phase 3 (codex_recs_1.md §22.29): four-KV-group worker
 * parallelism, reusing the persistent pool -- no new threads. Isolated from Phase 4 (GQA fusion):
 * each worker still does the exact same per-head QK/softmax/AV as the Phase 2 serial code, just on
 * a different thread; K/V is still re-read once per query head sharing a KV head (unchanged, that
 * redundancy is Phase 4's target, not this one's). Worker tn handles KV heads tn, tn+attn_nt,
 * tn+2*attn_nt, ... and each such KV head's gpr query heads, in ascending head order -- with
 * attn_nt==nkv (the production target once/if promoted) this gives exactly the plan's "ideal
 * division" (worker k = KV head k) since each worker's stride equals nkv. Every pool thread always
 * gets woken and always increments g_pool_done exactly once per round (matching the existing
 * lin_mm_hp/lin_mm_i8 wait discipline unchanged); threads with tn>=attn_nt do no attention work
 * that round, a fast no-op, so testing attn_nt=1/2/4 needs no change to the wait logic. */
static float* g_attn_scratch[MAXNT]; /* private per-worker score buffer, >=ctx floats, allocated
    once (lazily, on first use) not per-layer/token -- matches Phase 3's explicit requirement. */
static int g_attn_scratch_ctx=0;
static double g_w_qk[MAXNT], g_w_sm[MAXNT], g_w_av[MAXNT]; /* per-worker accumulators: each thread
    only ever writes its own index, so these are race-free without locking; summed into the global
    gT_attn_qk/sm/av by the single dispatching thread after every worker has finished its round. */
static void attn_scratch_ensure(int ctx){
    if(g_attn_scratch_ctx>=ctx) return;
    for(int i=0;i<MAXNT;i++){ free(g_attn_scratch[i]); g_attn_scratch[i]=malloc((size_t)ctx*4); }
    g_attn_scratch_ctx=ctx;
}
static void attn_worker_run(int tn){
    if(tn>=g_pool_work.aattn_nt){ g_w_qk[tn]=g_w_sm[tn]=g_w_av[tn]=0; return; }
    int nkv=g_pool_work.ankv, gpr=g_pool_work.agpr, hd=g_pool_work.ahd, pos=g_pool_work.apos, actx=g_pool_work.actx;
    float scale=g_pool_work.ascale;
    const float*q=g_pool_work.aq, *Kc=g_pool_work.aKc, *Vc=g_pool_work.aVc; float*att=g_pool_work.aatt;
    float*sc=g_attn_scratch[tn];
    double qk=0, sm=0, av=0;
    for(int kvh=tn; kvh<nkv; kvh+=g_pool_work.aattn_nt){
        const float*Kh=Kc+(size_t)kvh*actx*hd, *Vh=Vc+(size_t)kvh*actx*hd;
        for(int qi=0; qi<gpr; qi++){
            int hh=kvh*gpr+qi; const float*qh=q+(size_t)hh*hd;
            double t0=gT_on?now():0;
            for(int j=0;j<=pos;j++){ const float*kj=Kh+(size_t)j*hd; sc[j]=vdot_f32(qh,kj,hd)*scale; }
            double t1=gT_on?now():0; if(gT_on) qk+=t1-t0;
            softmax(sc,pos+1);
            double t2=gT_on?now():0; if(gT_on) sm+=t2-t1;
            float*oh=att+(size_t)hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
            for(int j=0;j<=pos;j++){ const float*vj=Vh+(size_t)j*hd; vaxpy_f32(oh,vj,sc[j],hd); }
            if(gT_on) av += now()-t2;
        }
    }
    g_w_qk[tn]=qk; g_w_sm[tn]=sm; g_w_av[tn]=av;
}
static void attn_dispatch(const float*q,const float*Kc,const float*Vc,float*att,int pos,float scale,
        int hd,int nkv,int gpr,int actx,int attn_nt){
    attn_scratch_ensure(actx);
    g_pool_work.kind=2; g_pool_work.aq=q; g_pool_work.aKc=Kc; g_pool_work.aVc=Vc; g_pool_work.aatt=att;
    g_pool_work.apos=pos; g_pool_work.ascale=scale; g_pool_work.ahd=hd; g_pool_work.ankv=nkv;
    g_pool_work.agpr=gpr; g_pool_work.actx=actx; g_pool_work.aattn_nt=attn_nt;
    double _tdisp=gT_on?now():0;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release); /* wake workers 1..g_pool_nt-1 */
    attn_worker_run(0); /* main thread does its own share (tn=0) */
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < g_pool_nt-1) { /* spin */ }
    if(gT_on){
        double maxw=0, sumqk=0,sumsm=0,sumav=0;
        for(int i=0;i<attn_nt;i++){
            double w=g_w_qk[i]+g_w_sm[i]+g_w_av[i]; if(w>maxw) maxw=w;
            sumqk+=g_w_qk[i]; sumsm+=g_w_sm[i]; sumav+=g_w_av[i];
        }
        gT_attn_qk+=sumqk; gT_attn_sm+=sumsm; gT_attn_av+=sumav;
        gT_attn_dispatch += (now()-_tdisp) - maxw;
    }
}
/* SwiGLU (codex_recs_1.md §22.18): g_swiglu_fast selects exact SiLU(x)*u (default, `expf`-based,
 * matches the original engine exactly) or a candidate fast approximation, gated behind the flag
 * per standing instruction ("avoid approximate SwiGLU until broader quality tests exist" --
 * that broader test now exists, codex_recs_1.md §22.15/17). Unlike every other RVV port in this
 * file, the fast path is NOT validated bit-exact against the exact path -- it's a deliberate lossy
 * approximation; its quality is judged by the multi-prompt harness against the gates predeclared
 * in §22.18, not an oracle. The RVV implementation itself IS validated against a scalar reference
 * of the SAME formula (bench/swiglu_hswish_probe.c) to catch vectorization bugs, separately from
 * asking whether the approximation itself is good enough. */
static void swiglu_exact(float*g,const float*u,int n){
    for(int i=0;i<n;i++){ float x=g[i]; g[i]=(x/(1.0f+expf(-x)))*u[i]; }
}
/* hard-swish (Howard et al. 2019, MobileNetV3): x*hard_sigmoid(x), hard_sigmoid(x)=clamp((x+3)/6,0,1).
 * No transcendentals -- pure add/mul/clamp, so this is genuinely RVV-vectorized (not relying on
 * gcc autovec, matching this file's explicit-vectorization-only policy, codex_recs_1.md §22.16). */
static void swiglu_hswish_rvv(float*g,const float*u,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(g+i,vl), vu=__riscv_vle32_v_f32m1(u+i,vl);
        vfloat32m1_t vh=__riscv_vfadd_vf_f32m1(vx,3.0f,vl);
        vh=__riscv_vfmul_vf_f32m1(vh,1.0f/6.0f,vl);
        vh=__riscv_vfmax_vf_f32m1(vh,0.0f,vl);
        vh=__riscv_vfmin_vf_f32m1(vh,1.0f,vl);
        vfloat32m1_t vy=__riscv_vfmul_vv_f32m1(vx,vh,vl);
        vy=__riscv_vfmul_vv_f32m1(vy,vu,vl);
        __riscv_vse32_v_f32m1(g+i,vy,vl); i+=vl; }
}
/* rational-Pade sigmoid (codex_recs_1.md §22.20): unlike hard-swish, this approximates the ACTUAL
 * sigmoid Qwen3 was trained against, not a different function -- tanh(y) ~= y*(27+y^2)/(27+9y^2)
 * (classic Pade[3/2] rational approximant, clamped to [-1,1] for the large-|y| asymptote), then
 * sigmoid(x) = 0.5*(1+tanh(x/2)). One division, no transcendentals. Python sanity check before
 * implementing: 2.5-5.2x lower mean abs error vs true SiLU than hard-swish across [-6,6]/[-3,3]/
 * [-20,20] input ranges (mean 0.011-0.022 vs hard-swish's 0.059, max 0.04 vs 0.14). */
static void swiglu_ratsig_rvv(float*g,const float*u,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(g+i,vl), vu=__riscv_vle32_v_f32m1(u+i,vl);
        vfloat32m1_t vhx=__riscv_vfmul_vf_f32m1(vx,0.5f,vl);
        vfloat32m1_t vhx2=__riscv_vfmul_vv_f32m1(vhx,vhx,vl);
        vfloat32m1_t vnum=__riscv_vfadd_vf_f32m1(vhx2,27.0f,vl);
        vnum=__riscv_vfmul_vv_f32m1(vhx,vnum,vl);
        vfloat32m1_t vden=__riscv_vfmul_vf_f32m1(vhx2,9.0f,vl);
        vden=__riscv_vfadd_vf_f32m1(vden,27.0f,vl);
        vfloat32m1_t vt=__riscv_vfdiv_vv_f32m1(vnum,vden,vl);
        vt=__riscv_vfmax_vf_f32m1(vt,-1.0f,vl);
        vt=__riscv_vfmin_vf_f32m1(vt,1.0f,vl);
        vfloat32m1_t vsig=__riscv_vfadd_vf_f32m1(vt,1.0f,vl);
        vsig=__riscv_vfmul_vf_f32m1(vsig,0.5f,vl);
        vfloat32m1_t vy=__riscv_vfmul_vv_f32m1(vx,vsig,vl);
        vy=__riscv_vfmul_vv_f32m1(vy,vu,vl);
        __riscv_vse32_v_f32m1(g+i,vy,vl); i+=vl; }
}
static int g_swiglu_fast=0; /* 0=exact(revert flag) 1=hard-swish(REJECTED, §22.19-20 -- 11.0% perplexity inflation, fails the 5% gate) 2=rational-Pade(DEFAULT since 2026-07-26, §22.21 -- passed the full quality harness §22.20 AND a bounded production A/B: 9.7->11.4 tok/s, swiglu bucket -92.1%, identical tokens, no regression in any other bucket) */
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
        if(pos<0 || pos>=kv->ctx){
            fprintf(stderr,"KV position overflow: pos=%d ctx=%d\n",pos,kv->ctx);
            abort();
        }
        /* attention_optimization_plan.md Phase 2 (codex_recs_1.md §22.28): head-major KV layout,
         * exact candidate vs the time-major baseline preserved at /tmp/qwen_moe_hp_kv_timemajor.c
         * (matches git HEAD before this change). Per-layer region is now [kv_head][position]
         * [head_dim] (stride ctx*hd per kv_head) instead of [position][kv_head][head_dim] (stride
         * kvd=nkv*hd per position, a 2KB stride for one head's history in the old layout) -- each
         * head's own K/V history is now contiguous. Same total per-layer size (ctx*kvd floats
         * either way), same values, only the addressing changes: the write becomes nkv separate
         * per-head memcpys (positions for different heads are no longer adjacent) instead of one
         * memcpy of all heads at a position; the read indexes each head's contiguous run directly
         * instead of striding by kvd per position. No math, no quantization, no threading change. */
        float*Kc=kv->Kc+(size_t)l*kv->ctx*kvd,*Vc=kv->Vc+(size_t)l*kv->ctx*kvd;
        for(int kvh=0;kvh<nkv;kvh++){
            memcpy(Kc+(size_t)kvh*kv->ctx*hd+(size_t)pos*hd, k+(size_t)kvh*hd, hd*4);
            memcpy(Vc+(size_t)kvh*kv->ctx*hd+(size_t)pos*hd, vv+(size_t)kvh*hd, hd*4);
        }
        float scale=1.0f/sqrtf(hd); double _at=gT_on?now():0;
        /* attention_optimization_plan.md Phase 3 (codex_recs_1.md §22.29): g_attn_nt<=1 (default)
         * keeps this exact serial loop byte-for-byte, unchanged since Phase 2 -- zero risk to the
         * already-verified path when parallelism is off. g_attn_nt>1 dispatches the same per-head
         * work across the persistent pool via attn_dispatch, isolated from Phase 2 (layout already
         * applied either way) and from Phase 4 (no GQA fusion -- each worker still does the exact
         * same per-head QK/softmax/AV, just on a different thread, one KV head's worth per stride
         * step, matching attn_worker_run's division). */
        if(g_attn_nt<=1){
            for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q+hh*hd,*sc=tmp;
                float*Kh=Kc+(size_t)kvh*kv->ctx*hd,*Vh=Vc+(size_t)kvh*kv->ctx*hd;
                double _tqk=gT_on?now():0;
                for(int j=0;j<=pos;j++){ float*kj=Kh+(size_t)j*hd; sc[j]=vdot_f32(qh,kj,hd)*scale; }
                double _tsm=gT_on?now():0; if(gT_on) gT_attn_qk += _tsm-_tqk;
                softmax(sc,pos+1);
                double _tav=gT_on?now():0; if(gT_on) gT_attn_sm += _tav-_tsm;
                float*oh=att+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
                for(int j=0;j<=pos;j++){ float*vj=Vh+(size_t)j*hd; vaxpy_f32(oh,vj,sc[j],hd); }
                if(gT_on) gT_attn_av += now()-_tav; }
        } else {
            attn_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,kv->ctx,g_attn_nt);
        }
        if(gT_on) gT_attn += now()-_at;
        g_lin_class=LIN_O; lin_mm(&ly->o,att,tmp,nt,Abuf); g_lin_class=LIN_NONE;
        for(int i=0;i<d;i++)h[i]+=tmp[i];
        rmsnorm(hn,h,ly->ffn_norm,d,m->eps);
        /* router precision mode (codex_recs_1.md §22.7-22.8, §22.15): int8-M1 is the DEFAULT
         * (g_router_mode=2) since 2026-07-26 -- passed the multi-prompt quality harness (6.1%
         * mismatch, -0.0034 nats/tok NLL delta, 2.1% divergence, 13.7% faster, all thresholds PASS).
         * int4 HP-Lin: 33% faster router-bucket but 58.9% of routing decisions get >=1/8 expert
         * swapped vs fp32 (avg 1.38/8) -- REJECTED, stays behind g_router_mode=1. fp32
         * (g_router_mode=0) remains available as an explicit exact revert flag. router shares its
         * int4-HP activation pack with eg/eu below regardless of mode (same hn, same K=d) -- pack
         * once, matching the P0.2 pattern used for q/k/v; int8 mode needs its own differently-shaped
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
              if(g_swiglu_fast==1) swiglu_hswish_rvv(g,u,moe);
              else if(g_swiglu_fast==2) swiglu_ratsig_rvv(g,u,moe);
              else swiglu_exact(g,u,moe);
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
/* Expanded 2026-07-26 per explicit direction to grow quality validation well past the 1,063-token
 * checkpoint (§22.20-21): teacher-forced positions roughly double (376->752-ish) emphasizing the
 * two long-context/free-running prompts, real-text corpus grows separately (see PPL_NTEXTS below)
 * with a heavier weight on code/multilingual/reasoning.
 *
 * MAXCTX MUST cover the max of (a) every HarnessPrompt's prefill+gen (worst case hp9/hp10:
 * 113+60=173) and (b) every PplText's raw length, since harness_eval_ppl() walks positions
 * 0..txt->n-2 over the SAME kv.ctx-sized cache (worst case ppl_code2: 355 tokens) -- confirmed
 * real bug (codex_recs_1.md §22.25): with the prior HARNESS_MAXCTX=200, harness_eval_ppl's forward()
 * calls at pos>=200 silently overran kv.Kc/kv.Vc (calloc'd to exactly nl*200*kvd floats), corrupting
 * unrelated heap allocations -- the eventual segfault surfaced much later, in an unrelated SwiGLU
 * phase, exactly the kind of delayed, misleading symptom heap corruption produces. 512 gives
 * comfortable headroom past the actual 355-token requirement. */
#define HARNESS_NP 10
#define HARNESS_MAXCTX 512
#define HARNESS_GEN_SHORT 80
#define HARNESS_GEN_LONG 60
#define HARNESS_GEN_MAX 80 /* = max(HARNESS_GEN_SHORT, HARNESS_GEN_LONG); sizes fixed local buffers */

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

/* n fields audited against actual array length 2026-07-26 (codex_recs_1.md §22.25) after ASan
 * caught a related mismatch elsewhere: hp3(15->16), hp4(22->24), hp9(113->124), hp10(113->155)
 * were all stale/under-declared -- silently truncating the prefill actually fed to the model
 * (not a memory-safety bug, since under-declaring stays within the array's real bounds, but a
 * real correctness bug: every harness run all session used a shorter prefill than the tokenized
 * text actually contains, especially hp10 -- 113 of 155 real tokens, over a quarter dropped).
 * Array *content* verified sane (coherent continuation, no duplication) before trusting it as the
 * correct length. */
typedef struct { const int*toks; int n; const char*name; int gen; } HarnessPrompt;
static const HarnessPrompt g_hprompts[HARNESS_NP] = {
    {hp1,12,"factual/capitals",HARNESS_GEN_SHORT}, {hp2,6,"factual/chemistry",HARNESS_GEN_SHORT},
    {hp3,16,"reasoning/syllogism",HARNESS_GEN_SHORT}, {hp4,24,"reasoning/sequence",HARNESS_GEN_SHORT},
    {hp5,23,"code/fibonacci",HARNESS_GEN_SHORT}, {hp6,24,"code/is_prime",HARNESS_GEN_SHORT},
    {hp7,7,"multilingual/french",HARNESS_GEN_SHORT}, {hp8,2,"multilingual/chinese",HARNESS_GEN_SHORT},
    {hp9,124,"long-context/narrative",HARNESS_GEN_LONG}, {hp10,155,"long-context/technical",HARNESS_GEN_LONG},
};

/* Real-text perplexity corpus (codex_recs_1.md §22.20 remediation, step 2: "perplexity-style
 * text... not only teacher-forced [against a model's own generated continuation]"). Independently
 * authored text (public-domain literature; hand-composed technical/code/reasoning prose), NOT
 * generated by any config being compared -- avoids the circularity of a model looking good at
 * predicting what it (or a close sibling config) already believed was likely. Pure forward-pass
 * NLL over real preceding context, no generation loop. */
#define PPL_NTEXTS 9
static const int ppl_lit[]={2132,572,279,1850,315,3039,11,432,572,279,11785,315,3039,11,432,572,279,4231,315,23389,11,432,572,279,4231,315,45237,2090,11,432,572,279,16342,315,16396,11,432,572,279,16342,315,9653,360,487,11,432,572,279,3200,315,8658,11,432,572,279,3200,315,53696,11,432,572,279,10464,315,3900,11,432,572,279,12406,315,45896,11,582,1030,4297,1573,601,11,582,1030,4302,1573,601,11,582,1033,678,2087,2118,311,31350,11,582,1033,678,2087,2118,279,1008,1616,13,2619,1033,264,11477,448,264,3460,16535,323,264,27906,448,264,14396,3579,11,389,279,43621,315,9448,26,1052,1033,264,11477,448,264,3460,16535,323,264,27906,448,264,6624,3579,11,389,279,43621,315,9625,13,758,2176,5837,432,572,48379,1091,25055,311,279,89113,315,279,3234,74898,315,775,4693,323,94361,11,429,2513,304,4586,1033,22378,369,3512,624};
static const int ppl_tech[]={785,431,27629,19625,7600,738,17646,374,5798,2163,264,2613,2331,7546,7600,738,11,11577,553,264,4013,315,10101,5297,19721,13,576,2331,738,18653,279,15811,34784,11,19819,11,323,2524,61313,11221,2500,389,1449,25879,287,8129,11,1393,19721,912,16928,1741,438,46444,323,12804,11,24510,4938,7525,11,3175,323,1990,16052,19057,1459,11,323,4621,8692,13,1096,43893,2884,15354,264,16392,14692,4211,1172,279,19721,9760,311,264,2169,3081,11,504,264,17377,22864,8003,7152,311,264,2480,3766,17654,13,576,4621,8894,304,3953,38919,264,3890,29325,4621,4161,1034,11,6693,7373,2374,374,11105,518,15592,1526,279,348,746,85,742,7600,4751,1091,8356,518,19192,882,11,10693,279,1852,19697,7868,311,1598,29720,3941,38337,448,2155,16533,4621,4161,64411,11,504,220,16,17,23,9472,311,220,16,15,17,19,9472,476,7797,624};
static const int ppl_code[]={1040,17718,96781,510,262,707,1304,2327,3804,721,982,286,656,12576,284,2240,271,262,707,5656,1193,11,897,982,286,421,656,12576,374,2240,510,310,656,12576,284,6018,3679,340,286,770,510,310,656,1436,4208,1193,12576,11,897,692,262,707,716,4208,1193,11,2436,11,897,982,286,421,897,366,2436,2824,510,310,421,2436,8272,374,2240,510,394,2436,8272,284,6018,3679,340,310,770,510,394,656,1436,4208,6958,8272,11,897,340,286,770,510,310,421,2436,9517,374,2240,510,394,2436,9517,284,6018,3679,340,310,770,510,394,656,1436,4208,6958,9517,11,897,692,262,707,5610,1193,11,897,982,286,1482,284,656,12576,198,286,1393,1482,374,537,2240,510,310,421,897,621,1482,2824,510,394,470,3007,198,310,1482,284,1482,8272,421,897,366,1482,2824,770,1482,9517,198,286,470,3557,198};
static const int ppl_reason[]={37175,264,5297,9530,315,32417,37402,5619,7411,11,17779,41047,1119,3040,27976,315,60659,20803,1817,13,1416,264,3175,3701,374,14764,518,4194,11,279,18927,315,13330,264,4746,374,6896,825,8338,11,2474,60659,315,279,32417,37402,7411,9173,311,429,7781,13,576,18927,315,13330,264,3579,3701,11,7290,264,25072,11,27906,11,476,11477,11,374,2326,270,404,665,306,4997,11,2474,1817,7781,42972,2326,1741,7411,13,4220,1378,4357,525,537,52479,13761,11,1576,2326,315,279,60659,22662,525,5577,3579,7411,13,2014,1477,279,18927,315,13330,264,3701,429,374,2987,264,4746,476,264,3579,3701,11,825,1969,912,279,3842,48216,323,1221,32256,279,18927,315,279,27248,11,30426,1990,25009,279,2326,7411,429,26553,2176,4682,24303,624};

/* Expansion (2026-07-26, per explicit direction "expand quality validation ... emphasizing code,
 * multilingual, reasoning and long free-running output"): five more independently-authored texts
 * -- a longer/harder code sample (graph algorithm, not the earlier BST), two genuinely multilingual
 * prose passages (French, Spanish -- neither language appears anywhere else in this harness), a
 * second longer reasoning sample (an induction proof, distinct proof shape from the probability
 * one above), and a second technical sample (thermal throttling in embedded SoCs -- chosen because
 * it is directly relevant to this project's own benchmark methodology, testing whether the model
 * reasons coherently about a domain adjacent to its own inference substrate). Tokenized with the
 * same real vendor tokenizer (`llama-tokenize --ids --no-bos`), same discipline as every other
 * array in this file -- not hand-encoded, not approximated. */
static const int ppl_code2[]={474,88522,271,750,1853,41808,13342,24312,11,1191,982,262,26552,284,314,3509,25,2224,492,13573,863,369,2436,304,4771,532,262,26552,28463,60,284,220,15,198,262,11994,284,738,741,262,17364,284,17826,15,11,1191,5563,262,1393,17364,510,286,1482,19464,11,1482,5084,284,88522,48035,676,453,73024,340,286,421,1482,5084,304,11994,510,310,3060,198,286,11994,1364,8762,5084,340,286,369,9565,11,4680,304,4771,25767,5084,936,3615,3932,310,421,9565,304,11994,510,394,3060,198,310,9144,284,1482,19464,488,4680,198,310,421,9144,366,26552,58,36469,10343,394,26552,58,36469,60,284,9144,198,394,88522,48035,676,1116,73024,11,320,46274,11,9565,1171,262,470,26552,271,750,43828,2638,68650,11,2169,982,262,1815,284,4167,262,2436,284,2169,198,262,1393,2436,374,537,2240,510,286,1815,2057,6958,340,286,2436,284,3681,670,6958,340,262,1815,32081,741,262,470,1815,271,750,1853,41808,13342,6615,2638,24312,11,1191,11,2169,982,262,26552,284,314,3509,25,2224,492,13573,863,369,2436,304,4771,532,262,3681,284,314,3509,25,2240,369,2436,304,4771,532,262,26552,28463,60,284,220,15,198,262,17364,284,17826,15,11,1191,5563,262,11994,284,738,741,262,1393,17364,510,286,1482,19464,11,1482,5084,284,88522,48035,676,453,73024,340,286,421,1482,5084,304,11994,510,310,3060,198,286,11994,1364,8762,5084,340,286,421,1482,5084,621,2169,510,310,1438,198,286,369,9565,11,4680,304,4771,25767,5084,936,3615,3932,310,9144,284,1482,19464,488,4680,198,310,421,9144,366,26552,58,36469,10343,394,26552,58,36469,60,284,9144,198,394,3681,58,36469,60,284,1482,5084,198,394,88522,48035,676,1116,73024,11,320,46274,11,9565,1171,262,470,26552,57033,1125,43828,2638,68650,11,2169,8};
static const int ppl_french[]={7839,9241,44789,34781,2667,550,11,512,2098,424,939,97285,25801,642,94099,15209,274,17321,2778,2111,41525,5165,963,13,11615,1063,261,1603,409,141910,7774,3952,64,1167,46738,85533,3541,435,1137,81412,14508,27700,3784,27700,272,15083,963,1187,1992,3784,939,66961,622,288,6995,3831,11,43729,3784,939,1329,11981,348,3341,86058,26486,19093,642,62384,14508,1875,72,1346,30532,13,11615,14132,1783,511,5822,9971,88710,34497,3845,293,10965,4003,7774,5908,85,14350,32570,326,25184,3760,11,3845,83651,3845,274,91906,5517,258,38623,326,53286,14093,285,1315,732,1515,446,480,478,15537,99064,1330,11,1842,3845,51950,38623,3541,78980,3845,14126,4225,5607,1167,5519,82357,1709,512,8322,2205,13,362,9635,76392,87153,11,3654,288,26414,34833,15632,306,409,42543,1524,3761,259,1038,84,3590,662,1186,7459,51885,326,6,80816,409,86225,99030,11,662,9333,309,10394,424,517,3541,7482,77411,13968,1842,662,3930,517,1187,34755,34428,6866,3541,35182,6990,8303,13,11615,99146,14789,304,963,6743,2200,25,44789,25801,642,37731,283,684,6185,16198,12815,304,56458,361,11,2635,13700,1346,650,3527,62859,409,86225,14132,1783,662,922,35993,294,22052,59597,409,17098,5519,61283,1238,11,53567,285,1709,294,80879,1976,11680,409,511,348,1776,47807,1114,11,39868,4788,294,6,1587,404,261,3541,2725,15398,1368,62746,17276,13,23769,6027,5525,13527,78251,6185,3405,5519,3460,1729,3761,7774,17191,650,38281,139819,478,17950,517,11,7906,47126,6362,939,68306,20115,8303,409,7042,13};
static const int ppl_spanish[]={1702,63459,11825,92823,409,97092,58763,11,1187,138747,1594,55462,511,6386,5508,5249,662,23895,409,2478,939,2577,2426,436,10918,33671,288,409,1187,3119,52407,77710,3362,13,3984,1884,66037,1613,7865,2123,409,5141,817,11437,3473,11,63459,56329,7437,5093,48176,29294,5690,86718,85100,11,6386,32963,2123,264,3619,27851,3725,436,6761,36016,409,6592,11431,272,380,46528,297,409,54204,11431,6051,3831,1709,435,5059,20563,39092,449,276,650,2629,258,15561,733,4942,13,386,49892,32086,11,2478,3619,27851,10918,27131,5553,913,8398,66057,40197,409,650,68529,51664,11,775,1709,97570,17090,5093,939,343,928,55844,1709,56303,300,56329,55170,346,662,2478,36535,35905,88844,15131,655,45915,13,1674,13259,436,19615,3530,19001,74109,2123,264,9342,277,390,88149,409,6427,5721,409,55462,409,9323,12058,685,379,390,12183,333,300,26435,263,11107,1709,5828,4814,939,258,1168,56398,655,79890,505,1603,6496,7437,46326,11155,264,5141,43426,3473,409,2953,4589,42664,436,13,9403,300,44023,51541,11,58366,2706,295,24358,300,11,88499,276,22106,8792,32086,409,30137,4589,1709,1562,268,650,81636,662,2478,2783,436,7953,409,11150,12430,1709,2453,47322,409,2478,59080,86163,84430,3348,4211,277,14493,58507,13,3984,26192,1531,650,296,11983,3955,409,2048,66247,1346,24119,642,1709,11,662,655,25937,409,2478,57533,11,452,344,1103,655,41076,7437,36220,385,409,80529,13};
static const int ppl_reason2[]={37175,279,3717,429,369,1449,6785,7546,308,11,279,2629,315,279,1156,308,10322,5109,16819,308,52263,13,2014,12118,419,553,37056,11,1156,10146,279,2331,1142,25,979,308,16819,825,11,279,2629,315,279,1156,10322,1372,374,4936,825,11,323,825,52263,374,1083,825,11,773,279,2331,1142,9982,13,9295,11,9658,279,5114,374,830,369,1045,24168,6785,7546,595,11,7290,279,2629,315,279,1156,595,10322,5109,16819,595,52263,26,419,24335,374,2598,279,304,67143,30078,13,576,5795,374,311,1473,429,279,5114,1221,9982,369,595,5519,825,13,576,2629,315,279,1156,595,5519,825,10322,5109,374,279,2629,315,279,1156,595,10322,5109,5519,279,595,10,16,7563,10322,1372,13,576,595,7563,10322,1372,304,279,8500,374,2661,553,1378,3039,595,27283,825,11,773,279,595,10,16,7563,10322,1372,374,1378,3039,595,10,16,27283,825,11,892,15491,9606,311,1378,595,5519,825,13,3216,279,304,67143,30078,11,279,2629,315,279,1156,595,10322,5109,374,595,52263,11,773,279,2629,315,279,1156,595,5519,825,10322,5109,9044,595,52263,5519,1378,595,5519,825,13,1096,7493,9363,62166,1119,595,10,16,52263,11,892,374,6896,279,897,279,3717,55878,369,308,6144,311,595,5519,825,13,8704,279,2331,1142,9982,323,279,304,67143,3019,12440,23377,279,8046,315,279,5114,504,595,311,595,5519,825,11,279,17508,315,35972,37056,35655,429,279,5114,9982,369,1449,6785,7546,308,13,1096,11064,5383,11,10146,264,2331,1142,323,1221,1473,279,3019,4637,74898,279,3343,11,24724,14971,6814,3614,17272,1211,11,1372,10126,11,323,279,6358,315,30819,25185,11,892,374,949,315,3170,432,374,3545,825,315,279,1156,46899,11064,12538,15599,311,4143,315,37596,13};
static const int ppl_tech2[]={83466,5942,10326,11582,573,429,3769,5248,97782,12564,26968,8630,264,3175,2746,3579,264,24999,23504,1948,28659,63762,323,28387,1968,2966,13,362,16392,2578,28141,264,20524,8866,11639,1632,3403,1128,1181,23189,323,26917,6291,646,13879,55234,11,38561,389,264,28387,19044,311,3019,33773,1495,3055,389,1737,645,25092,5312,264,12171,13,1096,7709,374,5990,29447,2337,2805,62019,11,892,4583,1573,279,19044,30857,288,11,714,9044,279,24456,8168,304,894,53596,4303,369,22008,315,6486,476,5021,11,1741,438,264,4128,1614,47116,264,1293,2033,13,362,49665,28431,37052,429,10953,1172,264,22955,315,11211,1283,264,9255,1191,646,8916,1895,5109,429,525,59726,5080,1091,1128,264,1196,1035,3139,304,6588,11,4936,1576,279,50592,702,537,3602,8643,1181,24020,20733,10350,9315,13,576,4396,65760,374,311,37867,458,2856,8205,5239,3241,323,6629,1172,279,24020,20733,5537,315,264,38944,1293,1598,11,323,11,1380,3204,11,311,1487,2746,9315,476,8866,11639,16263,63762,773,429,264,28387,2852,31511,1598,646,387,38475,504,264,35197,2155,5068,17484,13,17439,93208,14431,912,264,4623,3112,35144,25,26968,646,42166,28135,11,773,264,53596,429,28635,2795,1948,26968,949,3117,1526,264,1598,1231,1473,264,3019,2297,304,63762,429,702,4302,311,653,448,279,3162,1815,4429,323,4297,311,653,448,892,10652,6932,311,387,37291,518,429,4445,13};

/* ppl_reason's declared length was stale (176) vs its real 149 elements -- ASan caught the
 * resulting global-buffer-overflow (codex_recs_1.md §22.25): harness_eval_ppl read 27 tokens past
 * the array's end, into ppl_code's memory, for every real-text-perplexity evaluation that included
 * ppl_reason this whole session (§22.20/§22.21 among them). Fixed here; content verified as a
 * legitimate 149-token text, not truncated or duplicated. */
typedef struct { const int*toks; int n; const char*name; } PplText;
static const PplText g_ppltexts[PPL_NTEXTS] = {
    {ppl_lit,176,"literature(public-domain)"}, {ppl_tech,168,"technical(hand-composed)"},
    {ppl_code,171,"code(hand-composed)"}, {ppl_reason,149,"reasoning(hand-composed)"},
    {ppl_code2,355,"code2(hand-composed,graph-algo)"}, {ppl_french,290,"multilingual(french)"},
    {ppl_spanish,269,"multilingual(spanish)"}, {ppl_reason2,345,"reasoning2(induction-proof)"},
    {ppl_tech2,268,"technical2(thermal-throttling)"},
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
/* SwiGLU evaluation (§22.18): reference generation under int8-M1 router (g_router_mode=2 -- NOT
 * fp32/mode=0 like harness_run_fp32 above, which is the original router-promotion ground truth,
 * kept as-is for history) with EXACT SwiGLU, hardcoded (g_swiglu_fast=0) regardless of whatever
 * the CLI/production default currently is -- this is the fixed gold reference every SwiGLU
 * candidate is measured against, so it must stay pinned to exact math even after §22.21 promoted
 * rational-Pade to the production default. This is deliberately a separate function rather than a
 * parameterized reuse of harness_run_fp32: the SwiGLU question is "does adding fast SwiGLU on top
 * of what's already shipping cause a problem", not a re-litigation of the router decision, so the
 * router must match current production exactly while SwiGLU stays exact. */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void harness_run_prod_reference(Model*m,const HarnessPrompt*hp,int*out_toks,float*out_selfnll,Kv*kv,
        float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*gg,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2,float*logits){
    memset(kv->Kc,0,(size_t)m->nl*kv->ctx*kv->kvd*4); memset(kv->Vc,0,(size_t)m->nl*kv->ctx*kv->kvd*4);
    g_router_mode=2; g_router_validate=0; g_swiglu_fast=0;
    for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    int cur=argmax(logits,m->vocab); out_toks[0]=cur; out_selfnll[0]=harness_nll(logits,m->vocab,cur);
    for(int s=1;s<hp->gen;s++){
        forward(m,cur,hp->n+s-1,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
        cur=argmax(logits,m->vocab); out_toks[s]=cur; out_selfnll[s]=harness_nll(logits,m->vocab,cur);
    }
}
/* teacher-forces under the given (router_mode, swiglu_fast) config -- same teacher-forcing
 * methodology as the router harness (feed the reference token regardless of what this config
 * would itself pick, so one divergence never compounds into a different context for later
 * positions). router_mode is a parameter (not hardcoded to production) so this same function can
 * measure either "SwiGLU's marginal effect on top of production routing" (router_mode=2) or "the
 * full combined stack's total deviation from the original fp32-exact engine" (router_mode=2
 * teacher-forced against a router_mode=0 reference) -- codex_recs_1.md §22.20 remediation step 3.
 * No router validation needed here (not testing router quality), so no do_validate split like the
 * router version required. */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void harness_run_swiglu_teacherforced(Model*m,const HarnessPrompt*hp,const int*ref,int router_mode,int swiglu_fast,int*out_div,float*out_nll,Kv*kv,
        float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*gg,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2,float*logits){
    memset(kv->Kc,0,(size_t)m->nl*kv->ctx*kv->kvd*4); memset(kv->Vc,0,(size_t)m->nl*kv->ctx*kv->kvd*4);
    g_router_mode=router_mode; g_router_validate=0; g_swiglu_fast=swiglu_fast;
    for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    for(int s=0;s<hp->gen;s++){
        out_div[s]=(argmax(logits,m->vocab)!=ref[s]); out_nll[s]=harness_nll(logits,m->vocab,ref[s]);
        forward(m,ref[s],hp->n+s,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    }
    g_swiglu_fast=0;
}
/* real-text perplexity: pure forward pass over independently-authored text under the given
 * (router_mode, swiglu_fast) config, accumulating NLL of each REAL next token given the REAL
 * preceding context -- not teacher-forced against any model's own generation. Returns avg
 * nats/token; caller exponentiates for a perplexity figure. */
__attribute__((noinline,optimize("no-tree-vectorize")))
static double harness_eval_ppl(Model*m,const PplText*txt,int router_mode,int swiglu_fast,Kv*kv,
        float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*gg,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2,float*logits){
    memset(kv->Kc,0,(size_t)m->nl*kv->ctx*kv->kvd*4); memset(kv->Vc,0,(size_t)m->nl*kv->ctx*kv->kvd*4);
    g_router_mode=router_mode; g_router_validate=0; g_swiglu_fast=swiglu_fast;
    double total_nll=0; int count=0;
    for(int p=0;p<txt->n-1;p++){
        forward(m,txt->toks[p],p,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
        total_nll+=harness_nll(logits,m->vocab,txt->toks[p+1]); count++;
    }
    g_swiglu_fast=0;
    return count?total_nll/count:0.0;
}
/* free-running generation under the given (router_mode, swiglu_fast) config -- NOT teacher-forced,
 * each config generates its own independent continuation. Qualitative spot-check: teacher-forcing
 * always resets to the reference token every step, which structurally cannot reveal whether a
 * candidate's errors compound differently over a longer, uncorrected horizon (codex_recs_1.md
 * §22.20 remediation step 2). */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void harness_generate_generic(Model*m,const HarnessPrompt*hp,int*out_toks,int router_mode,int swiglu_fast,Kv*kv,
        float*hn,float*q,float*k,float*vv,float*att,float*tmp,float*gg,float*u,float*eout,uint8_t*Abuf,uint8_t*Abuf2,float*logits){
    memset(kv->Kc,0,(size_t)m->nl*kv->ctx*kv->kvd*4); memset(kv->Vc,0,(size_t)m->nl*kv->ctx*kv->kvd*4);
    g_router_mode=router_mode; g_router_validate=0; g_swiglu_fast=swiglu_fast;
    for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
    int cur=argmax(logits,m->vocab); out_toks[0]=cur;
    for(int s=1;s<hp->gen;s++){
        forward(m,cur,hp->n+s-1,kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
        cur=argmax(logits,m->vocab); out_toks[s]=cur;
    }
    g_swiglu_fast=0;
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
static void run_quality_harness(Model*m,Gguf*gguf){
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

    /* ===== SwiGLU evaluation v2 (codex_recs_1.md §22.20, remediation after §22.19's retraction).
     * Thresholds tightened and RESTATED IN PERPLEXITY-MULTIPLIER TERMS per external review of the
     * first attempt -- §22.18's "0.3 nats/tok" bound was, unnoticed at the time, a 35% perplexity-
     * inflation ceiling, far too loose; hard-swish's actual 0.0969 nats = 10.2% inflation should
     * have read as an obvious warning sign, not a threshold pass. New bar: perplexity multiplier
     * < 1.05 (5% inflation, ln(1.05)=0.0488 nats/tok). Divergence and speed bars unchanged.
     * Evaluated two ways per candidate: marginal effect on top of production routing (int8 router
     * held constant, matching §22.18's original scope) AND full combined-stack deviation from the
     * ORIGINAL fp32-router+exact-SiLU ground truth (§22.18 never measured this). */
    static int swiglu_ref[HARNESS_NP][HARNESS_GEN_SHORT];
    static float swiglu_selfnll[HARNESS_NP][HARNESS_GEN_SHORT];
    printf("\n=== quality harness: SwiGLU phase 1 -- production reference (int8 router + exact SwiGLU) ===\n");
    gT_on=1; gT_tok=0; gT_actpack=gT_lin=gT_attn=gT_rope=gT_router=gT_swiglu=gT_rest=0;
    for(int i=0;i<HARNESS_NP;i++){
        harness_run_prod_reference(m,&g_hprompts[i],swiglu_ref[i],swiglu_selfnll[i],&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
        printf("  [%2d/%d] %-24s done\n",i+1,HARNESS_NP,g_hprompts[i].name);
    }
    long tok_exact=gT_tok; double swiglu_exact_t=gT_swiglu;
    gT_on=0;

    const char* cand_names[3] = {"","hard-swish","rational-Pade"};
    int teacher_gate_pass[3] = {0,0,0};
    for(int cand_mode=1; cand_mode<=2; cand_mode++){
        printf("\n=== quality harness: SwiGLU phase 2 -- %s candidate ===\n", cand_names[cand_mode]);
        typedef struct { const char*name; int gen,divergent; float nll_fast,nll_exact; } SRes;
        SRes sres[HARNESS_NP];
        gT_on=1; gT_tok=0; gT_swiglu=0;
        for(int i=0;i<HARNESS_NP;i++){
            int div[HARNESS_GEN_SHORT]; float nll[HARNESS_GEN_SHORT];
            harness_run_swiglu_teacherforced(m,&g_hprompts[i],swiglu_ref[i],2,cand_mode,div,nll,&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
            SRes*r=&sres[i]; r->name=g_hprompts[i].name; r->gen=g_hprompts[i].gen; r->divergent=0; r->nll_fast=0; r->nll_exact=0;
            for(int s=0;s<r->gen;s++){ r->divergent+=div[s]; r->nll_fast+=nll[s]; r->nll_exact+=swiglu_selfnll[i][s]; }
            printf("  [%2d/%d] %-24s divergent=%2d/%-2d  nll_delta=%+.4f\n",i+1,HARNESS_NP,r->name,r->divergent,r->gen,(r->nll_fast-r->nll_exact)/r->gen);
        }
        long tok_fast=gT_tok; double swiglu_fast_t=gT_swiglu;
        gT_on=0;

        int total_gen2=0,total_div2=0; double total_nll_fast=0,total_nll_exact2=0;
        for(int i=0;i<HARNESS_NP;i++){ total_gen2+=sres[i].gen; total_div2+=sres[i].divergent; total_nll_fast+=sres[i].nll_fast; total_nll_exact2+=sres[i].nll_exact; }
        double div_rate2=100.0*total_div2/total_gen2, nll_delta2=(total_nll_fast-total_nll_exact2)/total_gen2;
        double exact_ms=tok_exact?swiglu_exact_t/tok_exact*1e3:0, fast_ms=tok_fast?swiglu_fast_t/tok_fast*1e3:0;
        double swiglu_speedup=exact_ms?100.0*(exact_ms-fast_ms)/exact_ms:0;
        double ppl_mult=exp(nll_delta2);

        /* full-stack deviation: teacher-force the SAME candidate against the ORIGINAL fp32+exact
         * ground truth (ref_toks/self_nll, computed in the router phase above) -- remediation
         * step 3, never measured in §22.18. */
        int total_div3=0,total_gen3=0; double total_nll_fast3=0,total_nll_exact3=0;
        for(int i=0;i<HARNESS_NP;i++){
            int div3[HARNESS_GEN_SHORT]; float nll3[HARNESS_GEN_SHORT];
            harness_run_swiglu_teacherforced(m,&g_hprompts[i],ref_toks[i],2,cand_mode,div3,nll3,&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
            int gen=g_hprompts[i].gen;
            for(int s=0;s<gen;s++){ total_div3+=div3[s]; total_nll_fast3+=nll3[s]; total_nll_exact3+=self_nll[i][s]; }
            total_gen3+=gen;
        }
        double div_rate3=100.0*total_div3/total_gen3, nll_delta3=(total_nll_fast3-total_nll_exact3)/total_gen3;
        double ppl_mult3=exp(nll_delta3);

        printf("\n--- %s vs int8+exact PRODUCTION (marginal SwiGLU effect) ---\n", cand_names[cand_mode]);
        printf("  divergence %d/%d (%.1f%%), NLL delta %+.4f nats/tok, perplexity multiplier x%.4f (%.1f%% inflation)\n",
            total_div2,total_gen2,div_rate2,nll_delta2,ppl_mult,100.0*(ppl_mult-1.0));
        printf("--- %s vs ORIGINAL fp32+exact ground truth (full combined-stack deviation) ---\n", cand_names[cand_mode]);
        printf("  divergence %d/%d (%.1f%%), NLL delta %+.4f nats/tok, perplexity multiplier x%.4f (%.1f%% inflation)\n",
            total_div3,total_gen3,div_rate3,nll_delta3,ppl_mult3,100.0*(ppl_mult3-1.0));
        printf("speed: exact swiglu %.2fms/tok, %s %.2fms/tok (%.1f%% time reduction, %.2fx faster)\n",
            exact_ms,cand_names[cand_mode],fast_ms,swiglu_speedup,fast_ms?exact_ms/fast_ms:0.0);

        printf("--- %s promotion thresholds (perplexity multiplier restated per external review, codex_recs_1.md §22.20) ---\n", cand_names[cand_mode]);
        int q1=ppl_mult<1.05, q2=div_rate2<15.0, q3=swiglu_speedup>=15.0;
        printf("  [%s] perplexity multiplier < 1.05 (5%% inflation ceiling), vs production  (got x%.4f, %.1f%%)\n",q1?"PASS":"FAIL",ppl_mult,100.0*(ppl_mult-1.0));
        printf("  [%s] token divergence < 15%% (vs production)                              (got %.1f%%)\n",q2?"PASS":"FAIL",div_rate2);
        printf("  [%s] swiglu bucket >=15%% faster than exact                                (got %.1f%%)\n",q3?"PASS":"FAIL",swiglu_speedup);
        printf("  (informational, not gated) full-stack perplexity multiplier vs fp32+exact ground truth: x%.4f (%.1f%%)\n",ppl_mult3,100.0*(ppl_mult3-1.0));
        teacher_gate_pass[cand_mode]=q1&&q2&&q3;
        printf("PRELIMINARY VERDICT: %s %s teacher-forced gates; final eligibility also requires the real-text gates below.\n",
            cand_names[cand_mode],teacher_gate_pass[cand_mode]?"PASSES":"FAILS");
    }

    /* ===== real-text perplexity: independently-authored text, 4 configs (codex_recs_1.md §22.20
     * remediation step 2/3) -- pure forward-pass NLL over REAL preceding context, not teacher-
     * forced against any model's own generated continuation. ===== */
    printf("\n=== quality harness: real-text perplexity (%d independently-authored texts) ===\n",PPL_NTEXTS);
    struct { const char*name; int router_mode, swiglu_mode; } cfgs[4] = {
        {"fp32+exact (original ground truth)",0,0}, {"int8+exact (production)",2,0},
        {"int8+hard-swish",2,1}, {"int8+rational-Pade",2,2},
    };
    double ppl_agg[4], ppl_by_text[4][PPL_NTEXTS];
    for(int c=0;c<4;c++){
        double sum=0; int total_toks=0;
        printf("  --- %s ---\n",cfgs[c].name);
        for(int t=0;t<PPL_NTEXTS;t++){
            double nll=harness_eval_ppl(m,&g_ppltexts[t],cfgs[c].router_mode,cfgs[c].swiglu_mode,&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
            ppl_by_text[c][t]=nll;
            printf("      %-28s NLL=%.4f nats/tok  perplexity=%.3f\n",g_ppltexts[t].name,nll,exp(nll));
            sum+=nll*(g_ppltexts[t].n-1); total_toks+=g_ppltexts[t].n-1;
        }
        ppl_agg[c]=sum/total_toks;
        printf("    aggregate: NLL=%.4f nats/tok  perplexity=%.3f  (%d tokens)\n",ppl_agg[c],exp(ppl_agg[c]),total_toks);
    }
    printf("  real-text perplexity multiplier vs fp32+exact ground truth: int8+exact x%.4f | int8+hard-swish x%.4f | int8+rational-Pade x%.4f\n",
        exp(ppl_agg[1]-ppl_agg[0]), exp(ppl_agg[2]-ppl_agg[0]), exp(ppl_agg[3]-ppl_agg[0]));
    printf("  real-text perplexity multiplier vs int8+exact production:                      int8+hard-swish x%.4f | int8+rational-Pade x%.4f\n",
        exp(ppl_agg[2]-ppl_agg[1]), exp(ppl_agg[3]-ppl_agg[1]));

    /* Final promotion gates were fixed before running the larger evaluation: aggregate real-text
     * perplexity inflation must stay below 5%, and no individual corpus may exceed 10%.  These
     * supplement (not replace) the teacher-forced quality/speed gates above. */
    printf("\n=== SwiGLU final promotion gates (predeclared before larger board run) ===\n");
    for(int cand_mode=1;cand_mode<=2;cand_mode++){
        int cfg=cand_mode+1;
        double agg_mult=exp(ppl_agg[cfg]-ppl_agg[1]);
        double worst_mult=0; int worst_text=0;
        for(int t=0;t<PPL_NTEXTS;t++){
            double mult=exp(ppl_by_text[cfg][t]-ppl_by_text[1][t]);
            if(mult>worst_mult){ worst_mult=mult; worst_text=t; }
        }
        int real_gate=agg_mult<1.05 && worst_mult<1.10;
        int final_pass=teacher_gate_pass[cand_mode] && real_gate;
        printf("  %s: aggregate x%.4f (<1.05), worst corpus x%.4f (<1.10, %s) -- %s\n",
            cand_names[cand_mode],agg_mult,worst_mult,g_ppltexts[worst_text].name,
            final_pass?"ELIGIBLE FOR PRODUCTION A/B":"NOT ELIGIBLE");
    }

    /* ===== free-running generation, not teacher-forced (remediation step 2) ===== */
    printf("\n=== quality harness: free-running generation (independent, qualitative spot-check) ===\n");
    int freerun_idx[2] = {0,8}; /* factual/capitals (short); long-context/narrative (long) */
    for(int fi=0; fi<2; fi++){
        int i=freerun_idx[fi];
        printf("  prompt: %s\n",g_hprompts[i].name);
        for(int c=1;c<4;c++){
            int toks[HARNESS_GEN_SHORT];
            harness_generate_generic(m,&g_hprompts[i],toks,cfgs[c].router_mode,cfgs[c].swiglu_mode,&kv,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2,logits);
            printf("    %-24s: ",cfgs[c].name);
            for(int s=0;s<g_hprompts[i].gen;s++) tok_print(gguf,toks[s]);
            printf("\n");
        }
    }

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
    g_swiglu_fast=(c>7)?atoi(v[7]):2; /* 0=exact(revert flag) 1=hard-swish(REJECTED §22.19-20 -- 11.0% perplexity inflation vs production on the expanded 1063-token/4-methodology eval, fails the 5% gate) 2=rational-Pade(DEFAULT since 2026-07-26, §22.21 -- passed the harness §22.20 AND a bounded production A/B: 2 paired trials, 9.7->11.4 tok/s, swiglu bucket 15.85ms->1.25ms (-92.1%), identical generated tokens both trials, no regression in any other bucket) */
    Model m; double tl=now(); int cached=0;
    if(cache_load(&m,cpath,nt)){ cached=1; fprintf(stderr,"loaded from cache in %.1fs  (%s)\n",now()-tl,cpath); }
    else { model_load(&m,&g,nt); fprintf(stderr,"requant loaded in %.1fs\n",now()-tl); }

    /* QWEN_CTXLEN (env var, same "avoid another positional production arg" convention as
     * QWEN_HARNESS -- codex_recs_1.md §22.23 context-length scaling benchmark): when set >0,
     * synthesizes a prefill of that many tokens (tiling the harness's real 113-token narrative
     * prompt, hp9, declared above) instead of the hardcoded 12-token ' Tokyo' prompt, and grows the
     * KV cache to fit. Lets the same binary/build measure how the attention bucket scales with
     * context length without a second benchmark harness. */
    int ctxlen_req=0; { const char*cv=getenv("QWEN_CTXLEN"); if(cv) ctxlen_req=atoi(cv); }
    /* QWEN_ATTN_NT (env var, same convention): number of pool threads that participate in
     * attention (attention_optimization_plan.md Phase 3, codex_recs_1.md §22.29). DEFAULT since
     * 2026-07-26 is min(nt,nkv) -- promoted after passing every predeclared keep criterion: tokens
     * identical at every worker count and context tested, ASan+UBSan clean, attention bucket
     * -74.4%/-74.9% at ctx=1024/512 (>=20% required), tok/s +160.4%/+117.1% (>=10% required),
     * short-context wall time *improved* 8.25% (>=2% regression tolerated). `QWEN_ATTN_NT=1`
     * remains an explicit serial revert flag, byte-identical to Phase 2's code path. Values above
     * g_pool_nt are clamped (no more workers than exist). */
    g_attn_nt = nt<m.nkv ? nt : m.nkv;
    { const char*av=getenv("QWEN_ATTN_NT"); if(av){ g_attn_nt=atoi(av); if(g_attn_nt>nt) g_attn_nt=nt; if(g_attn_nt<1) g_attn_nt=1; } }
    /* ctx must cover prefill+ngen regardless of which prompt path is taken -- the default path's
     * prefill is a fixed 12 tokens (see `np` below), NOT just the QWEN_CTXLEN path. Previously only
     * the QWEN_CTXLEN branch grew ctx, so any ngen (2nd CLI arg, directly user-controlled, no
     * bound) past ~52 on the default 12-token prompt silently overran the KV cache -- caught by the
     * new forward() bounds guard (codex_recs_1.md §22.25) the first time a longer decode was
     * actually requested, not hypothetically. */
    int base_np=(ctxlen_req>0)?ctxlen_req:12;
    int ctx=64; if(base_np+ngen+4>ctx) ctx=base_np+ngen+4;
    Kv kv; kv.kvd=m.nkv*m.hd; kv.ctx=ctx; kv.Kc=calloc((size_t)m.nl*ctx*kv.kvd,4); kv.Vc=calloc((size_t)m.nl*ctx*kv.kvd,4);
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
     * previously read past argv[6]. Root-caused in §22.16 to the unsafe tree-vectorized build;
     * the mandatory -fno-tree-vectorize flag fixes it.  The env trigger remains the stable harness
     * interface and avoids adding another positional production argument. */
    { const char*hv=getenv("QWEN_HARNESS"); if(hv && atoi(hv)){ run_quality_harness(&m,&g); return 0; } }

    static int prompt[1536]; int np;
    if(ctxlen_req>0){ np=ctxlen_req; for(int i=0;i<np;i++) prompt[i]=hp9[i%113]; }
    else { int base[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374}; np=12; memcpy(prompt,base,sizeof(base)); }
    double tp=now(); int first=0;
    for(int p=0;p<np;p++){ forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2); if(p==np-1)first=argmax(logits,m.vocab); }
    int echo_np=np<=64?np:64; /* QWEN_CTXLEN can make np huge (1024+); don't spam the log */
    printf("\nprompt      : "); for(int i=0;i<echo_np;i++)tok_print(&g,prompt[i]); if(np>echo_np)printf(" ...(%d more)",np-echo_np);
    printf("\nfirst argmax: %d ('",first); tok_print(&g,first); printf("')  expect 26194 (' Tokyo') -> %s%s\n", first==26194?"PASS":"FAIL", ctxlen_req>0?" (ctxlen test: mismatch expected, prompt isn't the capitals prompt)":"");
    printf("prefill %.2fs (%d tok)\n",now()-tp,np);
    printf("generation  : "); for(int i=0;i<echo_np;i++)tok_print(&g,prompt[i]); if(np>echo_np)printf(" ..."); tok_print(&g,first);
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
        /* At attn_nt=1 (serial), QK+softmax+AV+dispatch sums to ~the attention bucket, same
         * reconciliation check as Phase 1. At attn_nt>1, QK/softmax/AV are summed CPU-time across
         * all participating workers (so they naturally exceed wall-clock time by ~attn_nt) --
         * "attention bucket" (gT_attn) is the true wall-clock figure and the one that matters for
         * the A/B decision; "total work" is diagnostic (e.g. confirms parallel workers aren't
         * doing duplicate work: it should stay close to the attn_nt=1 total, not grow with nt). */
        printf("  attention breakdown (avg/%ld tok, ms): QK %.2f | softmax %.2f | AV %.2f (total work, %s) | dispatch %.2f | attention bucket (wall) %.2f, attn_nt=%d\n",
            gT_tok, gT_attn_qk/gT_tok*1e3, gT_attn_sm/gT_tok*1e3, gT_attn_av/gT_tok*1e3,
            g_attn_nt<=1?"= wall time":"summed across workers", gT_attn_dispatch/gT_tok*1e3, gT_attn/gT_tok*1e3, g_attn_nt);
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
