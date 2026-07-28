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
 * Build: gcc -O3 -fno-tree-vectorize -march=rv64gcv_zfh_zvfh_xsmtvdotii -fopenmp -o qwen_moe_hp qwen_moe_hp.c -lm -lpthread
 * Run  : LD_LIBRARY_PATH=/usr/lib ./qwen_moe_hp /root/models/Qwen3-30B-A3B-Q4_0.gguf [ngen] [nt]
 *
 * `zfh` (scalar half-precision) is REQUIRED on top of the long-standing `zvfh` (vector half-
 * precision) since the M-batch track's run_hp_m4 (codex_recs_1.md §22.35-36) uses a genuine scalar
 * `fmul.h` instruction -- `zvfh` alone does not imply it, confirmed by matching the vendor's own
 * build flags (`reference/spacemit-backend` compile_commands.json uses
 * `rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii`). Applies to every build of this file now,
 * not just M-batch-specific ones, since run_hp_m4 is always compiled in (only invoked when
 * QWEN_MBATCH_TEST=1 requests it).
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
/* M-batch track milestone 1 (codex_recs_1.md §22.35), validated in bench/vendor_ime_m4_probe.c:
 * M4 vs its own reconstructed reference has 0.119% mean error, essentially identical to M1's own
 * 0.119% (ratio 0.999x) -- the port is numerically correct. Ground truth: vendor's own packer
 * quantize_a_4row_i8_hp (reference/spacemit-backend/rvv_kernels.cpp:2100, vlenb==128 branch).
 * KEY property, different from every OTHER fusion in this file: the per-subblock quantization
 * scale is the MAX ABSOLUTE VALUE ACROSS ALL 4 BATCHED ROWS JOINTLY, not per row -- so this is NOT
 * expected to be bit-identical to 4 separate pack_A_hp_m1+run_hp_m1 calls on the same rows, a
 * genuine (small, principled) quantization tradeoff, not a bug. Record layout: 1160 bytes = 8
 * subblocks x 136B ([8B scale area, only offset+0 used -- one shared fp16 scale] + [4x int8[32]
 * row payloads, row-major]) + 64B a_sum trailer (4 rows x 8 fp16 values, PRE-SCALED as
 * -true_asum*8.0, same convention as M1) + 8B scale_avg area (only offset+0 used). */
static void pack_A_hp_m4(const float*a0,const float*a1,const float*a2,const float*a3,uint8_t*out /* 1160 bytes */){
    float scale_temp[NSUB]; float scale_avg=0.0f;
    for(int kk=0;kk<NSUB;kk++){
        float amax=1e-6f;
        for(int i=0;i<32;i++){
            float v0=fabsf(a0[kk*32+i]),v1=fabsf(a1[kk*32+i]),v2=fabsf(a2[kk*32+i]),v3=fabsf(a3[kk*32+i]);
            float mm=v0>v1?v0:v1; float nn=v2>v3?v2:v3; if(nn>mm)mm=nn;
            if(mm>amax) amax=mm;
        }
        scale_temp[kk]=amax/127.0f; scale_avg+=scale_temp[kk];
    }
    scale_avg/=NSUB;
    float scale_factor = scale_avg? 1.0f/scale_avg : 0.0f;
    uint16_t blkscale=f32_to_f16(scale_avg); memcpy(out+1152,&blkscale,2);

    const float*rows[4]={a0,a1,a2,a3};
    for(int kk=0;kk<NSUB;kk++){
        uint8_t*subblk=out+kk*136;
        float rep_scale = scale_temp[kk]? 1.0f/scale_temp[kk] : 0.0f;
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor); memcpy(subblk,&ssub,2);
        int8_t*quant_blk=(int8_t*)(subblk+8);
        for(int r=0;r<4;r++){
            int32_t sum=0; int8_t q[32];
            for(int i=0;i<32;i++){ int v=(int)lrintf(rows[r][kk*32+i]*rep_scale); if(v>127)v=127; if(v<-127)v=-127; q[i]=(int8_t)v; sum+=v; }
            memcpy(quant_blk+r*32,q,32);
            uint16_t as=f32_to_f16(-(float)sum*8.0f);
            memcpy(out+1088+r*16+kk*2,&as,2);
        }
    }
}
/* verbatim asm port of gemm_kernel_i8i4_hp_m4's no-zp branch (reference/spacemit-backend/
 * ime2_kernels.cpp:3360) -- confirmed the live branch for this engine's Q4_0 weights via
 * block_type_has_zp<block_q4_0>()==false (ime.cpp:107), so quant_b_zp==nullptr at the real vendor
 * call site (ime.cpp:276/583). ldc is the row stride in dst_c (in floats, not bytes -- matches
 * l->N, the full row width, since the caller writes one N-tile's worth per call but the 4 rows'
 * destinations are offset by the FULL row width, not just this tile's 32 columns). */
static void run_hp_m4(const uint8_t*a_data, const uint8_t*b_data, float*dst_c, long k_blks, long ldc){
    __asm__ volatile(
        "mv             t5, %[BK]                 \n\t"
        "mv             t6, %[A]                  \n\t"
        "mv             s5, %[B]                  \n\t"
        "li             t1, 0x4c00                \n\t"
        "fmv.h.x        fa6, t1                    \n\t"
        "vsetvli        t0, x0, e32, m1           \n\t"
        "vxor.vv        v28, v28, v28             \n\t"
        "vxor.vv        v29, v29, v29             \n\t"
        "vxor.vv        v30, v30, v30             \n\t"
        "vxor.vv        v31, v31, v31             \n\t"
        "li             t4, 8                     \n\t"
        "addi           t2, t6, 1088              \n\t"

        ".align 4                                 \n\t"
        "_M4_BLK_LPST%=:                          \n\t"
        "flh            fa1, 64(t2)               \n\t"
        "vsetvli        t0, x0, e32, m1           \n\t"
        "vxor.vv        v18, v30, v30             \n\t"
        "vxor.vv        v19, v31, v31             \n\t"
        "vxor.vv        v20, v30, v30             \n\t"
        "vxor.vv        v21, v31, v31             \n\t"
        "_M4_KsubBLK_LPST%=:                      \n\t"
        "flh            fa0,   0(t6)              \n\t"

        "vsetvli        t0, x0, e16, mf2          \n\t"
        "vle16.v        v12, (s5)                 \n\t"

        "fmul.h         fa2, fa0, fa6              \n\t"

        "vsetvli        t0, x0, e16, mf2          \n\t"
        "vfmul.vf       v16, v12, fa0             \n\t"
        "vfmul.vf       v17, v12, fa2             \n\t"

        "flh            ft1, 0(t2)                \n\t"
        "flh            ft2, 16(t2)                \n\t"
        "flh            ft3, 32(t2)                \n\t"
        "flh            ft4, 48(t2)                \n\t"

        "addi           t3, t6, 8                 \n\t"
        "vsetvli        t0, x0, e8, m1            \n\t"
        "vl1r.v         v0, (t3)                  \n\t"
        "addi           t3, s5, 64                \n\t"
        "vl4r.v         v4, (t3)                  \n\t"

        "vsetvli        t0, x0, e8, m1            \n\t"
        "vsrl.vi        v1, v0, 4                 \n\t"
        "vnpack4.vv     v12, v0, v1, 3            \n\t"
        "vpack.vv       v0, v17, v16, 3           \n\t"
        "vupack.vv      v2, v12, v12, 2           \n\t"

        "vsetvli        t0, x0, e16, mf2          \n\t"
        "vfmul.vf       v12, v16, ft1             \n\t"
        "vfmul.vf       v13, v16, ft2             \n\t"
        "vfmul.vf       v24, v16, ft3             \n\t"
        "vfmul.vf       v25, v16, ft4             \n\t"

        "vsetvli        t0, x0, e16, mf2          \n\t"
        "vfwmacc.vf     v28, fa1, v12             \n\t"
        "vfwmacc.vf     v29, fa1, v13             \n\t"
        "vfwmacc.vf     v30, fa1, v24             \n\t"
        "vfwmacc.vf     v31, fa1, v25             \n\t"

        "vsetvli        t0, x0, e32, m1           \n\t"
        "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
        "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
        "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
        "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
        "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
        "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
        "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
        "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"

        "addi           t4, t4, -1                \n\t"

        "addi           t6, t6, 8+128             \n\t"
        "addi           t2, t2, 2                 \n\t"
        "addi           s5, s5, 64+512            \n\t"
        "bgtz           t4, _M4_KsubBLK_LPST%=    \n\t"

        "vsetvli        t0, x0, e16, m1           \n\t"
        "vpack.vv       v8, v18, v19, 1           \n\t"
        "vpack.vv       v12, v20, v21, 1          \n\t"
        "vpack.vv       v26, v8, v12, 2           \n\t"

        "vsetvli        t0, x0, e16, m1           \n\t"
        "vfwmacc.vf     v28, fa1, v26             \n\t"
        "vfwmacc.vf     v30, fa1, v27             \n\t"

        "li             t4, 8                     \n\t"
        "addi           t5, t5, -1                \n\t"
        "addi           t6, t6, 72                \n\t"
        "addi           t2, t6, 1088              \n\t"
        "bgtz           t5, _M4_BLK_LPST%=        \n\t"

        "_M4_BLK_LPND%=:                          \n\t"
        "vsetvli        t0, x0, e32, m1           \n\t"
        "add            t2, %[LDC], %[DST]        \n\t"
        "vse32.v        v28, (%[DST])             \n\t"
        "add            t3, %[LDC], t2            \n\t"
        "vse32.v        v29, (t2)                 \n\t"
        "add            t2, %[LDC], t3            \n\t"
        "vse32.v        v30, (t3)                 \n\t"
        "vse32.v        v31, (t2)                 \n\t"
        : [A] "+r"(a_data), [B] "+r"(b_data)
        : [DST] "r"(dst_c), [LDC] "r"(ldc*4), [BK] "r"(k_blks)
        : "t0","t1","t2","t3","t4","t5","t6","s5","v0","v1","v2","v3","v4","v5","v6","v7","v8","v10",
          "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v24","v25","v26",
          "v27","v28","v29","v30","v31","fa0","fa1","fa2","fa6","ft1","ft2","ft3","ft4");
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
#define AREC_M4 1160
static void pack_act_hp_m4(const float*x0,const float*x1,const float*x2,const float*x3,int K,uint8_t*Abuf){
    int Sb=K/256;
    for(int sb=0;sb<Sb;sb++) pack_A_hp_m4(x0+sb*256,x1+sb*256,x2+sb*256,x3+sb*256, Abuf+(size_t)sb*AREC_M4);
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
/* g_lin_nt: Phase 6 (codex_recs_1.md §22.33) safety cap -- linear/IME GEMM work (kind 0/1) must
 * stay on the primary, non-contended harts (8/10/12/14 at the historical nt=4 default) regardless
 * of how many TOTAL pool threads exist (g_pool_nt, which Phase 6 may grow up to 8 so attention can
 * also use the paired harts 9/11/13/15). IME-2 is shared per hart-pair; using both harts of a pair
 * concurrently for IME-2 work is measured contended (docs/HARDWARE.md, pin_once's own comment).
 * Set once in main() to min(nt,4) -- nt<4 (an explicit, if untested, caller request for fewer
 * linear workers) is honored as before; nt>4 is capped at 4, never silently expanded to more than
 * the historical linear worker count regardless of how large the attention-only pool grows. */
static int g_lin_nt=4;
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
    if(tn>=g_lin_nt) return; /* Phase 6: extra pool threads (harts 9/11/13/15) idle during every
        linear/IME dispatch, never participate in GEMM work -- see g_lin_nt's own comment. */
    int Np=g_pool_work.l->N/32;
    if(g_pool_work.kind==0){
        for(int np=tn; np<Np; np+=g_lin_nt)
            run_hp_m1(g_pool_work.Abuf, g_pool_work.l->B+(size_t)np*g_pool_work.kb*BSUPER, g_pool_work.y+np*32, g_pool_work.kb);
    } else if(g_pool_work.kind==1){
        for(int np=tn; np<Np; np+=g_lin_nt)
            run_i8_m1(g_pool_work.Abuf, g_pool_work.l->B+(size_t)np*g_pool_work.kb*BREC_I8, g_pool_work.y+np*32, g_pool_work.kb);
    } else /* kind==3: M-batch track milestone 1 (codex_recs_1.md §22.35) -- M4-batched dense
        linear, dispatches gemm_kernel_i8i4_hp_m4 (validated bench/vendor_ime_m4_probe.c) across
        N-tiles the exact same way kind==0 dispatches M1, just with a 4-row-batched A-record
        (1160B, pack_A_hp_m4) and a 4-row-batched output write (ldc=l->N so each of the 4 rows'
        destinations land in its own contiguous N-wide slice of y). */ {
        long ldc=g_pool_work.l->N;
        for(int np=tn; np<Np; np+=g_lin_nt)
            run_hp_m4(g_pool_work.Abuf, g_pool_work.l->B+(size_t)np*g_pool_work.kb*BSUPER, g_pool_work.y+np*32, g_pool_work.kb, ldc);
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
static void lin_mm_pool_init(int total){
    /* Phase 6 (codex_recs_1.md §22.33): `total` is the FULL pool size (may exceed g_lin_nt when
     * attention wants more workers than linear/IME does) -- every created thread is woken on every
     * dispatch regardless of kind, so the wait bound in lin_mm_hp/lin_mm_i8/attn_dispatch must all
     * agree on g_pool_nt, not some kind-specific subset. */
    g_pool_nt=total; pin_once(0);
    for(int i=1;i<total;i++) pthread_create(&g_pool_threads[i],NULL,lin_mm_hp_worker,(void*)(intptr_t)i);
}
static void lin_mm_hp(const Lin*l,const uint8_t*Abuf,float*y,int nt){
    (void)nt; /* superseded by g_pool_nt (Phase 6) -- kept as a parameter so every call site (many,
        threaded through forward()) doesn't need touching; the actual worker count used for both
        the linear-specific stride (g_lin_nt) and the pool-wide wait bound (g_pool_nt) are globals
        now, decoupled from this historically-named-the-same local. */
    g_pool_work.kind=0; g_pool_work.l=l; g_pool_work.Abuf=Abuf; g_pool_work.y=y; g_pool_work.kb=l->K/256;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release); /* wake workers 1..g_pool_nt-1 */
    lin_mm_hp_worker_run(0); /* main thread does its own share (tn=0) */
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < g_pool_nt-1) { /* spin */ }
}
static void lin_mm_i8(const Lin*l,const uint8_t*Abuf,float*y,int nt){
    (void)nt; /* superseded by g_pool_nt (Phase 6) -- see lin_mm_hp's own comment */
    g_pool_work.kind=1; g_pool_work.l=l; g_pool_work.Abuf=Abuf; g_pool_work.y=y; g_pool_work.kb=l->K/32;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release);
    lin_mm_hp_worker_run(0);
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < g_pool_nt-1) { /* spin */ }
}
/* M-batch track milestone 1 (codex_recs_1.md §22.35): M4-batched dense linear. `Abuf4` is a single
 * packed 4-row activation (from pack_act_hp_m4, one call covering all K/256 blocks). `y4` is a
 * 4*l->N contiguous float buffer, row-major: row r's N outputs live at y4+r*l->N (matches
 * run_hp_m4's own ldc=l->N addressing, driven straight through the kind==3 dispatch path above).
 * Same weight matrix `l` as lin_mm_hp -- the whole point of M-batching is that this ONE weight
 * stream read now serves 4 independent sequences' activation rows instead of 1. */
static void lin_mm_hp_m4(const Lin*l,const uint8_t*Abuf4,float*y4){
    g_pool_work.kind=3; g_pool_work.l=l; g_pool_work.Abuf=Abuf4; g_pool_work.y=y4; g_pool_work.kb=l->K/256;
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release);
    lin_mm_hp_worker_run(0);
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < g_pool_nt-1) { /* spin */ }
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
/* attention_optimization_plan.md Phase 5 (codex_recs_1.md §22.32): RVV-vectorize softmax's max
 * reduction and final normalization, per explicit instruction. expf itself stays scalar/exact --
 * no approximate exponential -- so the exp+sum pass is untouched; only the two purely-arithmetic
 * passes either side of it are RVV candidates. Max: an elementwise-max reduction is order-
 * independent for the same finite-float set (no NaNs occur here), so RVV vfmax accumulation
 * followed by a single vfredmax fold produces the identical scalar to the linear scalar scan.
 * Normalization: x[i]*=inv is a pure per-element multiply; IEEE-754 single-precision multiply
 * gives the same bit pattern regardless of scalar-loop vs RVV-lane execution. Both are therefore
 * expected bit-exact to the scalar path -- verified, not assumed, in bench/softmax_rvv_probe.c and
 * the production QWEN_SOFTMAX_VALIDATE path below. */
static float rvv_max_f32(const float*x,int n){
    vfloat32m1_t vacc=__riscv_vfmv_v_f_f32m1(-1e30f,__riscv_vsetvlmax_e32m1());
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl);
        vacc=__riscv_vfmax_vv_f32m1(vacc,vx,vl); i+=vl; }
    vfloat32m1_t vid=__riscv_vfmv_v_f_f32m1(-1e30f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t vred=__riscv_vfredmax_vs_f32m1_f32m1(vacc,vid,__riscv_vsetvlmax_e32m1());
    return __riscv_vfmv_f_s_f32m1_f32(vred);
}
static void rvv_scale_f32(float*x,float s,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl);
        vx=__riscv_vfmul_vf_f32m1(vx,s,vl);
        __riscv_vse32_v_f32m1(x+i,vx,vl); i+=vl; }
}
/* g_softmax_rvv: Phase 5 (codex_recs_1.md §22.32), KEPT after A/B -- default is now 1 (RVV
 * max+normalize, scalar expf unchanged in between). 0 = the original fully-scalar softmax below,
 * byte-for-byte, kept as the explicit revert path. Set via QWEN_SOFTMAX_RVV env var, same
 * convention as QWEN_QK_FUSE/QWEN_AV_FUSE. A/B (2 trials, short/128/512/1024): wall time 0%/-0.5%/
 * -1.2%/-2.2%, softmax summed-work -8.5%/-13.9%/-15.0%/-15.6%, token-identical, 0/32,740,245
 * integration-validation mismatches. Modest relative to QK/AV (softmax's own cost is now the
 * remaining sub-bucket after both fusions, but still smaller in absolute ms than QK or AV were). */
static int g_softmax_rvv=1;
/* g_softmax_validate / QWEN_SOFTMAX_VALIDATE=1: unlike QK/AV, softmax's correctness does not
 * depend on the multi-threaded dispatch machinery (it's a pure, deterministic per-call function
 * whether invoked from the serial path or a pool worker), so validation can run both the scalar
 * and RVV computations on every real call, in place, without needing to replay the whole attention
 * dispatch twice. Thread-local scratch (grown once per thread, not malloc/free per call) makes
 * this safe to call concurrently from every pool worker without locking or cross-thread races. */
static int g_softmax_validate=0;
static _Thread_local float* g_smval_scratch=NULL;
static _Thread_local int g_smval_scratch_cap=0;
static long g_smval_cmp=0, g_smval_mismatch=0; static double g_smval_max_abs=0, g_smval_max_rel=0;
static void softmax(float*x,int n){
    if(g_softmax_validate){
        if(g_smval_scratch_cap<n){ free(g_smval_scratch); g_smval_scratch=malloc((size_t)n*4); g_smval_scratch_cap=n; }
        float*ref=g_smval_scratch;
        memcpy(ref,x,(size_t)n*4);
        { float m=-1e30f; for(int i=0;i<n;i++)if(ref[i]>m)m=ref[i];
          float s=0; for(int i=0;i<n;i++){ref[i]=expf(ref[i]-m);s+=ref[i];}
          float inv=1.0f/s; for(int i=0;i<n;i++)ref[i]*=inv; }
        { float m=rvv_max_f32(x,n);
          float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
          rvv_scale_f32(x,1.0f/s,n); }
        for(int i=0;i<n;i++){
            double a=ref[i], b=x[i]; double ad=fabs(a-b); double rd=fabs(a)>1e-12?ad/fabs(a):ad;
            if(ad>g_smval_max_abs) g_smval_max_abs=ad;
            if(rd>g_smval_max_rel) g_smval_max_rel=rd;
            if(ad>1e-4) g_smval_mismatch++;
            g_smval_cmp++;
        }
        if(!g_softmax_rvv) memcpy(x,ref,(size_t)n*4); /* leave x correct for whatever's configured */
        return;
    }
    if(g_softmax_rvv){
        float m=rvv_max_f32(x,n);
        float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
        rvv_scale_f32(x,1.0f/s,n);
    } else {
        float m=-1e30f; for(int i=0;i<n;i++)if(x[i]>m)m=x[i]; float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} float inv=1.0f/s; for(int i=0;i<n;i++)x[i]*=inv;
    }
}
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
#define QK8_MAXGPR 8
/* attention_optimization_plan.md Phase 4.1 (codex_recs_1.md §22.30): multi-Q QK, exact GQA reuse.
 * For a fixed KV head, the current per-head loop calls vdot_f32(qh,kj,hd) once per query head,
 * re-reading the SAME kj (K vector at position j) from memory up to 8 times per position (once per
 * query head sharing this KV head) -- each re-read is likely a real DRAM/L2 fetch, not an L1 hit,
 * since a full ctx-length sweep for one query head runs between touches of the same kj by the next
 * query head. This kernel instead loads each 32-lane chunk of kj exactly once per position and
 * immediately uses it to update ALL `n` query-head accumulators before moving to the next chunk --
 * "load once, update eight independent accumulators" per the plan's literal wording, not just a
 * cache-friendly loop reorder.
 *
 * Per-query-head numerics are intentionally IDENTICAL to vdot_f32: same chunk boundaries (vl
 * schedule driven by the same n=hd, same __riscv_vsetvl_e32m1 sequence), same vfmacc_vv order into
 * a per-qi running accumulator, same final vfredusum reduction -- the only thing that changes is
 * that kj's chunk is loaded once and shared across the `n` vfmacc calls instead of being loaded
 * fresh inside `n` separate vdot_f32 calls. Because each acc[qi]'s own update sequence is untouched
 * by what happens to the OTHER accumulators in between, this is expected to be bit-exact to calling
 * vdot_f32(qh[qi],kj,hd) for qi=0..n-1 -- verified, not just assumed, in bench/qk_multiq_probe.c
 * before this is trusted. */
static void qk8_dot(const float*const qh[QK8_MAXGPR],const float*kj,int hd,int n,float*out){
    /* RVV vector types are sizeless ("scalable"), so they cannot be array elements (`type v[N]` is
     * a compile error) -- 8 named accumulators instead, matching QK8_MAXGPR exactly (this model's
     * gpr is always 8; n<8 simply leaves the extra accumulators unused). */
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t a0=z,a1=z,a2=z,a3=z,a4=z,a5=z,a6=z,a7=z;
    int i=0;
    while(i<hd){
        size_t vl=__riscv_vsetvl_e32m1(hd-i);
        vfloat32m1_t vk=__riscv_vle32_v_f32m1(kj+i,vl); /* kj chunk loaded ONCE */
        if(n>0) a0=__riscv_vfmacc_vv_f32m1(a0,__riscv_vle32_v_f32m1(qh[0]+i,vl),vk,vl);
        if(n>1) a1=__riscv_vfmacc_vv_f32m1(a1,__riscv_vle32_v_f32m1(qh[1]+i,vl),vk,vl);
        if(n>2) a2=__riscv_vfmacc_vv_f32m1(a2,__riscv_vle32_v_f32m1(qh[2]+i,vl),vk,vl);
        if(n>3) a3=__riscv_vfmacc_vv_f32m1(a3,__riscv_vle32_v_f32m1(qh[3]+i,vl),vk,vl);
        if(n>4) a4=__riscv_vfmacc_vv_f32m1(a4,__riscv_vle32_v_f32m1(qh[4]+i,vl),vk,vl);
        if(n>5) a5=__riscv_vfmacc_vv_f32m1(a5,__riscv_vle32_v_f32m1(qh[5]+i,vl),vk,vl);
        if(n>6) a6=__riscv_vfmacc_vv_f32m1(a6,__riscv_vle32_v_f32m1(qh[6]+i,vl),vk,vl);
        if(n>7) a7=__riscv_vfmacc_vv_f32m1(a7,__riscv_vle32_v_f32m1(qh[7]+i,vl),vk,vl);
        i+=vl;
    }
    vfloat32m1_t vzero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    if(n>0) out[0]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a0,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>1) out[1]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a1,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>2) out[2]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a2,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>3) out[3]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a3,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>4) out[4]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a4,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>5) out[5]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a5,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>6) out[6]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a6,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>7) out[7]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a7,vzero,__riscv_vsetvlmax_e32m1()));
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
/* attention_optimization_plan.md Phase 4.2 (codex_recs_1.md §22.31): multi-Q AV, exact GQA reuse
 * on V. For a fixed KV head, the unfused AV loop calls vaxpy_f32(oh,vj,sc[j],hd) once per (query
 * head, position) pair, re-reading the SAME vj (V vector at position j) from memory up to 8 times
 * per position -- the AV-side analog of qk8_dot's redundant K re-reads. This kernel instead
 * processes ONE hd-chunk (32 lanes) at a time across the WHOLE position loop, holding 8 named
 * accumulators (one per query head) live in registers for that chunk, loading each V chunk once
 * per position and using it to update all 8 accumulators before advancing to the next position;
 * only after the full position sweep for that chunk does it write the 8 accumulators back to
 * memory. Chunk-outer/position-inner (rather than qk8_dot's single-position/chunk-inner) because
 * AV's reduction axis is position, not head_dim -- the accumulator must live across the position
 * loop, not be reduced away within one. Holding only 8 accumulators live at a time (never 8 heads
 * x 4 chunks = 32 at once) keeps the same register-pressure profile as qk8_dot, already known-good
 * at -O3 (the production optimization level) after Phase 4.1's sanitizer work.
 *
 * Per-query-head numerics are intentionally IDENTICAL to vaxpy_f32: for any fixed query head and
 * hd-chunk, the accumulation order across positions (j=0,1,2,...,pos) is unchanged from the
 * unfused loop -- IEEE-754 addition order determines the result exactly, and holding a running sum
 * in a register instead of round-tripping it through memory each iteration does not change that
 * order. Expected bit-exact to the unfused path -- verified, not just assumed, in
 * bench/av_multiq_probe.c before this is trusted. Writes oh[qi][coff..coff+vl) directly (not an
 * accumulate-into-existing-memory), so callers do not need to pre-zero oh before calling this for
 * every hd-chunk. */
static void av8_chunk(const float*const scw[QK8_MAXGPR],const float*Vh,int hd,int pos,int gpr,
        float*const oh[QK8_MAXGPR],int coff,size_t vl){
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,vl);
    vfloat32m1_t a0=z,a1=z,a2=z,a3=z,a4=z,a5=z,a6=z,a7=z;
    for(int j=0;j<=pos;j++){
        vfloat32m1_t vv=__riscv_vle32_v_f32m1(Vh+(size_t)j*hd+coff,vl); /* V chunk loaded ONCE */
        if(gpr>0) a0=__riscv_vfmacc_vf_f32m1(a0,scw[0][j],vv,vl);
        if(gpr>1) a1=__riscv_vfmacc_vf_f32m1(a1,scw[1][j],vv,vl);
        if(gpr>2) a2=__riscv_vfmacc_vf_f32m1(a2,scw[2][j],vv,vl);
        if(gpr>3) a3=__riscv_vfmacc_vf_f32m1(a3,scw[3][j],vv,vl);
        if(gpr>4) a4=__riscv_vfmacc_vf_f32m1(a4,scw[4][j],vv,vl);
        if(gpr>5) a5=__riscv_vfmacc_vf_f32m1(a5,scw[5][j],vv,vl);
        if(gpr>6) a6=__riscv_vfmacc_vf_f32m1(a6,scw[6][j],vv,vl);
        if(gpr>7) a7=__riscv_vfmacc_vf_f32m1(a7,scw[7][j],vv,vl);
    }
    if(gpr>0) __riscv_vse32_v_f32m1(oh[0]+coff,a0,vl);
    if(gpr>1) __riscv_vse32_v_f32m1(oh[1]+coff,a1,vl);
    if(gpr>2) __riscv_vse32_v_f32m1(oh[2]+coff,a2,vl);
    if(gpr>3) __riscv_vse32_v_f32m1(oh[3]+coff,a3,vl);
    if(gpr>4) __riscv_vse32_v_f32m1(oh[4]+coff,a4,vl);
    if(gpr>5) __riscv_vse32_v_f32m1(oh[5]+coff,a5,vl);
    if(gpr>6) __riscv_vse32_v_f32m1(oh[6]+coff,a6,vl);
    if(gpr>7) __riscv_vse32_v_f32m1(oh[7]+coff,a7,vl);
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
static float* g_attn_scratch_multi[MAXNT]; /* Phase 4.1: gpr*ctx floats per worker (row-major
    [qi][j]), holds all gpr query heads' scores for one KV head at once -- needed because the fused
    QK kernel computes every query head's score for a position together, so softmax/AV for query
    head 0 can't start until the whole fused QK sweep (all positions, all heads) has finished,
    unlike the unfused path where one query head's full score row is ready before its softmax. */
static int g_attn_scratch_multi_ctx=0, g_attn_scratch_multi_gpr=0;
static double g_w_qk[MAXNT], g_w_sm[MAXNT], g_w_av[MAXNT]; /* per-worker accumulators: each thread
    only ever writes its own index, so these are race-free without locking; summed into the global
    gT_attn_qk/sm/av by the single dispatching thread after every worker has finished its round. */
/* g_qk_fuse: Phase 4.1 fused multi-Q QK (codex_recs_1.md §22.30), KEPT after A/B -- default is now
 * 1 (fused, qk8_dot). 0 = byte-for-byte the Phase 3 code path below, kept as the explicit revert
 * path. softmax/AV/layout/scheduling unchanged either way. Set via QWEN_QK_FUSE env var, same
 * convention as QWEN_ATTN_NT/QWEN_CTXLEN. A/B (2 trials, short/128/512/1024): wall time -0.1%/
 * -1.13%/-3.12%/-5.80%, attn bucket -6.6%/-9.8%/-9.0%/-10.3%, token-identical, 0/60,426,240
 * integration-validation mismatches. */
static int g_qk_fuse=1;
/* g_qk_capture: when non-NULL, points to an nh*actx-float buffer that BOTH the fused and unfused
 * branches below additionally write their post-scale, pre-softmax QK score for (hh,j) into (index
 * hh*actx+j), on top of their normal scratch writes -- lets a validation driver run the exact same
 * real dispatch path (real pool, real scratch, real KV-head mapping, real scaling) twice, once per
 * mode, on the SAME real activations, and diff the two captures. NULL (default) costs one branch
 * per QK write, negligible, and is what every production/benchmark run uses. */
static float* g_qk_capture=NULL;
/* g_av_fuse: Phase 4.2 fused multi-Q AV (codex_recs_1.md §22.31), KEPT after A/B -- default is now
 * 1 (fused, av8_chunk). 0 = byte-for-byte the Phase 4.1 code path below, kept as the explicit
 * revert path. Only meaningful inside the g_qk_fuse==1 branch: AV fusion needs all gpr query
 * heads' softmax weights ready before it starts, which is only true once QK has already been
 * computed for the whole KV head (the fused-QK branch's structure), not in the unfused-QK branch
 * where each head's own softmax+AV happens immediately after that head's own QK. Set via
 * QWEN_AV_FUSE env var, same convention as QWEN_QK_FUSE. A/B (2 trials, short/128/512/1024): wall
 * time -0.2%/-4.5%/-11.6%/-17.8%, attn bucket -31.5%/-35.1%/-33.7%/-34.6%, token-identical,
 * 0/55,050,240 integration-validation mismatches. */
static int g_av_fuse=1;
/* Exact-path closure (codex_recs_1.md §22.34), under A/B: per-worker attention scratch buffers
 * (g_attn_scratch/g_attn_scratch_multi) are small enough (e.g. 32KB at ctx=1024) to go through
 * glibc's normal heap allocator rather than its mmap-backed large-allocation path (which returns
 * page-aligned memory already) -- plain malloc only guarantees ~16-byte alignment, not a full
 * 64-byte cache line. g_scratch_align gates 64-byte-aligned allocation (posix_memalign) vs the
 * original malloc, to test whether cache-line alignment measurably helps the RVV loads/stores that
 * walk these buffers. 0 (default until promoted) = original malloc. */
static int g_scratch_align=0;
/* M-batch track milestone 3 (codex_recs_1.md §22.37): expert-FFN batching statistics -- how often
 * all 4 sequences in forward4_dense_batch's batch happen to independently select the SAME expert
 * out of 128 (the only overlap case this milestone batches; see forward4_dense_batch's own MoE-FFN
 * comment for why partial 2-3-way overlaps are deliberately not padded-and-batched). File-scope so
 * run_mbatch_test can read and report them after a full run. */
static long g_moe4_hits=0, g_moe4_total_expert_slots=0;
static int g_prefill_moe4=1; /* QWEN_PREFILL_MOE4=0: ablation for prefill_chunk4 only (does NOT
    affect forward4_dense_batch/decode) -- forces the ecount==4 MoE-FFN M4 path off, isolating how
    much of batched prefill's speedup comes from dense-layer batching alone vs the expert-FFN
    grouping on top, matching how milestones 2 vs 3 separated the same two contributions for
    decode. Default 1 (both mechanisms on, the real production/benchmark configuration). */
static long g_moe_ecount_hist[5]={0,0,0,0,0}; /* histogram of ecount[e] in {0,1,2,3,4} across every
    expert/layer/decode-step visited -- diagnostic only, answers whether PARTIAL overlap (2-3 of 4
    sequences picking the same expert) is common enough to be worth a future padded-M4 batching
    strategy, separate from the exact-4-way case this milestone actually batches. */
static void* scratch_alloc(size_t n){
    if(!g_scratch_align) return malloc(n);
    void*p=NULL; size_t sz=((n+63)/64)*64; if(sz==0) sz=64;
    return posix_memalign(&p,64,sz)==0 ? p : NULL;
}
static void attn_scratch_ensure(int ctx){
    if(g_attn_scratch_ctx>=ctx) return;
    for(int i=0;i<MAXNT;i++){ free(g_attn_scratch[i]); g_attn_scratch[i]=scratch_alloc((size_t)ctx*4); }
    g_attn_scratch_ctx=ctx;
}
static void attn_scratch_multi_ensure(int ctx,int gpr){
    if(g_attn_scratch_multi_ctx>=ctx && g_attn_scratch_multi_gpr>=gpr) return;
    int gg=gpr>g_attn_scratch_multi_gpr?gpr:g_attn_scratch_multi_gpr;
    int cc=ctx>g_attn_scratch_multi_ctx?ctx:g_attn_scratch_multi_ctx;
    for(int i=0;i<MAXNT;i++){ free(g_attn_scratch_multi[i]); g_attn_scratch_multi[i]=scratch_alloc((size_t)gg*cc*4); }
    g_attn_scratch_multi_ctx=cc; g_attn_scratch_multi_gpr=gg;
}
/* Phase 6 (codex_recs_1.md §22.33): attention worker count may now exceed nkv (4) -- up to 8, using
 * the paired A100 harts (9/11/13/15) alongside the primary four (8/10/12/14). Since each KV head's
 * work was previously assigned to exactly one worker atomically, attn_nt>nkv needs each KV head's
 * gpr (8) query heads SPLIT across `gpk` sub-workers (gpk=attn_nt/nkv, e.g. 2 at attn_nt=8) instead
 * of just striding across more KV heads (there are only nkv=4 of those, so attn_nt=8 would
 * otherwise leave workers 4-7 permanently idle no-ops). At attn_nt<=nkv (gpk=1) this reduces
 * exactly to the pre-Phase-6 behavior byte-for-byte: kvh_stride=attn_nt, qi_start=0,
 * qi_count=gpr -- so nothing changes for the already-KEPT attn_nt=1/2/4 configurations. */
static void attn_worker_run(int tn){
    if(tn>=g_pool_work.aattn_nt){ g_w_qk[tn]=g_w_sm[tn]=g_w_av[tn]=0; return; }
    int nkv=g_pool_work.ankv, gpr=g_pool_work.agpr, hd=g_pool_work.ahd, pos=g_pool_work.apos, actx=g_pool_work.actx;
    int attn_nt=g_pool_work.aattn_nt;
    float scale=g_pool_work.ascale;
    const float*q=g_pool_work.aq, *Kc=g_pool_work.aKc, *Vc=g_pool_work.aVc; float*att=g_pool_work.aatt;
    double qk=0, sm=0, av=0;
    int gpk = (attn_nt>nkv) ? (attn_nt/nkv) : 1;      /* subgroups per KV head */
    int qi_count = gpr/gpk;                            /* query heads this worker owns per KV head */
    int kvh_stride = (gpk==1) ? attn_nt : nkv;
    int kvh_init = (gpk==1) ? tn : (tn/gpk);
    int qi_start = (gpk==1) ? 0 : (tn%gpk)*qi_count;
    if(g_qk_fuse && gpr<=QK8_MAXGPR){
        float*scm=g_attn_scratch_multi[tn]; /* [local_qi*actx+j], local_qi in [0,qi_count) */
        /* qh/scw/ohp declared ONCE at function scope (not re-entered every kvh-loop iteration) --
         * only [0,qi_count) of each QK8_MAXGPR(8)-sized array is ever populated/read/written;
         * slots [qi_count,QK8_MAXGPR) are simply unused, never touched, for the lifetime of the
         * call. This mirrors qk8_dot/av8_chunk's own `if(n>K)` guard discipline exactly. */
        const float*qh[QK8_MAXGPR]={0};
        const float*scw[QK8_MAXGPR]={0}; float*ohp[QK8_MAXGPR]={0};
        for(int kvh=kvh_init; kvh<nkv; kvh+=kvh_stride){
            const float*Kh=Kc+(size_t)kvh*actx*hd, *Vh=Vc+(size_t)kvh*actx*hd;
            for(int li=0;li<qi_count;li++) qh[li]=q+(size_t)(kvh*gpr+qi_start+li)*hd;
            double t0=gT_on?now():0;
            for(int j=0;j<=pos;j++){
                const float*kj=Kh+(size_t)j*hd;
                float out8[QK8_MAXGPR];
                qk8_dot(qh,kj,hd,qi_count,out8);
                for(int li=0;li<qi_count;li++){
                    float s=out8[li]*scale; scm[(size_t)li*actx+j]=s;
                    if(g_qk_capture) g_qk_capture[(size_t)(kvh*gpr+qi_start+li)*actx+j]=s;
                }
            }
            double t1=gT_on?now():0; if(gT_on) qk+=t1-t0;
            for(int li=0;li<qi_count;li++) softmax(scm+(size_t)li*actx,pos+1);
            double t2=gT_on?now():0; if(gT_on) sm+=t2-t1;
            if(g_av_fuse){
                /* Phase 4.2 (codex_recs_1.md §22.31): all qi_count heads' softmax weights for this
                 * KV head (sub)group are ready together, so AV can process them all per hd-chunk,
                 * loading each V chunk once instead of qi_count times. */
                for(int li=0;li<qi_count;li++){ ohp[li]=att+(size_t)(kvh*gpr+qi_start+li)*hd; scw[li]=scm+(size_t)li*actx; }
                int coff=0;
                while(coff<hd){
                    size_t vl=__riscv_vsetvl_e32m1(hd-coff);
                    av8_chunk(scw,Vh,hd,pos,qi_count,ohp,coff,vl);
                    coff+=(int)vl;
                }
            } else {
                for(int li=0;li<qi_count;li++){
                    int hh=kvh*gpr+qi_start+li; float*sc=scm+(size_t)li*actx;
                    float*oh=att+(size_t)hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
                    for(int j=0;j<=pos;j++){ const float*vj=Vh+(size_t)j*hd; vaxpy_f32(oh,vj,sc[j],hd); }
                }
            }
            if(gT_on) av += now()-t2;
        }
    } else {
        /* Phase 3 code, unchanged -- explicit revert path when g_qk_fuse==0. qi ranges over
         * [qi_start,qi_start+qi_count) directly (no local/global remap needed: g_attn_scratch[tn]
         * is a plain ctx-sized row reused serially per query head, not gpr-dimensioned). */
        float*sc=g_attn_scratch[tn];
        for(int kvh=kvh_init; kvh<nkv; kvh+=kvh_stride){
            const float*Kh=Kc+(size_t)kvh*actx*hd, *Vh=Vc+(size_t)kvh*actx*hd;
            for(int qi=qi_start; qi<qi_start+qi_count; qi++){
                int hh=kvh*gpr+qi; const float*qh=q+(size_t)hh*hd;
                double t0=gT_on?now():0;
                for(int j=0;j<=pos;j++){
                    const float*kj=Kh+(size_t)j*hd; float s=vdot_f32(qh,kj,hd)*scale; sc[j]=s;
                    if(g_qk_capture) g_qk_capture[(size_t)hh*actx+j]=s;
                }
                double t1=gT_on?now():0; if(gT_on) qk+=t1-t0;
                softmax(sc,pos+1);
                double t2=gT_on?now():0; if(gT_on) sm+=t2-t1;
                float*oh=att+(size_t)hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
                for(int j=0;j<=pos;j++){ const float*vj=Vh+(size_t)j*hd; vaxpy_f32(oh,vj,sc[j],hd); }
                if(gT_on) av += now()-t2;
            }
        }
    }
    g_w_qk[tn]=qk; g_w_sm[tn]=sm; g_w_av[tn]=av;
}
static void attn_dispatch(const float*q,const float*Kc,const float*Vc,float*att,int pos,float scale,
        int hd,int nkv,int gpr,int actx,int attn_nt){
    if(g_qk_fuse && gpr<=QK8_MAXGPR) attn_scratch_multi_ensure(actx,gpr); else attn_scratch_ensure(actx);
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
/* Phase 4.1 integration validation (codex_recs_1.md §22.30, per explicit review): the standalone
 * probe (bench/qk_multiq_probe.c) validates qk8_dot's math on synthetic data, copied out of the
 * production file -- it cannot catch bugs in how the REAL engine wires the kernel in: scratch
 * indexing (g_attn_scratch_multi's row layout), KV-head mapping (kvh*gpr+qi), scaling, or the pool
 * dispatch itself. This runs the exact real attn_dispatch path TWICE per call site, once per mode,
 * on the SAME real q/Kc/Vc from the actual decode (not synthetic data), capturing every (head,
 * position) QK score via g_qk_capture and diffing the two captures -- exercises every one of those
 * integration points for real, using the real multi-threaded pool both times. */
static int g_qk_validate=0; /* set via QWEN_QK_VALIDATE=1 */
static float *g_qkval_unfused=NULL, *g_qkval_fused=NULL; static int g_qkval_cap=0;
static long g_qkval_cmp=0, g_qkval_mismatch=0; static double g_qkval_max_abs=0, g_qkval_max_rel=0;
static void attn_qk_validate_and_dispatch(const float*q,const float*Kc,const float*Vc,float*att,int pos,float scale,
        int hd,int nkv,int gpr,int actx,int attn_nt){
    int nh=nkv*gpr;
    if(g_qkval_cap < nh*actx){
        free(g_qkval_unfused); free(g_qkval_fused);
        g_qkval_unfused=malloc((size_t)nh*actx*4); g_qkval_fused=malloc((size_t)nh*actx*4);
        g_qkval_cap=nh*actx;
    }
    int real_fuse=g_qk_fuse; /* whatever the run is actually configured to use */

    g_qk_fuse=0; g_qk_capture=g_qkval_unfused;
    attn_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,actx,attn_nt);

    g_qk_fuse=1; g_qk_capture=g_qkval_fused;
    attn_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,actx,attn_nt);

    for(int hh=0;hh<nh;hh++){
        for(int j=0;j<=pos;j++){
            double u=g_qkval_unfused[(size_t)hh*actx+j], f=g_qkval_fused[(size_t)hh*actx+j];
            double ad=fabs(u-f); double rd=fabs(u)>1e-12?ad/fabs(u):ad;
            if(ad>g_qkval_max_abs) g_qkval_max_abs=ad;
            if(rd>g_qkval_max_rel) g_qkval_max_rel=rd;
            if(ad>1e-4) g_qkval_mismatch++;
            g_qkval_cmp++;
        }
    }
    g_qk_capture=NULL; g_qk_fuse=real_fuse;
    if(real_fuse!=1) attn_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,actx,attn_nt); /* leave att correct for the configured mode -- the fused call above is only reused as-is when real_fuse==1 */
}
/* Phase 4.2 integration validation (codex_recs_1.md §22.31): simpler than the QK version above --
 * AV's fused output goes directly into the real `att` buffer already (no intermediate quantity
 * needs a separate capture hook), so this just runs the real attn_dispatch path twice into two
 * SEPARATE output buffers, once per mode, on the same real q/Kc/Vc from actual decode, and diffs
 * the two buffers directly. */
static int g_av_validate=0; /* set via QWEN_AV_VALIDATE=1 */
static float *g_avval_unfused=NULL, *g_avval_fused=NULL; static int g_avval_cap=0;
static long g_avval_cmp=0, g_avval_mismatch=0; static double g_avval_max_abs=0, g_avval_max_rel=0;
static void attn_av_validate_and_dispatch(const float*q,const float*Kc,const float*Vc,float*att,int pos,float scale,
        int hd,int nkv,int gpr,int actx,int attn_nt){
    int nh=nkv*gpr;
    if(g_avval_cap < nh*hd){
        free(g_avval_unfused); free(g_avval_fused);
        g_avval_unfused=malloc((size_t)nh*hd*4); g_avval_fused=malloc((size_t)nh*hd*4);
        g_avval_cap=nh*hd;
    }
    int real_av=g_av_fuse; /* whatever the run is actually configured to use */

    g_av_fuse=0; attn_dispatch(q,Kc,Vc,g_avval_unfused,pos,scale,hd,nkv,gpr,actx,attn_nt);
    g_av_fuse=1; attn_dispatch(q,Kc,Vc,g_avval_fused,pos,scale,hd,nkv,gpr,actx,attn_nt);

    for(int i=0;i<nh*hd;i++){
        double u=g_avval_unfused[i], f=g_avval_fused[i];
        double ad=fabs(u-f); double rd=fabs(u)>1e-12?ad/fabs(u):ad;
        if(ad>g_avval_max_abs) g_avval_max_abs=ad;
        if(rd>g_avval_max_rel) g_avval_max_rel=rd;
        if(ad>1e-4) g_avval_mismatch++;
        g_avval_cmp++;
    }
    g_av_fuse=real_av;
    memcpy(att, real_av?g_avval_fused:g_avval_unfused, (size_t)nh*hd*4); /* leave att correct for the configured mode -- reuse rather than a third dispatch */
}
/* Phase 6 integration validation (codex_recs_1.md §22.33): the sub-KV-head work partitioning added
 * for attn_nt>nkv (see attn_worker_run's own comment) is a pure work-REDISTRIBUTION change -- each
 * individual query head's QK/softmax/AV math is unchanged, only which worker computes it and how
 * many OTHER heads it's batched alongside in the same qk8_dot/av8_chunk call (which by construction
 * doesn't affect any one head's own result, per Phase 4.1/4.2's own validation). Still verified,
 * not assumed: runs the real attn_dispatch path at attn_nt=1 (the serial baseline, already proven
 * correct in Phase 2/3) and at the actually-configured attn_nt, on the same real q/Kc/Vc, into
 * separate buffers, and diffs them. */
static int g_workers_validate=0; /* set via QWEN_WORKERS_VALIDATE=1 */
static float *g_wval_ref=NULL, *g_wval_cand=NULL; static int g_wval_cap=0;
static long g_wval_cmp=0, g_wval_mismatch=0; static double g_wval_max_abs=0, g_wval_max_rel=0;
static void attn_workers_validate_and_dispatch(const float*q,const float*Kc,const float*Vc,float*att,int pos,float scale,
        int hd,int nkv,int gpr,int actx,int attn_nt){
    int nh=nkv*gpr;
    if(g_wval_cap < nh*hd){
        free(g_wval_ref); free(g_wval_cand);
        g_wval_ref=malloc((size_t)nh*hd*4); g_wval_cand=malloc((size_t)nh*hd*4);
        g_wval_cap=nh*hd;
    }
    attn_dispatch(q,Kc,Vc,g_wval_ref,pos,scale,hd,nkv,gpr,actx,1); /* serial baseline */
    attn_dispatch(q,Kc,Vc,g_wval_cand,pos,scale,hd,nkv,gpr,actx,attn_nt);

    for(int i=0;i<nh*hd;i++){
        double r=g_wval_ref[i], c=g_wval_cand[i];
        double ad=fabs(r-c); double rd=fabs(r)>1e-12?ad/fabs(r):ad;
        if(ad>g_wval_max_abs) g_wval_max_abs=ad;
        if(rd>g_wval_max_rel) g_wval_max_rel=rd;
        if(ad>1e-4) g_wval_mismatch++;
        g_wval_cmp++;
    }
    memcpy(att, g_wval_cand, (size_t)nh*hd*4); /* leave att correct for the configured attn_nt */
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
        } else if(g_qk_validate){
            attn_qk_validate_and_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,kv->ctx,g_attn_nt);
        } else if(g_av_validate){
            attn_av_validate_and_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,kv->ctx,g_attn_nt);
        } else if(g_workers_validate){
            attn_workers_validate_and_dispatch(q,Kc,Vc,att,pos,scale,hd,nkv,gpr,kv->ctx,g_attn_nt);
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
/* M-batch track milestone 1 (codex_recs_1.md §22.35): batched decode for 4 GENUINELY INDEPENDENT
 * sequences -- each has its own token, position, and KV cache (own attention history), but they
 * share every DENSE weight-stream read (QKV, O, router, lm_head) via one M4 dispatch instead of 4
 * separate M1 dispatches. Per the explicit scoping decision (only dense layers batched this
 * milestone): the MoE expert FFN (gate/up/down) stays per-sequence via the existing M1 path,
 * because each sequence's router independently selects its own top-8 of 128 experts from its own
 * hidden state -- different sequences will generally select different, only-partially-overlapping
 * expert sets, so batching THAT would need a genuinely different mechanism (grouping activations
 * by which expert they routed to, gather/scatter across the batch), a separate, comparably-sized
 * undertaking not attempted here. Attention is also per-sequence for the same reason M1 always was
 * (each sequence's own KV history). Mirrors forward()'s own per-layer structure closely by design,
 * to minimize the risk of introducing new bugs relative to the already-validated serial path. */
static void forward4_dense_batch(Model*m,int tok[4],int pos[4],Kv*kv[4],float*logits[4],
        float*h[4],float*hn[4],float*q[4],float*k[4],float*vv[4],float*att[4],float*tmp[4],
        float*g[4],float*u[4],float*eout[4],uint8_t*Abuf[4],uint8_t*Abuf4,uint8_t*Abuf2_4){
    int d=m->d,nh=m->nh,nkv=m->nkv,hd=m->hd,nt=m->nt,gpr=nh/nkv,moe=m->moe,ne=m->n_exp,na=m->n_act;
    for(int s=0;s<4;s++) memcpy(h[s],m->tok_embd+(size_t)tok[s]*d,d*4);
    float cosb[4][256],sinb[4][256];
    for(int s=0;s<4;s++) rope_table(cosb[s],sinb[s],hd,pos[s],m->rope_base);
    /* Large per-call scratch, function-local static (matches forward()'s own static float*h=NULL
     * convention for reused large buffers) -- allocated once on first use, not per-call/per-layer,
     * and NOT on the stack (avoids risking stack overflow at these sizes, e.g. y4lm alone is up to
     * 4*vocab floats). */
    static float *y4q=NULL,*y4k=NULL,*y4v=NULL,*y4o=NULL,*y4r=NULL,*y4lm=NULL,*y4eg=NULL,*y4eu=NULL,*y4ed=NULL;
    if(!y4q){ int maxqkv=nh*hd>nkv*hd?nh*hd:nkv*hd; y4q=malloc((size_t)4*maxqkv*4); y4k=malloc((size_t)4*maxqkv*4); y4v=malloc((size_t)4*maxqkv*4);
        y4o=malloc((size_t)4*d*4); y4r=malloc((size_t)4*256*4); y4lm=malloc((size_t)4*(size_t)m->vocab*4);
        y4eg=malloc((size_t)4*moe*4); y4eu=malloc((size_t)4*moe*4); y4ed=malloc((size_t)4*d*4); }

    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        for(int s=0;s<4;s++) rmsnorm(hn[s],h[s],ly->attn_norm,d,m->eps);
        pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
        lin_mm_hp_m4(&ly->q,Abuf4,y4q); lin_mm_hp_m4(&ly->k,Abuf4,y4k); lin_mm_hp_m4(&ly->v,Abuf4,y4v);
        for(int s=0;s<4;s++){ memcpy(q[s],y4q+s*ly->q.N,ly->q.N*4); memcpy(k[s],y4k+s*ly->k.N,ly->k.N*4); memcpy(vv[s],y4v+s*ly->v.N,ly->v.N*4); }
        for(int s=0;s<4;s++){
            for(int hh=0;hh<nh;hh++){ rmsnorm(q[s]+hh*hd,q[s]+hh*hd,ly->q_norm,hd,m->eps); rope_apply(q[s]+hh*hd,hd,cosb[s],sinb[s]); }
            for(int hh=0;hh<nkv;hh++){ rmsnorm(k[s]+hh*hd,k[s]+hh*hd,ly->k_norm,hd,m->eps); rope_apply(k[s]+hh*hd,hd,cosb[s],sinb[s]); }
        }
        float scale=1.0f/sqrtf(hd);
        for(int s=0;s<4;s++){
            if(pos[s]<0 || pos[s]>=kv[s]->ctx){ fprintf(stderr,"KV position overflow: pos=%d ctx=%d\n",pos[s],kv[s]->ctx); abort(); }
            int kvd_s=nkv*hd;
            float*Kc=kv[s]->Kc+(size_t)l*kv[s]->ctx*kvd_s,*Vc=kv[s]->Vc+(size_t)l*kv[s]->ctx*kvd_s;
            for(int kvh=0;kvh<nkv;kvh++){
                memcpy(Kc+(size_t)kvh*kv[s]->ctx*hd+(size_t)pos[s]*hd, k[s]+(size_t)kvh*hd, hd*4);
                memcpy(Vc+(size_t)kvh*kv[s]->ctx*hd+(size_t)pos[s]*hd, vv[s]+(size_t)kvh*hd, hd*4);
            }
            /* attention stays per-sequence, exactly the existing serial/pool-dispatched path --
             * each sequence has its own KV history, nothing to batch here (see this function's own
             * top comment). */
            if(g_attn_nt<=1){
                for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q[s]+hh*hd,*sc=tmp[s];
                    float*Kh=Kc+(size_t)kvh*kv[s]->ctx*hd,*Vh=Vc+(size_t)kvh*kv[s]->ctx*hd;
                    for(int j=0;j<=pos[s];j++){ float*kj=Kh+(size_t)j*hd; sc[j]=vdot_f32(qh,kj,hd)*scale; }
                    softmax(sc,pos[s]+1);
                    float*oh=att[s]+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
                    for(int j=0;j<=pos[s];j++){ float*vj=Vh+(size_t)j*hd; vaxpy_f32(oh,vj,sc[j],hd); } }
            } else {
                attn_dispatch(q[s],Kc,Vc,att[s],pos[s],scale,hd,nkv,gpr,kv[s]->ctx,g_attn_nt);
            }
        }
        pack_act_hp_m4(att[0],att[1],att[2],att[3],ly->o.K,Abuf4); /* att is qd(=nh*hd)-wide, NOT
            d-wide -- ly->o.K is the O-projection's real input width (qd here); packing with `d`
            would silently under-pack (miss (qd-d)/256 blocks) since qd>d for this model's GQA
            shape (nh*hd=4096 vs d=2048). */
        lin_mm_hp_m4(&ly->o,Abuf4,y4o);
        for(int s=0;s<4;s++){ for(int i=0;i<d;i++) h[s][i]+=y4o[s*ly->o.N+i]; }
        for(int s=0;s<4;s++) rmsnorm(hn[s],h[s],ly->ffn_norm,d,m->eps);
        pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
        int sel[4][32]; float sw[4][32];
        if(g_router_mode==1) lin_mm_hp_m4(&ly->router_hp,Abuf4,y4r);
        else if(g_router_mode==2){
            /* int8-M1 router has no M4 port in this milestone (scoped to the int4-HP path only,
             * matching the ALREADY-validated kernel) -- falls back to 4 separate M1 dispatches.
             * Still correct, just not batched; router is a small bucket (~4ms/token) regardless. */
            for(int s=0;s<4;s++){ uint8_t Ai8[3000]; pack_act_i8(hn[s],d,Ai8); lin_mm_i8(&ly->router_i8,Ai8,y4r+s*ly->router_i8.N,nt); }
        } else {
            for(int s=0;s<4;s++) for(int e=0;e<ne;e++) y4r[s*ne+e]=vdot_f32(ly->router+(size_t)e*d,hn[s],d);
        }
        for(int s=0;s<4;s++){
            float rl[256]; memcpy(rl,y4r+s*ne,(size_t)ne*4); softmax(rl,ne);
            for(int a=0;a<na;a++){ int bi=-1; float bv=-1e30f; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(sel[s][b]==e)used=1; if(!used&&rl[e]>bv){bv=rl[e];bi=e;} } sel[s][a]=bi; sw[s][a]=bv; }
            float ssum=0; for(int a=0;a<na;a++)ssum+=sw[s][a]; for(int a=0;a<na;a++)sw[s][a]/=ssum;
        }
        /* MoE expert FFN (codex_recs_1.md §22.37, M-batch milestone 3): each sequence independently
         * selects its own top-8-of-128 experts from its own hidden state, so most experts are NOT
         * shared across the 4-sequence batch. This section batches the one case worth batching --
         * an expert selected by ALL 4 sequences at once -- via M4, and falls back to the existing
         * per-sequence M1 path for everything else (an expert selected by only 1-3 sequences).
         * Deliberately NOT padding 2- or 3-way overlaps up to a full M4 call: milestone 2
         * (codex_recs_1.md §22.36) found M4's own arithmetic work scales with M rather than being a
         * fixed cost, so a padded call with 1-3 real rows would do CLOSE TO the same arithmetic as
         * a full 4-row call while only saving 1-3 weight-stream reads vs 1-3 separate M1 calls --
         * plausibly a net loss, not clearly a win, and not worth the added complexity/risk without
         * first knowing (from THIS milestone's own overlap statistics, reported by the caller)
         * whether partial overlaps are even common enough to matter. */
        /* esel[e][s] = the slot index a such that sel[s][a]==e (i.e. sequence s's own selection
         * order for expert e), or -1 if sequence s did not select expert e; ecount[e] = how many
         * of the 4 sequences selected expert e (0..4). */
        int esel[128][4]; int ecount[128];
        for(int e=0;e<ne;e++){ ecount[e]=0; for(int s=0;s<4;s++) esel[e][s]=-1; }
        for(int s=0;s<4;s++) for(int a=0;a<na;a++){ int e=sel[s][a]; esel[e][s]=a; ecount[e]++; }
        int processed[4][32]={{0}};
        for(int s=0;s<4;s++) for(int i=0;i<d;i++)eout[s][i]=0;
        for(int e=0;e<ne;e++){
            g_moe4_total_expert_slots += ecount[e]>0 ? 1 : 0;
            g_moe_ecount_hist[ecount[e]]++;
            if(ecount[e]!=4) continue;
            g_moe4_hits++;
            pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
            lin_mm_hp_m4(&ly->eg[e],Abuf4,y4eg); lin_mm_hp_m4(&ly->eu[e],Abuf4,y4eu);
            for(int s=0;s<4;s++){
                float*gs=y4eg+(size_t)s*moe,*us=y4eu+(size_t)s*moe;
                if(g_swiglu_fast==1) swiglu_hswish_rvv(gs,us,moe);
                else if(g_swiglu_fast==2) swiglu_ratsig_rvv(gs,us,moe);
                else swiglu_exact(gs,us,moe);
            }
            pack_act_hp_m4(y4eg+0*(size_t)moe,y4eg+1*(size_t)moe,y4eg+2*(size_t)moe,y4eg+3*(size_t)moe,moe,Abuf4);
            lin_mm_hp_m4(&ly->ed[e],Abuf4,y4ed);
            for(int s=0;s<4;s++){
                int a=esel[e][s]; float w=sw[s][a];
                for(int i=0;i<d;i++) eout[s][i]+=w*y4ed[s*ly->ed[e].N+i];
                processed[s][a]=1;
            }
        }
        /* Each sequence's OWN hn must be packed into M1's (290B/block) format before the fallback
         * loop -- this is the direct equivalent of forward()'s own `pack_act_hp(hn,d,Abuf2)` call,
         * just done once per sequence into that sequence's own Abuf2_4 slot instead of once for the
         * single stream. */
        for(int s=0;s<4;s++) pack_act_hp(hn[s],d,Abuf2_4+(size_t)s*3000);
        for(int s=0;s<4;s++){
            for(int a=0;a<na;a++){ if(processed[s][a]) continue; int e=sel[s][a]; float w=sw[s][a];
                lin_mm_hp(&ly->eg[e],Abuf2_4+(size_t)s*3000,g[s],nt); lin_mm_hp(&ly->eu[e],Abuf2_4+(size_t)s*3000,u[s],nt);
                if(g_swiglu_fast==1) swiglu_hswish_rvv(g[s],u[s],moe);
                else if(g_swiglu_fast==2) swiglu_ratsig_rvv(g[s],u[s],moe);
                else swiglu_exact(g[s],u[s],moe);
                lin_mm(&ly->ed[e],g[s],tmp[s],nt,Abuf[s]);
                for(int i=0;i<d;i++)eout[s][i]+=w*tmp[s][i]; }
            for(int i=0;i<d;i++)h[s][i]+=eout[s][i];
        }
    }
    for(int s=0;s<4;s++) rmsnorm(hn[s],h[s],m->out_norm,d,m->eps);
    pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
    lin_mm_hp_m4(&m->lm,Abuf4,y4lm);
    for(int s=0;s<4;s++) memcpy(logits[s],y4lm+s*m->lm.N,(size_t)m->lm.N*4);
}

/* Batched prefill track (codex_recs_1.md §22.38): batches the DENSE layers (QKV, O, router,
 * lm_head) across 4 CONSECUTIVE positions of one prompt/sequence via the same validated M4 kernel
 * used for decode-phase M-batching (§22.35-37), instead of the current token-at-a-time prefill
 * (N sequential forward() calls). Unlike decode's M-batch, all 4 positions' input tokens are known
 * upfront from the prompt -- no cross-position autoregressive dependency for the dense layers, so
 * batching them is a strictly simpler case than decode's own (there, 4 independent sequences had
 * to genuinely coexist; here, 4 positions of the SAME known prompt trivially do).
 *
 * Attention stays per-position, exact, UNBATCHED this milestone -- a genuine flash-attention-style
 * tiled/blocked causal computation (sharing K/V reads across the 4 chunk positions) is a separate,
 * larger undertaking with its own numerical-tiling risk, deliberately not attempted here, matching
 * the same "batch what clearly batches, defer the harder fused piece" scoping used for decode's
 * own dense-vs-expert split. Correctness of the per-position split is straightforward: each
 * position's own K/V depends only on that position's own (now M4-batched) QKV output, not on any
 * OTHER position in the chunk, so all 4 positions' K/V can be written into the cache up front; each
 * position's own attention then reads with its OWN correct causal bound (`j<=pos0+i`), which is
 * exactly equivalent to writing+attending one position at a time -- the causal bound alone
 * guarantees position pos0+i's attention never reads position pos0+i+1..3's K/V, independent of
 * whether that later K/V happens to already be present in memory.
 *
 * MoE expert FFN stays per-position via the existing M1 path, same reasoning as decode's own
 * scoping (§22.37): each position's own router, from its own hidden state, generally selects a
 * different expert set; whether WITHIN-sequence positions (which share context/content, unlike
 * decode's independent sequences) show meaningfully more expert-selection overlap than across-
 * sequence decode did is an open, checkable question, deferred to this milestone's own measurement
 * report rather than assumed. */
static void prefill_chunk4(Model*m,const int toks[4],int pos0,Kv*kv,float*logits[4],
        float*h[4],float*hn[4],float*q[4],float*k[4],float*vv[4],float*att[4],float*tmp[4],
        float*g[4],float*u[4],float*eout[4],uint8_t*Abuf[4],uint8_t*Abuf4,uint8_t*Abuf2_4){
    int d=m->d,nh=m->nh,nkv=m->nkv,hd=m->hd,nt=m->nt,gpr=nh/nkv,moe=m->moe,ne=m->n_exp,na=m->n_act;
    for(int i=0;i<4;i++) memcpy(h[i],m->tok_embd+(size_t)toks[i]*d,d*4);
    float cosb[4][256],sinb[4][256];
    for(int i=0;i<4;i++) rope_table(cosb[i],sinb[i],hd,pos0+i,m->rope_base); /* absolute position,
        NOT chunk-relative -- rope angle depends on the token's real position in the sequence. */
    static float *y4q=NULL,*y4k=NULL,*y4v=NULL,*y4o=NULL,*y4r=NULL,*y4lm=NULL,*y4eg=NULL,*y4eu=NULL,*y4ed=NULL;
    if(!y4q){ int maxqkv=nh*hd>nkv*hd?nh*hd:nkv*hd; y4q=malloc((size_t)4*maxqkv*4); y4k=malloc((size_t)4*maxqkv*4); y4v=malloc((size_t)4*maxqkv*4);
        y4o=malloc((size_t)4*d*4); y4r=malloc((size_t)4*256*4); y4lm=malloc((size_t)4*(size_t)m->vocab*4);
        y4eg=malloc((size_t)4*moe*4); y4eu=malloc((size_t)4*moe*4); y4ed=malloc((size_t)4*d*4); }

    for(int l=0;l<m->nl;l++){ Layer*ly=&m->L[l];
        for(int i=0;i<4;i++) rmsnorm(hn[i],h[i],ly->attn_norm,d,m->eps);
        pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
        lin_mm_hp_m4(&ly->q,Abuf4,y4q); lin_mm_hp_m4(&ly->k,Abuf4,y4k); lin_mm_hp_m4(&ly->v,Abuf4,y4v);
        for(int i=0;i<4;i++){ memcpy(q[i],y4q+i*ly->q.N,ly->q.N*4); memcpy(k[i],y4k+i*ly->k.N,ly->k.N*4); memcpy(vv[i],y4v+i*ly->v.N,ly->v.N*4); }
        for(int i=0;i<4;i++){
            for(int hh=0;hh<nh;hh++){ rmsnorm(q[i]+hh*hd,q[i]+hh*hd,ly->q_norm,hd,m->eps); rope_apply(q[i]+hh*hd,hd,cosb[i],sinb[i]); }
            for(int hh=0;hh<nkv;hh++){ rmsnorm(k[i]+hh*hd,k[i]+hh*hd,ly->k_norm,hd,m->eps); rope_apply(k[i]+hh*hd,hd,cosb[i],sinb[i]); }
        }
        int kvd_s=nkv*hd;
        float*Kc=kv->Kc+(size_t)l*kv->ctx*kvd_s,*Vc=kv->Vc+(size_t)l*kv->ctx*kvd_s;
        for(int i=0;i<4;i++){
            int pos=pos0+i;
            if(pos<0 || pos>=kv->ctx){ fprintf(stderr,"KV position overflow: pos=%d ctx=%d\n",pos,kv->ctx); abort(); }
            for(int kvh=0;kvh<nkv;kvh++){
                memcpy(Kc+(size_t)kvh*kv->ctx*hd+(size_t)pos*hd, k[i]+(size_t)kvh*hd, hd*4);
                memcpy(Vc+(size_t)kvh*kv->ctx*hd+(size_t)pos*hd, vv[i]+(size_t)kvh*hd, hd*4);
            }
        }
        float scale=1.0f/sqrtf(hd);
        for(int i=0;i<4;i++){
            /* exact per-position causal attention, unbatched -- see this function's own top
             * comment for why. The pos bound alone guarantees correctness even though positions
             * pos0+i+1..pos0+3's K/V are already resident in the cache by this point. */
            int pos=pos0+i;
            if(g_attn_nt<=1){
                for(int hh=0;hh<nh;hh++){ int kvh=hh/gpr; float*qh=q[i]+hh*hd,*sc=tmp[i];
                    float*Kh=Kc+(size_t)kvh*kv->ctx*hd,*Vh=Vc+(size_t)kvh*kv->ctx*hd;
                    for(int j=0;j<=pos;j++){ float*kj=Kh+(size_t)j*hd; sc[j]=vdot_f32(qh,kj,hd)*scale; }
                    softmax(sc,pos+1);
                    float*oh=att[i]+hh*hd; for(int t=0;t<hd;t++)oh[t]=0;
                    for(int j=0;j<=pos;j++){ float*vj=Vh+(size_t)j*hd; vaxpy_f32(oh,vj,sc[j],hd); } }
            } else {
                attn_dispatch(q[i],Kc,Vc,att[i],pos,scale,hd,nkv,gpr,kv->ctx,g_attn_nt);
            }
        }
        pack_act_hp_m4(att[0],att[1],att[2],att[3],ly->o.K,Abuf4);
        lin_mm_hp_m4(&ly->o,Abuf4,y4o);
        for(int i=0;i<4;i++){ for(int t=0;t<d;t++) h[i][t]+=y4o[i*ly->o.N+t]; }
        for(int i=0;i<4;i++) rmsnorm(hn[i],h[i],ly->ffn_norm,d,m->eps);
        pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
        int sel[4][32]; float sw[4][32];
        if(g_router_mode==1) lin_mm_hp_m4(&ly->router_hp,Abuf4,y4r);
        else if(g_router_mode==2){
            for(int i=0;i<4;i++){ uint8_t Ai8[3000]; pack_act_i8(hn[i],d,Ai8); lin_mm_i8(&ly->router_i8,Ai8,y4r+i*ly->router_i8.N,nt); }
        } else {
            for(int i=0;i<4;i++) for(int e=0;e<ne;e++) y4r[i*ne+e]=vdot_f32(ly->router+(size_t)e*d,hn[i],d);
        }
        for(int i=0;i<4;i++){
            float rl[256]; memcpy(rl,y4r+i*ne,(size_t)ne*4); softmax(rl,ne);
            for(int a=0;a<na;a++){ int bi=-1; float bv=-1e30f; for(int e=0;e<ne;e++){ int used=0; for(int b=0;b<a;b++)if(sel[i][b]==e)used=1; if(!used&&rl[e]>bv){bv=rl[e];bi=e;} } sel[i][a]=bi; sw[i][a]=bv; }
            float ssum=0; for(int a=0;a<na;a++)ssum+=sw[i][a]; for(int a=0;a<na;a++)sw[i][a]/=ssum;
        }
        /* MoE expert FFN: same exact-4-way-overlap batching as decode's own §22.37, plus the same
         * per-sequence (here, per-position) M1 fallback for everything else. Reuses the identical
         * grouping logic; see forward4_dense_batch's own MoE-FFN comment for the full rationale. */
        int esel[128][4]; int ecount[128];
        for(int e=0;e<ne;e++){ ecount[e]=0; for(int i=0;i<4;i++) esel[e][i]=-1; }
        for(int i=0;i<4;i++) for(int a=0;a<na;a++){ int e=sel[i][a]; esel[e][i]=a; ecount[e]++; }
        int processed[4][32]={{0}};
        for(int i=0;i<4;i++) for(int t=0;t<d;t++)eout[i][t]=0;
        for(int e=0;e<ne;e++){
            g_moe4_total_expert_slots += ecount[e]>0 ? 1 : 0;
            g_moe_ecount_hist[ecount[e]]++;
            if(ecount[e]!=4 || !g_prefill_moe4) continue;
            g_moe4_hits++;
            pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
            lin_mm_hp_m4(&ly->eg[e],Abuf4,y4eg); lin_mm_hp_m4(&ly->eu[e],Abuf4,y4eu);
            for(int i=0;i<4;i++){
                float*gs=y4eg+(size_t)i*moe,*us=y4eu+(size_t)i*moe;
                if(g_swiglu_fast==1) swiglu_hswish_rvv(gs,us,moe);
                else if(g_swiglu_fast==2) swiglu_ratsig_rvv(gs,us,moe);
                else swiglu_exact(gs,us,moe);
            }
            pack_act_hp_m4(y4eg+0*(size_t)moe,y4eg+1*(size_t)moe,y4eg+2*(size_t)moe,y4eg+3*(size_t)moe,moe,Abuf4);
            lin_mm_hp_m4(&ly->ed[e],Abuf4,y4ed);
            for(int i=0;i<4;i++){
                int a=esel[e][i]; float w=sw[i][a];
                for(int t=0;t<d;t++) eout[i][t]+=w*y4ed[i*ly->ed[e].N+t];
                processed[i][a]=1;
            }
        }
        for(int i=0;i<4;i++) pack_act_hp(hn[i],d,Abuf2_4+(size_t)i*3000);
        for(int i=0;i<4;i++){
            for(int a=0;a<na;a++){ if(processed[i][a]) continue; int e=sel[i][a]; float w=sw[i][a];
                lin_mm_hp(&ly->eg[e],Abuf2_4+(size_t)i*3000,g[i],nt); lin_mm_hp(&ly->eu[e],Abuf2_4+(size_t)i*3000,u[i],nt);
                if(g_swiglu_fast==1) swiglu_hswish_rvv(g[i],u[i],moe);
                else if(g_swiglu_fast==2) swiglu_ratsig_rvv(g[i],u[i],moe);
                else swiglu_exact(g[i],u[i],moe);
                lin_mm(&ly->ed[e],g[i],tmp[i],nt,Abuf[i]);
                for(int t=0;t<d;t++)eout[i][t]+=w*tmp[i][t]; }
            for(int t=0;t<d;t++)h[i][t]+=eout[i][t];
        }
    }
    for(int i=0;i<4;i++) rmsnorm(hn[i],h[i],m->out_norm,d,m->eps);
    pack_act_hp_m4(hn[0],hn[1],hn[2],hn[3],d,Abuf4);
    lin_mm_hp_m4(&m->lm,Abuf4,y4lm);
    for(int i=0;i<4;i++) memcpy(logits[i],y4lm+i*m->lm.N,(size_t)m->lm.N*4);
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

/* M-batch track milestone 1 (codex_recs_1.md §22.35), QWEN_MBATCH_TEST=1: real 4-sequence batched
 * decode test using genuinely independent prompts (4 of the harness's own real, distinct prompts,
 * not synthetic data), sharing weight-stream reads at QKV/O/router/lm_head via forward4_dense_batch
 * while attention and MoE-FFN stay per-sequence. Validates every batched sequence against a
 * SEPARATE run of the same prompt through the existing, already-proven M=1 forward() path (own KV
 * cache, own decode loop, identical prefill), then reports the required real metrics: aggregate
 * tok/s, per-sequence tok/s, latency, memory, and M=1 regression -- never a synthetic GEMM number
 * described as model tok/s, per the explicit instruction. */
static void run_mbatch_test(Model*m){
    const int NSEQ=4, NGEN=16;
    static const int seq_idx[4]={0,1,2,3}; /* hp1..hp4: factual/capitals, factual/chemistry,
        reasoning/syllogism, reasoning/sequence -- 4 real, distinct prompts, genuinely different
        content/length, not 4 copies of one prompt. */
    int d=m->d,qd=m->nh*m->hd,moe=m->moe,maxk=qd>moe?(qd>d?qd:d):(moe>d?moe:d); if(d>maxk)maxk=d;
    int ctx=256;

    /* --- Path 1: 4 SEPARATE M=1 decodes, own Kv/buffers each, the already-proven serial path --- */
    Kv kv1[4]; float*h1[4],*hn1[4],*q1[4],*k1[4],*vv1[4],*att1[4],*tmp1[4],*g1[4],*u1[4],*eout1[4],*logits1[4]; uint8_t*Abuf1[4],*Abuf1b[4];
    for(int s=0;s<4;s++){
        kv1[s].kvd=m->nkv*m->hd; kv1[s].ctx=ctx;
        kv1[s].Kc=calloc((size_t)m->nl*ctx*kv1[s].kvd,4); kv1[s].Vc=calloc((size_t)m->nl*ctx*kv1[s].kvd,4);
        h1[s]=malloc(d*4); hn1[s]=malloc(d*4); q1[s]=malloc(qd*4); k1[s]=malloc(kv1[s].kvd*4); vv1[s]=malloc(kv1[s].kvd*4);
        att1[s]=malloc(qd*4); tmp1[s]=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4);
        g1[s]=malloc(moe*4); u1[s]=malloc(moe*4); eout1[s]=malloc(d*4); logits1[s]=malloc((size_t)m->vocab*4);
        Abuf1[s]=malloc((size_t)(maxk/256)*AREC); Abuf1b[s]=malloc((size_t)(maxk/256)*AREC);
    }
    int toks1[4][256]; int ntoks1[4];
    static float step1_logits_m1[4][160000];
    /* prefill and decode timed SEPARATELY -- both paths pay the SAME (per-sequence, unbatched;
     * this milestone explicitly scopes batching to the DECODE phase, see this function's own top
     * comment) prefill cost, which dilutes an aggregate "total wall time" comparison for these
     * short test prompts where prefill is a substantial fraction of total time. Decode-only timing
     * isolates what M-batching itself actually contributes. */
    double t_m1_prefill=0, t_m1_decode_0;
    for(int s=0;s<4;s++){
        const HarnessPrompt*hp=&g_hprompts[seq_idx[s]];
        double tp0=now();
        for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,&kv1[s],logits1[s],hn1[s],q1[s],k1[s],vv1[s],att1[s],tmp1[s],g1[s],u1[s],eout1[s],Abuf1[s],Abuf1b[s]);
        t_m1_prefill += now()-tp0;
    }
    t_m1_decode_0=now();
    for(int s=0;s<4;s++){
        const HarnessPrompt*hp=&g_hprompts[seq_idx[s]];
        ntoks1[s]=0;
        int cur=argmax(logits1[s],m->vocab); toks1[s][ntoks1[s]++]=cur;
        for(int step=1;step<NGEN;step++){
            forward(m,cur,hp->n+step-1,&kv1[s],logits1[s],hn1[s],q1[s],k1[s],vv1[s],att1[s],tmp1[s],g1[s],u1[s],eout1[s],Abuf1[s],Abuf1b[s]);
            if(step==1) memcpy(step1_logits_m1[s],logits1[s],(size_t)m->vocab*4); /* first post-prefill
                decode step, BEFORE any compounding from a possibly-different chosen token -- prefill
                itself is identical M=1 in both paths, so this isolates ONE 48-layer M4-batched pass's
                own divergence from a SAME-input M1 pass, separate from later steps' extra divergence
                from potentially different tokens/KV entries feeding back in. */
            cur=argmax(logits1[s],m->vocab); toks1[s][ntoks1[s]++]=cur;
        }
    }
    double t_m1_decode=now()-t_m1_decode_0;
    double t_m1=t_m1_prefill+t_m1_decode;

    /* --- Path 2: 1 BATCHED M=4 decode, all 4 sequences advance in lockstep --- */
    Kv kv4[4]; float*h4[4],*hn4[4],*q4[4],*k4[4],*vv4[4],*att4[4],*tmp4[4],*g4[4],*u4[4],*eout4[4],*logits4[4]; uint8_t*Abuf4arr[4];
    for(int s=0;s<4;s++){
        kv4[s].kvd=m->nkv*m->hd; kv4[s].ctx=ctx;
        kv4[s].Kc=calloc((size_t)m->nl*ctx*kv4[s].kvd,4); kv4[s].Vc=calloc((size_t)m->nl*ctx*kv4[s].kvd,4);
        h4[s]=malloc(d*4); hn4[s]=malloc(d*4); q4[s]=malloc(qd*4); k4[s]=malloc(kv4[s].kvd*4); vv4[s]=malloc(kv4[s].kvd*4);
        att4[s]=malloc(qd*4); tmp4[s]=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4);
        g4[s]=malloc(moe*4); u4[s]=malloc(moe*4); eout4[s]=malloc(d*4); logits4[s]=malloc((size_t)m->vocab*4);
        Abuf4arr[s]=malloc((size_t)(maxk/256)*AREC);
    }
    int maxk4=qd>d?qd:d; /* forward4_dense_batch's Abuf4 must cover the widest K it ever packs --
        att is qd-wide (O-projection input), hn is d-wide (QKV/router/lm_head input) */
    uint8_t*Abuf4buf=malloc((size_t)(maxk4/256)*AREC_M4);
    uint8_t*Abuf2_4=malloc((size_t)4*3000);
    int toks4[4][256]; int ntoks4[4]; for(int s=0;s<4;s++) ntoks4[s]=0;

    /* prefill: still per-sequence (this milestone scopes batching to DECODE steps, matching the
     * "M-batch is a decode-phase lever" framing throughout this whole track -- prefill is its own
     * separate, not-yet-attempted track per the ladder's own next item). Timed separately from
     * decode for the SAME reason as path 1 above. */
    double t_m4_prefill=0;
    for(int s=0;s<4;s++){
        const HarnessPrompt*hp=&g_hprompts[seq_idx[s]];
        double tp0=now();
        for(int p=0;p<hp->n;p++) forward(m,hp->toks[p],p,&kv4[s],logits4[s],hn4[s],q4[s],k4[s],vv4[s],att4[s],tmp4[s],g4[s],u4[s],eout4[s],Abuf4arr[s],Abuf2_4);
        t_m4_prefill += now()-tp0;
    }
    int cur4[4], pos4[4];
    for(int s=0;s<4;s++){ cur4[s]=argmax(logits4[s],m->vocab); toks4[s][ntoks4[s]++]=cur4[s]; pos4[s]=g_hprompts[seq_idx[s]].n; }
    Kv*kvp[4]={&kv4[0],&kv4[1],&kv4[2],&kv4[3]};
    static float step1_logits_m4[4][160000];
    double t_m4_decode_0=now();
    for(int step=1;step<NGEN;step++){
        int tokv[4]; for(int s=0;s<4;s++) tokv[s]=cur4[s];
        forward4_dense_batch(m,tokv,pos4,kvp,logits4,h4,hn4,q4,k4,vv4,att4,tmp4,g4,u4,eout4,Abuf4arr,Abuf4buf,Abuf2_4);
        if(step==1) for(int s=0;s<4;s++) memcpy(step1_logits_m4[s],logits4[s],(size_t)m->vocab*4);
        for(int s=0;s<4;s++){ cur4[s]=argmax(logits4[s],m->vocab); toks4[s][ntoks4[s]++]=cur4[s]; pos4[s]++; }
    }
    double t_m4_decode=now()-t_m4_decode_0;
    double t_m4=t_m4_prefill+t_m4_decode;

    /* first-decode-step-only comparison: isolates ONE 48-layer M4-batched forward pass's own
     * numerical divergence from a same-input M1 pass, separate from later steps' extra divergence
     * once a possibly-different chosen token starts feeding back into subsequent steps. */
    double step1_max_abs=0, step1_sum_abs=0; long step1_cmp=0; long step1_tok_match=0;
    for(int s=0;s<4;s++){
        for(int e=0;e<m->vocab;e++){ double ad=fabs((double)step1_logits_m1[s][e]-(double)step1_logits_m4[s][e]);
            if(ad>step1_max_abs) step1_max_abs=ad; step1_sum_abs+=ad; step1_cmp++; }
        if(argmax(step1_logits_m1[s],m->vocab)==argmax(step1_logits_m4[s],m->vocab)) step1_tok_match++;
    }
    printf("first-decode-step-only (isolates one 48-layer M4 pass, no compounding yet): max_abs_diff=%e mean_abs_diff=%e, argmax match %ld/4 sequences\n",
        step1_max_abs, step1_sum_abs/step1_cmp, step1_tok_match);

    /* --- validation: token agreement + logit closeness, M=4 vs the separate M=1 run --- */
    long tok_match=0, tok_total=0; double max_logit_abs=0, sum_logit_abs=0; long logit_cmp=0;
    printf("per-step token match (1=agree, 0=diverge), one row per sequence:\n");
    for(int s=0;s<4;s++){
        printf("  seq%d: ",s);
        for(int i=0;i<NGEN;i++){ tok_total++; int match=(toks1[s][i]==toks4[s][i]); if(match) tok_match++; printf("%d",match); }
        printf("\n");
    }
    /* one more logits comparison at the FINAL step, on the buffers already in hand */
    for(int s=0;s<4;s++) for(int e=0;e<m->vocab;e++){
        double ad=fabs((double)logits1[s][e]-(double)logits4[s][e]);
        if(ad>max_logit_abs) max_logit_abs=ad; sum_logit_abs+=ad; logit_cmp++;
    }

    printf("\n=== M-batch test: 4 real, independent sequences, M=1 (separate) vs M=4 (batched dense layers) ===\n");
    printf("sequences: ");
    for(int s=0;s<4;s++) printf("%s%s", g_hprompts[seq_idx[s]].name, s<3?", ":"\n");
    printf("token agreement (batched vs separate M=1, argmax): %ld/%ld (%.2f%%)\n", tok_match, tok_total, 100.0*tok_match/tok_total);
    printf("final-step logits: max_abs_diff=%e mean_abs_diff=%e (%ld comparisons) -- expected small, nonzero (M4's shared-scale quantization, not a bug, see codex_recs_1.md %s22.35)\n",
        max_logit_abs, sum_logit_abs/logit_cmp, logit_cmp, "§");
    printf("--- prefill (NOT batched this milestone -- same per-sequence M1 cost paid by both paths, shown for transparency) ---\n");
    printf("M=1 path prefill: %.3fs total (4 sequences) | M=4 path prefill: %.3fs total (4 sequences)\n", t_m1_prefill, t_m4_prefill);
    printf("--- decode (the actual M-batching lever this milestone claims) ---\n");
    printf("M=1 (4 separate decodes) wall: %.3fs for %d total tokens (%d seq x %d gen) -> aggregate %.2f tok/s, %.2f tok/s/sequence\n",
        t_m1_decode, NSEQ*NGEN, NSEQ, NGEN, (NSEQ*NGEN)/t_m1_decode, NGEN/t_m1_decode);
    printf("M=4 (1 batched decode)   wall: %.3fs for %d total tokens (%d seq x %d gen) -> aggregate %.2f tok/s, %.2f tok/s/sequence\n",
        t_m4_decode, NSEQ*NGEN, NSEQ, NGEN, (NSEQ*NGEN)/t_m4_decode, NGEN/t_m4_decode);
    printf("DECODE-ONLY aggregate speedup: %.3fx | per-sequence decode latency: M=1 %.2fms/tok/seq vs M=4 %.2fms/tok(shared across the batch)\n",
        t_m1_decode/t_m4_decode, 1000.0*t_m1_decode/(NSEQ*NGEN), 1000.0*t_m4_decode/NGEN);
    printf("--- totals (prefill+decode combined, included for completeness -- NOT the primary metric since prefill dilutes it) ---\n");
    printf("M=1 total: %.3fs -> %.2f tok/s aggregate | M=4 total: %.3fs -> %.2f tok/s aggregate | total speedup: %.3fx\n",
        t_m1, (NSEQ*NGEN)/t_m1, t_m4, (NSEQ*NGEN)/t_m4, t_m1/t_m4);
    size_t extra_mem = (size_t)(maxk4/256)*AREC_M4 + 4*3000 + 4*(size_t)qd*4*3; /* Abuf4buf + Abuf2_4 + y4q/y4k/y4v, the main new allocations vs the M=1 path */
    printf("additional memory for the batched path (beyond 4x the existing per-sequence buffers): ~%.1f KB\n", extra_mem/1024.0);
    printf("MoE expert-FFN batching (milestone 3): %ld of %ld (expert, layer, decode-step) slots with >=1 selecting sequence had all 4 sequences select the SAME expert (%.2f%%) -- only this case is M4-batched, the rest fall back to per-sequence M1\n",
        g_moe4_hits, g_moe4_total_expert_slots, g_moe4_total_expert_slots>0 ? 100.0*g_moe4_hits/g_moe4_total_expert_slots : 0.0);
    { long tot4=g_moe_ecount_hist[1]+g_moe_ecount_hist[2]+g_moe_ecount_hist[3]+g_moe_ecount_hist[4];
      printf("full selecting-sequence-count histogram (diagnostic, answers whether PARTIAL overlap is common enough to matter for a future padded-batching strategy):\n");
      printf("  count=1 (no overlap): %ld (%.2f%%) | count=2: %ld (%.2f%%) | count=3: %ld (%.2f%%) | count=4 (batched): %ld (%.2f%%)\n",
        g_moe_ecount_hist[1], 100.0*g_moe_ecount_hist[1]/tot4, g_moe_ecount_hist[2], 100.0*g_moe_ecount_hist[2]/tot4,
        g_moe_ecount_hist[3], 100.0*g_moe_ecount_hist[3]/tot4, g_moe_ecount_hist[4], 100.0*g_moe_ecount_hist[4]/tot4);
    }

    for(int s=0;s<4;s++){
        free(h1[s]);free(hn1[s]);free(q1[s]);free(k1[s]);free(vv1[s]);free(att1[s]);free(tmp1[s]);free(g1[s]);free(u1[s]);free(eout1[s]);free(logits1[s]);free(Abuf1[s]);free(Abuf1b[s]);
        free(kv1[s].Kc);free(kv1[s].Vc);
        free(h4[s]);free(hn4[s]);free(q4[s]);free(k4[s]);free(vv4[s]);free(att4[s]);free(tmp4[s]);free(g4[s]);free(u4[s]);free(eout4[s]);free(logits4[s]);free(Abuf4arr[s]);
        free(kv4[s].Kc);free(kv4[s].Vc);
    }
    free(Abuf4buf); free(Abuf2_4);
}

/* Batched prefill (codex_recs_1.md §22.38), QWEN_PREFILL_TEST=1: validates and benchmarks
 * prefill_chunk4 (4-consecutive-position dense-layer M4 batching within ONE sequence) against the
 * existing token-at-a-time sequential prefill (repeated forward() calls), per the ladder's own
 * explicit requirement: "Benchmark prompt lengths 128/512/1024. Preserve exact causal masking and
 * compare logits against sequential prefill." Test lengths: 16 (short, matches this whole session's
 * own short/128/512/1024 A/B convention), 128/512/1024 (the explicitly required lengths), and 19 --
 * deliberately NOT a multiple of 4, included specifically to exercise and validate the N-mod-4
 * remainder fallback path (the other four lengths are all exact multiples of 4, so alone they'd
 * never touch that code path). Prompts synthesized via the same hp9-tiling convention already
 * established for QWEN_CTXLEN benchmarks (prompt[i]=hp9[i%113]).
 *
 * Unlike the decode M-batch milestones (codex_recs_1.md §22.35-37), prefill has NO argmax-choice
 * compounding: every position's input token is fixed by the prompt itself, not fed back from a
 * possibly-different model prediction, so the divergence source here is purely M4's own per-chunk
 * shared-scale quantization noise (plus its ordinary propagation through the KV cache into later
 * positions' attention) -- expected to be much smaller than milestone 2's 70.31% figure, and this
 * harness reports the real number rather than assuming it. */
static void run_prefill_test(Model*m){
    const int TESTLENS[]={16,19,128,512,1024}; int NLEN=5;
    /* QWEN_PREFILL_SANITIZE=1: caps the length sweep at N=512 -- sanitizer runs are slow (ASan
     * especially), and N=1024 exercises the exact same code paths as N=512 (same chunked/remainder
     * structure, just more repetitions), so it adds runtime without adding memory-safety coverage.
     * Not used for the real speed benchmark, only to bound sanitizer wall-clock time. */
    { const char*sz=getenv("QWEN_PREFILL_SANITIZE"); if(sz && atoi(sz)) NLEN=4; }
    { const char*pm=getenv("QWEN_PREFILL_MOE4"); if(pm) g_prefill_moe4=atoi(pm); }
    int d=m->d,qd=m->nh*m->hd,moe=m->moe,maxk=qd>moe?(qd>d?qd:d):(moe>d?moe:d); if(d>maxk)maxk=d;
    int maxk4=qd>d?qd:d;
    static int prompt[1536];
    for(int L=0;L<NLEN;L++){
        int np=TESTLENS[L]; for(int i=0;i<np;i++) prompt[i]=hp9[i%113];
        int ctx=np+4;

        /* --- Path 1: sequential exact prefill, own Kv/buffers, the already-proven forward() path --- */
        Kv kv1; kv1.kvd=m->nkv*m->hd; kv1.ctx=ctx; kv1.Kc=calloc((size_t)m->nl*ctx*kv1.kvd,4); kv1.Vc=calloc((size_t)m->nl*ctx*kv1.kvd,4);
        float*hn1=malloc(d*4),*q1=malloc(qd*4),*k1=malloc(kv1.kvd*4),*vv1=malloc(kv1.kvd*4),*att1=malloc(qd*4),
             *tmp1=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4),*g1=malloc(moe*4),*u1=malloc(moe*4),*eout1=malloc(d*4),*logits1=malloc((size_t)m->vocab*4);
        uint8_t*Abuf1=malloc((size_t)(maxk/256)*AREC),*Abuf1b=malloc((size_t)(maxk/256)*AREC);
        static int argmax1[1536];
        double t1_0=now();
        for(int p=0;p<np;p++){ forward(m,prompt[p],p,&kv1,logits1,hn1,q1,k1,vv1,att1,tmp1,g1,u1,eout1,Abuf1,Abuf1b); argmax1[p]=argmax(logits1,m->vocab); }
        double t1=now()-t1_0;
        static float flogits1[160000]; memcpy(flogits1,logits1,(size_t)m->vocab*4);

        /* --- Path 2: chunked M4 prefill (4 consecutive positions at a time) + remainder fallback --- */
        Kv kv2; kv2.kvd=m->nkv*m->hd; kv2.ctx=ctx; kv2.Kc=calloc((size_t)m->nl*ctx*kv2.kvd,4); kv2.Vc=calloc((size_t)m->nl*ctx*kv2.kvd,4);
        float*h2[4],*hn2[4],*q2[4],*k2[4],*vv2[4],*att2[4],*tmp2[4],*g2[4],*u2[4],*eout2[4],*logits2[4]; uint8_t*Abuf2arr[4];
        for(int s=0;s<4;s++){
            h2[s]=malloc(d*4); hn2[s]=malloc(d*4); q2[s]=malloc(qd*4); k2[s]=malloc(kv2.kvd*4); vv2[s]=malloc(kv2.kvd*4);
            att2[s]=malloc(qd*4); tmp2[s]=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4);
            g2[s]=malloc(moe*4); u2[s]=malloc(moe*4); eout2[s]=malloc(d*4); logits2[s]=malloc((size_t)m->vocab*4);
            Abuf2arr[s]=malloc((size_t)(maxk/256)*AREC);
        }
        uint8_t*Abuf4buf=malloc((size_t)(maxk4/256)*AREC_M4);
        uint8_t*Abuf2_4=malloc((size_t)4*3000);
        static int argmax2[1536];
        long moe4_hits0=g_moe4_hits, moe4_slots0=g_moe4_total_expert_slots;
        long hist0[5]; for(int i=0;i<5;i++) hist0[i]=g_moe_ecount_hist[i];
        int nchunks=np/4, rem=np%4;
        double t2_0=now();
        for(int c=0;c<nchunks;c++){
            int toks4[4]; for(int i=0;i<4;i++) toks4[i]=prompt[c*4+i];
            prefill_chunk4(m,toks4,c*4,&kv2,logits2,h2,hn2,q2,k2,vv2,att2,tmp2,g2,u2,eout2,Abuf2arr,Abuf4buf,Abuf2_4);
            for(int i=0;i<4;i++) argmax2[c*4+i]=argmax(logits2[i],m->vocab);
        }
        for(int r=0;r<rem;r++){
            int p=nchunks*4+r;
            forward(m,prompt[p],p,&kv2,logits2[0],hn2[0],q2[0],k2[0],vv2[0],att2[0],tmp2[0],g2[0],u2[0],eout2[0],Abuf2arr[0],Abuf2_4);
            argmax2[p]=argmax(logits2[0],m->vocab);
        }
        double t2=now()-t2_0;
        static float flogits2[160000];
        if(rem>0) memcpy(flogits2,logits2[0],(size_t)m->vocab*4); else memcpy(flogits2,logits2[3],(size_t)m->vocab*4);

        long match=0; for(int p=0;p<np;p++) if(argmax1[p]==argmax2[p]) match++;
        double fmax=0,fsum=0; for(int e=0;e<m->vocab;e++){ double ad=fabs((double)flogits1[e]-(double)flogits2[e]); if(ad>fmax)fmax=ad; fsum+=ad; }

        printf("\n=== batched prefill test: N=%d (%d M4 chunks + %d remainder tokens) ===\n", np, nchunks, rem);
        printf("token agreement (chunked-M4 vs sequential, argmax over all %d positions): %ld/%d (%.2f%%)\n", np, match, np, 100.0*match/np);
        printf("final-position logits: max_abs_diff=%e mean_abs_diff=%e (%d comparisons) -- expected small, nonzero (M4's shared-scale quantization, not a bug)\n",
            fmax, fsum/m->vocab, m->vocab);
        printf("sequential prefill: %.3fs (%.1f tok/s) | chunked-M4 prefill: %.3fs (%.1f tok/s) | speedup: %.3fx\n",
            t1, np/t1, t2, np/t2, t1/t2);
        if(nchunks>0){
            long h1=g_moe_ecount_hist[1]-hist0[1],h2c=g_moe_ecount_hist[2]-hist0[2],h3=g_moe_ecount_hist[3]-hist0[3],h4=g_moe_ecount_hist[4]-hist0[4];
            long tot=h1+h2c+h3+h4;
            printf("MoE expert-FFN 4-way overlap WITHIN this prefill chunk sequence: %ld/%ld expert-slots (%.2f%%) -- count=1:%ld(%.2f%%) count=2:%ld(%.2f%%) count=3:%ld(%.2f%%) count=4:%ld(%.2f%%)\n",
                g_moe4_hits-moe4_hits0, g_moe4_total_expert_slots-moe4_slots0,
                (g_moe4_total_expert_slots-moe4_slots0)>0?100.0*(g_moe4_hits-moe4_hits0)/(g_moe4_total_expert_slots-moe4_slots0):0.0,
                h1,tot>0?100.0*h1/tot:0.0, h2c,tot>0?100.0*h2c/tot:0.0, h3,tot>0?100.0*h3/tot:0.0, h4,tot>0?100.0*h4/tot:0.0);
        }
        fflush(stdout); /* under ASan, LeakSanitizer's atexit-time report can call a raw _exit()
            that skips normal stdio flushing -- without this, a real crash or a leak-detector exit
            partway through the length sweep would silently discard every printf above it, exactly
            the failure mode that hid this harness's own real output on its first sanitizer run. */

        for(int s=0;s<4;s++){ free(h2[s]);free(hn2[s]);free(q2[s]);free(k2[s]);free(vv2[s]);free(att2[s]);free(tmp2[s]);free(g2[s]);free(u2[s]);free(eout2[s]);free(logits2[s]);free(Abuf2arr[s]); }
        free(Abuf4buf); free(Abuf2_4); free(kv2.Kc); free(kv2.Vc);
        free(hn1);free(q1);free(k1);free(vv1);free(att1);free(tmp1);free(g1);free(u1);free(eout1);free(logits1);free(Abuf1);free(Abuf1b);
        free(kv1.Kc); free(kv1.Vc);
    }
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
     * attention (attention_optimization_plan.md Phase 3, codex_recs_1.md §22.29). Phase 3 promoted
     * min(nt,nkv)=4 as the default (2026-07-26). Phase 6 (codex_recs_1.md §22.33, 2026-07-28) KEPT
     * eight-worker attention -- attn_nt=8 (using the paired harts 9/11/13/15 alongside 8/10/12/14,
     * decoupled from `nt`, the linear/IME worker count) reproducibly beat attn_nt=4: attention
     * bucket -36.2%/-46.0%/-44.6%/-39.3% at short/128/512/1024, wall time -0.4%/-4.5%/-10.8%/-14.9%,
     * tok/s +0.3%/+4.7%/+12.2%/+17.5% -- growing with context, tokens identical throughout, no
     * short-context regression. DEFAULT is now 8. `QWEN_ATTN_NT=1` remains the serial revert
     * (byte-identical to Phase 2); `QWEN_ATTN_NT=4` reverts to the Phase 3-5 four-worker
     * configuration (byte-identical, since gpk==1 at attn_nt<=nkv reduces to the pre-Phase-6 code
     * path exactly). Clamped at MAXNT's practical limit of 8 (g_hart_order only has 8 entries). */
    g_attn_nt = 8;
    { const char*av=getenv("QWEN_ATTN_NT"); if(av){ g_attn_nt=atoi(av); if(g_attn_nt>8) g_attn_nt=8; if(g_attn_nt<1) g_attn_nt=1; } }
    /* g_lin_nt / pool sizing (Phase 6, codex_recs_1.md §22.33): linear/IME stays capped at min(nt,4)
     * regardless of how large attention's own request grows the total pool. The pool itself must be
     * created with enough threads for whichever of the two is larger, since every dispatch (linear
     * or attention) wakes ALL g_pool_nt-1 secondary threads unconditionally -- see lin_mm_pool_init
     * and lin_mm_hp_worker_run's own comments for why each no-ops correctly when it isn't this
     * round's active kind. */
    g_lin_nt = nt<4 ? nt : 4;
    int pool_total = g_attn_nt>nt ? g_attn_nt : nt;
    /* QWEN_QK_FUSE (env var, same convention): Phase 4.1 multi-Q QK (attention_optimization_plan.md,
     * codex_recs_1.md §22.30). Default is now 1 (fused, KEPT after A/B); QWEN_QK_FUSE=0 is the
     * explicit unfused revert to the Phase 3 code path. Softmax/AV/layout/scheduling are unaffected
     * either way. */
    { const char*qf=getenv("QWEN_QK_FUSE"); if(qf) g_qk_fuse=atoi(qf); }
    /* QWEN_QK_VALIDATE=1 (env var, same convention): integration validation for Phase 4.1, per
     * explicit review -- runs BOTH the unfused and fused paths on every real attention dispatch
     * (via attn_qk_validate_and_dispatch), diffing real captured QK scores. Costs ~2-3x the
     * attention bucket while active; meant for a short bounded run before benchmarking, not left on
     * during the real A/B. Overrides whatever QWEN_QK_FUSE requested for att's final content --
     * the actually-configured mode still wins, validation is a side comparison, not a behavior
     * change. */
    { const char*qv=getenv("QWEN_QK_VALIDATE"); if(qv) g_qk_validate=atoi(qv); }
    /* QWEN_AV_FUSE (env var, same convention): Phase 4.2 multi-Q AV (attention_optimization_plan.md,
     * codex_recs_1.md §22.31), KEPT after A/B -- default is now 1 (fused). QWEN_AV_FUSE=0 is the
     * explicit unfused revert to the Phase 4.1 code path. Only takes effect when QWEN_QK_FUSE is
     * also 1 (see g_av_fuse's own comment for why). */
    { const char*vf=getenv("QWEN_AV_FUSE"); if(vf) g_av_fuse=atoi(vf); }
    /* QWEN_AV_VALIDATE=1 (env var, same convention): integration validation for Phase 4.2, mirrors
     * QWEN_QK_VALIDATE -- runs both AV modes on every real attention dispatch, diffing the real
     * output buffers. Not meant to be left on during the real A/B. */
    { const char*vv=getenv("QWEN_AV_VALIDATE"); if(vv) g_av_validate=atoi(vv); }
    /* QWEN_SOFTMAX_RVV (env var, same convention): Phase 5 (attention_optimization_plan.md,
     * codex_recs_1.md §22.32), KEPT after A/B -- default is now 1 (RVV). QWEN_SOFTMAX_RVV=0 is the
     * explicit scalar revert. */
    { const char*sr=getenv("QWEN_SOFTMAX_RVV"); if(sr) g_softmax_rvv=atoi(sr); }
    /* QWEN_SOFTMAX_VALIDATE=1: runs both scalar and RVV softmax on every real call and diffs them
     * in place (see softmax()'s own comment) -- not meant to be left on during the real A/B. */
    { const char*sv=getenv("QWEN_SOFTMAX_VALIDATE"); if(sv) g_softmax_validate=atoi(sv); }
    /* QWEN_WORKERS_VALIDATE=1: Phase 6 (attention_optimization_plan.md, codex_recs_1.md §22.33) --
     * runs the real attn_dispatch path at attn_nt=1 (serial baseline) and at the actually-
     * configured g_attn_nt on every real call, diffing the two output buffers. Not meant to be
     * left on during the real A/B. */
    { const char*wv=getenv("QWEN_WORKERS_VALIDATE"); if(wv) g_workers_validate=atoi(wv); }
    /* QWEN_SCRATCH_ALIGN (env var, same convention): exact-path closure (attention_optimization_plan.md,
     * codex_recs_1.md §22.34), under A/B -- default 0 (plain malloc). 1 = 64-byte-aligned
     * (posix_memalign) attention scratch buffers. */
    { const char*sa=getenv("QWEN_SCRATCH_ALIGN"); if(sa) g_scratch_align=atoi(sa); }
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
    lin_mm_pool_init(pool_total); /* PR8: persistent workers, spawned once, spin-dispatched per
        lin_mm_hp call. Phase 6 (codex_recs_1.md §22.33): pool_total = max(nt,g_attn_nt), so the
        common case (no QWEN_ATTN_NT override, or an override <=nt) creates exactly `nt` threads,
        byte-identical resource footprint to before -- extra threads for harts 9/11/13/15 are only
        created when an attention worker count above `nt` is explicitly requested. */

    /* QWEN_HARNESS=1 (env var, not an 8th CLI arg) triggers the multi-prompt quality harness --
     * see codex_recs_1.md §22.15. NOTE: an 8th positional CLI arg was tried first and reproducibly
     * read back as a corrupted/wild pointer by the time execution reached here (valid at main()
     * entry, clobbered somewhere during cache_load/model setup) -- a real, pre-existing memory
     * bug this exposed, not a bug in the harness itself; never manifested before because nothing
     * previously read past argv[6]. Root-caused in §22.16 to the unsafe tree-vectorized build;
     * the mandatory -fno-tree-vectorize flag fixes it.  The env trigger remains the stable harness
     * interface and avoids adding another positional production argument. */
    { const char*hv=getenv("QWEN_HARNESS"); if(hv && atoi(hv)){ run_quality_harness(&m,&g); return 0; } }
    /* QWEN_MBATCH_TEST=1: M-batch track milestone 1 test (codex_recs_1.md §22.35) -- real 4-
     * sequence batched decode vs 4 separate M=1 decodes, see run_mbatch_test's own comment. */
    { const char*mv=getenv("QWEN_MBATCH_TEST"); if(mv && atoi(mv)){ run_mbatch_test(&m); return 0; } }
    /* QWEN_PREFILL_TEST=1: batched prefill test (codex_recs_1.md §22.38) -- chunked-M4 prefill
     * (prefill_chunk4) vs sequential per-token prefill, see run_prefill_test's own comment. */
    { const char*pv=getenv("QWEN_PREFILL_TEST"); if(pv && atoi(pv)){ run_prefill_test(&m); return 0; } }
    /* QWEN_PREFILL_CHUNK (env var, same convention): batched prefill (codex_recs_1.md §22.38) --
     * chunked-M4 prefill (prefill_chunk4, 4 consecutive positions batched via the M-batch track's
     * already-validated dense-layer M4 kernel) vs the sequential token-at-a-time baseline.
     * Validated (QWEN_PREFILL_TEST=1, run_prefill_test): 96-100% per-position token agreement (no
     * argmax-feedback compounding during prefill, unlike decode M-batch -- every position's input
     * token is fixed by the prompt, not fed back from a possibly-different prediction, so the only
     * divergence source is M4's own shared-scale quantization noise propagating through the KV
     * cache), ~1.11-1.12x prefill speedup at N=16/19/128/512/1024, reproducible, no length-
     * dependent degradation. Default 0 (sequential, byte-identical to every prior session) -- this
     * has NOT yet been run through the full multi-prompt NLL/perplexity quality harness that gated
     * router/swiglu's own promotion (codex_recs_1.md §22.15/22.20-21), so it stays an explicit
     * opt-in pending that review rather than a promoted default. */
    int g_prefill_chunk=0; { const char*pc=getenv("QWEN_PREFILL_CHUNK"); if(pc) g_prefill_chunk=atoi(pc); }

    static int prompt[1536]; int np;
    if(ctxlen_req>0){ np=ctxlen_req; for(int i=0;i<np;i++) prompt[i]=hp9[i%113]; }
    else { int base[]={785,6722,315,9625,374,12095,13,576,6722,315,6323,374}; np=12; memcpy(prompt,base,sizeof(base)); }
    double tp=now(); int first=0;
    if(g_prefill_chunk){
        float*h4[4],*hn4[4],*q4[4],*k4[4],*vv4[4],*att4[4],*tmp4[4],*g4[4],*u4[4],*eout4[4],*logits4[4]; uint8_t*Abuf4arr[4];
        for(int s=0;s<4;s++){
            h4[s]=malloc(d*4); hn4[s]=malloc(d*4); q4[s]=malloc(qd*4); k4[s]=malloc(kv.kvd*4); vv4[s]=malloc(kv.kvd*4);
            att4[s]=malloc(qd*4); tmp4[s]=malloc((size_t)(moe>ctx?(moe>d?moe:d):(ctx>d?ctx:d))*4);
            g4[s]=malloc(moe*4); u4[s]=malloc(moe*4); eout4[s]=malloc(d*4); logits4[s]=malloc((size_t)m.vocab*4);
            Abuf4arr[s]=malloc((size_t)(maxk/256)*AREC);
        }
        int maxk4b=qd>d?qd:d;
        uint8_t*Abuf4buf=malloc((size_t)(maxk4b/256)*AREC_M4);
        uint8_t*Abuf2_4=malloc((size_t)4*3000);
        int nchunks=np/4, rem=np%4;
        for(int c=0;c<nchunks;c++){
            int toks4[4]; for(int i=0;i<4;i++) toks4[i]=prompt[c*4+i];
            prefill_chunk4(&m,toks4,c*4,&kv,logits4,h4,hn4,q4,k4,vv4,att4,tmp4,g4,u4,eout4,Abuf4arr,Abuf4buf,Abuf2_4);
            if(c==nchunks-1 && rem==0){ memcpy(logits,logits4[3],(size_t)m.vocab*4); first=argmax(logits,m.vocab); }
        }
        for(int r=0;r<rem;r++){
            int p=nchunks*4+r;
            forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2);
            if(p==np-1) first=argmax(logits,m.vocab);
        }
        for(int s=0;s<4;s++){ free(h4[s]);free(hn4[s]);free(q4[s]);free(k4[s]);free(vv4[s]);free(att4[s]);free(tmp4[s]);free(g4[s]);free(u4[s]);free(eout4[s]);free(logits4[s]);free(Abuf4arr[s]); }
        free(Abuf4buf); free(Abuf2_4);
    } else {
        for(int p=0;p<np;p++){ forward(&m,prompt[p],p,&kv,logits,hn,q,k,vv,att,tmp,gg,u,eout,Abuf,Abuf2); if(p==np-1)first=argmax(logits,m.vocab); }
    }
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
        printf("  attention breakdown (avg/%ld tok, ms): QK %.2f | softmax %.2f | AV %.2f (total work, %s) | dispatch %.2f | attention bucket (wall) %.2f, attn_nt=%d, qk_fuse=%d, av_fuse=%d, softmax_rvv=%d\n",
            gT_tok, gT_attn_qk/gT_tok*1e3, gT_attn_sm/gT_tok*1e3, gT_attn_av/gT_tok*1e3,
            g_attn_nt<=1?"= wall time":"summed across workers", gT_attn_dispatch/gT_tok*1e3, gT_attn/gT_tok*1e3, g_attn_nt, g_qk_fuse, g_av_fuse, g_softmax_rvv);
    }
    if(g_router_validate){
        printf("  router int4-HP-vs-fp32: %ld comparisons, %ld expert-set mismatches (avg %.2f/%d experts differ per mismatch), max abs logit delta %e, max rel %e\n",
            g_rtr_cmp, g_rtr_hp_mismatch, g_rtr_hp_mismatch?(double)g_rtr_hp_diffcount/g_rtr_hp_mismatch:0.0, m.n_act, g_rtr_hp_maxabs, g_rtr_hp_maxrel);
        printf("  router int8-M1-vs-fp32: %ld comparisons, %ld expert-set mismatches (avg %.2f/%d experts differ per mismatch), max abs logit delta %e, max rel %e\n",
            g_rtr_cmp, g_rtr_i8_mismatch, g_rtr_i8_mismatch?(double)g_rtr_i8_diffcount/g_rtr_i8_mismatch:0.0, m.n_act, g_rtr_i8_maxabs, g_rtr_i8_maxrel);
    }
    if(g_qk_validate){
        printf("  qk_fuse integration validate: %ld comparisons, %ld mismatches(>1e-4 abs), max_abs=%e, max_rel=%e -- %s\n",
            g_qkval_cmp, g_qkval_mismatch, g_qkval_max_abs, g_qkval_max_rel, g_qkval_mismatch==0?"PASS":"FAIL");
    }
    if(g_av_validate){
        printf("  av_fuse integration validate: %ld comparisons, %ld mismatches(>1e-4 abs), max_abs=%e, max_rel=%e -- %s\n",
            g_avval_cmp, g_avval_mismatch, g_avval_max_abs, g_avval_max_rel, g_avval_mismatch==0?"PASS":"FAIL");
    }
    if(g_softmax_validate){
        printf("  softmax_rvv validate: %ld comparisons, %ld mismatches(>1e-4 abs), max_abs=%e, max_rel=%e -- %s\n",
            g_smval_cmp, g_smval_mismatch, g_smval_max_abs, g_smval_max_rel, g_smval_mismatch==0?"PASS":"FAIL");
    }
    if(g_workers_validate){
        printf("  attn_workers validate (vs serial): %ld comparisons, %ld mismatches(>1e-4 abs), max_abs=%e, max_rel=%e -- %s\n",
            g_wval_cmp, g_wval_mismatch, g_wval_max_abs, g_wval_max_rel, g_wval_mismatch==0?"PASS":"FAIL");
    }
    if(!cached){ fprintf(stderr,"saving requant cache -> %s ...\n",cpath); double ts=now(); cache_save(&m,cpath); fprintf(stderr,"cache saved in %.1fs\n",now()-ts); }
    return 0;
}
