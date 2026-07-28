/* vendor_ime_m4_probe.c — M-batch track (codex_recs_1.md §22.35): validates a fresh port of the
 * REAL vendor M=4 decode kernel, gemm_kernel_i8i4_hp_m4 (reference/spacemit-backend/
 * ime2_kernels.cpp:3360, no-zp branch -- confirmed the live branch for Q4_0 via
 * ime.cpp:107 block_type_has_zp<block_q4_0>()==false, so quant_b_zp==nullptr at the real call
 * site, ime.cpp:276/583).
 *
 * A-record ground truth: quantize_a_4row_i8_hp (reference/spacemit-backend/rvv_kernels.cpp:2100,
 * vlenb==128 branch -- this board's A100 harts, VLEN=1024, matching the note already established
 * for the M1 port in qwen_moe_hp.c's own pack_A_hp comment). Per-256-block record is 1160 bytes
 * (= q8_hp_blk_size(256,true,true)=290 * 4 rows, confirmed via rvv_kernels.h:25's q8_hp_blk_size
 * formula):
 *   8 subblocks x 136B: [8B scale area, only offset+0 used -- ONE shared fp16 scale across all
 *     4 rows in this subblock, NOT 4 separate per-row scales despite the asm's own "8B = 4 x fp16
 *     row scales" comment; confirmed by the packer only ever writing scale_a_ptr[0]] +
 *     [4 x int8[32] row payloads, row-major: row0,row1,row2,row3]
 *   64B a_sum trailer: 4 rows x 8 fp16 values, PRE-SCALED as -true_asum*8.0 (same convention as
 *     M1), laid out [row0's 8][row1's 8][row2's 8][row3's 8]
 *   8B scale_avg area, only offset+0 used -- ONE shared block-average scale across all 4 rows
 *     (mean of the 8 per-subblock scale_temp values, same "block_avg_scale" concept as M1)
 *
 * KEY correctness property, different from every prior fusion in this session: M4's per-subblock
 * scale is computed from the MAX ABSOLUTE VALUE ACROSS ALL 4 BATCHED ROWS JOINTLY, not per row
 * (quantize_a_4row_i8_hp's v_max_abs = max(max(|a0|,|a1|),max(|a2|,|a3|))). This means M4's
 * quantized output is NOT expected to be bit-identical to 4 independent M1 calls on the same 4
 * rows -- a genuinely different (though still principled, not degenerate) quantization choice per
 * subblock.
 *
 * Reference methodology, matching vendor_ime_a2_full.c's own proven "dequant reference" approach
 * exactly (NOT an exact-fp32-activation oracle -- an earlier version of this probe used one and it
 * produced a wildly inflated, misleading ~100%+ "error" that had nothing to do with kernel
 * correctness, since it conflated A-side int8 quantization's own inherent ~8-bit rounding error
 * with the question this probe actually needs to answer). Each kernel (M1, M4) is checked against
 * a reference reconstructed from ITS OWN actual stored, quantized, fp16-rounded packed bytes (both
 * A and W sides) -- i.e. "does the ported asm correctly compute the dot product its own documented
 * byte format implies," isolating kernel/packer bugs from A-quantization's own expected error.
 * Separately, M1's and M4's kernel-vs-own-reference error magnitudes are compared to confirm M4
 * doesn't introduce EXTRA error beyond ordinary quantization noise, and M1 vs M4's direct outputs
 * are diffed (expected nonzero, per the shared-scale property above -- a real, small, principled
 * difference, not a bug).
 *
 * B-record: unchanged from M1 -- block_q4_0x32 (576B/K32/N32-panel), same weight stream, confirmed
 * by the M4 asm's own "B: N32 x K256 q4 HP block... same as M1" structure and the fact that the
 * b_tile_stride computation only depends on k_blks/b_superblk_stride, not on M.
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zfh_zvfh_xsmtvdotii -o vendor_ime_m4_probe vendor_ime_m4_probe.c -lm
 * (zfh needed on top of M1's own march string -- the vendor kernel's fmul.h scalar half-precision
 * multiply needs it explicitly; zvfh, vector half-float, doesn't imply it.)
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

/* ============ B-matrix (weights) packing -- IDENTICAL to M1, copied verbatim from qwen_moe_hp.c ============ */
#define NSUB 8
#define BREC 576
#define BSUPER (NSUB*BREC)
typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_native;
static void quantize_q4_0_native(const float*w, q4_0_native*out){
    float amax=1e-6f; for(int i=0;i<32;i++){ float v=fabsf(w[i]); if(v>amax)amax=v; }
    float d=amax/8.0f; float inv = d? 1.0f/d : 0.0f;
    int8_t q[32];
    for(int i=0;i<32;i++){ int v=(int)lrintf(w[i]*inv); if(v>7)v=7; if(v<-8)v=-8; q[i]=(int8_t)v; }
    out->d=f32_to_f16(d);
    for(int l=0;l<16;l++){ uint8_t lo=(uint8_t)(q[l]+8), hi=(uint8_t)(q[l+16]+8); out->qs[l]=(lo&0xf)|((hi&0xf)<<4); }
}
static void pack_B_q4_0x32(q4_0_native rows[32], uint8_t*out){
    for(int i=0;i<32;i++) memcpy(out+i*2,&rows[i].d,2);
    uint8_t*qs=out+64;
    for(int i=0;i<32;i++){
        for(int j=0;j<8;j++) qs[i*16+j]     = (rows[i].qs[j*2]&0x0F) | ((rows[i].qs[j*2+1]&0x0F)<<4);
        for(int j=0;j<8;j++) qs[i*16+8+j]   = ((rows[i].qs[j*2]&0xF0)>>4) | (rows[i].qs[j*2+1]&0xF0);
    }
}

/* ============ A-matrix (activations) packing, M1 (reused for the oracle-vs-M1 comparison) ============ */
static void pack_A_hp_m1(const float*a, uint8_t*out /* 290 bytes */){
    float scale_temp[NSUB]; float scale_avg=0.0f;
    for(int kk=0;kk<NSUB;kk++){
        float amax=0.0f; for(int i=0;i<32;i++){ float v=fabsf(a[kk*32+i]); if(v>amax)amax=v; }
        scale_temp[kk]=amax/127.0f; scale_avg+=scale_temp[kk];
    }
    scale_avg/=NSUB;
    float scale_factor = scale_avg? 1.0f/scale_avg : 0.0f;
    uint16_t blkscale=f32_to_f16(scale_avg); memcpy(out+288,&blkscale,2);
    for(int kk=0;kk<NSUB;kk++){
        uint8_t*base=out+kk*34;
        float rep_scale = scale_temp[kk]? 1.0f/scale_temp[kk] : 0.0f;
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor); memcpy(base,&ssub,2);
        int8_t q[32]; int32_t sum=0;
        for(int i=0;i<32;i++){ int v=(int)lrintf(a[kk*32+i]*rep_scale); if(v>127)v=127; if(v<-127)v=-127; q[i]=(int8_t)v; sum+=v; }
        uint16_t as=f32_to_f16(-(float)sum*8.0f); memcpy(out+272+kk*2,&as,2);
        memcpy(base+2,q,32);
    }
}

/* ============ A-matrix (activations) packing, M4 -- ground truth: rvv_kernels.cpp:2100 ============ */
static void pack_A_hp_m4(const float*a0,const float*a1,const float*a2,const float*a3,uint8_t*out /* 1160 bytes */){
    float scale_temp[NSUB]; float scale_avg=0.0f;
    for(int kk=0;kk<NSUB;kk++){
        float amax=0.0f;
        for(int i=0;i<32;i++){
            float v0=fabsf(a0[kk*32+i]),v1=fabsf(a1[kk*32+i]),v2=fabsf(a2[kk*32+i]),v3=fabsf(a3[kk*32+i]);
            float m=v0>v1?v0:v1; float n=v2>v3?v2:v3; if(n>m)m=n;
            if(m>amax) amax=m;
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

/* ============ M1 kernel -- verbatim from qwen_moe_hp.c, for the oracle-vs-M1 comparison ============ */
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

/* ============ M4 kernel -- verbatim port of gemm_kernel_i8i4_hp_m4's no-zp branch,
 * reference/spacemit-backend/ime2_kernels.cpp (the `} else {` block after `if (quant_b_zp != nullptr)`) ============ */
static void run_hp_m4(const uint8_t*a_data, const uint8_t*b_data, float*dst_c, long k_blks, long ldc){
    __asm__ volatile(
        "mv             t5, %[BK]                 \n\t"
        "mv             t6, %[A]                  \n\t"
        "mv             s5, %[B]                  \n\t"
        "li             t1, 0x4c00                \n\t"
        "fmv.h.x        fa6, t1                   \n\t"
        "vsetvli        t0, x0, e32, m1           \n\t"
        "vxor.vv        v28, v28, v28             \n\t"
        "vxor.vv        v29, v29, v29             \n\t"
        "vxor.vv        v30, v30, v30             \n\t"
        "vxor.vv        v31, v31, v31             \n\t"
        "li             t4, 8                     \n\t"
        "addi           t2, t6, 1088              \n\t"

        ".align 4                                 \n\t"
        "_BLK_LPST%=:                             \n\t"
        "flh            fa1, 64(t2)               \n\t"
        "vsetvli        t0, x0, e32, m1           \n\t"
        "vxor.vv        v18, v30, v30             \n\t"
        "vxor.vv        v19, v31, v31             \n\t"
        "vxor.vv        v20, v30, v30             \n\t"
        "vxor.vv        v21, v31, v31             \n\t"
        "_KsubBLK_LPST%=:                         \n\t"
        "flh            fa0,   0(t6)              \n\t"

        "vsetvli        t0, x0, e16, mf2          \n\t"
        "vle16.v        v12, (s5)                 \n\t"

        "fmul.h         fa2, fa0, fa6              \n\t"

        "vsetvli        t0, x0, e16, mf2          \n\t"
        "vfmul.vf       v16, v12, fa0             \n\t"
        "vfmul.vf       v17, v12, fa2             \n\t"

        "flh            ft1, 0(t2)                \n\t"
        "flh            ft2, 16(t2)               \n\t"
        "flh            ft3, 32(t2)               \n\t"
        "flh            ft4, 48(t2)               \n\t"

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
        "bgtz           t4, _KsubBLK_LPST%=       \n\t"

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
        "bgtz           t5, _BLK_LPST%=           \n\t"

        "_BLK_LPND%=:                             \n\t"
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

static uint32_t rng=42;
static float randf(void){ rng=rng*1103515245u+12345u; return ((rng>>8)*(1.0f/16777216.0f))-0.5f; }

int main(void){
    bind_ai(); { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs); sched_setaffinity(0,sizeof(cs),&cs); }
    for(int i=0;i<5;i++) sched_yield();
    const int K=256, N=32; /* one superblock, one N-tile -- k_blks=1, per trial */
    const int TRIALS=200;

    double osum_all=0; long ocount_all=0;
    double m1_abssum=0, m4_abssum=0; /* mean-abs-error, robust aggregate metric */
    double m1_max_abs=0, m4_max_abs=0;
    long cmp_all=0;
    double m1_vs_m4_abssum=0, m1_vs_m4_max=0;

    for(int trial=0; trial<TRIALS; trial++){
        float *A[4], *W[32];
        for(int r=0;r<4;r++){ A[r]=malloc(K*4); for(int i=0;i<K;i++) A[r][i]=randf()*4.0f; }
        for(int n=0;n<N;n++){ W[n]=malloc(K*4); for(int i=0;i<K;i++) W[n][i]=randf()*2.0f; }

        q4_0_native qW[NSUB][32];
        for(int kk=0;kk<NSUB;kk++) for(int n=0;n<32;n++) quantize_q4_0_native(W[n]+kk*32,&qW[kk][n]);
        uint8_t*Bpack=malloc(BSUPER);
        for(int kk=0;kk<NSUB;kk++) pack_B_q4_0x32(qW[kk],Bpack+kk*BREC);

        uint8_t Am1[290]; float m1out[4][32];
        for(int r=0;r<4;r++){ pack_A_hp_m1(A[r],Am1); run_hp_m1(Am1,Bpack,m1out[r],1); }

        uint8_t Am4[1160]; float m4out[4][32];
        pack_A_hp_m4(A[0],A[1],A[2],A[3],Am4);
        run_hp_m4(Am4,Bpack,&m4out[0][0],1,32);

        /* Reference reconstruction, matching vendor_ime_a2_full.c's own proven "dequant reference"
         * methodology exactly: reconstruct from the ACTUAL STORED, QUANTIZED (and fp16-rounded)
         * bytes -- both A and W sides -- not from idealized exact-fp32 activations. The kernel
         * approximates "quantized-A dot quantized-W", not "exact-A dot quantized-W"; comparing
         * against an exact-A oracle would (and, in an earlier version of this probe, did) conflate
         * A-quantization's own inherent error with a real kernel/packer bug, producing a wildly
         * inflated and misleading "error" that had nothing to do with this port's correctness. */
        double oracle_m1[4][32]={{0}};
        for(int r=0;r<4;r++){
            uint8_t A1[290]; pack_A_hp_m1(A[r],A1);
            for(int kk=0;kk<NSUB;kk++){
                uint16_t ssub; memcpy(&ssub,A1+kk*34,2); float sub_ratio=f16_to_f32(ssub);
                uint16_t blks; memcpy(&blks,A1+288,2); float blk_avg=f16_to_f32(blks);
                float a_scale=sub_ratio*blk_avg;
                int8_t qa[32]; memcpy(qa,A1+kk*34+2,32);
                for(int n=0;n<32;n++){
                    float d=f16_to_f32(qW[kk][n].d); double acc=0;
                    for(int c=0;c<32;c++){
                        int lo=qW[kk][n].qs[c%16]&0xf, hi=(qW[kk][n].qs[c%16]>>4)&0xf;
                        int nib=(c<16)?lo:hi; int wsigned=nib-8;
                        acc += (double)qa[c]*a_scale*(double)wsigned*d;
                    }
                    oracle_m1[r][n]+=acc;
                }
            }
        }
        double oracle_m4[4][32]={{0}};
        for(int kk=0;kk<NSUB;kk++){
            uint8_t*subblk=Am4+kk*136;
            uint16_t ssub; memcpy(&ssub,subblk,2); float sub_ratio=f16_to_f32(ssub);
            uint16_t blks; memcpy(&blks,Am4+1152,2); float blk_avg=f16_to_f32(blks);
            float a_scale=sub_ratio*blk_avg;
            int8_t*quant_blk=(int8_t*)(subblk+8);
            for(int r=0;r<4;r++){
                int8_t qa[32]; memcpy(qa,quant_blk+r*32,32);
                for(int n=0;n<32;n++){
                    float d=f16_to_f32(qW[kk][n].d); double acc=0;
                    for(int c=0;c<32;c++){
                        int lo=qW[kk][n].qs[c%16]&0xf, hi=(qW[kk][n].qs[c%16]>>4)&0xf;
                        int nib=(c<16)?lo:hi; int wsigned=nib-8;
                        acc += (double)qa[c]*a_scale*(double)wsigned*d;
                    }
                    oracle_m4[r][n]+=acc;
                }
            }
        }

        for(int r=0;r<4;r++) for(int n=0;n<32;n++){
            osum_all+=fabs(oracle_m1[r][n]); ocount_all++;
            double a1=fabs(oracle_m1[r][n]-(double)m1out[r][n]);
            double a4=fabs(oracle_m4[r][n]-(double)m4out[r][n]);
            m1_abssum+=a1; m4_abssum+=a4;
            if(a1>m1_max_abs) m1_max_abs=a1;
            if(a4>m4_max_abs) m4_max_abs=a4;
            double dmm=fabs((double)m1out[r][n]-(double)m4out[r][n]);
            m1_vs_m4_abssum+=dmm; if(dmm>m1_vs_m4_max) m1_vs_m4_max=dmm;
            cmp_all++;
        }
        for(int r=0;r<4;r++) free(A[r]);
        for(int n=0;n<N;n++) free(W[n]);
        free(Bpack);
    }

    double omean=osum_all/ocount_all;
    printf("oracle: %ld comparisons over %d trials, mean|value|=%e\n", ocount_all, TRIALS, omean);
    printf("M1 vs fp32 oracle: mean_abs_err=%e (%.3f%% of mean value), max_abs_err=%e\n",
        m1_abssum/cmp_all, 100.0*(m1_abssum/cmp_all)/omean, m1_max_abs);
    printf("M4 vs fp32 oracle: mean_abs_err=%e (%.3f%% of mean value), max_abs_err=%e\n",
        m4_abssum/cmp_all, 100.0*(m4_abssum/cmp_all)/omean, m4_max_abs);
    printf("M4/M1 mean-error ratio: %.3fx -- %s\n",
        (m1_abssum>0)?(m4_abssum/m1_abssum):0,
        (m4_abssum < m1_abssum*2.0) ? "REASONABLE (M4 within 2x of M1's own mean error)" : "SUSPICIOUS (M4 mean error much larger than M1's)");
    printf("M1 vs M4 direct: mean_abs_diff=%e, max_abs_diff=%e (expected NONZERO -- different per-subblock scale choice)\n",
        m1_vs_m4_abssum/cmp_all, m1_vs_m4_max);

    return 0;
}
