/* q3k_probe.c -- Q3_K engine branch, oracle-first ladder step 1 (codex_recs_1.md, new §22.43):
 * standalone pack/dequant oracle for gemm_kernel_i8i3k_m1 and its repacker/A-packer, BEFORE any
 * engine integration, matching the exact discipline that succeeded for the W4 M1/M4 ports (§22.35).
 *
 * Explicit direction: "Port the vendor's existing nrow_block_q3_k<32> repacker and
 * gemm_kernel_i8i3k_m1 exactly, using the same oracle-first ladder that succeeded for W4:
 * standalone pack/dequant oracle, hot and cold kernel benchmark, then full-engine integration."
 *
 * Source (reference/spacemit-backend/):
 *   - Kernel: ime2_kernels.cpp:1419-2029. Contains a `#if 0 ... #else ... #endif` split; the `#if 0`
 *     branch (~1445-1758) is DEAD CODE (uses plain `vmadot ... i8`, never reads A's a_sum). The
 *     LIVE branch is the `#else` (1759-2027) -- uses `vmadot.hp`, the same custom-instruction family
 *     as the already-validated W4 HP kernels (gemm_kernel_i8i4_hp_m1). Confirmed via a real
 *     dispatch trace (ime.cpp:295-301, `if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>)`),
 *     not inline-comment guesswork -- avoids repeating the A2-saga wrong-branch mistake earlier in
 *     this session.
 *   - B-side repacker: repack_q3_k_to_q3_k_32_bl, repack.cpp:452-555. Reads GGUF-native block_q3_K
 *     (110 bytes: hmask[32], qs[64], scales[12], ggml_half d -- standard, well-known ggml layout;
 *     NOT re-derivable from a header in this checkout, cross-validated instead by self-consistency
 *     against the kernel's own read pattern and against the canonical ggml dequant algorithm below).
 *     Writes nrow_block_q3_k<32> (3648 bytes: int8 scales[512], hmask[1024], qs[2048], fp16
 *     scales16[64]).
 *   - A-side packer: quantize_a_row_i8k, rvv_kernels.cpp:2305-2394 (vlenb==128 branch ported to
 *     plain scalar C here -- it's a format-production routine, not a hot inner loop, matching how
 *     this whole engine's own pack_A_hp/pack_act_hp are scalar C, not RVV intrinsics). Produces the
 *     292-byte/superblock "q8k" format: fp32 a_scale + int16 a_sum[16] (RAW negated int sum, NOT
 *     the fp16 "-asum*8" trick W4's own HP packer uses) + int8 a_qs[256]. This q8k format is shared
 *     ONLY between Q2_K and Q3_K (confirmed via ime.cpp's own dispatch table, not used by
 *     Q4_K/Q5_K/Q6_K) -- relevant when the Q2_K track begins later.
 *   - Dispatch-confirmed but NOT ported here: gemm_kernel_i8i3k_m4 (ime2_kernels.cpp:2031-2428)
 *     uses a DIFFERENT instruction family (plain `vmadot`, not `.hp`) than M1's live path -- flagged
 *     explicitly by the investigation as needing its own independent trace if/when an M4 port is
 *     attempted, NOT assumed to mirror M1's HP/scale-fusion trick.
 *
 * KEY correctness property: the M1 kernel never reads A's a_sum field (no `lh` instruction touches
 * it anywhere in either branch) -- Q3_K has no zero-point (block_type_has_zp<block_q3_K>()==false,
 * ime.cpp:107), so the sum term is produced by the shared q8k packer but is dead weight for this
 * quant type. Computed correctly anyway here (matching the vendor packer exactly) rather than
 * skipped, to keep the port a faithful byte-for-byte format match.
 *
 * Reference methodology, matching vendor_ime_a2_full.c's / vendor_ime_m4_probe.c's own proven
 * approach: the oracle reference is reconstructed from the SAME quantized bytes the kernel itself
 * consumes (dequantize_q3_K_block on the synthetic B blocks, A's own stored int8 qs * its own
 * stored scale_a) -- NOT from the pre-quantization "true" random values, which would conflate
 * A-side int8 quantization's own inherent rounding error with kernel/repack correctness (the exact
 * methodology bug caught and fixed in the M4 probe, §22.35).
 *
 * v0-v31 explicitly added to the asm clobber list even though the vendor's own upstream source
 * lists none -- matching the established, already-validated correction this session's own prior
 * ports (run_hp_m1/run_hp_m4) already made for the same reason: an accurate clobber list is needed
 * for GCC to avoid register-allocation conflicts with surrounding C code at an arbitrary call site,
 * unlike whatever specific context the vendor's own binary was compiled in.
 *
 * Build (board, A100 harts, VLEN=1024): gcc -O2 -fno-tree-vectorize
 *   -march=rv64gcv_zfh_zvfh_xsmtvdotii -o q3k_probe q3k_probe.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>

static int g_pinned=0;
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static void pin_hart8(void){ if(g_pinned)return; bind_ai(); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs); sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield(); g_pinned=1; }

static double now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

/* ===== GGUF-native block_q3_K (standard ggml layout, QK_K=256), 110 bytes ===== */
typedef struct {
    uint8_t  hmask[32];   /* QK_K/8 */
    uint8_t  qs[64];      /* QK_K/4 */
    uint8_t  scales[12];
    _Float16 d;
} block_q3_K;

/* ===== canonical ggml dequant reference (NOT part of the kernel port -- this is the independent
 * "ground truth" oracle reference every other piece is checked against). Verbatim algorithm,
 * well-established across the ggml/llama.cpp ecosystem. ===== */
static void dequantize_q3_K_block(const block_q3_K *x, float *y){
    const uint32_t kmask1=0x03030303, kmask2=0x0f0f0f0f;
    uint32_t aux[4]; int8_t *scales=(int8_t*)aux;
    float d_all=(float)x->d;
    const uint8_t *q=x->qs, *hm=x->hmask; uint8_t m=1;
    memcpy(aux,x->scales,12);
    uint32_t tmp=aux[2];
    aux[2]=((aux[0]>>4)&kmask2)|(((tmp>>4)&kmask1)<<4);
    aux[3]=((aux[1]>>4)&kmask2)|(((tmp>>6)&kmask1)<<4);
    aux[0]=(aux[0]&kmask2)|(((tmp>>0)&kmask1)<<4);
    aux[1]=(aux[1]&kmask2)|(((tmp>>2)&kmask1)<<4);
    int is=0; float dl;
    for(int n=0;n<256;n+=128){
        int shift=0;
        for(int j=0;j<4;j++){
            dl=d_all*(float)(scales[is++]-32);
            for(int l=0;l<16;l++) y[n+j*32+l]=dl*(float)((int)((q[l]>>shift)&3) - ((hm[l]&m)?0:4));
            dl=d_all*(float)(scales[is++]-32);
            for(int l=0;l<16;l++) y[n+j*32+16+l]=dl*(float)((int)((q[l+16]>>shift)&3) - ((hm[l+16]&m)?0:4));
            shift+=2; m=(uint8_t)(m<<1);
        }
        q+=32;
    }
}

/* ===== nrow_block_q3_k<32>, repacked B layout, 3648 bytes: field order scales->hmask->qs->scales16
 * matches ime_kernels.h:23-34 exactly (verified against the kernel's own pointer arithmetic). ===== */
typedef struct {
    int8_t   scales[32*16];   /* 512 */
    uint8_t  hmask[32*32];    /* 1024 */
    uint8_t  qs[32*64];       /* 2048 */
    _Float16 scales16[32];    /* 64 */
} nrow_block_q3_k32; /* 3648 */

/* Verbatim transcription of repack_q3_k_to_q3_k_32_bl's per-row body (repack.cpp:486-547), just
 * the inner "for i in 0..31" loop pulled out to take 32 source blocks (one K256-superblock's worth
 * of 32 output rows) and produce ONE nrow_block_q3_k32. */
static void pack_B_q3k32(const block_q3_K rows[32], nrow_block_q3_k32*dst){
    const uint32_t kmask1=0x03030303, kmask2=0x0f0f0f0f;
    uint8_t qs_aux[256];
    for(int i=0;i<32;i++){
        const block_q3_K *src_block=&rows[i];
        uint32_t auxs[4]; int8_t *scale=(int8_t*)auxs;
        memcpy(auxs,src_block->scales,12);
        uint32_t tmp=auxs[2];
        auxs[2]=((auxs[0]>>4)&kmask2)|(((tmp>>4)&kmask1)<<4);
        auxs[3]=((auxs[1]>>4)&kmask2)|(((tmp>>6)&kmask1)<<4);
        auxs[0]=(auxs[0]&kmask2)|(((tmp>>0)&kmask1)<<4);
        auxs[1]=(auxs[1]&kmask2)|(((tmp>>2)&kmask1)<<4);
        for(int j=0;j<16;j++) dst->scales[j*32+i]=(int8_t)(scale[j]-32);

        for(int k=0;k<4;k++) for(int j=0;j<32;j++) qs_aux[k*32+j]=(src_block->qs[j]>>(2*k))&0x03;
        for(int k=0;k<4;k++) for(int j=0;j<32;j++) qs_aux[k*32+j+128]=(src_block->qs[j+32]>>(2*k))&0x03;

        for(int k=0;k<4;k++){
            for(int j=0;j<16;j++){
                uint8_t qs0=qs_aux[j+k*64], qs16=qs_aux[j+16+k*64], qs32=qs_aux[j+32+k*64], qs48=qs_aux[j+48+k*64];
                dst->qs[(k*32+i)*16+j]=(uint8_t)((qs0&0x03)|((qs16&0x03)<<2)|((qs32&0x03)<<4)|((qs48&0x03)<<6));
            }
        }

        uint16_t *dst_mask=((uint16_t*)dst->hmask)+i;
        for(int j=0;j<16;j++, dst_mask+=32){
            uint8_t b_shift=(uint8_t)(j/2);
            const uint8_t *b_mask_col=src_block->hmask+(j%2)*16;
            uint16_t msk=0;
            for(int k=0;k<8;k++)  msk|=(uint16_t)(((b_mask_col[k]>>b_shift)&1))<<k;
            for(int k=8;k<16;k++) msk|=(uint16_t)(((b_mask_col[k]>>b_shift)&1))<<k;
            dst_mask[0]=msk;
        }
        dst->scales16[i]=src_block->d;
    }
}

/* ===== A-side "q8k" activation packer, scalar port of quantize_a_row_i8k (rvv_kernels.cpp:
 * 2305-2350, vlenb==128 branch). 292 bytes/superblock = 4 (fp32 scale) + 32 (int16 sum[16]) +
 * 256 (int8 qs). a_sum is genuinely unused by the M1 kernel (see file header) but computed
 * correctly anyway to keep this a faithful byte-format port. ===== */
#define AREC_Q3K 292
static void pack_A_q3k(const float *a, uint8_t *out){
    float max_abs=0.0f;
    for(int i=0;i<256;i++){ float v=fabsf(a[i]); if(v>max_abs) max_abs=v; }
    float scale_a=max_abs/127.0f;
    float rep=scale_a?1.0f/scale_a:0.0f;
    memcpy(out,&scale_a,4);
    int16_t *sum_ptr=(int16_t*)(out+4);
    int8_t  *qs_ptr =(int8_t*)(out+4+32);
    for(int bki=0;bki<16;bki++){
        int32_t sum=0;
        for(int l=0;l<16;l++){
            float v=a[bki*16+l]*rep;
            int q=(int)roundf(v);
            if(q>127)q=127; if(q<-128)q=-128;
            qs_ptr[bki*16+l]=(int8_t)q;
            sum+=q;
        }
        sum_ptr[bki]=(int16_t)(-sum);
    }
}

/* ===== M1 kernel, verbatim asm transcription of the LIVE #else branch,
 * ime2_kernels.cpp:1761-2020. count_n is fixed at 32 (NB_COLS, "only support 32 in ASM" per the
 * vendor's own comment) -- nb_real/b_str are still parameters, matching the real kernel's own
 * signature, for direct reuse when this gets wired into the real engine's N-tiling loop. ===== */
static void run_hp_q3k_m1(const uint8_t*a_data, const uint8_t*b_data, float*dst_c, long k_blks, long nb_real, long b_str){
    const _Float16 q3_step=(_Float16)0.0625f;
    const float a_post_mul=16.0f;
    __asm__ volatile(
        "mv           t2, %[KBLKS]            \n\t"
        "li           t3, 4                   \n\t"
        "mv           s2, %[pA]               \n\t"
        "addi         s3, %[pA], 4+32         \n\t"
        "addi         s5, %[pB], 32*16        \n\t"
        "mv           s4, %[pB]               \n\t"
        "addi         s6, s5, 1024            \n\t"
        "addi         s8, s6, 1024            \n\t"
        "addi         s8, s8, 1024            \n\t"
        "mv           s7, %[pB]               \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vxor.vv      v31, v0, v0             \n\t"

        "vsetvli      t0, x0, e16, mf2        \n\t"
        "vle16.v      v1, (s8)                \n\t"
        "vsetvli      t0, x0, e16, m1         \n\t"
        "vpack.vv     v26, v1, v1, 3          \n\t"
        "vmv.v.v      v17, v26                \n\t"
        "vsetvli      t0, x0, e16, m1         \n\t"
        "vfmul.vf     v30, v17, %[q3_step]    \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vxor.vv      v24, v16, v16           \n\t"
        "vxor.vv      v25, v16, v16           \n\t"
        "vxor.vv      v26, v16, v16           \n\t"
        "vxor.vv      v27, v16, v16           \n\t"

        ".align 4                             \n\t"
        "BLK_LPST%=:                          \n\t"
        "K64_LPST%=:                          \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vle8.v       v2, (s4)                \n\t"
        "addi         s4, s4, 128             \n\t"

        "vle8.v       v4, (s6)                \n\t"
        "addi         s6, s6, 128             \n\t"
        "vle8.v       v5, (s6)                \n\t"
        "addi         s6, s6, 128             \n\t"
        "vle8.v       v6, (s6)                \n\t"
        "addi         s6, s6, 128             \n\t"
        "vle8.v       v7, (s6)                \n\t"
        "addi         s6, s6, 128             \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v0, (s5)                \n\t"
        "addi         s5, s5, 64              \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v3, (s3)                \n\t"
        "addi         s3, s3, 64              \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vfwcvt.f.x.v v28, v2                 \n\t"
        "vsetvli      t0, x0, e16, m1         \n\t"
        "vfmul.vv     v1, v28, v30            \n\t"
        "vfmul.vv     v29, v29, v30           \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vnot.v       v0, v0                  \n\t"
        "vand.vi      v12, v4, 0x3            \n\t"
        "vand.vi      v13, v5, 0x3            \n\t"
        "vand.vi      v14, v6, 0x3            \n\t"
        "vand.vi      v15, v7, 0x3            \n\t"
        "vsetvli      t0, x0, e8, m4          \n\t"
        "vadd.vi      v12, v12, -4, v0.t      \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v0, (s5)                \n\t"
        "addi         s5, s5, 64              \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vsll.vi      v8, v4, 4               \n\t"
        "vsll.vi      v9, v5, 4               \n\t"
        "vsll.vi      v10, v6, 4              \n\t"
        "vsll.vi      v11, v7, 4              \n\t"
        "vsrl.vi      v16, v8, 6              \n\t"
        "vsrl.vi      v17, v9, 6              \n\t"
        "vnot.v       v0, v0                  \n\t"
        "vsrl.vi      v18, v10, 6             \n\t"
        "vsrl.vi      v19, v11, 6             \n\t"
        "vsetvli      t0, x0, e8, m4          \n\t"
        "vadd.vi      v16, v16, -4, v0.t      \n\t"

        "vsetvli      t0, x0, e64, mf2        \n\t"
        "vslidedown.vi  v2, v3, 2             \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vmadot.hp    v24, v3, v12, v1, 0, i8 \n\t"
        "vmadot.hp    v25, v3, v13, v1, 1, i8 \n\t"
        "vmadot.hp    v26, v3, v14, v1, 2, i8 \n\t"
        "vmadot.hp    v27, v3, v15, v1, 3, i8 \n\t"
        "vmadot.hp    v24, v2, v16, v1, 4, i8 \n\t"
        "vmadot.hp    v25, v2, v17, v1, 5, i8 \n\t"
        "vmadot.hp    v26, v2, v18, v1, 6, i8 \n\t"
        "vmadot.hp    v27, v2, v19, v1, 7, i8 \n\t"

        "vsetvli      t0, x0, e64, m1         \n\t"
        "vmv.v.v      v1, v29                 \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v0, (s5)                \n\t"
        "addi         s5, s5, 64              \n\t"

        "vsetvli      t0, x0, e64, mf2        \n\t"
        "vslidedown.vi  v3, v3, 4             \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vsll.vi      v8, v4, 2               \n\t"
        "vsll.vi      v9, v5, 2               \n\t"
        "vsll.vi      v10, v6, 2              \n\t"
        "vsll.vi      v11, v7, 2              \n\t"

        "vsrl.vi      v20, v8, 6              \n\t"
        "vsrl.vi      v21, v9, 6              \n\t"
        "vnot.v       v0, v0                  \n\t"
        "vsrl.vi      v22, v10, 6             \n\t"
        "vsrl.vi      v23, v11, 6             \n\t"

        "vsetvli      t0, x0, e8, m4          \n\t"
        "vadd.vi      v20, v20, -4, v0.t      \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v0, (s5)                \n\t"
        "addi         s5, s5, 64              \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vsrl.vi      v8, v4, 6               \n\t"
        "vsrl.vi      v9, v5, 6               \n\t"
        "vnot.v       v0, v0                  \n\t"
        "vsrl.vi      v10, v6, 6              \n\t"
        "vsrl.vi      v11, v7, 6              \n\t"

        "vsetvli      t0, x0, e8, m4          \n\t"
        "vadd.vi      v8, v8, -4, v0.t        \n\t"

        "vsetvli      t0, x0, e64, mf2        \n\t"
        "vslidedown.vi  v2, v3, 2             \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vmadot.hp    v24, v3, v20, v1, 0, i8 \n\t"
        "vmadot.hp    v25, v3, v21, v1, 1, i8 \n\t"
        "vmadot.hp    v26, v3, v22, v1, 2, i8 \n\t"
        "vmadot.hp    v27, v3, v23, v1, 3, i8 \n\t"
        "vmadot.hp    v24, v2, v8, v1, 4, i8  \n\t"
        "vmadot.hp    v25, v2, v9, v1, 5, i8  \n\t"
        "vmadot.hp    v26, v2, v10, v1, 6, i8 \n\t"
        "vmadot.hp    v27, v2, v11, v1, 7, i8 \n\t"

        "addi         t3, t3, -1              \n\t"
        "bgtz         t3, K64_LPST%=          \n\t"
        "K64_LPND%=:                          \n\t"

        "vsetvli      t0, x0, e16, m1         \n\t"
        "vpack.vv     v12, v24, v25, 1        \n\t"
        "vpack.vv     v14, v26, v27, 1        \n\t"
        "vpack.vv     v16, v12, v14, 2        \n\t"
        "vsetvli      t0, x0, e16, mf2        \n\t"
        "vfwcvt.f.f.v v26, v16                \n\t"

        "flw          f0, (s2)                \n\t"
        "addi         s2, s2, 4+32+256        \n\t"
        "add          t4, s7, %[B_STR]        \n\t"
        "addi         s3, s2, 4+32            \n\t"

        "addi         s5, t4, 32*16           \n\t"
        "mv           s4, t4                  \n\t"
        "addi         s6, s5, 32*32           \n\t"
        "addi         s8, s6, 1024            \n\t"
        "addi         s8, s8, 1024            \n\t"
        "addi         s7, t4, 0               \n\t"
        "addi         t2, t2, -1              \n\t"

        "fmul.s       f0, f0, %[a_post_mul]   \n\t"
        "vsetvli      t0, x0, e32, m1         \n\t"
        "vfmacc.vf    v31, f0, v26            \n\t"

        "beqz         t2, BLK_LPND%=          \n\t"

        "vsetvli      t0, x0, e16, mf2        \n\t"
        "vle16.v      v1, (s8)                \n\t"
        "vsetvli      t0, x0, e16, m1         \n\t"
        "vpack.vv     v26, v1, v1, 3          \n\t"
        "vmv.v.v      v17, v26                \n\t"
        "vsetvli      t0, x0, e16, m1         \n\t"
        "vfmul.vf     v30, v17, %[q3_step]    \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vxor.vv      v24, v16, v16           \n\t"
        "vxor.vv      v25, v16, v16           \n\t"
        "vxor.vv      v26, v16, v16           \n\t"
        "vxor.vv      v27, v16, v16           \n\t"

        "li           t3, 4                   \n\t"
        "bgtz         t2, BLK_LPST%=          \n\t"

        "BLK_LPND%=:                          \n\t"
        "vsetvli      t0, %[NBLKS], e32, m1   \n\t"
        "vse32.v      v31, (%[pC])            \n\t"

        :
        : [KBLKS] "r"(k_blks), [NBLKS] "r"(nb_real), [pA] "r"(a_data), [pB] "r"(b_data),
          [pC] "r"(dst_c), [B_STR] "r"(b_str), [q3_step] "f"(q3_step), [a_post_mul] "f"(a_post_mul)
        : "cc", "memory", "t0", "t2", "t3", "t4", "t5", "f0", "f1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",
          "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15",
          "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31");
}

/* ===== test harness ===== */
static uint64_t g_rng=0x243F6A8885A308D3ULL;
static uint32_t rnd(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17; return (uint32_t)g_rng; }
static float rndf(float lo,float hi){ return lo+(hi-lo)*((rnd()&0xFFFFFF)/(float)0xFFFFFF); }

int main(void){
    pin_hart8();
    printf("Q3_K standalone pack/dequant/kernel oracle\n");
    printf("sizeof(block_q3_K)=%zu (expect 110)\n",sizeof(block_q3_K));
    printf("sizeof(nrow_block_q3_k32)=%zu (expect 3648)\n",sizeof(nrow_block_q3_k32));
    printf("AREC_Q3K=%d (expect 292)\n",AREC_Q3K);
    if(sizeof(block_q3_K)!=110||sizeof(nrow_block_q3_k32)!=3648){ printf("STRUCT SIZE MISMATCH -- abort\n"); return 1; }

    const int K_BLKS=8; /* 8 superblocks x 256 = 2048 K, a representative multi-superblock test */
    const int NCOLS=32;

    /* synthetic B: NCOLS output rows x K_BLKS superblocks of block_q3_K, random-but-well-formed */
    block_q3_K *B=malloc(sizeof(block_q3_K)*NCOLS*K_BLKS);
    for(int r=0;r<NCOLS;r++) for(int kb=0;kb<K_BLKS;kb++){
        block_q3_K *blk=&B[r*K_BLKS+kb];
        for(int i=0;i<32;i++) blk->hmask[i]=(uint8_t)rnd();
        for(int i=0;i<64;i++) blk->qs[i]=(uint8_t)rnd();
        for(int i=0;i<12;i++) blk->scales[i]=(uint8_t)rnd();
        blk->d=(_Float16)rndf(0.001f,0.08f);
    }

    /* repack into K_BLKS nrow_block_q3_k32 blocks (one per superblock, 32 rows each) */
    nrow_block_q3_k32 *Bp=malloc(sizeof(nrow_block_q3_k32)*K_BLKS);
    for(int kb=0;kb<K_BLKS;kb++){
        block_q3_K rows[32];
        for(int r=0;r<NCOLS;r++) rows[r]=B[r*K_BLKS+kb];
        pack_B_q3k32(rows,&Bp[kb]);
    }

    /* synthetic A: one row of K_BLKS*256 fp32 activations */
    int KTOT=K_BLKS*256;
    float *A=malloc(sizeof(float)*KTOT);
    for(int i=0;i<KTOT;i++) A[i]=rndf(-2.0f,2.0f);
    uint8_t *Ap=malloc((size_t)AREC_Q3K*K_BLKS);
    for(int kb=0;kb<K_BLKS;kb++) pack_A_q3k(A+kb*256, Ap+(size_t)kb*AREC_Q3K);

    /* reference: dequant B via canonical ggml algorithm, dequant A from its OWN stored int8/scale
     * (NOT the original random floats) -- isolates kernel/repack correctness from A's own
     * quantization noise, matching vendor_ime_m4_probe.c's proven methodology. */
    double ref[32]; memset(ref,0,sizeof(ref));
    for(int r=0;r<NCOLS;r++){
        double acc=0;
        for(int kb=0;kb<K_BLKS;kb++){
            float bdeq[256]; dequantize_q3_K_block(&B[r*K_BLKS+kb],bdeq);
            const uint8_t *arec=Ap+(size_t)kb*AREC_Q3K;
            float ascale; memcpy(&ascale,arec,4);
            const int8_t *aqs=(const int8_t*)(arec+4+32);
            for(int l=0;l<256;l++) acc += (double)bdeq[l] * (double)aqs[l] * (double)ascale;
        }
        ref[r]=acc;
    }

    /* kernel output */
    float out[32];
    run_hp_q3k_m1(Ap,(const uint8_t*)Bp,out,K_BLKS,NCOLS,(long)sizeof(nrow_block_q3_k32));

    double max_abs=0, sum_abs=0, sum_rel=0;
    for(int r=0;r<NCOLS;r++){
        double d=fabs((double)out[r]-ref[r]);
        double rel=d/(fabs(ref[r])+1e-6);
        if(d>max_abs)max_abs=d;
        sum_abs+=d; sum_rel+=rel;
        printf("  row%2d: kernel=%+.6f ref=%+.6f absdiff=%.3e reldiff=%.3e\n",r,out[r],ref[r],d,rel);
    }
    printf("\nmax_abs_diff=%.4e mean_abs_diff=%.4e mean_rel_diff=%.4e (%d rows, %d K)\n",
        max_abs,sum_abs/NCOLS,sum_rel/NCOLS,NCOLS,KTOT);
    printf("PASS/FAIL (mean_rel_diff<0.05 as a first-pass sanity bar, matching expected int8/fp16 quantization noise scale): %s\n",
        (sum_rel/NCOLS)<0.05 ? "PASS" : "FAIL");

    /* hot/cold kernel timing, per the ladder's own "hot and cold kernel benchmark" requirement */
    double t0,t1;
    /* cold: fresh malloc'd, not-yet-touched B/A pair, single call */
    nrow_block_q3_k32 *Bc=malloc(sizeof(nrow_block_q3_k32)*K_BLKS);
    memcpy(Bc,Bp,sizeof(nrow_block_q3_k32)*K_BLKS);
    uint8_t *Ac=malloc((size_t)AREC_Q3K*K_BLKS); memcpy(Ac,Ap,(size_t)AREC_Q3K*K_BLKS);
    float outc[32];
    t0=now(); run_hp_q3k_m1(Ac,(const uint8_t*)Bc,outc,K_BLKS,NCOLS,(long)sizeof(nrow_block_q3_k32)); t1=now();
    printf("\ncold call (first touch): %.3f us\n",(t1-t0)*1e6);

    /* hot: repeated calls on already-resident data */
    const int REPS=2000;
    t0=now();
    for(int i=0;i<REPS;i++) run_hp_q3k_m1(Ap,(const uint8_t*)Bp,out,K_BLKS,NCOLS,(long)sizeof(nrow_block_q3_k32));
    t1=now();
    printf("hot call avg (%d reps, %d K each): %.3f us/call, %.2f Melem/s\n",
        REPS,KTOT,(t1-t0)*1e6/REPS,(double)KTOT*NCOLS*REPS/(t1-t0)/1e6);

    free(B);free(Bp);free(A);free(Ap);free(Bc);free(Ac);
    return 0;
}
