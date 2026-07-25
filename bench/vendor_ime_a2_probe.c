/* vendor_ime_a2_probe.c — A2 step 0: empirically pin down the vmadotsu.hp/vmadotu.hp zero-point
 * sign convention and nibble->K mapping. The public riscv-ime-extension-spec covers plain
 * vmadotsu/vmadotu (C[i,j] += A[i,k]*B[j,k], no built-in zero point) but does NOT document
 * vnpack4 or the .hp variants at all -- those are vendor-internal. So: single-known-element
 * test vectors, read the hardware's actual answer, compare against every candidate hypothesis.
 *
 * Layout (from reference/spacemit-backend/ime2_kernels.cpp, gemm_kernel_i8i4_m1 active branch):
 *   A record (per K32 group): [4B fp32 ascale][2B int16 asum][32B int8 Adata], stride 38B.
 *   B record (per K32 group, one N32 panel): [64B: 32x fp16 bscale][512B: 4x128B N8K32 int4
 *   data (v4..v7)], stride 576B. Byte j (0..15) of an N8 block, row n: holds K=j in the low
 *   nibble, K=j+16 in the high nibble (same K-pairing our own pack_w_int4 uses).
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o vendor_ime_a2_probe vendor_ime_a2_probe.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* one K32-group A record + one K32-group / N32-panel B record, N32=1 real row (rest zero) */
static uint8_t Arec[38];
static uint8_t Brec[576];
static float   Cout[32];

/* ported verbatim from gemm_kernel_i8i4_m1's active (#else) branch, k_blks=1, nblks=32 */
static void run_hp_kernel(void){
    long cnt=1; long nblks=32;
    uint8_t*pA=Arec; uint8_t*pB=Brec; float*pC=Cout;
    __asm__ volatile(
        "mv           t3, %[BCK]              \n\t"
        "mv           t4, %[NBLKS]            \n\t"
        "vsetvli      t0, x0, e16, m1         \n\t"
        "vmv.v.i      v0, 1                   \n\t"
        "mv           s2, %[pA]               \n\t"
        "addi         s3, %[pA], 4+2          \n\t"
        "mv           s4, %[pB]               \n\t"
        "addi         s5, %[pB], 32*2         \n\t"
        "mv           s6, %[pC]               \n\t"

        "vsll.vi      v1, v0, 4               \n\t"
        "vxor.vv      v2, v0, v0              \n\t"
        "vfcvt.f.x.v  v0, v0                  \n\t"
        "vfcvt.f.x.v  v1, v1                  \n\t"

        ".align 4                             \n\t"
        "_K_LPST%=:                           \n\t"

        "vsetvli      t0, x0, e8, m1          \n\t"
        "vl4r.v       v4, (s5)                \n\t"
        "addi         s5, s5, 128*4+64        \n\t"

        "vsetvli      t0, x0, e8, mf2         \n\t"
        "vle8.v       v30, (s4)               \n\t"
        "addi         s4, s4, 64+128*4        \n\t"

        "vsetvli      t0, x0, e8, mf4         \n\t"
        "vle8.v       v3, (s3)                \n\t"
        "addi         s3, s3, 32+6            \n\t"

        "flw          f0, (s2)                \n\t"
        "lh           t2, 4(s2)               \n\t"
        "addi         s2, s2, 6+32            \n\t"

        "vsetvli      t0, x0, e16, m1         \n\t"
        "vmv.v.i      v28, 8                  \n\t"
        "vsetvli      t0, x0, e8, m1          \n\t"
        "vsrl.vi      v24, v3, 4              \n\t"

        "vsetvli      t0, x0, e16, m1         \n\t"
        "vmul.vx      v26, v28, t2            \n\t"
        "vnpack4.vv   v8, v3, v3, 3            \n\t"
        "vnpack4.vv   v10, v24, v24, 3         \n\t"

        "vfcvt.f.x.v  v16, v26                \n\t"
        "vadd.vi      v18, v16, 0             \n\t"
        "vadd.vi      v20, v16, 0             \n\t"
        "vadd.vi      v22, v16, 0             \n\t"

        "vmadotsu.hp  v16, v10, v4, v1, 0, i4  \n\t"
        "vmadotsu.hp  v18, v10, v5, v1, 0, i4  \n\t"
        "vmadotsu.hp  v20, v10, v6, v1, 0, i4  \n\t"
        "vmadotsu.hp  v22, v10, v7, v1, 0, i4  \n\t"
        "vmadotu.hp   v16, v8, v4, v0, 0, i4   \n\t"
        "vmadotu.hp   v18, v8, v5, v0, 0, i4   \n\t"
        "vmadotu.hp   v20, v8, v6, v0, 0, i4   \n\t"
        "vmadotu.hp   v22, v8, v7, v0, 0, i4   \n\t"

        "vpack.vv     v24, v16, v18, 1        \n\t"
        "vpack.vv     v26, v20, v22, 1        \n\t"
        "vpack.vv     v16, v24, v26, 2        \n\t"

        "vsetvli      t0, x0, e16, mf2        \n\t"
        "vfwmul.vv     v31, v30, v16          \n\t"

        "vsetvli      t0, x0, e32, m1         \n\t"
        "vfmacc.vf    v2, f0, v31             \n\t"

        "addi         t3, t3, -1              \n\t"
        "bgtz         t3, _K_LPST%=           \n\t"
        "_K_LPND%=:                           \n\t"

        "vsetvli      t0, t4, e32, m1         \n\t"
        "vse32.v      v2, (s6)                \n\t"

        : : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(pA), [pB] "r"(pB), [pC] "r"(pC)
        : "cc","memory","t0","t2","t3","t4","f0","s2","s3","s4","s5","s6",
          "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
          "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31");
}

/* set up: A scale=1.0, one nonzero A element = av at k=kidx (0..31), rest 0.
 * B: only row n=0 nonzero, unsigned nibble bval at k=kidx, rest 0 (=unsigned 0, i.e. signed -8).
 * b_scale[0]=1.0 (fp16), others irrelevant. */
static void setup(int kidx,int8_t av,uint8_t bval_unsigned){
    memset(Arec,0,sizeof Arec); memset(Brec,0,sizeof Brec); memset(Cout,0,sizeof Cout);
    float ascale=1.0f; memcpy(Arec,&ascale,4);
    int8_t Adata[32]={0}; Adata[kidx]=av;
    int16_t asum=0; for(int i=0;i<32;i++) asum+=Adata[i];
    memcpy(Arec+4,&asum,2); memcpy(Arec+6,Adata,32);

    /* B scale block: 32 x fp16, row0 = 1.0, rest 0 (doesn't matter, only row0/lane0 read) */
    uint16_t one_fp16=0x3C00; /* IEEE754 half 1.0 */
    memcpy(Brec,&one_fp16,2);
    /* B data block: 4x128B (v4..v7). Row0 lives in block v4 (rows 0-7), byte = 0*16+j where j=kidx%16,
     * nibble = lo if kidx<16 else hi. */
    uint8_t*blk0=Brec+64; /* start of v4 data */
    int j=kidx%16; int byte_idx = 0*16+j; /* n=0 */
    uint8_t cur=blk0[byte_idx];
    if(kidx<16) blk0[byte_idx]=(cur&0xf0)|(bval_unsigned&0xf);
    else        blk0[byte_idx]=(cur&0x0f)|((bval_unsigned&0xf)<<4);
}

int main(void){
    bind_ai(); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs);
    sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield();

    printf("=== Test 1: k=0 (lo nibble), A=1, B_unsigned=13 (candidate signed=5 if zp=8-sub, or -11 if zp=8-add-wrong) ===\n");
    setup(0,1,13);
    run_hp_kernel();
    printf("C[0]=%g  (hyp raw_minus_zp: %d, hyp raw_plus_zp: %d, hyp raw_only: %d)\n",
        Cout[0], 13-8, 13+8, 13);

    printf("\n=== Test 2: k=16 (hi nibble), A=1, B_unsigned=13 ===\n");
    setup(16,1,13);
    run_hp_kernel();
    printf("C[0]=%g\n", Cout[0]);

    printf("\n=== Test 3: k=0, A=1, B_unsigned=8 (should be the true zero point -> signed 0) ===\n");
    setup(0,1,8);
    run_hp_kernel();
    printf("C[0]=%g  (expect 0 if zp=8 subtraction is correct)\n", Cout[0]);

    printf("\n=== Test 4: k=0, A=1, B_unsigned=0 (signed should be -8) ===\n");
    setup(0,1,0);
    run_hp_kernel();
    printf("C[0]=%g  (expect -8 if zp=8 subtraction correct)\n", Cout[0]);

    printf("\n=== Test 5: k=0, A=1, B_unsigned=15 (signed should be +7) ===\n");
    setup(0,1,15);
    run_hp_kernel();
    printf("C[0]=%g  (expect 7 if zp=8 subtraction correct)\n", Cout[0]);

    printf("\n=== Test 6: k=0, A=2 (scale still 1.0), B_unsigned=13 (signed 5) -> expect 10 if linear ===\n");
    setup(0,2,13);
    run_hp_kernel();
    printf("C[0]=%g\n", Cout[0]);

    printf("\n=== Test 7: asum sanity — k=5 (mid lo), A=1, B_unsigned=13, all else in A still 0 ===\n");
    setup(5,1,13);
    run_hp_kernel();
    printf("C[0]=%g  (if same as test1 => nibble mapping symmetric across lo range)\n", Cout[0]);

    return 0;
}
