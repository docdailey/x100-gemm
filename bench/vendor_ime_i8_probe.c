/* vendor_ime_i8_probe.c — A1-style toolchain probe for the vendor int8 M1 kernel
 * (gemm_kernel_i8i8_m1, ime2_kernels.cpp:4773), before building anything on top of it.
 * New instructions vs the int4 .hp kernel: plain typed `vmadot ...,i8` (signed x signed, no
 * zero-point trickery), `vupack.vv` (operand-shape unpack for the wider int8 case), and
 * `vslidedown.vi` (splits A into two halves via a slide instead of a nibble shift). Pass/fail
 * here is just "no SIGILL, no crash" -- numerical correctness is a separate step.
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o vendor_ime_i8_probe vendor_ime_i8_probe.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* one K32-group A record (38B: 4B fp32 scale + 2B int16 asum + 32B int8 data) +
 * one K32-group / N32-panel B record (1088B: 64B fp16 scales + 1024B int8 data) */
static uint8_t Arec[38];
static uint8_t Brec[1088];
static float   Cout[32];

__attribute__((noinline,optimize("no-tree-vectorize")))
static void run_i8_m1_probe(void){
    long cnt=1; long nblks=32;
    uint8_t*pA=Arec; uint8_t*pB=Brec; float*pC=Cout;
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
        : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(pA), [pB] "r"(pB), [pC] "r"(pC)
        : "cc","t0","t3","t4","f0","s2","s3","s4","s5","s6",
          "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
          "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory");
}

int main(void){
    bind_ai(); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs);
    int r=sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield();
    printf("[A1-i8] setaffinity=%d cpu=%d\n", r, sched_getcpu());

    for(int i=0;i<38;i++) Arec[i]=(uint8_t)((i*7+3)&0xff);
    for(int i=0;i<1088;i++) Brec[i]=(uint8_t)((i*13+5)&0xff);
    memcpy(Arec,&(float){1.0f},4); /* fp32 scale = 1.0 for sanity */

    run_i8_m1_probe();
    printf("[A1-i8] Cout[0..7]="); for(int i=0;i<8;i++) printf("%g ",Cout[i]);
    printf("\n[A1-i8] PASS (no SIGILL/crash)\n");
    return 0;
}
