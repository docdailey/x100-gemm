/* vendor_ime_probe.c — A1 (research_feed_paths.md §4.4): assemble+run the vendor-shaped IME-2
 * typed-int4 instructions (vnpack4.vv, vmadotsu ... i4, vmadotsu.hp/vmadotu.hp) in isolation on
 * the A100 (hart 8). Pass/fail is just "no SIGILL, prints something" — correctness against the
 * grouped-fold oracle is A2, not here. Source of the exact mnemonics/operand order: reference/
 * spacemit-backend/ime2_kernels.cpp, gemm_kernel_i8i4_m1 (the active #else / .hp branch).
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o vendor_ime_probe vendor_ime_probe.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <setjmp.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

static sigjmp_buf jb;
static volatile sig_atomic_t last_sig=0;
static void on_sig(int sig){ last_sig=sig; siglongjmp(jb,1); }

/* returns 0 on success (no signal), else the signal number */
static int guarded(const char*name, void(*fn)(void)){
    struct sigaction sa={0}; sa.sa_handler=on_sig;
    sigaction(SIGILL,&sa,NULL); sigaction(SIGSEGV,&sa,NULL);
    last_sig=0;
    if(sigsetjmp(jb,1)==0){ fn(); printf("[A1] %-28s OK\n",name); return 0; }
    printf("[A1] %-28s FAILED (signal %d)\n",name,last_sig);
    return last_sig;
}

/* ---- probe 1: vnpack4.vv — split a e8,m1 register into nibble-extracted halves ---- */
static int8_t g_a[32]; static int8_t g_lo[32], g_hi[32];
static void probe_vnpack4(void){
    /* v3 = raw nibble-packed byte (hi nibble = second int4, lo nibble = first, same as our own
     * pack_w_int4 nibble packing); v24 = v3>>4 (hi nibbles as low nibbles of a byte).
     * vnpack4.vv v8,v3,v3,3 / vnpack4.vv v10,v24,v24,3 per the vendor kernel. */
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t"
        "vle8.v v3,(%0)\n\t"
        "vsrl.vi v24,v3,4\n\t"
        "vnpack4.vv v8,v3,v3,3\n\t"
        "vnpack4.vv v10,v24,v24,3\n\t"
        "vse8.v v8,(%1)\n\t"
        "vse8.v v10,(%2)\n\t"
        : : "r"(g_a),"r"(g_lo),"r"(g_hi) : "t0","v3","v8","v9","v10","v11","v24","v25","memory");
}

/* ---- probe 2: typed vmadotsu/vmadotu ... i4 (ordinary, non-hp, the #if 0 branch's core insn) ---- */
static int8_t g_A4[32];      /* A: M1 x K32 int8-shaped nibble source (hi/lo already split by vnpack4) */
static int8_t g_B4[4*128];   /* B: 4 VRF regs x 128B = N32 x K32 int4, vendor packed */
static int32_t g_C4[4*8];    /* 4x int32 accumulator groups (N8 each) -> N32 */
static void probe_vmadotsu_i4(void){
    __asm__ volatile(
        "vsetvli t0,x0,e8,mf4\n\t vle8.v v3,(%0)\n\t"          /* A M1K32 int8, 32B */
        "vsetvli t0,x0,e8,m1\n\t vsrl.vi v24,v3,4\n\t"
        "vnpack4.vv v8,v3,v3,3\n\t vnpack4.vv v10,v24,v24,3\n\t"
        "vl4r.v v4,(%1)\n\t"                                    /* B N32K32 int4, 512B */
        "vsetvli t0,x0,e32,m1\n\t"
        "vxor.vv v16,v16,v16\n\t vxor.vv v18,v18,v18\n\t vxor.vv v20,v20,v20\n\t vxor.vv v22,v22,v22\n\t"
        "vmadotsu v16,v10,v4,i4\n\t vmadotsu v18,v10,v5,i4\n\t vmadotsu v20,v10,v6,i4\n\t vmadotsu v22,v10,v7,i4\n\t"
        "vsll.vi v16,v16,4\n\t vsll.vi v18,v18,4\n\t vsll.vi v20,v20,4\n\t vsll.vi v22,v22,4\n\t"
        "vmadotu v16,v8,v4,i4\n\t vmadotu v18,v8,v5,i4\n\t vmadotu v20,v8,v6,i4\n\t vmadotu v22,v8,v7,i4\n\t"
        "vse32.v v16,(%2)\n\t"
        : : "r"(g_A4),"r"(g_B4),"r"(g_C4)
        : "t0","v3","v4","v5","v6","v7","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","memory");
}

/* ---- probe 3: vmadotsu.hp / vmadotu.hp (the active branch in gemm_kernel_i8i4_m1) ---- */
static _Float16 g_Bscale[32];  /* 32 x fp16 group scales (one N32 panel) */
static void probe_vmadotsu_hp(void){
    __asm__ volatile(
        "vsetvli t0,x0,e16,m1\n\t vmv.v.i v0,1\n\t"
        "vsll.vi v1,v0,4\n\t vxor.vv v2,v0,v0\n\t"
        "vfcvt.f.x.v v0,v0\n\t vfcvt.f.x.v v1,v1\n\t"
        "vsetvli t0,x0,e8,m1\n\t vle8.v v3,(%0)\n\t vsrl.vi v24,v3,4\n\t"
        "vnpack4.vv v8,v3,v3,3\n\t vnpack4.vv v10,v24,v24,3\n\t"
        "vl4r.v v4,(%1)\n\t"
        "vmadotsu.hp v16,v10,v4,v1,0,i4\n\t vmadotsu.hp v18,v10,v5,v1,0,i4\n\t"
        "vmadotsu.hp v20,v10,v6,v1,0,i4\n\t vmadotsu.hp v22,v10,v7,v1,0,i4\n\t"
        "vmadotu.hp  v16,v8,v4,v0,0,i4\n\t  vmadotu.hp  v18,v8,v5,v0,0,i4\n\t"
        "vmadotu.hp  v20,v8,v6,v0,0,i4\n\t  vmadotu.hp  v22,v8,v7,v0,0,i4\n\t"
        "vpack.vv v24,v16,v18,1\n\t vpack.vv v26,v20,v22,1\n\t vpack.vv v16,v24,v26,2\n\t"
        "vsetvli t0,x0,e16,mf2\n\t vle8.v v30,(%2)\n\t"
        "vfwmul.vv v31,v30,v16\n\t"
        "vsetvli t0,x0,e32,m1\n\t vfmacc.vf v2,%3,v31\n\t"
        "vse32.v v2,(%4)\n\t"
        : : "r"(g_A4),"r"(g_B4),"r"(g_Bscale),"f"(1.0f),"r"(g_C4)
        : "t0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
          "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v30","v31","memory");
}

int main(void){
    bind_ai(); cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(8,&cs);
    int r=sched_setaffinity(0,sizeof(cs),&cs); for(int i=0;i<5;i++)sched_yield();
    unsigned long vlenb=0; __asm__ volatile("csrr %0, vlenb":"=r"(vlenb));
    printf("[A1] setaffinity=%d cpu=%d vlenb=%lu (VLEN=%lu bits)\n", r, sched_getcpu(), vlenb, vlenb*8);

    for(int i=0;i<32;i++){ g_a[i]=(int8_t)((i*7-11)&0xff); }
    for(int i=0;i<4*128;i++) g_B4[i]=(int8_t)((i*13+5)&0xff);
    for(int i=0;i<32;i++) g_Bscale[i]=(_Float16)1.0f;

    int f1=guarded("vnpack4.vv",probe_vnpack4);
    int f2=guarded("vmadotsu/vmadotu ...i4",probe_vmadotsu_i4);
    int f3=guarded("vmadotsu.hp/.hp",probe_vmadotsu_hp);

    printf("[A1] lo[0..7]="); for(int i=0;i<8;i++)printf("%d ",g_lo[i]); printf("\n");
    printf("[A1] hi[0..7]="); for(int i=0;i<8;i++)printf("%d ",g_hi[i]); printf("\n");
    printf("[A1] C4[0..7]="); for(int i=0;i<8;i++)printf("%d ",g_C4[i]); printf("\n");
    printf("[A1] summary: vnpack4=%s typed_i4=%s hp=%s\n",
        f1?"FAIL":"PASS", f2?"FAIL":"PASS", f3?"FAIL":"PASS");
    return f1||f2||f3;
}
