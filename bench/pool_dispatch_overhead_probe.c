#define _GNU_SOURCE
/* Isolates pool-dispatch overhead specifically (Codex's directive: "measure dispatch separately
 * only if kernel-only scaling approaches 4x" -- it did, 3.90x, see run_hp_m1_scaling_probe.c).
 * Replicates qwen_moe_hp.c's EXACT persistent spin-dispatch pool mechanism (atomic gen/done
 * counters, main thread as tn=0 doing inline work, workers 1..nt-1 spinning on the generation
 * counter) around the SAME cache-resident hot per-thread buffers as the kernel-only scaling
 * probe. The only new variable vs that probe is the dispatch machinery itself -- if throughput
 * matches ~41 GB/s (this probe's nt=4 kernel-only number), dispatch overhead is negligible; if
 * it's materially lower, dispatch is a real contributor to the production 2.90x-not-4x gap. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdatomic.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static uint16_t f32_to_f16(float f){
    uint32_t bits; memcpy(&bits,&f,4);
    uint32_t sign=(bits>>16)&0x8000; int32_t exp=((bits>>23)&0xff)-127+15; uint32_t mant=bits&0x7fffff;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; mant|=0x800000; uint32_t shift=14-exp; uint32_t r=mant>>shift; if((mant>>(shift-1))&1) r++; return (uint16_t)(sign|r); }
    if(exp>=31) return (uint16_t)(sign|0x7c00);
    uint32_t r=mant>>13; if(mant&0x1000){ r++; if(r==0x400){ r=0; exp++; if(exp>=31) return (uint16_t)(sign|0x7c00);} }
    return (uint16_t)(sign|(exp<<10)|r);
}

#define NSUB 8
#define AREC 290
#define BREC 576
#define BSUPER (NSUB*BREC)
#define KB 8      /* K=2048 -> kb=8 */
#define PANELS 6  /* panels/thread/dispatch round, matching eg/eu's Np=24 at nt=4 */

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
static void pack_A_hp_scalar(const float*a, uint8_t*out){
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
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor);
        memcpy(out+kk*34,&ssub,2); memcpy(out+kk*34+2,qa[kk],32);
    }
    for(int kk=0;kk<NSUB;kk++){ uint16_t as=f32_to_f16(-(float)asum[kk]*8.0f); memcpy(out+272+kk*2,&as,2); }
    uint16_t blkscale=f32_to_f16(scale_avg);
    memcpy(out+288,&blkscale,2);
}
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
static uint32_t rng_state=42;
static float randf(void){ rng_state=rng_state*1103515245u+12345u; return ((int32_t)(rng_state>>8))/8388608.0f; }

/* ===== replica of qwen_moe_hp.c's persistent spin-dispatch pool ===== */
#define MAXNT 4
static int hart_order[MAXNT] = {8,10,12,14};
static _Atomic int g_pool_gen=0, g_pool_done=0;
static int g_pool_nt=0;
static pthread_t g_pool_threads[MAXNT];
static uint8_t* g_Abuf[MAXNT];   /* each thread's private hot A buffer */
static uint8_t* g_Bbuf[MAXNT];   /* each thread's private hot B buffer (PANELS panels worth) */
static float g_dst[MAXNT][PANELS*32];
static long g_rounds_done=0;
#define ROUNDS 200000

static void make_buffers(int tid){
    g_Abuf[tid] = malloc((size_t)KB*AREC);
    for(int sb=0; sb<KB; sb++){
        float a[256]; for(int i=0;i<256;i++) a[i]=randf()*2.0f;
        pack_A_hp_scalar(a, g_Abuf[tid]+(size_t)sb*AREC);
    }
    g_Bbuf[tid] = malloc((size_t)PANELS*KB*BSUPER);
    for(int p=0;p<PANELS;p++) for(int sb=0; sb<KB; sb++) for(int sub=0; sub<NSUB; sub++){
        q4_0_native rows[32];
        for(int r=0;r<32;r++){ float wv[32]; for(int i=0;i<32;i++) wv[i]=randf(); quantize_q4_0_native(wv,&rows[r]); }
        pack_B_q4_0x32(rows, g_Bbuf[tid]+((size_t)p*KB+sb)*BSUPER+(size_t)sub*BREC);
    }
}
static void worker_run(int tn){
    for(int p=0;p<PANELS;p++)
        run_hp_m1(g_Abuf[tn], g_Bbuf[tn]+(size_t)p*KB*BSUPER, g_dst[tn]+p*32, KB);
}
static void* pool_worker(void* arg){
    int tn = (int)(intptr_t)arg;
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart_order[tn],&cs); sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
    make_buffers(tn);
    int last=0;
    for(;;){
        int gen;
        while((gen=atomic_load_explicit(&g_pool_gen,memory_order_acquire))==last) { /* spin */ }
        last=gen;
        if(gen<0) return NULL;
        worker_run(tn);
        atomic_fetch_add_explicit(&g_pool_done,1,memory_order_release);
    }
}
static void dispatch_round(int nt){
    atomic_store_explicit(&g_pool_done,0,memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool_gen,1,memory_order_release);
    worker_run(0); /* main/tn=0 does its own share inline, exactly like lin_mm_hp */
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < nt-1) { /* spin */ }
}

int main(void){
    for(int nt=1; nt<=4; nt*=2){
        g_pool_nt=nt;
        bind_ai();
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart_order[0],&cs); sched_setaffinity(0,sizeof(cs),&cs);
        for(int i=0;i<5;i++) sched_yield();
        make_buffers(0);
        atomic_store_explicit(&g_pool_gen,0,memory_order_relaxed);
        for(int i=1;i<nt;i++) pthread_create(&g_pool_threads[i],NULL,pool_worker,(void*)(intptr_t)i);
        usleep(200000); /* let workers finish their own make_buffers() before timing */

        double t0=now();
        for(long r=0;r<ROUNDS;r++) dispatch_round(nt);
        double t1=now();

        double bytes = (double)nt*PANELS*KB*BSUPER*ROUNDS;
        double gbps = bytes/(t1-t0)/1e9;
        printf("nt=%d (pool-dispatched, %d rounds, %d panels/thread/round): %.2f GB/s\n", nt, ROUNDS, PANELS, gbps);

        atomic_store_explicit(&g_pool_gen,-1,memory_order_release); /* shutdown */
        for(int i=1;i<nt;i++) pthread_join(g_pool_threads[i],NULL);
    }
    return 0;
}
