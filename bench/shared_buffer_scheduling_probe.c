#define _GNU_SOURCE
/* Decisive follow-up: the cold_streaming_probe gave each thread its OWN private buffer -- real
 * production has all nt threads striping through the SAME shared Lin.B buffer, cyclically
 * (lin_mm_hp_worker_run: np=tn; np<Np; np+=nt -- thread 0 takes panels 0,4,8..., thread 1 takes
 * 1,5,9..., etc). That per-thread stride could defeat prefetch/DRAM-row locality in a way private
 * per-thread buffers never could. Tests, on the SAME cold (large, freshly-built) shared Lin
 * buffers, through the REAL persistent-pool dispatch mechanism (already shown negligible
 * overhead in pool_dispatch_overhead_probe.c):
 *   CYCLIC:  thread tn takes panels tn, tn+nt, tn+2nt, ... (production's actual assignment)
 *   BLOCKED: thread tn takes one contiguous adjacent range [tn*Np/nt, (tn+1)*Np/nt)
 * Interpretation per the decision tree: blocked>>cyclic -> stride defeats locality, change
 * production's scheduling. Both ~= cyclic's low number -> shared-buffer fabric/delivery itself is
 * the limit, not the schedule. Both ~= the earlier private-buffer ~30GB/s -> look elsewhere
 * (short-call/tail imbalance, output stores, A-buffer traffic, timing/byte accounting). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sched.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdatomic.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

#define BREC 576
#define NSUB 8
#define BSUPER (NSUB*BREC)
#define AREC 290

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

static void fastfill(uint8_t* p, size_t n, uint32_t seed){
    uint32_t s = seed?seed:1;
    for(size_t i=0;i<n;i+=4){ s^=s<<13; s^=s>>17; s^=s<<5; memcpy(p+i,&s,(n-i)>=4?4:(n-i)); }
}

typedef struct { uint8_t* buf; size_t bytes; int Np, kb; } LinBuf;
static LinBuf make_lin(int N, int K, uint32_t seed){
    int Np=N/32, kb=K/256;
    size_t bytes=(size_t)Np*kb*BSUPER;
    uint8_t* p = malloc(bytes);
    fastfill(p, bytes, seed);
    LinBuf lb = { p, bytes, Np, kb };
    return lb;
}
#define NLINS_PER_LAYER 28 /* q,k,v,o + 8x(eg,eu,ed) */
static void build_layer(LinBuf out[NLINS_PER_LAYER], uint32_t seedbase){
    int i=0;
    out[i++]=make_lin(4096,2048,seedbase+1);
    out[i++]=make_lin(512, 2048,seedbase+2);
    out[i++]=make_lin(512, 2048,seedbase+3);
    out[i++]=make_lin(2048,4096,seedbase+4);
    for(int e=0;e<8;e++){
        out[i++]=make_lin(768,2048,seedbase+10+e*3+0);
        out[i++]=make_lin(768,2048,seedbase+10+e*3+1);
        out[i++]=make_lin(2048,768,seedbase+10+e*3+2);
    }
}
static void free_layer(LinBuf lin[NLINS_PER_LAYER]){ for(int i=0;i<NLINS_PER_LAYER;i++) free(lin[i].buf); }
static size_t layer_bytes(LinBuf lin[NLINS_PER_LAYER]){ size_t s=0; for(int i=0;i<NLINS_PER_LAYER;i++) s+=lin[i].bytes; return s; }

#define NLAYERS 10
static LinBuf g_layers[NLAYERS][NLINS_PER_LAYER];
static uint8_t g_A[AREC*16];

typedef enum { SCHED_CYCLIC, SCHED_BLOCKED } Sched;

/* ===== real persistent spin-dispatch pool, replica of qwen_moe_hp.c ===== */
#define MAXNT 4
static int hart_order[MAXNT] = {8,10,12,14};
static _Atomic int g_pool_gen=0, g_pool_done=0;
static int g_pool_nt=0;
static pthread_t g_pool_threads[MAXNT];
static Sched g_sched;
static LinBuf* g_cur_lin; /* the Lin currently being dispatched */
/* real per-panel-unique output array, shared across threads and reused across Lin calls -- matches
 * production exactly: run_hp_m1(Abuf, B+..., y+np*32, kb), where y is the Lin's full N-sized
 * output (q/tmp/g/u/logits etc, reused across different Lin calls in forward(), not a tiny
 * per-thread scratch that stays permanently cache-hot regardless of Np). Sized for the largest N
 * in this test set (lm_head, N=151936). */
static float* g_Y;

static void worker_run(int tn){
    int Np = g_cur_lin->Np, kb = g_cur_lin->kb;
    if(g_sched==SCHED_CYCLIC){
        for(int np=tn; np<Np; np+=g_pool_nt)
            run_hp_m1(g_A, g_cur_lin->buf+(size_t)np*kb*BSUPER, g_Y+np*32, kb);
    } else {
        int lo = (int)((long)Np*tn/g_pool_nt), hi = (int)((long)Np*(tn+1)/g_pool_nt);
        for(int np=lo; np<hi; np++)
            run_hp_m1(g_A, g_cur_lin->buf+(size_t)np*kb*BSUPER, g_Y+np*32, kb);
    }
}
static void* pool_worker(void* arg){
    int tn = (int)(intptr_t)arg;
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart_order[tn],&cs); sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
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
    worker_run(0);
    while(atomic_load_explicit(&g_pool_done,memory_order_acquire) < nt-1) { /* spin */ }
}

static void start_pool(int nt){
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart_order[0],&cs); sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
    atomic_store_explicit(&g_pool_gen,0,memory_order_relaxed);
    for(int i=1;i<nt;i++) pthread_create(&g_pool_threads[i],NULL,pool_worker,(void*)(intptr_t)i);
    usleep(50000);
}
static void stop_pool(int nt){
    atomic_store_explicit(&g_pool_gen,-1,memory_order_release);
    for(int i=1;i<nt;i++) pthread_join(g_pool_threads[i],NULL);
}

static double run_config(int nt, Sched sched){
    g_pool_nt=nt; g_sched=sched;
    start_pool(nt);

    size_t total=0;
    double t0=now();
    for(int L=0; L<NLAYERS; L++)
        for(int i=0;i<NLINS_PER_LAYER;i++){
            g_cur_lin = &g_layers[L][i];
            total += g_cur_lin->bytes;
            dispatch_round(nt);
        }
    double t1=now();

    stop_pool(nt);
    return (double)total/(t1-t0)/1e9;
}

/* lm_head in isolation: one huge Lin (N=151936, K=2048), called exactly once, unlike the smaller
 * Lins above which repeat NLAYERS times -- tests whether a single large cold call behaves
 * differently from many smaller cold calls to the same aggregate byte count. */
static double run_config_lmhead(int nt, Sched sched, LinBuf* lmhead){
    g_pool_nt=nt; g_sched=sched;
    start_pool(nt);
    g_cur_lin = lmhead;
    double t0=now();
    dispatch_round(nt);
    double t1=now();
    stop_pool(nt);
    return (double)lmhead->bytes/(t1-t0)/1e9;
}

int main(void){
    fastfill((uint8_t*)g_A, sizeof(g_A), 42);
    g_Y = malloc((size_t)151936*4); /* sized for lm_head's N -- also covers every smaller Lin */

    size_t total_bytes=0;
    for(int L=0; L<NLAYERS; L++){ build_layer(g_layers[L], 1000+L*1000); total_bytes += layer_bytes(g_layers[L]); }
    printf("shared cold working set: %.0f MB (built once, reused across all configs below)\n", total_bytes/1e6);
    printf("output buffer: now a REAL per-panel-unique write into a full N-sized array (was a\n");
    printf("tiny reused per-thread scratch before) -- isolates whether output-store footprint\n");
    printf("explains the gap to production's 26.13 GB/s.\n\n");

    for(int nt=1; nt<=4; nt*=2){
        double cyclic  = run_config(nt, SCHED_CYCLIC);
        double blocked = run_config(nt, SCHED_BLOCKED);
        printf("nt=%d: cyclic(production)=%.2f GB/s   blocked(contiguous-range)=%.2f GB/s\n", nt, cyclic, blocked);
    }

    printf("\n=== lm_head in isolation (N=151936, K=2048, one huge call vs many small ones) ===\n");
    LinBuf lmhead = make_lin(151936, 2048, 99999);
    printf("lm_head buffer: %.0f MB\n", lmhead.bytes/1e6);
    for(int nt=1; nt<=4; nt*=2){
        double cyclic  = run_config_lmhead(nt, SCHED_CYCLIC, &lmhead);
        double blocked = run_config_lmhead(nt, SCHED_BLOCKED, &lmhead);
        printf("nt=%d: cyclic=%.2f GB/s   blocked=%.2f GB/s\n", nt, cyclic, blocked);
    }
    free(lmhead.buf);

    for(int L=0; L<NLAYERS; L++) free_layer(g_layers[L]);
    return 0;
}
