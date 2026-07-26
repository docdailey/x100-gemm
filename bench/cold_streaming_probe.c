#define _GNU_SOURCE
/* Decisive cold-working-set probe, per Codex's design after the run_hp_m1/pool-dispatch probes
 * both scaled ~3.9x on small REUSED (cache-resident) buffers -- that ruled out kernel/tensor-unit
 * contention and dispatch overhead, but neither probe touched a working set anywhere near
 * production's real size, so "cold access is the remaining cause" was still a guess, not a
 * measurement. This probe: >=256MB/thread (far past any cache level), each N32xK256 superblock
 * touched exactly ONCE per timed pass (no reuse), nt=1/2/4, no dispatch (isolate raw cold
 * throughput first). Four variants to separate "cold access is just slow" from "production's
 * specific layout/order/page-size is the problem":
 *   A. contiguous  -- one big buffer per thread, walked sequentially start to end
 *   B. production   -- separate malloc per Lin (q,k,v,o,eg/eu/ed x8), production sizes, visited
 *                       in the real per-layer call order, repeated over simulated layers
 *   C. randomized   -- same buffer set as B, but the per-layer visit order is shuffled
 *   D. hugepage     -- same as B, but madvise(MADV_HUGEPAGE) on every buffer before touching
 * Reports WALL bandwidth for a single cold pass, not repeated-rep hot-cache GB/s. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sched.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

#define BREC 576
#define NSUB 8
#define BSUPER (NSUB*BREC) /* 4608 */
#define AREC 290
#define THREAD_BUDGET (256UL*1024*1024) /* >= 256MB/thread, per the request */

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

static uint32_t xs_state=1;
static void fastfill(uint8_t* p, size_t n, uint32_t seed){
    uint32_t s = seed?seed:1;
    for(size_t i=0;i<n;i+=4){ s^=s<<13; s^=s>>17; s^=s<<5; memcpy(p+i, &s, (n-i)>=4?4:(n-i)); }
}
/* fixed "A" activation record, reused across every kernel call (matches real usage -- one
 * activation panel gets reused across every N-panel of a given Lin). Sized for the largest kb
 * used anywhere (o: K=4096 -> kb=16) -- run_hp_m1 reads kb*AREC bytes from A; a smaller buffer
 * here was an out-of-bounds read for o's calls (root cause of the first segfault). */
static uint8_t g_A[AREC*16];

/* one Lin's worth of B data: production N,K -> Np panels x kb superblocks x BSUPER bytes */
typedef struct { uint8_t* buf; size_t bytes; int Np, kb; } LinBuf;

static LinBuf make_lin(int N, int K, uint32_t seed){
    int Np=N/32, kb=K/256;
    size_t bytes=(size_t)Np*kb*BSUPER;
    uint8_t* p = malloc(bytes);
    fastfill(p, bytes, seed);
    LinBuf lb = { p, bytes, Np, kb };
    return lb;
}
static void walk_lin(LinBuf* lb, float* dst){
    for(int np=0; np<lb->Np; np++)
        run_hp_m1(g_A, lb->buf+(size_t)np*lb->kb*BSUPER, dst, lb->kb);
}

/* one simulated layer's Lin set, production shapes: d=2048 qd=4096 kvd=512 moe_ffn=768 */
#define NLINS_PER_LAYER (4+8*3) /* q,k,v,o + 8x(eg,eu,ed) = 28 */
static void build_layer(LinBuf out[NLINS_PER_LAYER], uint32_t seedbase){
    int i=0;
    out[i++]=make_lin(4096,2048,seedbase+1);  /* q */
    out[i++]=make_lin(512, 2048,seedbase+2);  /* k */
    out[i++]=make_lin(512, 2048,seedbase+3);  /* v */
    out[i++]=make_lin(2048,4096,seedbase+4);  /* o */
    for(int e=0;e<8;e++){
        out[i++]=make_lin(768,2048,seedbase+10+e*3+0); /* eg */
        out[i++]=make_lin(768,2048,seedbase+10+e*3+1); /* eu */
        out[i++]=make_lin(2048,768,seedbase+10+e*3+2); /* ed */
    }
}
static void free_layer(LinBuf lin[NLINS_PER_LAYER]){ for(int i=0;i<NLINS_PER_LAYER;i++) free(lin[i].buf); }
static size_t layer_bytes(LinBuf lin[NLINS_PER_LAYER]){ size_t s=0; for(int i=0;i<NLINS_PER_LAYER;i++) s+=lin[i].bytes; return s; }

static void shuffle_order(int* order, int n, uint32_t seed){
    uint32_t s=seed?seed:1;
    for(int i=n-1;i>0;i--){ s^=s<<13; s^=s>>17; s^=s<<5; int j=s%(i+1); int t=order[i]; order[i]=order[j]; order[j]=t; }
}

typedef enum { VARIANT_CONTIGUOUS, VARIANT_PRODUCTION, VARIANT_RANDOM, VARIANT_HUGEPAGE } Variant;

typedef struct { int tid; Variant variant; double gbps; size_t bytes_done; } WArg;

static void* worker(void* arg){
    WArg* w = (WArg*)arg;
    int hart_order[4] = {8,10,12,14};
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart_order[w->tid],&cs); sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
    float dst[32];
    uint32_t seed = 1000*(w->tid+1);

    if(w->variant==VARIANT_CONTIGUOUS){
        size_t bytes=THREAD_BUDGET - (THREAD_BUDGET % BSUPER);
        uint8_t* buf = malloc(bytes);
        fastfill(buf, bytes, seed);
        double t0=now();
        for(size_t off=0; off<bytes; off+=8*BSUPER) /* kb=8 per call, matching KB elsewhere */
            run_hp_m1(g_A, buf+off, dst, 8);
        double t1=now();
        w->gbps = (double)bytes/(t1-t0)/1e9;
        w->bytes_done = bytes;
        free(buf);
        return NULL;
    }

    /* PRODUCTION / RANDOM / HUGEPAGE: repeated fresh layers until >= THREAD_BUDGET, all built
     * (fill included) BEFORE timing starts -- timing covers only the single cold read pass. */
    int nlayers = (int)((THREAD_BUDGET + 30*1024*1024 - 1) / (30UL*1024*1024)) + 1; /* ~31MB/layer */
    LinBuf** layers = malloc(sizeof(LinBuf*)*nlayers);
    int order[NLINS_PER_LAYER]; for(int i=0;i<NLINS_PER_LAYER;i++) order[i]=i;
    size_t total=0;
    for(int L=0; L<nlayers; L++){
        layers[L] = malloc(sizeof(LinBuf)*NLINS_PER_LAYER);
        build_layer(layers[L], seed + L*1000);
        if(w->variant==VARIANT_HUGEPAGE)
            for(int i=0;i<NLINS_PER_LAYER;i++) madvise(layers[L][i].buf, layers[L][i].bytes, MADV_HUGEPAGE);
        total += layer_bytes(layers[L]);
    }

    double t0=now();
    for(int L=0; L<nlayers; L++){
        int visit[NLINS_PER_LAYER]; memcpy(visit, order, sizeof(order));
        if(w->variant==VARIANT_RANDOM) shuffle_order(visit, NLINS_PER_LAYER, seed+L*7+1);
        for(int i=0;i<NLINS_PER_LAYER;i++) walk_lin(&layers[L][visit[i]], dst);
    }
    double t1=now();
    w->gbps = (double)total/(t1-t0)/1e9;
    w->bytes_done = total;

    for(int L=0; L<nlayers; L++){ free_layer(layers[L]); free(layers[L]); }
    free(layers);
    return NULL;
}

static const char* vname(Variant v){
    switch(v){ case VARIANT_CONTIGUOUS: return "contiguous"; case VARIANT_PRODUCTION: return "production-order";
        case VARIANT_RANDOM: return "randomized-order"; default: return "hugepage"; }
}

int main(void){
    fastfill((uint8_t*)g_A, sizeof(g_A), 42); /* shared activation record, built once */
    Variant variants[4] = { VARIANT_CONTIGUOUS, VARIANT_PRODUCTION, VARIANT_RANDOM, VARIANT_HUGEPAGE };
    for(int vi=0; vi<4; vi++){
        Variant v = variants[vi];
        for(int nt=1; nt<=4; nt*=2){
            pthread_t th[4]; WArg args[4];
            for(int i=0;i<nt;i++){ args[i].tid=i; args[i].variant=v; pthread_create(&th[i],NULL,worker,&args[i]); }
            for(int i=0;i<nt;i++) pthread_join(th[i],NULL);
            double agg=0; size_t totalbytes=0;
            for(int i=0;i<nt;i++){ agg+=args[i].gbps; totalbytes+=args[i].bytes_done; }
            printf("%-18s nt=%d: aggregate=%.2f GB/s (bytes/thread~%.0fMB)\n",
                vname(v), nt, agg, args[0].bytes_done/1e6);
        }
        printf("\n");
    }
    return 0;
}
