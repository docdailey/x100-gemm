#define _GNU_SOURCE
/* Real achievable concurrent sequential-read bandwidth from nt=4 A100 harts (same pinning as
 * production qwen_moe_hp's pool: harts 8,10,12,14), to compare against linear(kernel)'s measured
 * ~3.26 GB/s effective bandwidth -- is that close to the hardware ceiling, or is there real
 * headroom left on the table? Measure before optimizing the hottest, most validated kernel in the
 * file (per PROGRESS.md's subclass breakdown: expert gate/up/down is 61% of linear(kernel)). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

#define BUFSZ (256UL*1024*1024)  /* 256MB per thread -- far bigger than any cache level */
#define REPS 4
static int harts[4] = {8,10,12,14};
static char* bufs[4];
static volatile uint64_t sinks[4];
static double t_lo[4], t_hi[4];
static pthread_barrier_t bar;

static void* worker(void* arg){
    long id = (long)arg;
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(harts[id],&cs); sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
    char* buf = bufs[id];
    pthread_barrier_wait(&bar); /* all threads pinned before any starts timing */
    uint64_t sink=0;
    t_lo[id]=now();
    for(int r=0;r<REPS;r++)
        for(size_t off=0; off<BUFSZ; off+=64) sink += *(volatile uint64_t*)(buf+off);
    t_hi[id]=now();
    sinks[id]=sink;
    return NULL;
}

static double solo_run(int hart){
    bind_ai();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hart,&cs); sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<5;i++) sched_yield();
    char* buf = malloc(BUFSZ); memset(buf,1,BUFSZ);
    uint64_t sink=0;
    double t0=now();
    for(int r=0;r<REPS;r++)
        for(size_t off=0; off<BUFSZ; off+=64) sink += *(volatile uint64_t*)(buf+off);
    double t1=now();
    free(buf);
    if(sink==0xdeadbeef) printf("(unreachable, keeps sink live)\n");
    return (double)BUFSZ*REPS/(t1-t0);
}

int main(void){
    printf("=== solo (1 thread, hart 8) streaming-read bandwidth ===\n");
    double solo = solo_run(8);
    printf("solo: %.2f GB/s\n", solo/1e9);

    printf("=== concurrent nt=4 (harts 8,10,12,14) streaming-read bandwidth ===\n");
    for(int i=0;i<4;i++){ bufs[i]=malloc(BUFSZ); memset(bufs[i],(int)(i+1),BUFSZ); }
    pthread_barrier_init(&bar,NULL,4);
    pthread_t th[4];
    for(long i=0;i<4;i++) pthread_create(&th[i],NULL,worker,(void*)i);
    for(int i=0;i<4;i++) pthread_join(th[i],NULL);

    double lo=t_lo[0], hi=t_hi[0];
    uint64_t total_bytes=0;
    for(int i=0;i<4;i++){
        if(t_lo[i]<lo) lo=t_lo[i];
        if(t_hi[i]>hi) hi=t_hi[i];
        double per_thread_gbps = (double)BUFSZ*REPS/(t_hi[i]-t_lo[i])/1e9;
        printf("  thread %d (hart %d): %.2f GB/s\n", i, harts[i], per_thread_gbps);
        total_bytes += (uint64_t)BUFSZ*REPS;
    }
    double aggregate = (double)total_bytes/(hi-lo)/1e9;
    printf("aggregate (sum bytes / span): %.2f GB/s\n", aggregate);
    printf("\n=== comparison ===\n");
    printf("linear(kernel) measured effective bandwidth: 3.26 GB/s (191.1MB/tok @ 58.6ms)\n");
    printf("this probe's nt=4 concurrent ceiling: %.2f GB/s -> linear(kernel) is using %.0f%% of it\n",
        aggregate, 100.0*3.26/aggregate);
    return 0;
}
