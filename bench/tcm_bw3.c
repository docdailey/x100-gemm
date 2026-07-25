/* tcm_bw3.c — measure the REAL A100-direct TCM read BW via the proper libspine_tcm
 * acquire path (cacheable), vs my earlier raw /dev/tcm mmap (uncached 0.41 GB/s) and DRAM.
 * Block IDs are affinity-bound to A100 pairs (tcm-smi: id0/1->harts8/9, ...). Bind to hart 8,
 * acquire block 0, and vle8-read it.
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o tcm_bw3 tcm_bw3.c -lspine_tcm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>
#include "spine_tcm.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* dependency-free vle8 read of `bytes`, 4 independent accumulators, reps passes */
static void read_bw(const int8_t*p,size_t bytes,int reps,double*gbps){
    double t=now();
    for(int r=0;r<reps;r++)
        __asm__ volatile(
            "mv t1,%0\n\t mv t2,%1\n\t vsetvli t0,zero,e8,m2\n\t"
            "vmv.v.i v16,0\n\t vmv.v.i v18,0\n\t vmv.v.i v20,0\n\t vmv.v.i v22,0\n\t slli t3,t0,2\n\t"
            "1:\n\t vle8.v v0,(t1)\n\t add t4,t1,t0\n\t vle8.v v2,(t4)\n\t"
            "add t4,t4,t0\n\t vle8.v v4,(t4)\n\t add t4,t4,t0\n\t vle8.v v6,(t4)\n\t"
            "vadd.vv v16,v16,v0\n\t vadd.vv v18,v18,v2\n\t vadd.vv v20,v20,v4\n\t vadd.vv v22,v22,v6\n\t"
            "add t1,t1,t3\n\t sub t2,t2,t3\n\t bgtz t2,1b\n\t"
            :: "r"(p),"r"(bytes) : "t0","t1","t2","t3","t4","v0","v2","v4","v6","v16","v18","v20","v22","memory");
    double dt=now()-t; *gbps=(double)bytes*reps/dt/1e9;
}

int main(void){
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    printf("hart=%d\n", sched_getcpu());

    if (spine_tcm_open_handle(NULL)!=0 || !spine_tcm_is_available()){ printf("spine_tcm unavailable\n"); return 1; }
    spine_tcm_mem_info_t mi; spine_tcm_mem_info(&mi);
    printf("spine_tcm: blk_size=%zu blk_num=%zu fake=%d\n", mi.blk_size, mi.blk_num, mi.is_fake_tcm);

    void*tcm = spine_tcm_mem_get(0);            /* block 0 -> harts 8/9 (my pair) */
    if(!tcm){ printf("mem_get(0) failed\n"); return 1; }
    size_t sz = mi.blk_size;
    printf("acquired block 0: %p (%zu KB)\n", tcm, sz/1024);
    memset(tcm,0x5a,sz);

    int8_t*dram=aligned_alloc(64,sz); memset(dram,0x5a,sz);
    /* raw uncached mmap for the apples-to-apples comparison */
    int fd=open("/dev/tcm",O_RDWR); void*raw=mmap(NULL,0x300000,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(raw!=MAP_FAILED) memset(raw,0x5a,sz);

    double g;
    read_bw(tcm,sz,50,&g);
    read_bw((int8_t*)tcm,sz,2000,&g);                          printf("spine_tcm (cacheable) read: %7.2f GB/s\n", g);
    read_bw(dram,sz,2000,&g);                                  printf("DRAM read                 : %7.2f GB/s\n", g);
    if(raw!=MAP_FAILED){ read_bw((int8_t*)raw,sz,300,&g);      printf("raw-mmap /dev/tcm (uncached): %6.2f GB/s\n", g); }

    spine_tcm_mem_free(0);
    return 0;
}
