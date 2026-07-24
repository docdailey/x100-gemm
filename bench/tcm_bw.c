/* tcm_bw.c — can we mmap /dev/tcm directly, and how fast is it vs DRAM?
 * dmesg: "tcm 0.tcm: direct mmap phys 0x0 size 0x300000 block_size 0x60000 block_num 8".
 * Stage weights in TCM to dodge the ~20 GB/s shared-LPDDR5 wall. Measure read BW from
 * an A100 core for TCM vs a DRAM buffer of the same size.
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o tcm_bw tcm_bw.c
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

#define TCM_SIZE 0x300000UL   /* 3 MB */

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* vector sum-read of `bytes` from p, repeated; returns a checksum to prevent DCE */
static int64_t read_bw(const int8_t*p,size_t bytes,int reps,double*gbps){
    int64_t acc=0; double t=now();
    for(int r=0;r<reps;r++){
        int64_t s;
        __asm__ volatile(
            "mv t1,%1\n\t" "mv t2,%2\n\t" "vsetvli t0,zero,e8,m8\n\t"
            "vmv.v.i v8,0\n\t"
            "1:\n\t" "vle8.v v0,(t1)\n\t" "vadd.vv v8,v8,v0\n\t"
            "add t1,t1,t0\n\t" "sub t2,t2,t0\n\t" "bgtz t2,1b\n\t"
            "vsetvli t3,zero,e8,m1\n\t" "vmv.x.s %0,v8\n\t"
            : "=r"(s) : "r"(p),"r"(bytes) : "t0","t1","t2","t3","v0","v8","v9","v10","v11","v12","v13","v14","v15","memory");
        acc+=s;
    }
    double dt=now()-t; *gbps=(double)bytes*reps/dt/1e9; return acc;
}

int main(void){
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    printf("hart=%d\n", sched_getcpu());

    int fd=open("/dev/tcm",O_RDWR);
    if(fd<0){ perror("open /dev/tcm"); return 1; }
    void*tcm=mmap(NULL,TCM_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(tcm==MAP_FAILED){ perror("mmap /dev/tcm"); return 1; }
    printf("mmap /dev/tcm OK: %p (%lu KB)\n", tcm, TCM_SIZE/1024);

    /* prove read/write works */
    memset(tcm,0x5a,TCM_SIZE);
    volatile uint8_t*u=tcm; int ok=(u[0]==0x5a && u[TCM_SIZE-1]==0x5a);
    printf("write/read TCM: %s\n", ok?"OK":"FAIL");

    int8_t*dram=aligned_alloc(64,TCM_SIZE); memset(dram,0x5a,TCM_SIZE);

    double g; int reps=2000;
    read_bw(tcm,TCM_SIZE,50,&g);                 /* warm */
    read_bw((int8_t*)tcm,TCM_SIZE,reps,&g);  printf("TCM  read BW: %6.1f GB/s\n", g);
    read_bw(dram,TCM_SIZE,reps,&g);          printf("DRAM read BW: %6.1f GB/s\n", g);
    return 0;
}
