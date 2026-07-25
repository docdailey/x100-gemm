/* tcm_probe2.c — rigorous TCM characterization (the 0.4 GB/s smelled wrong).
 * Fixes the serialized-accumulator artifact: 4 INDEPENDENT accumulators, no dep chain.
 * Also: dumps the mapping's VmFlags (is it really uncached?), measures read BW, write BW,
 * and load-to-use latency (pointer-chase) for TCM vs DRAM on an A100 core.
 * Build: gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o tcm_probe2 tcm_probe2.c
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

#define TCM_SIZE 0x300000UL

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

/* dependency-free read: 4 independent m2 loads + 4 independent accumulators per iter (512 B/iter) */
static void read_bw(const int8_t*p,size_t bytes,int reps,double*gbps){
    double t=now();
    for(int r=0;r<reps;r++){
        __asm__ volatile(
            "mv t1,%0\n\t" "mv t2,%1\n\t" "vsetvli t0,zero,e8,m2\n\t"
            "vmv.v.i v16,0\n\t vmv.v.i v18,0\n\t vmv.v.i v20,0\n\t vmv.v.i v22,0\n\t"
            "slli t3,t0,2\n\t"                                   /* 4*VL bytes per iter */
            "1:\n\t"
            "vle8.v v0,(t1)\n\t" "add t4,t1,t0\n\t vle8.v v2,(t4)\n\t"
            "add t4,t4,t0\n\t vle8.v v4,(t4)\n\t add t4,t4,t0\n\t vle8.v v6,(t4)\n\t"
            "vadd.vv v16,v16,v0\n\t vadd.vv v18,v18,v2\n\t vadd.vv v20,v20,v4\n\t vadd.vv v22,v22,v6\n\t"
            "add t1,t1,t3\n\t" "sub t2,t2,t3\n\t" "bgtz t2,1b\n\t"
            :: "r"(p),"r"(bytes)
            : "t0","t1","t2","t3","t4","v0","v2","v4","v6","v16","v18","v20","v22","memory");
    }
    double dt=now()-t; *gbps=(double)bytes*reps/dt/1e9;
}
/* write BW: vector stores */
static void write_bw(int8_t*p,size_t bytes,int reps,double*gbps){
    double t=now();
    for(int r=0;r<reps;r++){
        __asm__ volatile(
            "mv t1,%0\n\t" "mv t2,%1\n\t" "vsetvli t0,zero,e8,m8\n\t" "vmv.v.i v8,7\n\t"
            "1:\n\t" "vse8.v v8,(t1)\n\t" "add t1,t1,t0\n\t" "sub t2,t2,t0\n\t" "bgtz t2,1b\n\t"
            :: "r"(p),"r"(bytes) : "t0","t1","t2","v8","v9","v10","v11","v12","v13","v14","v15","memory");
    }
    double dt=now()-t; *gbps=(double)bytes*reps/dt/1e9;
}
/* latency: dependent pointer chase through a ring of stride-4KB uint64 next-pointers */
static double latency_ns(uint64_t*p,size_t n,int chase){
    volatile uint64_t idx=0; double t=now();
    for(int i=0;i<chase;i++) idx=p[idx];
    double dt=now()-t; (void)idx; return dt/chase*1e9;
}

static void dump_vmflags(void*addr){
    FILE*f=fopen("/proc/self/smaps","r"); if(!f){return;}
    char line[512]; unsigned long a=(unsigned long)addr; int in=0;
    while(fgets(line,sizeof line,f)){
        unsigned long s,e;
        if(sscanf(line,"%lx-%lx",&s,&e)==2){ in=(a>=s&&a<e); }
        if(in && (strncmp(line,"VmFlags:",8)==0)){ printf("  TCM %s",line); break; }
    }
    fclose(f);
}

int main(void){
    bind_ai(); { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    printf("hart=%d\n", sched_getcpu());

    int fd=open("/dev/tcm",O_RDWR); if(fd<0){perror("open");return 1;}
    void*tcm=mmap(NULL,TCM_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(tcm==MAP_FAILED){perror("mmap");return 1;}
    printf("mmap /dev/tcm -> %p\n", tcm);
    dump_vmflags(tcm);
    memset(tcm,1,TCM_SIZE);
    int8_t*dram=aligned_alloc(64,TCM_SIZE); memset(dram,1,TCM_SIZE);

    double g;
    read_bw(tcm,TCM_SIZE,20,&g);  read_bw((int8_t*)tcm,TCM_SIZE,500,&g);  printf("TCM  read : %7.2f GB/s\n",g);
    read_bw(dram,TCM_SIZE,500,&g);                                        printf("DRAM read : %7.2f GB/s\n",g);
    write_bw(tcm,TCM_SIZE,500,&g);                                        printf("TCM  write: %7.2f GB/s\n",g);
    write_bw(dram,TCM_SIZE,500,&g);                                       printf("DRAM write: %7.2f GB/s\n",g);

    /* latency ring: stride 4KB next-pointers */
    size_t n=TCM_SIZE/sizeof(uint64_t); uint64_t*tt=(uint64_t*)tcm,*dd=(uint64_t*)dram;
    size_t stride=4096/sizeof(uint64_t);
    for(size_t i=0;i<n;i++){ tt[i]=(i+stride)%n; dd[i]=(i+stride)%n; }
    printf("TCM  latency: %6.1f ns/access\n", latency_ns(tt,n,200000));
    printf("DRAM latency: %6.1f ns/access\n", latency_ns(dd,n,200000));
    return 0;
}
