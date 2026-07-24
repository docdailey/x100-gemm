/* a100_probe.c — pin to a given CPU, report VLEN + vmadot throughput.
 * Used to test whether the A100 AI cores (harts 8-15) are reachable and what
 * their vector width / IME throughput is vs the X100 cores. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <time.h>

static void vmadot_loop(int8_t *a, int8_t *b, uint64_t n){
  __asm__ volatile("vsetvli t0,zero,e8,m1\n\t" "vle8.v v0,(%0)\n\t" "vle8.v v1,(%1)\n\t"
    "1:\n\t" "vmadot v28,v0,v1\n\t" "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
    : "+r"(a),"+r"(b),"+r"(n) :: "t0","v0","v1","v28","v29","cc","memory");
}
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int c, char **v){
  int cpu = (c>1)? atoi(v[1]) : 0;
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s);
  int r = sched_setaffinity(0,sizeof(s),&s);
  printf("cpu%-2d affinity=%-4s on cpu %-2d ", cpu, r?"FAIL":"ok", sched_getcpu());
  if (r){ printf("\n"); return 1; }
  unsigned long vb; __asm__ volatile("csrr %0,vlenb":"=r"(vb));
  printf("VLEN=%lu ", vb*8);
  int8_t a[512],b[512]; for(int i=0;i<512;i++){a[i]=(int8_t)(i-256); b[i]=(int8_t)((i*3)%7-3);}
  uint64_t it = 400000000ULL;
  double t=now(); vmadot_loop(a,b,it); double dt=now()-t;
  printf("| %.2e vmadot/s | %.1f GOP/s (assuming 128MAC/tile)\n", it/dt, (double)it*128*2/dt/1e9);
  return 0;
}
