/* ai_thread_probe.c — reach the A100 AI cores the SANCTIONED way.
 *
 * The A100 cores (harts 8-15, 1024-bit RVV + IME-2) reject sched_setaffinity/taskset
 * (EINVAL). SpaceMIT's backend instead writes "0" to /proc/set_ai_thread, and the
 * kernel migrates the calling thread onto an AI core. This probe replicates that and
 * measures what actually changed: which hart we land on, the VLEN, and vmadot rate.
 *
 * Build (on board): gcc -O3 -march=rv64gcv_zvfh_xsmtvdotii -o ai_thread_probe ai_thread_probe.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9; }

static unsigned long vlen_bits(void){
    unsigned long vb; __asm__ volatile("csrr %0,vlenb":"=r"(vb)); return vb*8; }

/* pure vmadot issue rate (int8 matrix tile), looped n times */
static void vmadot_loop(int8_t *a, int8_t *b, uint64_t n){
    __asm__ volatile(
        "vsetvli t0,zero,e8,m1\n\t" "vle8.v v0,(%0)\n\t" "vle8.v v1,(%1)\n\t"
        "1:\n\t" "vmadot v28,v0,v1\n\t" "addi %2,%2,-1\n\t" "bnez %2,1b\n\t"
        : "+r"(a),"+r"(b),"+r"(n) :: "t0","v0","v1","v28","v29","cc","memory");
}

static int bind_ai_thread(void){
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0){ perror("open /proc/set_ai_thread"); return -1; }
    int r = write(fd, "0", 1);
    close(fd);
    if (r < 0){ perror("write"); return -1; }
    return 0;
}

int main(void){
    printf("BEFORE bind: hart=%d  VLEN=%lu\n", sched_getcpu(), vlen_bits());

    if (bind_ai_thread() != 0){ printf("bind_ai_thread failed\n"); return 1; }
    /* give the scheduler a moment to migrate us */
    for (int i=0;i<3;i++) sched_yield();

    int hart = sched_getcpu();
    unsigned long vl = vlen_bits();
    printf("AFTER  bind: hart=%d  VLEN=%lu   %s\n", hart, vl,
           (hart>=8 ? ">>> ON AN A100 AI CORE <<<" : "(still on X100)"));

    int8_t a[512], b[512];
    for (int i=0;i<512;i++){ a[i]=(int8_t)(i-256); b[i]=(int8_t)((i*3)%7-3); }
    uint64_t it = 400000000ULL;
    double t=now(); vmadot_loop(a,b,it); double dt=now()-t;

    /* MAC/tile scales with VLEN: 256-bit=4x8x4=128 MAC; scale linearly by VLEN. */
    double mac_per_tile = 128.0 * (double)vl / 256.0;
    printf("vmadot: %.3e vmadot/s | hart=%d after loop | VLEN=%lu | ~%.1f GOP/s (%.0f MAC/tile)\n",
           it/dt, sched_getcpu(), vlen_bits(), it*mac_per_tile*2/dt/1e9, mac_per_tile);
    return 0;
}
