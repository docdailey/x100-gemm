#define _GNU_SOURCE
/* Validate an RVV port of the real vendor quantize_a_row_i8_hp (rvv_kernels.cpp:1989, vlenb==128
 * branch -- VLEN=1024, matches this board's A100 cores) against the scalar pack_A_hp already
 * shipping in qwen_moe_hp.c, byte-for-byte, before considering swapping it into production.
 * Per the session's standing discipline: port vendor ground truth verbatim, validate standalone,
 * only then integrate -- same playbook as the M1 kernel ports (A1-A3) and the int8 router kernel. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

/* qwen_moe_hp.c's bind_ai(): writing "0" to /proc/set_ai_thread is what actually grants this
 * thread access to harts 8-15 (A100/IME-2) -- sched_setaffinity(CPU_SET(8)) alone silently fails
 * under this session's cgroup (cpuset 0-7) without it. Found the hard way: this probe reported
 * vlenb=32 (X100) even after pinning to hart 8, because it skipped this call. */
static int bind_ai(void){ int fd=open("/proc/set_ai_thread",O_WRONLY); if(fd<0)return -1; int r=write(fd,"0",1); close(fd); return r<0?-1:0; }

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

/* ===== existing scalar pack_A_hp, verbatim copy from qwen_moe_hp.c (the oracle) ===== */
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

/* ===== RVV port of the real vendor quantize_a_row_i8_hp, vlenb==128 branch =====
 * Ground truth: /root/llama.cpp/ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:1989-2039 (this board).
 * NOTE: the vendor code does NOT clamp after vfncvt/vncvt (relies on the per-subblock scale
 * keeping values in-range by construction); the scalar port above added an explicit clamp as a
 * defensive addition. This probe's job is to find out whether that clamp ever actually fires. */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void pack_A_hp_rvv(const float*a, uint8_t*out){
    float scale_temp[NSUB];
    float scale_avg=0.0f;
    for(int kk=0;kk<NSUB;kk++){
        size_t vl=__riscv_vsetvl_e32m1(32);
        vfloat32m1_t v_a=__riscv_vle32_v_f32m1(a+kk*32,vl);
        vfloat32m1_t v_a_abs=__riscv_vfabs_v_f32m1(v_a,vl);
        vfloat32m1_t tmp=__riscv_vfmv_v_f_f32m1(0.0f,vl);
        vfloat32m1_t v_a_max=__riscv_vfredmax_vs_f32m1_f32m1(v_a_abs,tmp,vl);
        float max_abs_a=__riscv_vfmv_f_s_f32m1_f32(v_a_max);
        if(max_abs_a<1e-6f) max_abs_a=1e-6f; /* match pack_A_hp_scalar's amax=1e-6f floor (vendor
            code omits it -- fine numerically, an all-zero block contributes 0 regardless of the
            stored scale, but this makes the port byte-identical rather than merely equivalent) */
        scale_temp[kk]=max_abs_a/127.0f;
        scale_avg+=scale_temp[kk];
    }
    scale_avg/=NSUB;
    const float scale_factor = scale_avg? 1.0f/scale_avg : 0.0f;
    uint16_t blkscale=f32_to_f16(scale_avg);
    memcpy(out+288,&blkscale,2);

    for(int kk=0;kk<NSUB;kk++){
        uint8_t*base=out+kk*34;
        size_t vl=__riscv_vsetvl_e32m1(32);
        vfloat32m1_t v_a=__riscv_vle32_v_f32m1(a+kk*32,vl);
        float rep_scale_a = scale_temp[kk]? 1.0f/scale_temp[kk] : 0.0f;
        uint16_t ssub=f32_to_f16(scale_temp[kk]*scale_factor);
        memcpy(base,&ssub,2);

        vfloat32m1_t v_a_scale = __riscv_vfmul_vf_f32m1(v_a, rep_scale_a, vl);
        vint16mf2_t  v_a_quant = __riscv_vfncvt_x_f_w_i16mf2(v_a_scale, vl);
        vint8mf4_t   v_a_quant_i8 = __riscv_vncvt_x_x_w_i8mf4(v_a_quant, vl);

        vint16m1_t tmp_sum = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t v_a_sum = __riscv_vwredsum_vs_i8mf4_i16m1(v_a_quant_i8, tmp_sum, vl);
        int16_t a_sum = __riscv_vmv_x_s_i16m1_i16(v_a_sum);
        uint16_t as = f32_to_f16(-(float)a_sum*8.0f);
        memcpy(out+272+kk*2,&as,2);

        __riscv_vse8_v_i8mf4(base+2, v_a_quant_i8, vl);
    }
}

static uint32_t rng_state=12345;
static float randf(void){ rng_state=rng_state*1103515245u+12345u; return ((int32_t)(rng_state>>8))/8388608.0f; }

int main(void){
    bind_ai();
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(8,&s); sched_setaffinity(0,sizeof(s),&s); }
    for(int i=0;i<5;i++) sched_yield();
    printf("vlenb=%zu (expect 128 for A100/VLEN=1024)\n", __riscv_vlenb());
    int ntrials=200000;
    long mismatches=0, byte_mismatches=0;
    int worst_trial=-1, worst_bytediffs=0;
    for(int t=0;t<ntrials;t++){
        float a[256];
        int mode=t%5;
        for(int i=0;i<256;i++){
            if(mode==0) a[i]=randf()*4.0f;              /* generic random */
            else if(mode==1) a[i]=randf()*0.001f;        /* tiny magnitudes */
            else if(mode==2) a[i]=(i%37==0)?randf()*50.0f:randf()*0.01f; /* one dominant outlier per subblock-ish */
            else if(mode==3) a[i]=0.0f;                   /* all-zero row */
            else a[i]=(t+i)%2? randf()*8.0f : -randf()*8.0f; /* alternating sign */
        }
        uint8_t out_s[AREC], out_v[AREC];
        memset(out_s,0xAA,AREC); memset(out_v,0x55,AREC);
        pack_A_hp_scalar(a,out_s);
        pack_A_hp_rvv(a,out_v);
        if(memcmp(out_s,out_v,AREC)!=0){
            mismatches++;
            int bd=0; for(int i=0;i<AREC;i++) if(out_s[i]!=out_v[i]) bd++;
            byte_mismatches+=bd;
            if(bd>worst_bytediffs){ worst_bytediffs=bd; worst_trial=t; }
        }
    }
    printf("trials=%d mismatched_records=%ld (%.4f%%) total_byte_diffs=%ld\n",
        ntrials, mismatches, 100.0*mismatches/ntrials, byte_mismatches);
    if(mismatches){
        printf("worst trial=%d bytediffs=%d (mode=%d)\n", worst_trial, worst_bytediffs, worst_trial%5);
    } else {
        printf("BYTE-IDENTICAL across all %d trials (5 distributions incl. all-zero, tiny-mag, outlier, alternating-sign) -- RVV port is a drop-in replacement.\n", ntrials);
    }

    /* hot timing A/B, same pattern as A3 (kernel-only, in-cache, no pack overhead) */
    {
        float a[256]; uint8_t out[AREC];
        for(int i=0;i<256;i++) a[i]=randf()*4.0f;
        int reps=2000000;
        struct timespec t0,t1;
        pack_A_hp_scalar(a,out); pack_A_hp_rvv(a,out); /* warm */
        clock_gettime(CLOCK_MONOTONIC,&t0);
        for(int r=0;r<reps;r++) pack_A_hp_scalar(a,out);
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ns_scalar=((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/reps;
        clock_gettime(CLOCK_MONOTONIC,&t0);
        for(int r=0;r<reps;r++) pack_A_hp_rvv(a,out);
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double ns_rvv=((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/reps;
        printf("hot timing (256-wide record, %d reps): scalar %.1f ns/call, RVV %.1f ns/call -- %.2fx\n",
            reps, ns_scalar, ns_rvv, ns_scalar/ns_rvv);
    }
    return 0;
}
