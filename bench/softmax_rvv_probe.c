/* Validates the Phase 5 RVV-vectorized softmax (attention_optimization_plan.md,
 * codex_recs_1.md §22.32) against the original fully-scalar softmax, on deterministic synthetic
 * data, before trusting it for a production A/B. expf is NOT touched by this change -- only the
 * max reduction (before exp) and the final normalization (after exp) are RVV candidates, per
 * explicit instruction ("Keep expf exact... No approximate exponential"). Reports max abs/rel
 * error, not just an assertion. rvv_max_f32/rvv_scale_f32 are copied verbatim from qwen_moe_hp.c
 * (same discipline as bench/qk_multiq_probe.c and bench/av_multiq_probe.c) -- standalone, no
 * #include of the production file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

static float rvv_max_f32(const float*x,int n){
    vfloat32m1_t vacc=__riscv_vfmv_v_f_f32m1(-1e30f,__riscv_vsetvlmax_e32m1());
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl);
        vacc=__riscv_vfmax_vv_f32m1(vacc,vx,vl); i+=vl; }
    vfloat32m1_t vid=__riscv_vfmv_v_f_f32m1(-1e30f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t vred=__riscv_vfredmax_vs_f32m1_f32m1(vacc,vid,__riscv_vsetvlmax_e32m1());
    return __riscv_vfmv_f_s_f32m1_f32(vred);
}
static void rvv_scale_f32(float*x,float s,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl);
        vx=__riscv_vfmul_vf_f32m1(vx,s,vl);
        __riscv_vse32_v_f32m1(x+i,vx,vl); i+=vl; }
}
static void softmax_scalar(float*x,int n){
    float m=-1e30f; for(int i=0;i<n;i++)if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    float inv=1.0f/s; for(int i=0;i<n;i++)x[i]*=inv;
}
static void softmax_rvv(float*x,int n){
    float m=rvv_max_f32(x,n);
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    rvv_scale_f32(x,1.0f/s,n);
}

static uint32_t rng=42;
static float randf(void){ rng=rng*1103515245u+12345u; return ((rng>>8)*(1.0f/16777216.0f))-0.5f; }

int main(void){
    /* n=1 and n=33 exercise the tail (non-multiple-of-32) case; n=32/1024 exercise exact-multiple
     * cases; the rest match the ctx values used elsewhere in this session's A/Bs. */
    const int ns[] = {1, 12, 32, 33, 113, 128, 300, 512, 1024};
    long total_cmp=0, mismatches=0;
    double max_abs=0, max_rel=0;
    for(int ni=0; ni<9; ni++){
        int n=ns[ni];
        float*a=malloc((size_t)n*4);
        float*b=malloc((size_t)n*4);
        /* realistic QK-score-like magnitudes: scaled dot products, not tiny/huge outliers */
        for(int i=0;i<n;i++){ float v=randf()*20.0f; a[i]=v; b[i]=v; }

        softmax_scalar(a,n);
        softmax_rvv(b,n);

        double n_max_abs=0, n_max_rel=0; long n_cmp=0, n_mismatch=0;
        double sum_rvv=0;
        for(int i=0;i<n;i++){
            double sa=a[i], sb=b[i];
            double ad=fabs(sa-sb);
            double rd=fabs(sa)>1e-12 ? ad/fabs(sa) : ad;
            if(ad>n_max_abs) n_max_abs=ad;
            if(rd>n_max_rel) n_max_rel=rd;
            if(ad>1e-4) n_mismatch++;
            n_cmp++;
            sum_rvv+=sb;
        }
        printf("n=%5d: comparisons=%ld mismatches(>1e-4 abs)=%ld max_abs=%e max_rel=%e sum_rvv=%.6f -- %s\n",
            n, n_cmp, n_mismatch, n_max_abs, n_max_rel, sum_rvv, n_mismatch==0?"PASS":"FAIL");
        total_cmp+=n_cmp; mismatches+=n_mismatch;
        if(n_max_abs>max_abs) max_abs=n_max_abs;
        if(n_max_rel>max_rel) max_rel=n_max_rel;

        if(fabs(sum_rvv-1.0)>1e-3){
            printf("  WARNING: RVV softmax output sums to %.6f, not ~1.0 -- possible wiring bug\n", sum_rvv);
        }

        free(a); free(b);
    }
    printf("=== TOTAL: comparisons=%ld mismatches=%ld max_abs=%e max_rel=%e -- %s ===\n",
        total_cmp, mismatches, max_abs, max_rel, mismatches==0?"PASS":"FAIL");
    return mismatches==0 ? 0 : 1;
}
