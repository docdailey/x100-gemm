/* Validates the Phase 4.1 fused multi-Q QK kernel (attention_optimization_plan.md,
 * codex_recs_1.md §22.30) two ways, on deterministic synthetic data, before trusting it for a
 * production A/B:
 * (1) per-query-head numerics vs the ORIGINAL per-head loop (8 separate vdot_f32 calls per
 *     position) -- expected bit-exact, since qk8_dot uses the identical chunk boundaries,
 *     vfmacc_vv order, and final vfredusum reduction per query head as vdot_f32 does, just with
 *     the K chunk loaded once and shared across the 8 accumulators instead of reloaded inside 8
 *     separate calls. Reports max abs/rel error, not just an assertion.
 * (2) sanity that the fused path is not silently degenerate (e.g. all-zero output) by checking a
 *     nonzero fraction of scores differ from zero.
 * vdot_f32 and qk8_dot are copied verbatim from qwen_moe_hp.c (same discipline as
 * bench/swiglu_ratsig_probe.c) -- standalone, no #include of the production file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

#define QK8_MAXGPR 8

static float vdot_f32(const float*a,const float*b,int n){
    vfloat32m1_t vacc=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t va=__riscv_vle32_v_f32m1(a+i,vl), vb=__riscv_vle32_v_f32m1(b+i,vl);
        vacc=__riscv_vfmacc_vv_f32m1(vacc,va,vb,vl); i+=vl; }
    vfloat32m1_t vzero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t vsum=__riscv_vfredusum_vs_f32m1_f32m1(vacc,vzero,__riscv_vsetvlmax_e32m1());
    return __riscv_vfmv_f_s_f32m1_f32(vsum);
}
static void qk8_dot(const float*const qh[QK8_MAXGPR],const float*kj,int hd,int n,float*out){
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    vfloat32m1_t a0=z,a1=z,a2=z,a3=z,a4=z,a5=z,a6=z,a7=z;
    int i=0;
    while(i<hd){
        size_t vl=__riscv_vsetvl_e32m1(hd-i);
        vfloat32m1_t vk=__riscv_vle32_v_f32m1(kj+i,vl);
        if(n>0) a0=__riscv_vfmacc_vv_f32m1(a0,__riscv_vle32_v_f32m1(qh[0]+i,vl),vk,vl);
        if(n>1) a1=__riscv_vfmacc_vv_f32m1(a1,__riscv_vle32_v_f32m1(qh[1]+i,vl),vk,vl);
        if(n>2) a2=__riscv_vfmacc_vv_f32m1(a2,__riscv_vle32_v_f32m1(qh[2]+i,vl),vk,vl);
        if(n>3) a3=__riscv_vfmacc_vv_f32m1(a3,__riscv_vle32_v_f32m1(qh[3]+i,vl),vk,vl);
        if(n>4) a4=__riscv_vfmacc_vv_f32m1(a4,__riscv_vle32_v_f32m1(qh[4]+i,vl),vk,vl);
        if(n>5) a5=__riscv_vfmacc_vv_f32m1(a5,__riscv_vle32_v_f32m1(qh[5]+i,vl),vk,vl);
        if(n>6) a6=__riscv_vfmacc_vv_f32m1(a6,__riscv_vle32_v_f32m1(qh[6]+i,vl),vk,vl);
        if(n>7) a7=__riscv_vfmacc_vv_f32m1(a7,__riscv_vle32_v_f32m1(qh[7]+i,vl),vk,vl);
        i+=vl;
    }
    vfloat32m1_t vzero=__riscv_vfmv_v_f_f32m1(0.0f,__riscv_vsetvlmax_e32m1());
    if(n>0) out[0]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a0,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>1) out[1]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a1,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>2) out[2]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a2,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>3) out[3]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a3,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>4) out[4]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a4,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>5) out[5]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a5,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>6) out[6]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a6,vzero,__riscv_vsetvlmax_e32m1()));
    if(n>7) out[7]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(a7,vzero,__riscv_vsetvlmax_e32m1()));
}

static uint32_t rng=42;
static float randf(void){ rng=rng*1103515245u+12345u; return ((rng>>8)*(1.0f/16777216.0f))-0.5f; }

int main(void){
    const int hd=128, gpr=8;
    const int ctxs[] = {113,300,1024}; /* short/mid/long, exercising every hd chunking case (128/32=4 chunks exactly, no remainder, but varying ctx exercises the position loop bound itself, and the plan asks for a deterministic per-layer test, not tied to one specific length) */
    const float scale=1.0f/sqrtf((float)hd);
    long total_cmp=0, mismatches=0;
    double max_abs=0, max_rel=0;
    for(int ci=0; ci<3; ci++){
        int ctx=ctxs[ci];
        float*Q=malloc((size_t)gpr*hd*4);
        float*K=malloc((size_t)ctx*hd*4);
        for(int i=0;i<gpr*hd;i++) Q[i]=randf();
        for(int i=0;i<ctx*hd;i++) K[i]=randf();
        const float*qh[QK8_MAXGPR];
        for(int qi=0;qi<gpr;qi++) qh[qi]=Q+(size_t)qi*hd;

        float*orig=malloc((size_t)gpr*ctx*4);   /* [qi*ctx+j] */
        float*fused=malloc((size_t)gpr*ctx*4);

        for(int qi=0;qi<gpr;qi++)
            for(int j=0;j<ctx;j++)
                orig[(size_t)qi*ctx+j]=vdot_f32(qh[qi],K+(size_t)j*hd,hd)*scale;

        for(int j=0;j<ctx;j++){
            float out8[QK8_MAXGPR];
            qk8_dot(qh,K+(size_t)j*hd,hd,gpr,out8);
            for(int qi=0;qi<gpr;qi++) fused[(size_t)qi*ctx+j]=out8[qi]*scale;
        }

        double ctx_max_abs=0, ctx_max_rel=0; long ctx_cmp=0, ctx_mismatch=0;
        for(int qi=0;qi<gpr;qi++){
            for(int j=0;j<ctx;j++){
                double o=orig[(size_t)qi*ctx+j], f=fused[(size_t)qi*ctx+j];
                double ad=fabs(o-f);
                double rd=fabs(o)>1e-12 ? ad/fabs(o) : ad;
                if(ad>ctx_max_abs) ctx_max_abs=ad;
                if(rd>ctx_max_rel) ctx_max_rel=rd;
                if(ad>1e-4) ctx_mismatch++;
                ctx_cmp++;
            }
        }
        printf("ctx=%4d: comparisons=%ld mismatches(>1e-4 abs)=%ld max_abs=%e max_rel=%e -- %s\n",
            ctx, ctx_cmp, ctx_mismatch, ctx_max_abs, ctx_max_rel, ctx_mismatch==0?"PASS":"FAIL");
        total_cmp+=ctx_cmp; mismatches+=ctx_mismatch;
        if(ctx_max_abs>max_abs) max_abs=ctx_max_abs;
        if(ctx_max_rel>max_rel) max_rel=ctx_max_rel;

        /* sanity: fused output isn't degenerate (e.g. always zero from a wiring bug) */
        long nonzero=0;
        for(long i=0;i<(long)gpr*ctx;i++) if(fabsf(fused[i])>1e-6f) nonzero++;
        if(nonzero < (long)gpr*ctx/2){
            printf("  WARNING: only %ld/%ld fused outputs are nonzero -- possible wiring bug, not just numerical noise\n", nonzero, (long)gpr*ctx);
        }

        free(Q); free(K); free(orig); free(fused);
    }
    printf("=== TOTAL: comparisons=%ld mismatches=%ld max_abs=%e max_rel=%e -- %s ===\n",
        total_cmp, mismatches, max_abs, max_rel, mismatches==0?"PASS":"FAIL");
    return mismatches==0 ? 0 : 1;
}
