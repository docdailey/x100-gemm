/* Validates the Phase 4.2 fused multi-Q AV kernel (attention_optimization_plan.md,
 * codex_recs_1.md §22.31) two ways, on deterministic synthetic data, before trusting it for a
 * production A/B:
 * (1) per-query-head numerics vs the ORIGINAL per-head loop (vaxpy_f32 called once per (query
 *     head, position) pair) -- expected bit-exact, since av8_chunk accumulates each query head's
 *     output in the same position-ascending order as the unfused loop, just holding the running
 *     sum in a register instead of round-tripping it through memory each iteration (IEEE-754
 *     addition order, not storage location, determines the result). Reports max abs/rel error,
 *     not just an assertion.
 * (2) sanity that the fused path is not silently degenerate (e.g. all-zero output) by checking a
 *     nonzero fraction of outputs differ from zero.
 * vaxpy_f32 and av8_chunk are copied verbatim from qwen_moe_hp.c (same discipline as
 * bench/qk_multiq_probe.c) -- standalone, no #include of the production file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

#define QK8_MAXGPR 8

static void vaxpy_f32(float*y,const float*x,float scale,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(x+i,vl), vy=__riscv_vle32_v_f32m1(y+i,vl);
        vy=__riscv_vfmacc_vf_f32m1(vy,scale,vx,vl);
        __riscv_vse32_v_f32m1(y+i,vy,vl); i+=vl; }
}
static void av8_chunk(const float*const scw[QK8_MAXGPR],const float*Vh,int hd,int pos,int gpr,
        float*const oh[QK8_MAXGPR],int coff,size_t vl){
    vfloat32m1_t z=__riscv_vfmv_v_f_f32m1(0.0f,vl);
    vfloat32m1_t a0=z,a1=z,a2=z,a3=z,a4=z,a5=z,a6=z,a7=z;
    for(int j=0;j<=pos;j++){
        vfloat32m1_t vv=__riscv_vle32_v_f32m1(Vh+(size_t)j*hd+coff,vl);
        if(gpr>0) a0=__riscv_vfmacc_vf_f32m1(a0,scw[0][j],vv,vl);
        if(gpr>1) a1=__riscv_vfmacc_vf_f32m1(a1,scw[1][j],vv,vl);
        if(gpr>2) a2=__riscv_vfmacc_vf_f32m1(a2,scw[2][j],vv,vl);
        if(gpr>3) a3=__riscv_vfmacc_vf_f32m1(a3,scw[3][j],vv,vl);
        if(gpr>4) a4=__riscv_vfmacc_vf_f32m1(a4,scw[4][j],vv,vl);
        if(gpr>5) a5=__riscv_vfmacc_vf_f32m1(a5,scw[5][j],vv,vl);
        if(gpr>6) a6=__riscv_vfmacc_vf_f32m1(a6,scw[6][j],vv,vl);
        if(gpr>7) a7=__riscv_vfmacc_vf_f32m1(a7,scw[7][j],vv,vl);
    }
    if(gpr>0) __riscv_vse32_v_f32m1(oh[0]+coff,a0,vl);
    if(gpr>1) __riscv_vse32_v_f32m1(oh[1]+coff,a1,vl);
    if(gpr>2) __riscv_vse32_v_f32m1(oh[2]+coff,a2,vl);
    if(gpr>3) __riscv_vse32_v_f32m1(oh[3]+coff,a3,vl);
    if(gpr>4) __riscv_vse32_v_f32m1(oh[4]+coff,a4,vl);
    if(gpr>5) __riscv_vse32_v_f32m1(oh[5]+coff,a5,vl);
    if(gpr>6) __riscv_vse32_v_f32m1(oh[6]+coff,a6,vl);
    if(gpr>7) __riscv_vse32_v_f32m1(oh[7]+coff,a7,vl);
}

static uint32_t rng=42;
static float randf(void){ rng=rng*1103515245u+12345u; return ((rng>>8)*(1.0f/16777216.0f))-0.5f; }

int main(void){
    const int hd=128, gpr=8;
    const int ctxs[] = {113,300,1024}; /* same set as qk_multiq_probe.c, exercising every hd
        chunking case (128/32=4 chunks exactly) plus varying the position-loop bound itself */
    long total_cmp=0, mismatches=0;
    double max_abs=0, max_rel=0;
    for(int ci=0; ci<3; ci++){
        int ctx=ctxs[ci];
        float*V=malloc((size_t)ctx*hd*4);
        for(int i=0;i<ctx*hd;i++) V[i]=randf();
        /* softmax-like weight rows: don't need real softmax normalization for an arithmetic-
         * equivalence test, just gpr independent weight sequences of length ctx feeding both
         * computations identically */
        float*sc[QK8_MAXGPR];
        for(int qi=0;qi<gpr;qi++){ sc[qi]=malloc((size_t)ctx*4); for(int j=0;j<ctx;j++) sc[qi][j]=randf(); }

        float*orig[QK8_MAXGPR]; float*fused[QK8_MAXGPR];
        for(int qi=0;qi<gpr;qi++){ orig[qi]=calloc(hd,4); fused[qi]=malloc(hd*4); }

        for(int qi=0;qi<gpr;qi++)
            for(int j=0;j<ctx;j++)
                vaxpy_f32(orig[qi],V+(size_t)j*hd,sc[qi][j],hd);

        {
            const float*scw[QK8_MAXGPR]; float*ohp[QK8_MAXGPR];
            for(int qi=0;qi<gpr;qi++){ scw[qi]=sc[qi]; ohp[qi]=fused[qi]; }
            int coff=0;
            while(coff<hd){
                size_t vl=__riscv_vsetvl_e32m1(hd-coff);
                av8_chunk(scw,V,hd,ctx-1,gpr,ohp,coff,vl);
                coff+=(int)vl;
            }
        }

        double ctx_max_abs=0, ctx_max_rel=0; long ctx_cmp=0, ctx_mismatch=0;
        for(int qi=0;qi<gpr;qi++){
            for(int t=0;t<hd;t++){
                double o=orig[qi][t], f=fused[qi][t];
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

        long nonzero=0;
        for(int qi=0;qi<gpr;qi++) for(int t=0;t<hd;t++) if(fabsf(fused[qi][t])>1e-6f) nonzero++;
        if(nonzero < (long)gpr*hd/2){
            printf("  WARNING: only %ld/%ld fused outputs are nonzero -- possible wiring bug, not just numerical noise\n", nonzero, (long)gpr*hd);
        }

        free(V);
        for(int qi=0;qi<gpr;qi++){ free(sc[qi]); free(orig[qi]); free(fused[qi]); }
    }
    printf("=== TOTAL: comparisons=%ld mismatches=%ld max_abs=%e max_rel=%e -- %s ===\n",
        total_cmp, mismatches, max_abs, max_rel, mismatches==0?"PASS":"FAIL");
    return mismatches==0 ? 0 : 1;
}
