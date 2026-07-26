/* Validates the RVV hard-swish implementation against a scalar reference of the SAME formula --
 * this checks vectorization correctness, NOT approximation quality (that's the multi-prompt
 * harness's job, codex_recs_1.md §22.15/17/18). Unlike every other RVV port in this file,
 * hard-swish is a deliberate lossy approximation of the real SwiGLU, so there's no bit-exact
 * oracle to check against -- only "does the vector code compute exactly what the scalar hard-swish
 * formula says." */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

static void swiglu_hswish_scalar(const float*g,const float*u,float*out,int n){
    for(int i=0;i<n;i++){
        float x=g[i];
        float h=(x+3.0f)/6.0f; if(h<0.0f)h=0.0f; if(h>1.0f)h=1.0f;
        out[i]=x*h*u[i];
    }
}
static void swiglu_hswish_rvv(const float*g,const float*u,float*out,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(g+i,vl), vu=__riscv_vle32_v_f32m1(u+i,vl);
        vfloat32m1_t vh=__riscv_vfadd_vf_f32m1(vx,3.0f,vl);
        vh=__riscv_vfmul_vf_f32m1(vh,1.0f/6.0f,vl);
        vh=__riscv_vfmax_vf_f32m1(vh,0.0f,vl);
        vh=__riscv_vfmin_vf_f32m1(vh,1.0f,vl);
        vfloat32m1_t vy=__riscv_vfmul_vv_f32m1(vx,vh,vl);
        vy=__riscv_vfmul_vv_f32m1(vy,vu,vl);
        __riscv_vse32_v_f32m1(out+i,vy,vl); i+=vl; }
}

static uint32_t rng=1;
static float randf(void){ rng=rng*1103515245u+12345u; return ((int32_t)(rng>>8))/8388608.0f; }

int main(void){
    int ntrials=50000, n=768; /* moe_ffn width, the real per-call size */
    float*g=malloc(n*4),*u=malloc(n*4),*out_s=malloc(n*4),*out_v=malloc(n*4);
    long mismatches=0; float max_abs_diff=0;
    for(int t=0;t<ntrials;t++){
        int mode=t%4;
        for(int i=0;i<n;i++){
            if(mode==0) g[i]=(randf()-0.5f)*20.0f;      /* wide range, spans the clamp region */
            else if(mode==1) g[i]=(randf()-0.5f)*0.01f; /* near zero */
            else if(mode==2) g[i]=0.0f;                  /* exact zero */
            else g[i]=(randf()-0.5f)*1000.0f;             /* extreme, saturates the clamp hard */
            u[i]=(randf()-0.5f)*10.0f;
        }
        swiglu_hswish_scalar(g,u,out_s,n);
        swiglu_hswish_rvv(g,u,out_v,n);
        for(int i=0;i<n;i++){
            float d=fabsf(out_s[i]-out_v[i]);
            if(d>max_abs_diff) max_abs_diff=d;
            if(d>1e-5f) mismatches++;
        }
    }
    printf("trials=%d width=%d total_elems=%ld mismatches(>1e-5)=%ld max_abs_diff=%e\n",
        ntrials, n, (long)ntrials*n, mismatches, max_abs_diff);
    printf(mismatches==0 ? "PASS -- RVV vectorization is a correct implementation of the hard-swish formula\n"
                          : "FAIL -- RVV implementation diverges from the scalar formula\n");
    return mismatches==0 ? 0 : 1;
}
