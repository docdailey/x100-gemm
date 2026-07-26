/* Validates the rational-Pade sigmoid SwiGLU candidate (codex_recs_1.md §22.20) two ways:
 * (1) RVV vs scalar implementation of the SAME rational formula -- vectorization correctness,
 *     same discipline as bench/swiglu_hswish_probe.c.
 * (2) Numerical closeness to TRUE SiLU (expf-based) -- unlike hard-swish, this candidate is
 *     supposed to approximate the actual function Qwen3 was trained against, so (unlike
 *     hard-swish, which is a different function by construction) this check is meaningful and
 *     directly comparable against hard-swish's error on the same input distributions. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

static void swiglu_exact_scalar(const float*g,const float*u,float*out,int n){
    for(int i=0;i<n;i++){ float x=g[i]; out[i]=(x/(1.0f+expf(-x)))*u[i]; }
}
static void swiglu_hswish_scalar(const float*g,const float*u,float*out,int n){
    for(int i=0;i<n;i++){ float x=g[i]; float h=(x+3.0f)/6.0f; if(h<0.0f)h=0.0f; if(h>1.0f)h=1.0f; out[i]=x*h*u[i]; }
}
static void swiglu_ratsig_scalar(const float*g,const float*u,float*out,int n){
    for(int i=0;i<n;i++){
        float x=g[i]; float hx=x*0.5f; float hx2=hx*hx;
        float num=hx*(27.0f+hx2), den=27.0f+9.0f*hx2;
        float t=num/den; if(t<-1.0f)t=-1.0f; if(t>1.0f)t=1.0f;
        float sig=0.5f*(1.0f+t);
        out[i]=x*sig*u[i];
    }
}
static void swiglu_ratsig_rvv(const float*g,const float*u,float*out,int n){
    int i=0;
    while(i<n){ size_t vl=__riscv_vsetvl_e32m1(n-i);
        vfloat32m1_t vx=__riscv_vle32_v_f32m1(g+i,vl), vu=__riscv_vle32_v_f32m1(u+i,vl);
        vfloat32m1_t vhx=__riscv_vfmul_vf_f32m1(vx,0.5f,vl);
        vfloat32m1_t vhx2=__riscv_vfmul_vv_f32m1(vhx,vhx,vl);
        vfloat32m1_t vnum=__riscv_vfadd_vf_f32m1(vhx2,27.0f,vl);
        vnum=__riscv_vfmul_vv_f32m1(vhx,vnum,vl);
        vfloat32m1_t vden=__riscv_vfmul_vf_f32m1(vhx2,9.0f,vl);
        vden=__riscv_vfadd_vf_f32m1(vden,27.0f,vl);
        vfloat32m1_t vt=__riscv_vfdiv_vv_f32m1(vnum,vden,vl);
        vt=__riscv_vfmax_vf_f32m1(vt,-1.0f,vl);
        vt=__riscv_vfmin_vf_f32m1(vt,1.0f,vl);
        vfloat32m1_t vsig=__riscv_vfadd_vf_f32m1(vt,1.0f,vl);
        vsig=__riscv_vfmul_vf_f32m1(vsig,0.5f,vl);
        vfloat32m1_t vy=__riscv_vfmul_vv_f32m1(vx,vsig,vl);
        vy=__riscv_vfmul_vv_f32m1(vy,vu,vl);
        __riscv_vse32_v_f32m1(out+i,vy,vl); i+=vl; }
}

static uint32_t rng=7;
/* Uniform [0,1).  The first draft divided by 2^23, yielding [0,2) and biasing/clamping the
 * distributions below toward their upper endpoint. */
static float randf(void){ rng=rng*1103515245u+12345u; return (rng>>8)*(1.0f/16777216.0f); }

int main(void){
    int n=768;
    float*g=malloc(n*4),*u=malloc(n*4),*out_s=malloc(n*4),*out_v=malloc(n*4),
         *out_exact=malloc(n*4),*out_hs=malloc(n*4);

    /* (1) vectorization correctness: RVV vs scalar, same formula */
    int ntrials=50000; long mismatches=0; float max_diff=0;
    for(int t=0;t<ntrials;t++){
        int mode=t%4;
        for(int i=0;i<n;i++){
            if(mode==0) g[i]=(randf()-0.5f)*20.0f;
            else if(mode==1) g[i]=(randf()-0.5f)*0.01f;
            else if(mode==2) g[i]=0.0f;
            else g[i]=(randf()-0.5f)*1000.0f;
            u[i]=(randf()-0.5f)*10.0f;
        }
        swiglu_ratsig_scalar(g,u,out_s,n);
        swiglu_ratsig_rvv(g,u,out_v,n);
        for(int i=0;i<n;i++){ float d=fabsf(out_s[i]-out_v[i]); if(d>max_diff)max_diff=d; if(d>1e-4f)mismatches++; }
    }
    printf("(1) vectorization check: trials=%d elems=%ld mismatches(>1e-4)=%ld max_abs_diff=%e -- %s\n",
        ntrials, (long)ntrials*n, mismatches, max_diff, mismatches==0?"PASS":"FAIL");

    /* (2) numerical closeness to true SiLU, rational-Pade vs hard-swish, same distributions used
     * in the earlier Python sanity check -- direct apples-to-apples comparison in the real C
     * implementation, not just the Python prototype. */
    double ranges[3][2] = {{-6,6},{-3,3},{-20,20}};
    const char* labels[3] = {"typical [-6,6]", "near-zero [-3,3]", "wide/extreme [-20,20]"};
    for(int r=0;r<3;r++){
        double lo=ranges[r][0], hi=ranges[r][1];
        double sum_abs_rat=0, sum_abs_hs=0, max_abs_rat=0, max_abs_hs=0;
        long total=0;
        for(int t=0;t<20000;t++){
            for(int i=0;i<n;i++){ g[i]=(float)(lo+(hi-lo)*randf()); u[i]=1.0f; }
            swiglu_exact_scalar(g,u,out_exact,n);
            swiglu_hswish_scalar(g,u,out_hs,n);
            swiglu_ratsig_scalar(g,u,out_s,n);
            for(int i=0;i<n;i++){
                double dr=fabs(out_exact[i]-out_s[i]), dh=fabs(out_exact[i]-out_hs[i]);
                sum_abs_rat+=dr; sum_abs_hs+=dh;
                if(dr>max_abs_rat)max_abs_rat=dr; if(dh>max_abs_hs)max_abs_hs=dh;
                total++;
            }
        }
        printf("(2) %s: rational mean=%.5f max=%.5f | hswish mean=%.5f max=%.5f | ratio(hswish/rational)=%.2fx\n",
            labels[r], sum_abs_rat/total, max_abs_rat, sum_abs_hs/total, max_abs_hs, (sum_abs_hs/total)/(sum_abs_rat/total));
    }
    return mismatches==0 ? 0 : 1;
}
