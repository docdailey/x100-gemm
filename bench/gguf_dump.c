/* gguf_dump.c — minimal dependency-free GGUF reader: list metadata + tensor infos.
 * GGUF v3 layout: magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv |
 *   n_kv × { str key | u32 type | value } | n_tensors × { str name | u32 n_dims | u64 dims[] |
 *   u32 ggml_type | u64 offset } | pad to alignment | tensor data.
 * str = u64 len + bytes. Build: gcc -O2 -o gguf_dump gguf_dump.c   Usage: ./gguf_dump file.gguf [grep]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static const char* GT[]={"F32","F16","Q4_0","Q4_1","","","Q5_0","Q5_1","Q8_0","Q8_1","Q2_K","Q3_K",
    "Q4_K","Q5_K","Q6_K","Q8_K","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ1_S","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS",
    "I8","I16","I32","I64","F64","IQ1_M","BF16"};
static unsigned char*P; static size_t OFF;
static uint32_t u32(void){ uint32_t x; memcpy(&x,P+OFF,4); OFF+=4; return x; }
static uint64_t u64(void){ uint64_t x; memcpy(&x,P+OFF,8); OFF+=8; return x; }
/* read a gguf string (u64 len + bytes) into buf */
static void gstr(char*buf,int cap){ uint64_t n=u64(); int c=n<(uint64_t)cap-1?(int)n:cap-1; memcpy(buf,P+OFF,c); buf[c]=0; OFF+=n; }
/* skip one metadata value of the given type (recurses for arrays) */
static void skipval(uint32_t t){
    switch(t){
        case 0: case 1: case 7: OFF+=1; break;            /* u8/i8/bool */
        case 2: case 3: OFF+=2; break;                    /* u16/i16 */
        case 4: case 5: case 6: OFF+=4; break;            /* u32/i32/f32 */
        case 10: case 11: case 12: OFF+=8; break;         /* u64/i64/f64 */
        case 8: { uint64_t n=u64(); OFF+=n; } break;      /* string */
        case 9: { uint32_t et=u32(); uint64_t n=u64(); for(uint64_t i=0;i<n;i++) skipval(et); } break; /* array */
        default: OFF+=4; break;
    }
}
/* print a scalar metadata value (only for keys we care about) */
static void printval(uint32_t t){
    if(t==8){ char b[256]; gstr(b,256); printf("%s",b); }
    else if(t==4){ printf("%u",u32()); }
    else if(t==5){ printf("%d",(int32_t)u32()); }
    else if(t==6){ float f; uint32_t x=u32(); memcpy(&f,&x,4); printf("%g",f); }
    else if(t==10){ printf("%llu",(unsigned long long)u64()); }
    else if(t==11){ printf("%lld",(long long)u64()); }
    else if(t==9){ uint32_t et=u32(); uint64_t n=u64(); printf("[array type=%u n=%llu]",et,(unsigned long long)n); for(uint64_t i=0;i<n;i++) skipval(et); }
    else { skipval(t); printf("<t%u>",t); }
}

int main(int c,char**v){
    if(c<2){ printf("usage: %s file.gguf [namegrep]\n",v[0]); return 1; }
    const char*grep=(c>2)?v[2]:NULL;
    FILE*f=fopen(v[1],"rb"); if(!f){ perror("open"); return 1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    P=malloc(sz); if(fread(P,1,sz,f)!=(size_t)sz){ printf("read fail\n"); return 1; } fclose(f);
    OFF=0;
    char magic[5]={0}; memcpy(magic,P,4); OFF=4;
    uint32_t ver=u32(); uint64_t nt=u64(), nkv=u64();
    printf("magic=%s ver=%u  n_tensors=%llu  n_kv=%llu\n",magic,ver,(unsigned long long)nt,(unsigned long long)nkv);
    /* metadata */
    for(uint64_t i=0;i<nkv;i++){
        char key[256]; gstr(key,256); uint32_t t=u32();
        int show = strstr(key,"architecture")||strstr(key,"expert")||strstr(key,"block_count")||
                   strstr(key,"head_count")||strstr(key,"embedding_length")||strstr(key,"feed_forward")||
                   strstr(key,"context_length")||strstr(key,"vocab")||strstr(key,"rope")||strstr(key,"tokenizer.ggml.model")||
                   strstr(key,"attention.key_length")||strstr(key,"attention.value_length");
        if(show){ printf("  %-45s = ",key); printval(t); printf("\n"); }
        else skipval(t);
    }
    /* tensor infos */
    printf("--- tensors (name  type  dims  offset) ---\n");
    int shown=0;
    for(uint64_t i=0;i<nt;i++){
        char name[256]; gstr(name,256);
        uint32_t nd=u32(); uint64_t d[8]={1,1,1,1,1,1,1,1};
        for(uint32_t j=0;j<nd&&j<8;j++) d[j]=u64();
        uint32_t typ=u32(); uint64_t off=u64();
        if(!grep || strstr(name,grep)){
            if(shown<40 || grep){
                printf("  %-30s %-6s [",name, typ<sizeof(GT)/sizeof(GT[0])?GT[typ]:"?");
                for(uint32_t j=0;j<nd;j++) printf("%llu%s",(unsigned long long)d[j], j+1<nd?"x":"");
                printf("]  @%llu\n",(unsigned long long)off);
            }
            shown++;
        }
    }
    printf("(%d tensors listed of %llu)\n",shown,(unsigned long long)nt);
    return 0;
}
