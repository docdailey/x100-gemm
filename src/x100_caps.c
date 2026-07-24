/* x100_caps.c — runtime capability detection for the SpaceMIT X100. */
#include "x100_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int isa_has(const char *isa, const char *tok)
{
    /* tokens in /proc/cpuinfo isa are '_'-separated (e.g. ..._zvfh_...) or
     * space-separated on the short line. Match as a whole token. */
    size_t n = strlen(tok);
    const char *p = isa;
    while ((p = strstr(p, tok))) {
        char before = (p == isa) ? '_' : p[-1];
        char after  = p[n];
        int lb = (before == '_' || before == ' ' || before == '\0');
        int la = (after  == '_' || after  == ' ' || after  == '\0');
        if (lb && la) return 1;
        p += n;
    }
    return 0;
}

void x100_detect(x100_caps_t *caps)
{
    memset(caps, 0, sizeof(*caps));

#if defined(__riscv)
    unsigned long vlenb = 0;
    /* vlenb CSR (0xC22) — bytes per vector register. Readable in U-mode when V on. */
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    caps->vlenb = (int)vlenb;
    caps->vlen_bits = (int)(vlenb * 8);
#endif

    /* Parse /proc/cpuinfo for the isa string + mvendorid. */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[2048];
        while (fgets(line, sizeof(line), f)) {
            if (!caps->isa[0] && (strncmp(line, "isa", 3) == 0)) {
                char *c = strchr(line, ':');
                if (c) { strncpy(caps->isa, c + 2, sizeof(caps->isa) - 1);
                         caps->isa[strcspn(caps->isa, "\n")] = 0; }
            }
            if (strncmp(line, "mvendorid", 9) == 0) {
                char *c = strchr(line, ':');
                if (c) caps->mvendorid = strtoul(c + 2, NULL, 0);
            }
        }
        fclose(f);
    }

    caps->has_v    = isa_has(caps->isa, "v") || isa_has(caps->isa, "rv64imafdcv")
                     || strstr(caps->isa, "cv") != NULL;
    caps->has_zvfh = isa_has(caps->isa, "zvfh");
    caps->has_zvbb = isa_has(caps->isa, "zvbb");
    /* IME (vmadot) is a SpaceMIT custom extension NOT surfaced in riscv,isa.
     * Infer from vendor==SpaceMIT + VLEN>=256; a true probe needs a trap-tolerant
     * test of a vmadot instruction. */
    caps->has_ime  = (caps->mvendorid == 0x710) && (caps->vlen_bits >= 256);
}

void x100_caps_print(const x100_caps_t *c)
{
    printf("== X100 capabilities ==\n");
    printf("  mvendorid : 0x%lx %s\n", c->mvendorid,
           c->mvendorid == 0x710 ? "(SpaceMIT)" : "");
    printf("  VLEN      : %d bits (%d bytes)\n", c->vlen_bits, c->vlenb);
    printf("  RVV(v)    : %s\n", c->has_v    ? "yes" : "no");
    printf("  Zvfh(f16) : %s\n", c->has_zvfh ? "yes" : "no");
    printf("  Zvbb      : %s\n", c->has_zvbb ? "yes" : "no");
    printf("  IME(vmadot): %s  (custom; not in riscv,isa — see docs/IME.md)\n",
           c->has_ime ? "likely" : "no");
    printf("  isa       : %.120s%s\n", c->isa, strlen(c->isa) > 120 ? "..." : "");
}
