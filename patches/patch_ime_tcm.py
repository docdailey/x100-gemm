#!/usr/bin/env python3
# Patch spacemit llama.cpp ime.cpp: fix the TCM setaffinity->A100-hart EINVAL abort.
# (1) call bind_ai_thread() unconditionally before setaffinity (proven-required unlock).
# (2) make setaffinity failure non-fatal: warn + run without TCM instead of GGML_ABORT.
import sys, shutil
F = "/root/llama.cpp/ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(F).read()
shutil.copy(F, F + ".orig")

# (1) insert bind_ai_thread() right after the use_tcm block opens
a_old = ("""        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));""")
a_new = ("""        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        bind_ai_thread();  // PATCH: unlock thread for A100-core affinity before setaffinity
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));""")

# (2) non-fatal setaffinity + guard TCM setup behind success
b_old = ('''        if (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\\n", thread_n, perfer_cpu_id);
        }

        int ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;''')
b_new = ('''        if (s != 0) {
            GGML_LOG_ERROR("PATCH: setaffinity failed thread_n %d cpu_id %d; running WITHOUT TCM\\n", thread_n, perfer_cpu_id);
        } else {
            int ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
            ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
                ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
            ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
                ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
        }''')

for old,new,tag in [(a_old,a_new,"unlock"),(b_old,b_new,"nonfatal")]:
    n = s.count(old)
    if n != 1: print(f"ABORT: {tag} matched {n} times (need 1)"); sys.exit(1)
    s = s.replace(old, new)
open(F,"w").write(s)
print("PATCHED ok: bind_ai_thread unlock + non-fatal setaffinity")
