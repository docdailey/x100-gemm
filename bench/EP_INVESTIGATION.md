# SpaceMIT EP investigation — is there a memory-direct IME path past the 2.3-TOPS vle8 ceiling?

Goal (fork #1): prove or kill a matrix instruction that feeds the IME-2 units *without* going
through `vle8`→registers. If `libspacemit_ep.so` (the vendor stack claiming 60 TOPS) beats
~2.3 TOPS on int8 matmul, such a path must exist; disassembly will show it.

## Confirmed so far
- `apt install spacemit-onnxruntime` → `/usr/lib/libspacemit_ep.so`, `onnxruntime_perf_test`.
- **The EP opens `/dev/ai_dma`, `/dev/aidma_list`, `/dev/dma_msi`** (strings + clean init) — the
  production path DOES drive the DMA engine (unlike llama.cpp, which uses CPU `memcpy1d`).
- **"spine-kernel route"** = `SpineComputeDispatch` / `SpineComputeContext` — the high-perf path.
- Supported spine ops (from strings): **`Conv`, `ConvBinaryFusion`, `ConvExpand`, `ConvTranspose`,
  `DynamicQuantizeMatMul`, `ConvWithBinary`**. NOT `MatMulInteger` (→ "kernel not found").
- `onnxruntime_perf_test -e spacemit` works; EP + ai_dma init OK. Needs a `test_data_set_0/input_0.pb`.

## READY-TO-RUN when board is back (home LAN)
### 1. Benchmark int8 matmul via a SUPPORTED op → TOPS (the decisive number)
Use `DynamicQuantizeMatMul` (com.microsoft) or a big `Conv` — NOT MatMulInteger. See
`bench/make_ep_model.py`. Then:
```
export LD_LIBRARY_PATH=/usr/lib
onnxruntime_perf_test -e spacemit -t 8 /tmp/dqm/model.onnx /tmp/res.txt
# TOPS = 2*M*N*K / avg_inference_s / 1e12   (for the matmul dims used)
```
If TOPS > ~2.3 → mem-direct path exists → disassemble to find the instruction.
If TOPS <= ~2.3 → vle8 ceiling holds; TCM/ai_dma value is decode bandwidth (fork #2).

### 2. Recover the ai_dma ioctl ABI (needed for fork #2 regardless) via strace
```
strace -f -e trace=ioctl,openat,mmap -s 200 \
  onnxruntime_perf_test -e spacemit -r 5 /tmp/dqm/model.onnx /tmp/res.txt 2>strace.txt
grep -E "/dev/ai_dma|/dev/tcm|ioctl" strace.txt   # exact ioctl cmd numbers + arg structs
```

### 3. Disassemble the spine matmul/conv kernel (the direct proof)
```
objdump -d /usr/lib/libspacemit_ep.so.2.0.5 > /tmp/ep.s   # large; may need a few min
# look for: vmadot/vmadotsu (register-fed, = 2.3 ceiling) vs any NON-vmadot custom matrix op
grep -oiE "vmadot[a-z]*|vmv|vle8|vsetvli|\.insn|\.word 0x[0-9a-f]+" /tmp/ep.s | sort | uniq -c | sort -rn
# if a novel custom opcode dominates the hot loop AND has memory operands -> mem-direct path
```

## Interpretation key
- Only `vmadot`/`vmadotsu` in the hot loop  → same register-fed path; 2.3 TOPS is the real ceiling;
  ai_dma just moves DRAM→TCM to cut CPU cost / feed private BW (helps decode, not prefill compute).
- A distinct custom matrix op with memory/TCM operands → the mem-direct datapath; replicate it.

## RESULTS (2026-07-25, board @192.168.68.88 home LAN)
- **Vendor EP int8 2048³ (DynamicQuantizeMatMul): ~2.79 TOPS** (162 inf/s, 6.15 ms), **CPU usage 110% (~1 core)**.
  => The matmul is OFFLOADED to the ai_dma/IME accelerator; CPU nearly idle. My 8-core-vmadot (0.97 TOPS,
  ~800% CPU) is the WRONG paradigm. FORK #1 verdict: **no 14-60 TOPS for real matmul; ~2.8 is the ceiling**
  (60 TOPS = peak/register-fed marketing).
- **ABI (strace):** opens /dev/tcm(×2, O_RDWR|O_SYNC), /dev/aidma_list, /dev/ai_dma, /dev/dma_msi.
  /dev/tcm ioctls: magic 0x63('c'), nr 0x7=query(_IOC_READ,4B), nr 0x9=acquire(_IOC_READ|WRITE,4B, 8× = per
  block). /dev/dma_msi: _IOC_NONE (irq). ai_dma SUBMISSION = mmap'd ring/doorbell (no per-transfer syscalls)
  -> needs objdump of libspacemit_ep.so to fully recover.
- **STRATEGIC FORK:** (A) RE the ai_dma ring -> custom offload engine (huge, ceiling ~2.8), or (B) USE the
  vendor EP for matmul offload + build value at decode/MoE/MTP/scheduling. Recommend B.
