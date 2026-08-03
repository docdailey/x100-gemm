# SpaceMIT OCR acceleration findings

Status: investigated on Jupiter 2 with `spacemit-ort 2.0.5`; the live OCR API
remains on `CPUExecutionProvider` because no accelerated candidate passed both
provider-placement and quality gates.

## Rosetta OCR compatibility probe (2026-07-30)

The official PaddleOCR Rosetta recognizers were exported to ONNX and quantized
with XQuant 2.0.4. Neither is safe to deploy through SpaceMIT EP 2.0.5:

- ResNet34 Rosetta (39 Conv, CTC, fixed `1x3x32x100`) segfaults while compiling
  its first residual `Add`. Prefixes through the two branch outputs load.
- Batch 16, batch 1, and tile-aligned width 128 all fail at the same graph
  feature, ruling out batch count and width-25 alignment as the cause.
- MobileNetV3 Rosetta also segfaults, so changing to the smaller Paddle
  backbone does not remove the runtime defect.
- Replacing residual `Add` with bit-identical `Mean(a,b) * 2` avoids the
  segfault but fails provider partition initialization inside the vendor
  runtime (`LoggingManager::DefaultLogger`), even with explicit ORT logging.
- Splitting ResNet34 works only when every computed branch is its own model:
  all 32 single-path branch models load with SpaceMIT EP. This is not
  deployable because every EP session reserves all eight AI cores. The first
  accelerates; the other 31 report zero available AI cores and fall back to
  CPU. End-to-end latency was 665-972 ms/crop and degraded across runs.

The convolution kernels are compatible, but EP 2.0.5 cannot safely compile or
partition the complete OCR topology and cannot share AI cores across the
sessions needed to route around it. The live API therefore remains on the
honest CPU PP-OCRv6 path. Provider-list presence is not acceleration; require
an ORT profile containing a `SpaceMITExecutionProvider_SpineSubgraph` for the
complete recognizer.

## What was proven

- The SpaceMIT execution provider is installed and functional. The official
  `resnet50.q.onnx` executes as one `SpaceMITExecutionProvider_SpineSubgraph`.
- A minimal ONNX convolution also loads successfully.
- PP-OCRv6 Tiny detector and recognizer both segfault during SpaceMIT session
  creation in their original dynamic form.
- Fixing all input shapes does not eliminate the crashes.
- SpaceMIT EP 2.0.5 has a reproducible parser bug: adding the optional,
  redundant `kernel_shape` attribute to an otherwise working Conv makes
  session creation segfault. Removing it fixes that minimal case.
- The detector then reaches its feature-pyramid merges, where combining two
  independently compilable branches at `Concat` or `Add` causes another EP
  session-creation segfault.

## XQuant results

SpaceMIT XQuant 2.0.4 was run on the x86-64 Docker host `ryzen`, using pinned
CPU-only PyTorch, NumPy 1.26.4, ONNX 1.16.2, and ONNX Runtime 1.20.1.

- Detector: a static 736x736 `.q.onnx` was produced. After removing redundant
  Conv `kernel_shape` attributes, compilation still crashes at the first
  multi-branch `Concat`.
- Recognizer: a static batch-16, 48x320 `.q.onnx` was produced. Its convolution
  trunk loads when split before the vocabulary Gemm head, but ORT profiling
  assigns all 75 trunk nodes to `CPUExecutionProvider`; listing SpaceMIT in the
  session provider list is not evidence that it claimed any work.
- The final Gemm head itself crashes SpaceMIT compilation and must remain CPU.
- Calibration from 96 real perspective-corrected text crops reached only 80%
  timestep argmax agreement with the FP32 recognizer on the checked batch.
  This fails the quality gate and was not promoted.

## Reference placement test

The official SpaceMIT INT8 ResNet model profiles as:

```text
Counter({'SpaceMITExecutionProvider': 1})
SpaceMITExecutionProvider_SpineSubgraph_..._kernel_time
```

The split OCR recognizer trunk profiles as:

```text
Counter({'CPUExecutionProvider': 75})
```

Therefore the present OCR models are not running on the accelerated backend,
even when the session reports both providers as available.

## Decision

Do not switch the live API to the current XQuant artifacts. A credible next
acceleration path requires one of:

1. a SpaceMIT EP/compiler fix for Paddle feature merges and Gemm;
2. a vendor-supplied OCR `.q.onnx` known to form a Spine subgraph; or
3. a recognizer architecture deliberately restricted to the operator topology
   demonstrated by SpaceMIT's accelerated model zoo, followed by retraining.

The CPU service remains the correctness baseline and rollback target.
