# Jupiter 2 Vision API

The initial endpoint runs official PP-OCRv6 Tiny ONNX detector and recognizer models.

```sh
curl -F image=@page.png http://192.168.68.24:8080/v1/ocr
curl http://192.168.68.24:8080/healthz
```

The service currently defaults to ONNX Runtime's CPU provider because SpaceMIT EP 2.0.5 crashes
during session creation for both official PP-OCRv6 Tiny graphs. This is recorded explicitly in
the health response. Set `VISION_PROVIDER=spacemit` only when testing a fixed provider build.

## Recognizer selection

`VISION_RECOGNIZER` picks which recognizer runs; the detector is always PP-OCRv6 Tiny on ORT CPU.

- `native-ppocrv6` (default) — PP-OCRv6 Tiny as a hand-written FP32 RVV engine (`ocr_rvv.c`) on the
  eight A100 AI harts, with no ONNX Runtime, SpaceMIT EP, XQuant or Paddle runtime in its path.
  Bit-equivalent to the ONNX path (100% argmax agreement, byte-identical text end to end) and
  1.7-2.5x faster per crop.
- `onnx-ppocrv6` — the same model through ONNX Runtime CPU. Rollback target.
- `native-rosetta` — native ResNet34+CTC. Correct, but a 36-character lowercase-only dictionary
  makes it unusable for documents.

`BARE_IME_OCR_PROGRESS.md` has the full correctness numbers, benchmarks, hart-scaling results and
the evidence behind the default. Note that the native engine pins its calling thread to an AI hart
for life — drive it from one dedicated thread, as `ocr_native.py` does.

