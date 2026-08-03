"""ctypes binding for the native RVV recognizer engine (libocr_rvv.so).

The shared library owns the AI-hart worker pool and the weight blob; this module
only marshals numpy arrays across. Input height and class count come from the
blob, so the same wrapper drives either recognizer. Calls are serialised inside
the library by its own mutex, and ctypes releases the GIL for the duration, so a
hart-bound spin dispatch does not block the FastAPI event loop.

Crops are inferred one at a time at their own natural width. Padding a batch to a
common width -- which the ONNX Runtime path must do -- makes every short crop pay
the widest crop's convolution cost, which for a fully convolutional CTC
recognizer is pure waste.
"""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


class OCRNative:
    def __init__(self, lib_path: str, blob_path: str, harts: int = 8,
                 width_align: int = 16, max_width: int = 3200):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ocr_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.ocr_load.restype = ctypes.c_int
        self.lib.ocr_out_steps.argtypes = [ctypes.c_int]
        self.lib.ocr_out_steps.restype = ctypes.c_int
        self.lib.ocr_infer.argtypes = [
            np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.ocr_infer.restype = ctypes.c_int
        for name in ("ocr_harts", "ocr_input_height", "ocr_n_classes"):
            getattr(self.lib, name).restype = ctypes.c_int

        # Every call into the library must come from ONE dedicated thread. The
        # engine makes its caller worker 0 and pins it to an AI hart for life,
        # because a thread that is not AI-bound cannot reliably see stores made
        # by AI-hart threads (see dispatch() in ocr_rvv.c). Driving it from the
        # request thread would therefore pin that thread too, and every bit of
        # OpenCV/numpy work around the call would be stuck on a single hart --
        # measured at 2-4x slower preprocessing and decoding.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-native")

        rc = self._pool.submit(self.lib.ocr_load, blob_path.encode(), harts).result()
        if rc != 0:
            raise RuntimeError(f"ocr_load({blob_path!r}) failed with rc={rc}")
        self.harts = self.lib.ocr_harts()
        self.height = self.lib.ocr_input_height()
        self.n_classes = self.lib.ocr_n_classes()
        self.width_align = width_align
        self.max_width = max_width
        self._out = None      # reused output buffer; see infer_array

    def _width_for(self, crop: np.ndarray) -> int:
        h, w = crop.shape[:2]
        target = int(np.ceil(self.height * w / max(h, 1)))
        target = int(np.ceil(target / self.width_align) * self.width_align)
        return max(self.width_align, min(self.max_width, target))

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop -> (3, H, W) float32 in [-1, 1] (Paddle recognizer norm)."""
        width = self._width_for(crop)
        resized = cv2.resize(crop, (width, self.height)).astype(np.float32) / 127.5 - 1.0
        return np.ascontiguousarray(np.transpose(resized, (2, 0, 1)))

    def infer_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize + normalize a BGR crop, then run it."""
        return self.infer_array(self.preprocess(crop))

    def infer_array(self, x: np.ndarray) -> np.ndarray:
        """x: preprocessed (3, H, W) float32 -> (steps, n_classes) probabilities.

        The returned array is a view of a buffer reused across calls -- consume
        it before the next call. At 6906 classes a wide crop's output is ~11 MB,
        and allocating that per crop costs more in page faults than the whole
        inference does.
        """
        width = x.shape[2]
        steps = self._pool.submit(self.lib.ocr_out_steps, width).result()
        if steps < 1:
            raise RuntimeError(f"ocr_out_steps({width}) returned {steps}")
        need = steps * self.n_classes
        if self._out is None or self._out.size < need:
            self._out = np.empty(need, np.float32)
        rc = self._pool.submit(self.lib.ocr_infer, x.reshape(-1), width, self._out).result()
        if rc < 0:
            raise RuntimeError(f"ocr_infer failed with rc={rc}")
        return self._out[:need].reshape(steps, self.n_classes)


NATIVE_MODELS = {
    "native-ppocrv6": ("ppocrv6_rec.bin", "PP-OCRv6_tiny_rec (native RVV FP32)"),
    "native-rosetta": ("rosetta_r34.bin", "rec_r34_vd_none_none_ctc_v2.0 (native RVV FP32)"),
}


def default_paths(recognizer: str) -> tuple[str, str, str]:
    root = os.environ.get("OCR_NATIVE_DIR", "/root/vision-api/models/ocr-native")
    blob, label = NATIVE_MODELS[recognizer]
    return os.path.join(root, "libocr_rvv.so"), os.path.join(root, blob), label
