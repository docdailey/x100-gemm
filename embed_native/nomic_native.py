#!/usr/bin/env python3
"""ctypes binding for libnomic_rvv.so, used by app-level Python (tokenization stays in Python,
same division of labor as vision_api/ocr_native.py -> ocr_rvv.c)."""
import ctypes
import os

_lib = None


def _load_lib(so_path=None):
    global _lib
    if _lib is not None:
        return _lib
    if so_path is None:
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnomic_rvv.so")
    _lib = ctypes.CDLL(so_path)
    _lib.nomic_load.restype = ctypes.c_void_p
    _lib.nomic_load.argtypes = [ctypes.c_char_p]
    _lib.nomic_free.argtypes = [ctypes.c_void_p]
    _lib.nomic_embed.restype = ctypes.c_double
    _lib.nomic_embed.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ]
    return _lib


class NomicEngine:
    def __init__(self, bin_path, so_path=None):
        lib = _load_lib(so_path)
        self.lib = lib
        self.ctx = lib.nomic_load(bin_path.encode())
        if not self.ctx:
            raise RuntimeError(f"nomic_load failed for {bin_path}")
        self.hidden = 768

    def embed(self, ids, nthreads=8):
        n = len(ids)
        ids_arr = (ctypes.c_int32 * n)(*ids)
        out = (ctypes.c_float * self.hidden)()
        dt = self.lib.nomic_embed(self.ctx, ids_arr, n, nthreads, out)
        return list(out), dt

    def close(self):
        if self.ctx:
            self.lib.nomic_free(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
