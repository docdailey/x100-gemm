#!/usr/bin/env python3
"""ctypes binding for libmxbai_rvv.so, mirrors nomic_native.py."""
import ctypes
import os

_lib = None


def _load_lib(so_path=None):
    global _lib
    if _lib is not None:
        return _lib
    if so_path is None:
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libmxbai_rvv.so")
    _lib = ctypes.CDLL(so_path)
    _lib.mxbai_load.restype = ctypes.c_void_p
    _lib.mxbai_load.argtypes = [ctypes.c_char_p]
    _lib.mxbai_free.argtypes = [ctypes.c_void_p]
    _lib.mxbai_embed.restype = ctypes.c_double
    _lib.mxbai_embed.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ]
    return _lib


class MxbaiEngine:
    def __init__(self, bin_path, so_path=None):
        lib = _load_lib(so_path)
        self.lib = lib
        self.ctx = lib.mxbai_load(bin_path.encode())
        if not self.ctx:
            raise RuntimeError(f"mxbai_load failed for {bin_path}")
        self.hidden = 1024

    def embed(self, ids, nthreads=8):
        n = len(ids)
        ids_arr = (ctypes.c_int32 * n)(*ids)
        out = (ctypes.c_float * self.hidden)()
        dt = self.lib.mxbai_embed(self.ctx, ids_arr, n, nthreads, out)
        return list(out), dt

    def close(self):
        if self.ctx:
            self.lib.mxbai_free(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
