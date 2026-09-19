# -*- coding: utf-8 -*-
"""Shared GPU availability helpers for the test suite.

The probe actually compiles and runs a trivial CUDA kernel -- a bare
``cuda.is_available()`` can report True (driver present) while kernel
compilation still fails (e.g. missing NVVM), and can report False on
some setups where kernels work.  The result is cached per process.
"""
import numpy as np

_cache = None


def _probe():
    try:
        from numba import cuda
    except ImportError:
        return False, 'no   (numba not installed)'

    try:
        if cuda.is_available():
            d = cuda.get_current_device()
            name = d.name.decode() if isinstance(d.name, bytes) else d.name

            # real probe: compile & run a trivial kernel
            @cuda.jit
            def _trivial(x):
                x[0] = 1.0

            a = cuda.device_array(1, dtype=np.float64)
            _trivial[1, 1](a)
            if a.copy_to_host()[0] == 1.0:
                return True, f'yes  ({name}, CC {d.compute_capability})'
            return False, 'no   (kernel ran but returned wrong value)'
        return False, 'no   (cuda.is_available() returned False)'
    except Exception as e:
        return False, f'no   ({type(e).__name__}: {e})'


def gpu_available():
    """Return True when a CUDA kernel can actually be compiled & run."""
    global _cache
    if _cache is None:
        _cache = _probe()
    return _cache[0]


def gpu_info():
    """Return a human-readable GPU status string (diagnostics)."""
    global _cache
    if _cache is None:
        _cache = _probe()
    return _cache[1]
