# -*- coding: utf-8 -*-
"""
GPU module of xinvert: CUDA-accelerated SOR iteration kernels.

Contains GPU implementations of the SOR iteration kernels using numba.cuda.
Uses Red-Black ordering for parallel SOR iteration on GPU.

Currently implemented:
  - invert_standard_2D_gpu: GPU version of invert_standard_2D (Poisson, etc.)

The GPU wrapper functions have the **same signature** as the numba kernels in
:mod:`xinvert.cpus`, so they can be used as drop-in replacements via the
``architect`` dispatch in :mod:`xinvert.core._make_kernel`.
"""
import numpy as np
from numba import cuda


# ---------------------------------------------------------------------------
# CUDA kernels for Red-Black SOR (standard 2D form)
# ---------------------------------------------------------------------------

@cuda.jit
def _sor_2d_rb(S, A, B, C, F, yc, xc, bcx_periodic,
               delxSqr, ratioQtr, ratioSqr, optArg, undef, color):
    """Red-Black SOR update for one color of the standard 2D form.

    color=0 → red points ((j+i) even), color=1 → black points.
    Handles interior points and periodic x-boundary points.

    The update formula matches the interior loop of
    :func:`xinvert.cpus.invert_standard_2D`.
    """
    j, i = cuda.grid(2)

    # Skip rows outside the interior (y-boundary rows are fixed or extend)
    if j < 1 or j >= yc - 1:
        return

    # Determine valid i-range
    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 1 or i >= xc - 1:
            return

    # Only process the requested color
    if (j + i) % 2 != color:
        return

    # Periodic wrapping for x-neighbours
    ip1 = i + 1
    im1 = i - 1
    if ip1 >= xc:
        ip1 = 0
    if im1 < 0:
        im1 = xc - 1

    # Check that all needed coefficients are defined
    if (F[j, i] == undef or
        A[j + 1, i] == undef or A[j, i] == undef or
        B[j, ip1] == undef or B[j, im1] == undef or
        B[j + 1, i] == undef or B[j - 1, i] == undef or
        C[j, ip1] == undef or C[j, i] == undef):
        return

    # Compute SOR residual (identical to numba kernel interior formula)
    temp = (
        (A[j + 1, i] * (S[j + 1, i] - S[j, i]) -
         A[j, i] * (S[j, i] - S[j - 1, i])) * ratioSqr +
        (B[j + 1, i] * (S[j + 1, ip1] - S[j + 1, im1]) -
         B[j - 1, i] * (S[j - 1, ip1] - S[j - 1, im1])) * ratioQtr +
        (B[j, ip1] * (S[j + 1, ip1] - S[j - 1, ip1]) -
         B[j, im1] * (S[j + 1, im1] - S[j - 1, im1])) * ratioQtr +
        (C[j, ip1] * (S[j, ip1] - S[j, i]) -
         C[j, i] * (S[j, i] - S[j, im1]))
    ) - F[j, i] * delxSqr

    denom = (A[j + 1, i] + A[j, i]) * ratioSqr + (C[j, ip1] + C[j, i])
    if denom != 0.0:
        S[j, i] += temp * optArg / denom


@cuda.jit
def _abs_norm_2d(S, undef, out):
    """Compute sum(|S|) and count of non-undef points (atomic reduction)."""
    j, i = cuda.grid(2)
    if j < S.shape[0] and i < S.shape[1]:
        if S[j, i] != undef:
            cuda.atomic.add(out, 0, abs(S[j, i]))
            cuda.atomic.add(out, 1, 1.0)


@cuda.jit
def _extend_y_boundary(S, yc, xc, undef):
    """Extend BC: copy y-interior boundary to y-outer boundary."""
    i = cuda.grid(1)
    if i < xc:
        if S[1, i] != undef:
            S[0, i] = S[1, i]
        if S[yc - 2, i] != undef:
            S[yc - 1, i] = S[yc - 2, i]


@cuda.jit
def _extend_x_boundary(S, yc, xc, undef):
    """Extend BC: copy x-interior boundary to x-outer boundary, plus corners.

    Folding corner handling into this kernel avoids a separate 1-block
    launch (which triggers a ``Grid size 1`` under-utilization warning)
    and saves one kernel launch per iteration.
    """
    j = cuda.grid(1)
    if j >= yc:
        return
    if j == 0:
        # top-left / top-right corners (diagonal neighbour)
        if S[1, 1] != undef:
            S[0, 0] = S[1, 1]
        if S[1, xc - 2] != undef:
            S[0, xc - 1] = S[1, xc - 2]
    elif j == yc - 1:
        # bottom-left / bottom-right corners
        if S[yc - 2, 1] != undef:
            S[yc - 1, 0] = S[yc - 2, 1]
        if S[yc - 2, xc - 2] != undef:
            S[yc - 1, xc - 1] = S[yc - 2, xc - 2]
    else:
        # interior rows: left and right edges
        if S[j, 1] != undef:
            S[j, 0] = S[j, 1]
        if S[j, xc - 2] != undef:
            S[j, xc - 1] = S[j, xc - 2]


# ---------------------------------------------------------------------------
# Shared host-side helpers (reused by 2D/3D/general GPU wrappers)
# ---------------------------------------------------------------------------

import os


def _parse_block_2d(default):
    """Read 2D thread-block shape from ``XINVERT_GPU_BLOCK2D`` env var.

    Accepts ``"32,8"`` or ``"32x8"``.  Falls back to *default* when unset or
    malformed.  The x-dimension should be a multiple of the warp size (32)
    for coalesced global-memory access.
    """
    raw = os.environ.get('XINVERT_GPU_BLOCK2D', '')
    if raw:
        try:
            parts = [int(x) for x in raw.replace('x', ',').split(',')]
            if len(parts) == 2 and all(p > 0 for p in parts):
                return (parts[0], parts[1])
        except ValueError:
            pass
    return default


# Default (16, 16): 256 threads/block, square shape → best cache locality
# for the 2-D stencil (neighbours in both x and y stay within the block).
# A block sweep (tests/benchmark_blocks.py) showed (16,16) is ~3 % faster
# than (32,8) at 4096^2 despite the latter being warp-coalesced, because
# stencil access is 2-D, not row-stride.  Override via env var for tuning:
#   XINVERT_GPU_BLOCK2D=32,8
def _block_2d():
    return _parse_block_2d((16, 16))


def _auto_bsize_1d(n):
    """Pick 1D thread-block size adaptively by problem extent *n*.

    Returns a warp-multiple in [32, 256] that gives good occupancy across
    typical grid sizes without exposing a user-facing parameter:
      - n < 64   -> 32   (single warp, avoids grid=1 under-utilisation)
      - n < 512  -> 128  (2-4 blocks, SMs start to fill)
      - n >= 512 -> 256  (sweet spot for all larger sizes)
    """
    if n < 64:
        return 32
    if n < 512:
        return 128
    return 256


def _compute_check_interval(mxLoop, tolerance):
    """Adaptive convergence-check interval based on mxLoop and tolerance.

    Scales the interval with the total iteration budget so that the number
    of host-device synchronisations stays roughly constant regardless of
    problem size, while keeping the worst-case "over-iteration after
    convergence" bounded.

    - tolerance > 0  (real convergence run): ~mxLoop/50, clamped to [10, 100]
      e.g. mxLoop= 1000 -> 20, 5000 -> 100, 10000 -> 100, 100000 -> 100
    - tolerance <= 0 (fixed-iteration / benchmark, no early exit): only an
      overflow guard is needed, so the interval can be larger:
      ~mxLoop/20, clamped to [50, 500]
      e.g. mxLoop= 1000 -> 50, 5000 -> 100, 10000 -> 200, 100000 -> 500
    """
    if tolerance > 0.0:
        return max(10, min(100, mxLoop // 50))
    else:
        return max(50, min(500, mxLoop // 20))


def _evaluate_gpu_norm(norm_sum, norm_count, norm_prev,
                       need_convergence, tolerance):
    """Reduce the [sum|S|, count] atomic accumulators to a convergence signal.

    Mirrors the CPU ``absNorm*`` metric: norm = sum(|S|)/count, and the
    convergence error is the relative change of this norm between two
    successive checks (``|norm - norm_prev| / norm_prev``).

    Returns
    -------
    norm : float
        Mean absolute value of S (nan if no valid points).
    error : float
        Relative change of the norm vs the previous check (0.0 on the
        first check, when norm_prev is still unset).
    overflow : bool
        True if norm is nan or exceeds 1e100.
    """
    if norm_count > 0:
        norm = norm_sum / norm_count
    else:
        norm = np.nan

    overflow = (np.isnan(norm) or norm > 1e100)

    if need_convergence and norm_prev < np.finfo(np.float64).max:
        error = abs(norm - norm_prev) / norm_prev
    else:
        error = 0.0

    return norm, error, overflow


# ---------------------------------------------------------------------------
# Python wrapper (same signature as numba kernel)
# ---------------------------------------------------------------------------

def invert_standard_2D_gpu(S, A, B, C, F, info,
                           yc, xc, BCy, BCx, delxSqr,
                           ratioQtr, ratioSqr, optArg, undef, flags,
                           mxLoop, tolerance, block_2d=None):
    r"""GPU implementation of ``invert_standard_2D`` using Red-Black SOR.

    Same signature as :func:`xinvert.cpus.invert_standard_2D` so it can be
    used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.

    The host-side Python code controls the iteration loop; each iteration
    launches a Red kernel followed by a Black kernel on the GPU, then
    computes the convergence norm via an atomic-reduction kernel.

    Parameters
    ----------
    S : numpy.ndarray (modified in-place)
        Solution array, shape (yc, xc).
    A, B, C : numpy.ndarray
        Coefficient arrays, same shape as S.
    F : numpy.ndarray
        Forcing array, same shape as S.
    info : any
        Information array (unused in computation, kept for signature compat).
    yc, xc : int
        Grid counts in y and x dimensions.
    BCy, BCx : str
        Boundary conditions ('fixed', 'extend', 'periodic').
    delxSqr, ratioQtr, ratioSqr : float
        Grid spacing parameters.
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (masked points).
    flags : numpy.ndarray, shape (3,)
        Output flags: [overflow, error, loop_count].
    mxLoop : int
        Maximum iteration count.
    tolerance : float
        Convergence tolerance.
    """
    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_F = cuda.to_device(F)

    bcx_periodic = (BCx == 'periodic')
    bcy_extend = (BCy == 'extend')

    # --- thread / block config ---
    # Block shape: per-call override (iParams['gpu_block2d']) > env var
    # (XINVERT_GPU_BLOCK2D) > built-in default (16, 16).  Override per-call:
    #   iParams['gpu_block2d'] = (32, 8)   # warp-coalesced
    #
    # Axis mapping (CUDA):
    #   j = cuda.grid(2)[0] = blockIdx.x*blockDim.x + threadIdx.x  (rows = yc)
    #   i = cuda.grid(2)[1] = blockIdx.y*blockDim.y + threadIdx.y  (cols = xc)
    # so gridDim.x (blocks[0]) pairs with blockDim.x (bx) to cover yc, and
    # gridDim.y (blocks[1]) pairs with blockDim.y (by) to cover xc.
    bx, by = block_2d if block_2d is not None else _block_2d()
    threads = (bx, by)
    blocks = (max((yc + bx - 1) // bx, 1), max((xc + by - 1) // by, 1))

    # 1D block sizes for boundary kernels, chosen adaptively by the extent
    # each kernel actually sweeps (xc for y-boundary, yc for x-boundary).
    bs1d_x = _auto_bsize_1d(xc)
    bs1d_y = _auto_bsize_1d(yc)
    bcount_x = max((xc + bs1d_x - 1) // bs1d_x, 1)
    bcount_y = max((yc + bs1d_y - 1) // bs1d_y, 1)

    d_norm = cuda.device_array(2, dtype=np.float64)

    # Check convergence / overflow only every check_interval iterations to
    # amortise the host-device sync cost.  The interval scales with mxLoop:
    #  - tolerance>0 (real convergence): ~mxLoop/50, clamped [10,100]
    #  - tolerance<=0 (fixed-iter / benchmark): only overflow guard needed,
    #    ~mxLoop/20, clamped [50,500]
    check_interval = _compute_check_interval(mxLoop, tolerance)
    need_convergence = (tolerance > 0.0)

    norm_prev = np.finfo(np.float64).max
    loop = 0
    overflow = False
    error = 0.0

    while True:
        # --- process boundaries ---
        if bcy_extend:
            _extend_y_boundary[bcount_x, bs1d_x](d_S, yc, xc, undef)
            if not bcx_periodic:
                # _extend_x_boundary now also handles the four corners, so a
                # separate 1-block corners kernel is no longer needed.
                _extend_x_boundary[bcount_y, bs1d_y](d_S, yc, xc, undef)

        # --- Red + Black SOR update ---
        _sor_2d_rb[blocks, threads](
            d_S, d_A, d_B, d_C, d_F, yc, xc, bcx_periodic,
            delxSqr, ratioQtr, ratioSqr, optArg, undef, 0)
        _sor_2d_rb[blocks, threads](
            d_S, d_A, d_B, d_C, d_F, yc, xc, bcx_periodic,
            delxSqr, ratioQtr, ratioSqr, optArg, undef, 1)

        loop += 1

        # --- convergence / overflow check (only every check_interval iters) ---
        is_check_iter = (loop % check_interval == 0) or (loop >= mxLoop)

        if not is_check_iter:
            continue  # fast path: skip norm computation entirely

        d_norm[0] = 0.0
        d_norm[1] = 0.0
        _abs_norm_2d[blocks, threads](d_S, undef, d_norm)
        norm_h = d_norm.copy_to_host()

        norm, error, overflow = _evaluate_gpu_norm(
            norm_h[0], norm_h[1], norm_prev, need_convergence, tolerance)

        if overflow:
            break

        if (need_convergence and error < tolerance) or loop >= mxLoop or norm == 0:
            break

        norm_prev = norm

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
