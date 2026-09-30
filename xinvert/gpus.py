# -*- coding: utf-8 -*-
"""
GPU module of xinvert: CUDA-accelerated SOR iteration kernels.

Contains GPU implementations of the SOR iteration kernels using numba.cuda.
Uses dependency-safe multi-color ordering for parallel SOR iteration on GPU:
red-black for nearest-neighbour stencils, four colors for mixed-derivative
nine-point stencils, and five colors for the biharmonic stencil.

Currently implemented:
  - invert_standard_2D_gpu:      GPU version of invert_standard_2D (Poisson, etc.)
  - invert_standard_2D_full_gpu: GPU version of invert_standard_2D_full
                                 (divergence form + Helmholtz term; Bretherton,
                                 Fofonoff, GillMatsunoFlux, StommelFlux)
  - invert_standard_1D_gpu:      GPU version of invert_standard_1D
                                 (GeoAdjustment, RefStateSWM)
  - invert_general_2D_gpu:       GPU version of invert_general_2D
                                 (GillMatsuno, Stommel, StommelArons)
  - invert_standard_3D_gpu:      GPU version of invert_standard_3D (omega)
  - invert_general_3D_gpu:       GPU version of invert_general_3D (3DOcean)
  - invert_general_bih_2D_gpu:   GPU version of invert_general_bih_2D
                                 (StommelMunk); 5-color SOR for the
                                 13-point biharmonic stencil

The GPU wrapper functions have the **same signature** as the numba kernels in
:mod:`xinvert.cpus`, so they can be used as drop-in replacements via the
``architect`` dispatch in :mod:`xinvert.core._make_kernel`.
"""
import numpy as np
import threading
import warnings

from numba import cuda

# Numba emits a ``NumbaPerformanceWarning`` at every kernel launch whose
# grid is too small to fill the GPU ("Grid size N will likely result in
# GPU under-utilization due to low occupancy").  In xinvert this can only
# happen for SMALL problems (fewer blocks than the GPU has SMs), where
# under-utilisation is inherent to the problem size rather than the
# implementation; for large problems the warning never fires.  Silencing
# it therefore loses no information.  Filter by message text because the
# warning category's module path varies across numba / numba-cuda
# versions, and it would otherwise fire thousands of times (once per
# unique grid size) during a single solve.
warnings.filterwarnings('ignore', message='.*GPU under-utilization.*')


# Default 2D thread-block shape: 256 threads/block, square shape → best
# cache locality for the 2-D stencil (neighbours in both x and y stay
# within the block).  A block sweep (tests/benchmark_blocks.py) showed
# (16,16) is ~3 % faster than (32,8) at 4096^2 despite the latter being
# warp-coalesced, because stencil access is 2-D, not row-stride.
# Override per call via ``iParams['gpu_block2d'] = (bx, by)``.
_DEFAULT_BLOCK_2D = (16, 16)


# ---------------------------------------------------------------------------
# CUDA kernels for adaptive two/four-color SOR (standard 2D forms)
# ---------------------------------------------------------------------------

@cuda.jit
def _sor_2d_rb(S, A, B, C, F, yc, xc, bcx_periodic,
               delxSqr, ratioQtr, ratioSqr, optArg, undef, color, n_color,
               metric, track_residual):
    """Multi-color SOR update for one color of the standard 2D form.

    A five-point operator uses classic red-black coloring.  When ``B`` is
    non-zero the mixed derivative adds diagonal neighbours, so the wrapper
    selects a four-color ``(j % 2, i % 2)`` scheme instead.  Points in one
    four-color pass then share neither edges nor corners.

    The update formula matches the interior loop of
    :func:`xinvert.cpus.invert_standard_2D`.

    Parameters
    ----------
    S : cuda.device_array (modified in-place)
        Solution array, shape (yc, xc).
    A, B, C : cuda.device_array
        Coefficient arrays, same shape as S.
    F : cuda.device_array
        Forcing array, same shape as S.
    yc, xc : int
        Grid counts in y and x dimensions.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSqr, ratioQtr, ratioSqr : float
        Grid spacing parameters.
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (points needing undefined coefficients are skipped).
    color : int
        0 = update red points ((j+i) even), 1 = black points ((j+i) odd).
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

    if n_color == 2:
        if (j + i) % 2 != color:
            return
    else:
        if (j % 2) * 2 + (i % 2) != color:
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
        delta = temp * optArg / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(metric, 1,
                            max(abs(S[j, i]), abs(S[j, i] + delta)))
        S[j, i] += delta


@cuda.jit
def _sor_2d_rb_full(S, A, B, C, D, E, F, yc, xc, bcx_periodic,
                    delxSqr, ratioQtr, ratioSqr, optArg, undef, color,
                    n_color, metric, track_residual):
    """Multi-color SOR update for one color of the full standard 2D form.

    Solves  (A ψy + B ψx)/y + (C ψy + D ψx)/x + E ψ = F  with coefficients
    at staggered (half-grid) positions -- the divergence form including the
    Helmholtz term E ψ.  This is the generalisation of :func:`_sor_2d_rb`
    used by Bretherton-Haidvogel, Fofonoff, GillMatsunoFlux and
    StommelFlux; setting C≡B and E≡0 recovers :func:`_sor_2d_rb`.

    The update formula matches the interior loop of
    :func:`xinvert.cpus.invert_standard_2D_full`.

    Parameters
    ----------
    S : cuda.device_array (modified in-place)
        Solution array, shape (yc, xc).
    A, B, C, D, E : cuda.device_array
        Coefficient arrays, same shape as S.
    F : cuda.device_array
        Forcing array, same shape as S.
    yc, xc : int
        Grid counts in y and x dimensions.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSqr, ratioQtr, ratioSqr : float
        Grid spacing parameters.
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (points needing undefined coefficients are skipped).
    color : int
        0 = update red points ((j+i) even), 1 = black points ((j+i) odd).
    """
    j, i = cuda.grid(2)

    if j < 1 or j >= yc - 1:
        return

    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 1 or i >= xc - 1:
            return

    if n_color == 2:
        if (j + i) % 2 != color:
            return
    else:
        if (j % 2) * 2 + (i % 2) != color:
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
        B[j + 1, i] == undef or B[j - 1, i] == undef or
        C[j, ip1] == undef or C[j, im1] == undef or
        D[j, ip1] == undef or D[j, i] == undef or
        E[j, i] == undef):
        return

    # Compute SOR residual (identical to numba kernel interior formula)
    temp = (
        (A[j + 1, i] * (S[j + 1, i] - S[j, i]) -
         A[j, i] * (S[j, i] - S[j - 1, i])) * ratioSqr +
        (B[j + 1, i] * (S[j + 1, ip1] - S[j + 1, im1]) -
         B[j - 1, i] * (S[j - 1, ip1] - S[j - 1, im1])) * ratioQtr +
        (C[j, ip1] * (S[j + 1, ip1] - S[j - 1, ip1]) -
         C[j, im1] * (S[j + 1, im1] - S[j - 1, im1])) * ratioQtr +
        (D[j, ip1] * (S[j, ip1] - S[j, i]) -
         D[j, i] * (S[j, i] - S[j, im1]))
    ) + (E[j, i] * S[j, i] - F[j, i]) * delxSqr

    denom = ((A[j + 1, i] + A[j, i]) * ratioSqr +
             (D[j, ip1] + D[j, i]) - E[j, i] * delxSqr)
    if denom != 0.0:
        delta = temp * optArg / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(metric, 1,
                            max(abs(S[j, i]), abs(S[j, i] + delta)))
        S[j, i] += delta


@cuda.jit
def _sor_2d_rb_general(S, A, B, C, D, E, F, G, yc, xc, bcx_periodic,
                       delxSqr, delx, ratio, ratioQtr, ratioSqr,
                       optArg, undef, color, n_color, metric, track_residual):
    """Multi-color SOR update for one color of the general 2D form.

    Solves  A ψyy + B ψyx + C ψxx + D ψy + E ψx + F ψ = G  with all
    coefficients at grid centers (non-divergence / point form).  The
    update formula matches the loops of
    :func:`xinvert.cpus.invert_general_2D` (interior and both periodic
    boundary points share one expression via wrapped indices).

    Parameters
    ----------
    S : cuda.device_array (modified in-place)
        Solution array, shape (yc, xc).
    A, B, C, D, E, F : cuda.device_array
        Coefficient arrays, same shape as S.
    G : cuda.device_array
        Forcing array, same shape as S.
    yc, xc : int
        Grid counts in y and x dimensions.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSqr, delx, ratio, ratioQtr, ratioSqr : float
        Grid spacing parameters.
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (points needing undefined coefficients are skipped).
    color : int
        0 = update red points ((j+i) even), 1 = black points ((j+i) odd).
    """
    j, i = cuda.grid(2)

    if j < 1 or j >= yc - 1:
        return

    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 1 or i >= xc - 1:
            return

    if n_color == 2:
        if (j + i) % 2 != color:
            return
    else:
        if (j % 2) * 2 + (i % 2) != color:
            return

    # Periodic wrapping for x-neighbours
    ip1 = i + 1
    im1 = i - 1
    if ip1 >= xc:
        ip1 = 0
    if im1 < 0:
        im1 = xc - 1

    # Check that all needed coefficients are defined (all at grid centers)
    if (G[j, i] == undef or A[j, i] == undef or B[j, i] == undef or
        C[j, i] == undef or D[j, i] == undef or E[j, i] == undef or
        F[j, i] == undef):
        return

    # Compute SOR residual (identical to numba kernel interior formula)
    temp = (
        A[j, i] * ((S[j + 1, i] - S[j, i]) - (S[j, i] - S[j - 1, i]))
    ) * ratioSqr + (
        B[j, i] * ((S[j + 1, ip1] - S[j - 1, ip1]) -
                   (S[j + 1, im1] - S[j - 1, im1]))
    ) * ratioQtr + (
        C[j, i] * ((S[j, ip1] - S[j, i]) - (S[j, i] - S[j, im1]))
    ) + (
        (D[j, i] * (S[j + 1, i] - S[j - 1, i])) * ratio +
        (E[j, i] * (S[j, ip1] - S[j, im1]))
    ) * delx / 2.0 + (F[j, i] * S[j, i] - G[j, i]) * delxSqr

    denom = (A[j, i] * ratioSqr + C[j, i]) * 2.0 - F[j, i] * delxSqr
    if denom != 0.0:
        delta = temp * optArg / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(metric, 1,
                            max(abs(S[j, i]), abs(S[j, i] + delta)))
        S[j, i] += delta


@cuda.jit
def _abs_norm_2d(S, undef, out):
    """Compute sum(|S|) and count of non-undef points (atomic reduction).

    Parameters
    ----------
    S : cuda.device_array
        Solution array, shape (yc, xc).
    undef : float
        Undefined value (masked points are excluded from the norm).
    out : cuda.device_array, shape (2,)
        Atomic accumulators: out[0] = sum(|S|), out[1] = valid point count.
    """
    j, i = cuda.grid(2)
    if j < S.shape[0] and i < S.shape[1]:
        if S[j, i] != undef:
            cuda.atomic.add(out, 0, abs(S[j, i]))
            cuda.atomic.add(out, 1, 1.0)


@cuda.jit
def _extend_y_boundary(S, yc, xc, undef, bcx_periodic,
                       anchor_side, anchor_idx):
    """Extend BC: copy y-interior boundary to y-outer boundary.

    The gauge-anchor column (singular extend systems) is skipped, so
    the pinned boundary point keeps its initial (uploaded) value.

    Parameters
    ----------
    S : cuda.device_array (modified in-place)
        Solution array, shape (yc, xc).
    yc, xc : int
        Grid counts in y and x dimensions.
    undef : float
        Undefined value (undef cells are not overwritten).
    anchor_side, anchor_idx : int
        Boundary gauge anchor (side 0 = north row, 1 = south row); the
        copy of S[anchor] is skipped.  side < 0 = no anchor.
    """
    i = cuda.grid(1)
    if i < xc and (bcx_periodic or (0 < i < xc - 1)):
        if S[1, i] != undef and not (anchor_side == 0 and i == anchor_idx):
            S[0, i] = S[1, i]
        if S[yc - 2, i] != undef and not (anchor_side == 1 and i == anchor_idx):
            S[yc - 1, i] = S[yc - 2, i]


@cuda.jit
def _extend_x_boundary(S, yc, xc, undef, bcy_extend,
                       anchor_side, anchor_idx):
    """Extend BC: copy x-interior boundary to x-outer boundary, plus corners.

    Folding corner handling into this kernel avoids a separate 1-block
    launch (which triggers a ``Grid size 1`` under-utilization warning)
    and saves one kernel launch per iteration.  The gauge-anchor row
    (side 2 = west column, 3 = east column) is skipped.

    Parameters
    ----------
    S : cuda.device_array (modified in-place)
        Solution array, shape (yc, xc).
    yc, xc : int
        Grid counts in y and x dimensions.
    undef : float
        Undefined value (undef cells are not overwritten).
    anchor_side, anchor_idx : int
        Boundary gauge anchor; the copy of S[anchor] is skipped.
        side < 0 = no anchor.
    """
    j = cuda.grid(1)
    if j >= yc:
        return
    if j == 0:
        if not bcy_extend:
            return
        # top-left / top-right corners (diagonal neighbour)
        if S[1, 1] != undef:
            S[0, 0] = S[1, 1]
        if S[1, xc - 2] != undef:
            S[0, xc - 1] = S[1, xc - 2]
    elif j == yc - 1:
        if not bcy_extend:
            return
        # bottom-left / bottom-right corners
        if S[yc - 2, 1] != undef:
            S[yc - 1, 0] = S[yc - 2, 1]
        if S[yc - 2, xc - 2] != undef:
            S[yc - 1, xc - 1] = S[yc - 2, xc - 2]
    else:
        # interior rows: left and right edges
        if S[j, 1] != undef and not (anchor_side == 2 and j == anchor_idx):
            S[j, 0] = S[j, 1]
        if S[j, xc - 2] != undef and not (anchor_side == 3 and j == anchor_idx):
            S[j, xc - 1] = S[j, xc - 2]


# ---------------------------------------------------------------------------
# CUDA kernels for Red-Black SOR (standard 1D form)
# ---------------------------------------------------------------------------

@cuda.jit
def _sor_1d_rb(S, A, B, F, xc, bcx_periodic,
               delxSqr, optArg, undef, color, metric, track_residual):
    """Red-Black SOR update for one color of the standard 1D form.

    Solves  (A psi_x)/x + B psi = F  with A at staggered (half-grid)
    positions.  The update formula matches the loops of
    :func:`xinvert.cpus.invert_standard_1D` (interior and both periodic
    boundary points share one expression via wrapped indices).

    Parameters
    ----------
    S : cuda.device_array (modified in-place), shape (xc,)
    A, B, F : cuda.device_array, shape (xc,)
        Coefficients and forcing.
    xc : int
        Grid count in x dimension.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSqr : float
        Squared grid spacing.
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (skipped points).
    color : int
        0 = red points (i even), 1 = black points (i odd).
    """
    i = cuda.grid(1)

    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 1 or i >= xc - 1:
            return

    if i % 2 != color:
        return

    ip1 = i + 1
    im1 = i - 1
    if ip1 >= xc:
        ip1 = 0
    if im1 < 0:
        im1 = xc - 1

    # Check that all needed coefficients are defined
    if F[i] == undef or A[i] == undef or A[ip1] == undef or B[i] == undef:
        return

    temp = (A[ip1] * (S[ip1] - S[i]) - A[i] * (S[i] - S[im1])) / delxSqr \
        + (B[i] * S[i] - F[i])

    denom = (A[ip1] + A[i]) / delxSqr - B[i]
    if denom != 0.0:
        delta = temp * optArg / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(metric, 1, max(abs(S[i]), abs(S[i] + delta)))
        S[i] += delta


@cuda.jit
def _abs_norm_1d(S, undef, out):
    """1D atomic reduction: out[0] = sum(|S|), out[1] = valid point count."""
    i = cuda.grid(1)
    if i < S.shape[0]:
        if S[i] != undef:
            cuda.atomic.add(out, 0, abs(S[i]))
            cuda.atomic.add(out, 1, 1.0)


@cuda.jit
def _extend_boundary_1d(S, xc, undef, anchor_side):
    """Extend BC for 1D: copy interior boundary to outer boundary.

    The gauge-anchor end (side 0 = S[0], 1 = S[-1]) is skipped so it
    keeps its initial (uploaded) value."""
    if cuda.grid(1) == 0:
        if S[1] != undef and anchor_side != 0:
            S[0] = S[1]
        if S[xc - 2] != undef and anchor_side != 1:
            S[xc - 1] = S[xc - 2]


# ---------------------------------------------------------------------------
# CUDA kernels for Red-Black SOR (standard 3D form)
# ---------------------------------------------------------------------------

@cuda.jit
def _sor_3d_rb(S, A, B, C, F, zc, yc, xc, bcx_periodic,
               delxSqr, ratio2Sqr, ratio1Sqr, optArg, undef, color,
               metric, track_residual):
    """Red-Black SOR update for one color of the standard 3D form.

    Solves  (A ψz)/z + (B ψy)/y + (C ψx)/x = F  with coefficients at
    staggered (half-grid) positions.  The update formula matches the
    interior loop of :func:`xinvert.cpus.invert_standard_3D`.

    The red/black coloring is the (k + j + i) parity: every neighbour in
    the 7-point stencil differs by ±1 in exactly one index and therefore
    has the opposite color.

    Parameters
    ----------
    S : cuda.device_array (modified in-place), shape (zc, yc, xc).
    A, B, C, F : cuda.device_array
        Coefficient / forcing arrays, same shape as S.
    zc, yc, xc : int
        Grid counts in z, y and x dimensions.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSqr, ratio2Sqr, ratio1Sqr : float
        Grid spacing parameters (ratio2 = delx/delz, ratio1 = delx/dely).
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (skipped points).
    color : int
        0 = red points ((k+j+i) even), 1 = black points.
    """
    k, j, i = cuda.grid(3)

    # z / y boundaries are fixed (provided via icbc) and never updated
    if k < 1 or k >= zc - 1:
        return
    if j < 1 or j >= yc - 1:
        return

    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 1 or i >= xc - 1:
            return

    if (k + j + i) % 2 != color:
        return

    # Periodic wrapping for x-neighbours
    ip1 = i + 1
    im1 = i - 1
    if ip1 >= xc:
        ip1 = 0
    if im1 < 0:
        im1 = xc - 1

    # Check that all needed coefficients are defined
    if (F[k, j, i] == undef or
        A[k + 1, j, i] == undef or A[k, j, i] == undef or
        B[k, j + 1, i] == undef or B[k, j, i] == undef or
        C[k, j, ip1] == undef or C[k, j, i] == undef):
        return

    # Compute SOR residual (identical to numba kernel interior formula)
    temp = (
        (A[k + 1, j, i] * (S[k + 1, j, i] - S[k, j, i]) -
         A[k, j, i] * (S[k, j, i] - S[k - 1, j, i])) * ratio2Sqr +
        (B[k, j + 1, i] * (S[k, j + 1, i] - S[k, j, i]) -
         B[k, j, i] * (S[k, j, i] - S[k, j - 1, i])) * ratio1Sqr +
        (C[k, j, ip1] * (S[k, j, ip1] - S[k, j, i]) -
         C[k, j, i] * (S[k, j, i] - S[k, j, im1]))
    ) - F[k, j, i] * delxSqr

    denom = ((A[k + 1, j, i] + A[k, j, i]) * ratio2Sqr +
             (B[k, j + 1, i] + B[k, j, i]) * ratio1Sqr +
             (C[k, j, ip1] + C[k, j, i]))
    if denom != 0.0:
        delta = temp * optArg / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(
                metric, 1, max(abs(S[k, j, i]), abs(S[k, j, i] + delta)))
        S[k, j, i] += delta


@cuda.jit
def _abs_norm_3d(S, undef, out):
    """3D atomic reduction: out[0] = sum(|S|), out[1] = valid point count."""
    k, j, i = cuda.grid(3)
    if k < S.shape[0] and j < S.shape[1] and i < S.shape[2]:
        if S[k, j, i] != undef:
            cuda.atomic.add(out, 0, abs(S[k, j, i]))
            cuda.atomic.add(out, 1, 1.0)


@cuda.jit
def _extend_y_boundary_3d(S, zc, yc, xc, undef, bcx_periodic):
    """3D extend BC: copy y-interior boundary to y-outer boundary.

    Sweeped over (k, i) for interior k levels; covers the full i range
    (corners are later overwritten by the x kernel, mirroring the CPU).
    """
    k, i = cuda.grid(2)
    if (k < 1 or k >= zc - 1 or i >= xc or
            (not bcx_periodic and (i == 0 or i == xc - 1))):
        return
    if S[k, 1, i] != undef:
        S[k, 0, i] = S[k, 1, i]
    if S[k, yc - 2, i] != undef:
        S[k, yc - 1, i] = S[k, yc - 2, i]


@cuda.jit
def _extend_x_boundary_3d(S, zc, yc, xc, undef, bcy_extend):
    """3D extend BC: copy x-interior boundary to x-outer boundary + corners.

    Sweeped over (k, j); folds corner handling in to avoid a separate
    1-block launch (mirrors the 2D :func:`_extend_x_boundary`).
    """
    k, j = cuda.grid(2)
    if k < 1 or k >= zc - 1 or j >= yc:
        return
    if j == 0:
        if not bcy_extend:
            return
        # top-left / top-right corners (diagonal neighbour)
        if S[k, 1, 1] != undef:
            S[k, 0, 0] = S[k, 1, 1]
        if S[k, 1, xc - 2] != undef:
            S[k, 0, xc - 1] = S[k, 1, xc - 2]
    elif j == yc - 1:
        if not bcy_extend:
            return
        # bottom-left / bottom-right corners
        if S[k, yc - 2, 1] != undef:
            S[k, yc - 1, 0] = S[k, yc - 2, 1]
        if S[k, yc - 2, xc - 2] != undef:
            S[k, yc - 1, xc - 1] = S[k, yc - 2, xc - 2]
    else:
        # interior rows: left and right edges
        if S[k, j, 1] != undef:
            S[k, j, 0] = S[k, j, 1]
        if S[k, j, xc - 2] != undef:
            S[k, j, xc - 1] = S[k, j, xc - 2]


@cuda.jit
def _sor_3d_rb_general(S, A, B, C, D, E, F, G, H, zc, yc, xc, bcx_periodic,
                       delxSqr, delx, ratio2, ratio1, ratio2Sqr, ratio1Sqr,
                       optArg, undef, color, metric, track_residual):
    """Red-Black SOR update for one color of the general 3D form.

    Solves  A ψzz + B ψyy + C ψxx + D ψz + E ψy + F ψx + G ψ = H  with all
    coefficients at grid centers (non-divergence / point form).  The
    update formula matches the interior loop of
    :func:`xinvert.cpus.invert_general_3D`.

    The red/black coloring is the (k + j + i) parity.

    Parameters
    ----------
    S : cuda.device_array (modified in-place), shape (zc, yc, xc).
    A, B, C, D, E, F, G : cuda.device_array
        Coefficient arrays, same shape as S.
    H : cuda.device_array
        Forcing array, same shape as S.
    zc, yc, xc : int
        Grid counts in z, y and x dimensions.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSqr, delx, ratio2, ratio1, ratio2Sqr, ratio1Sqr : float
        Grid spacing parameters.
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (skipped points).
    color : int
        0 = red points ((k+j+i) even), 1 = black points.
    """
    k, j, i = cuda.grid(3)

    if k < 1 or k >= zc - 1:
        return
    if j < 1 or j >= yc - 1:
        return

    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 1 or i >= xc - 1:
            return

    if (k + j + i) % 2 != color:
        return

    # Periodic wrapping for x-neighbours
    ip1 = i + 1
    im1 = i - 1
    if ip1 >= xc:
        ip1 = 0
    if im1 < 0:
        im1 = xc - 1

    # Check that all needed coefficients are defined (all at grid centers)
    if (H[k, j, i] == undef or A[k, j, i] == undef or B[k, j, i] == undef or
        C[k, j, i] == undef or D[k, j, i] == undef or E[k, j, i] == undef or
        F[k, j, i] == undef or G[k, j, i] == undef):
        return

    # Compute SOR residual (identical to numba kernel interior formula)
    temp = (
        A[k, j, i] * ((S[k + 1, j, i] - S[k, j, i]) -
                      (S[k, j, i] - S[k - 1, j, i]))
    ) * ratio2Sqr + (
        B[k, j, i] * ((S[k, j + 1, i] - S[k, j, i]) -
                      (S[k, j, i] - S[k, j - 1, i]))
    ) * ratio1Sqr + (
        C[k, j, i] * ((S[k, j, ip1] - S[k, j, i]) -
                      (S[k, j, i] - S[k, j, im1]))
    ) + (
        (D[k, j, i] * (S[k + 1, j, i] - S[k - 1, j, i])) * ratio2 +
        (E[k, j, i] * (S[k, j + 1, i] - S[k, j - 1, i])) * ratio1 +
        (F[k, j, i] * (S[k, j, ip1] - S[k, j, im1]))
    ) * delx / 2.0 + (G[k, j, i] * S[k, j, i] - H[k, j, i]) * delxSqr

    denom = ((A[k, j, i] * ratio2Sqr + B[k, j, i] * ratio1Sqr + C[k, j, i])
             * 2.0 - G[k, j, i] * delxSqr)
    if denom != 0.0:
        delta = temp * optArg / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(
                metric, 1, max(abs(S[k, j, i]), abs(S[k, j, i] + delta)))
        S[k, j, i] += delta


# ---------------------------------------------------------------------------
# CUDA kernels for multi-color SOR (general biharmonic 2D form)
# ---------------------------------------------------------------------------

@cuda.jit
def _sor_2d_mc_bih(S, A, B, C, D, E, F, G, H, I, J, yc, xc, bcx_periodic,
                   delxSSr, delxTr, delxSqr,
                   ratio, ratioSSr, ratioQtr, ratioSqr,
                   optArg, undef, color, metric, track_residual):
    """One color (of five) of the multi-color SOR update for the
    general biharmonic 2D form.

    Solves  A ψyyyy + B ψyyxx + C ψxxxx + D ψyy + E ψyx + F ψxx
            + G ψy + H ψx + I ψ = J  with all coefficients at grid
    centers.  The update formula matches the interior loop of
    :func:`xinvert.cpus.invert_general_bih_2D`.

    Coloring: the stencil couples axial neighbours at distances one and two,
    first diagonals, and ``(+/-2, +/-2)`` neighbours.  The previous
    ``(j+i) % 3`` scheme assigned ``(j+2, i-2)`` the same color as ``(j,i)``.
    ``(j + 2*i) % 5`` gives every possible non-zero stencil offset a non-zero
    color difference, so one kernel launch cannot read a point being updated
    by another thread in that launch.

    Parameters
    ----------
    S : cuda.device_array (modified in-place), shape (yc, xc).
    A, B, C, D, E, F, G, H, I : cuda.device_array
        Coefficient arrays, same shape as S.
    J : cuda.device_array
        Forcing array, same shape as S.
    yc, xc : int
        Grid counts in y and x dimensions.
    bcx_periodic : bool
        True if the x boundary is periodic (neighbours wrap around).
    delxSSr, delxTr, delxSqr : float
        delx**4, delx**3 and delx**2.
    ratio, ratioSSr, ratioQtr, ratioSqr : float
        Grid-spacing ratio parameters (ratio = delx/dely).
    optArg : float
        SOR relaxation factor omega (1 ~ 2).
    undef : float
        Undefined value (skipped points).
    color : int
        0 through 4: update points with ``(j + 2*i) % 5 == color``.
    """
    j, i = cuda.grid(2)

    # 2-cell-wide interior: the y boundary rows 0,1,yc-2,yc-1 are fixed
    if j < 2 or j >= yc - 2:
        return

    if bcx_periodic:
        if i < 0 or i >= xc:
            return
    else:
        if i < 2 or i >= xc - 2:
            return

    if (j + 2 * i) % 5 != color:
        return

    # Periodic wrapping for x-neighbours (up to +-2 cells)
    ip1 = i + 1
    im1 = i - 1
    ip2 = i + 2
    im2 = i - 2
    if ip1 >= xc:
        ip1 -= xc
    if im1 < 0:
        im1 += xc
    if ip2 >= xc:
        ip2 -= xc
    if im2 < 0:
        im2 += xc

    # Check that all needed coefficients are defined (all at grid centers)
    if (A[j, i] == undef or B[j, i] == undef or C[j, i] == undef or
        D[j, i] == undef or E[j, i] == undef or F[j, i] == undef or
        G[j, i] == undef or H[j, i] == undef or I[j, i] == undef or
        J[j, i] == undef):
        return

    # Compute SOR residual (identical to numba kernel interior formula)
    temp = (
        A[j, i] * (S[j + 2, i] - 4.0 * S[j + 1, i] + 6.0 * S[j, i]
                   - 4.0 * S[j - 1, i] + S[j - 2, i])
    ) * ratioSSr + (
        B[j, i] * (S[j + 2, ip2] - 2.0 * S[j + 2, i] + S[j + 2, im2]
                   - 2.0 * S[j, ip2] + 4.0 * S[j, i] - 2.0 * S[j, im2]
                   + S[j - 2, ip2] - 2.0 * S[j - 2, i] + S[j - 2, im2])
    ) * ratioSqr / 16.0 + (
        C[j, i] * (S[j, ip2] - 4.0 * S[j, ip1] + 6.0 * S[j, i]
                   - 4.0 * S[j, im1] + S[j, im2])
    ) + (
        D[j, i] * ((S[j + 1, i] - S[j, i]) - (S[j, i] - S[j - 1, i]))
    ) * ratioSqr * delxSqr + (
        E[j, i] * ((S[j + 1, ip1] - S[j - 1, ip1]) -
                   (S[j + 1, im1] - S[j - 1, im1]))
    ) * ratioQtr * delxSqr + (
        F[j, i] * ((S[j, ip1] - S[j, i]) - (S[j, i] - S[j, im1]))
    ) * delxSqr + (
        G[j, i] * (S[j + 1, i] - S[j - 1, i])
    ) * delxTr * ratio / 2.0 + (
        H[j, i] * (S[j, ip1] - S[j, im1])
    ) * delxTr / 2.0 + (
        I[j, i] * S[j, i] - J[j, i]
    ) * delxSSr

    denom = ((A[j, i] * ratioSSr + C[j, i]) * 6.0 +
             B[j, i] * ratioSqr / 4.0 +
             -(D[j, i] * ratioSqr + F[j, i]) * 2.0 * delxSqr +
             I[j, i] * delxSSr)
    if denom != 0.0:
        delta = temp * (-optArg) / denom
        if track_residual:
            cuda.atomic.max(metric, 0, abs(delta))
            cuda.atomic.max(metric, 1,
                            max(abs(S[j, i]), abs(S[j, i] + delta)))
        S[j, i] += delta


@cuda.jit
def _extend_y_boundary_2d_bih(S, yc, xc, undef, bcx_periodic,
                              anchor_side, anchor_idx):
    """Biharmonic extend BC: copy the 2-cell y boundary (rows 0,1 / yc-2,yc-1).

    Mirrors the CPU kernel: with periodic x, S[0] takes the old S[1];
    otherwise both boundary rows take S[2].  Non-periodic x skips the
    corner columns (handled by the x kernel).  The gauge-anchor column
    keeps its OUTER row pinned (skipped copy); the inner boundary row
    (S[1] / S[yc-2]) still extends.
    """
    i = cuda.grid(1)
    if i >= xc:
        return
    if not bcx_periodic and (i < 2 or i >= xc - 2):
        return  # corners handled by the x kernel
    if S[2, i] != undef:
        if not (anchor_side == 0 and i == anchor_idx):
            if bcx_periodic:
                S[0, i] = S[1, i]
            else:
                S[0, i] = S[2, i]
        S[1, i] = S[2, i]
    if S[yc - 3, i] != undef:
        if not (anchor_side == 1 and i == anchor_idx):
            S[yc - 1, i] = S[yc - 3, i]
        S[yc - 2, i] = S[yc - 3, i]


@cuda.jit
def _extend_x_boundary_2d_bih(S, yc, xc, undef, bcy_extend,
                              anchor_side, anchor_idx):
    """Biharmonic extend BC: copy the 2-cell x boundary (+ 2x2 corners).

    Mirrors the CPU kernel's non-periodic branch.  NOTE the corner blocks
    span TWO rows (0,1 and yc-2,yc-1), so the interior-row branch must
    start at j == 2: otherwise threads j == 1 / j == yc-2 would race with
    the corner threads on row 1 / row yc-2.  The gauge-anchor row keeps
    its OUTER column pinned (skipped copy).
    """
    j = cuda.grid(1)
    if j >= yc:
        return
    if 2 <= j < yc - 2:
        # interior rows: left and right 2-cell edges
        if S[j, 2] != undef:
            if not (anchor_side == 2 and j == anchor_idx):
                S[j, 0] = S[j, 2]
            S[j, 1] = S[j, 2]
        if S[j, xc - 3] != undef:
            if not (anchor_side == 3 and j == anchor_idx):
                S[j, xc - 1] = S[j, xc - 3]
            S[j, xc - 2] = S[j, xc - 3]
    elif j == 0:
        if not bcy_extend:
            return
        # top 2x2 corners from the (2, 2) / (2, xc-3) diagonal values
        if S[2, 2] != undef:
            S[0, 0] = S[2, 2]
            S[1, 0] = S[2, 2]
            S[0, 1] = S[2, 2]
            S[1, 1] = S[2, 2]
        if S[2, xc - 3] != undef:
            S[0, xc - 1] = S[2, xc - 3]
            S[1, xc - 1] = S[2, xc - 3]
            S[0, xc - 2] = S[2, xc - 3]
            S[1, xc - 2] = S[2, xc - 3]
    elif j == yc - 1:
        if not bcy_extend:
            return
        # bottom 2x2 corners from the (yc-3, 2) / (yc-3, xc-3) values
        if S[yc - 3, 2] != undef:
            S[yc - 1, 0] = S[yc - 3, 2]
            S[yc - 2, 0] = S[yc - 3, 2]
            S[yc - 1, 1] = S[yc - 3, 2]
            S[yc - 2, 1] = S[yc - 3, 2]
        if S[yc - 3, xc - 3] != undef:
            S[yc - 1, xc - 1] = S[yc - 3, xc - 3]
            S[yc - 2, xc - 1] = S[yc - 3, xc - 3]
            S[yc - 1, xc - 2] = S[yc - 3, xc - 3]
            S[yc - 2, xc - 2] = S[yc - 3, xc - 3]


def _find_boundary_anchor_2d_host(F, undef, allow_we):
    """Host-side BOUNDARY gauge-anchor lookup (mirrors
    cpus._find_boundary_anchor_2d).

    Returns (side, idx): side 0=N (S[0, idx]), 1=S (S[yc-1, idx]),
    2=W (S[idx, 0]), 3=E (S[idx, xc-1]); (-1, -1) if none.  The anchor
    is a BOUNDARY point whose extend-copy is skipped, so it keeps its
    initial (uploaded) value -- a Dirichlet gauge that removes the
    constant null space of singular extend systems while every interior
    point keeps updating normally.
    """
    valid_n = F[1, 1:-1] != undef
    if valid_n.any():
        return 0, int(np.argmax(valid_n)) + 1
    valid_s = F[-2, 1:-1] != undef
    if valid_s.any():
        return 1, int(np.argmax(valid_s)) + 1
    if allow_we:
        valid_w = F[1:-1, 1] != undef
        if valid_w.any():
            return 2, int(np.argmax(valid_w)) + 1
        valid_e = F[1:-1, -2] != undef
        if valid_e.any():
            return 3, int(np.argmax(valid_e)) + 1
    return -1, -1


def _find_boundary_anchor_1d_host(F, undef):
    """1D counterpart: 0 = pin S[0], 1 = pin S[-1], -1 = none."""
    if F[1] != undef:
        return 0
    if F[-2] != undef:
        return 1
    return -1


# ---------------------------------------------------------------------------
# Shared host-side helpers (reused by 2D/3D/general GPU wrappers)
# ---------------------------------------------------------------------------

def ensure_context():
    """Initialise the CUDA context in the calling thread.

    numba-cuda manages contexts thread-locally.  When the input dataset is
    dask-backed, ``dask='parallelized'`` runs each per-time-step solve in a
    dask worker thread; if the context has never been created in the main
    thread, the first launch inside a worker thread fails with
    ``CUDA_ERROR_NOT_INITIALIZED``.  Call this once from the main thread
    (``core._make_kernel`` does it when dispatching to GPU) before handing
    GPU tasks to dask.
    """
    # ``get_current_device`` is unavailable under Numba's CUDA simulator;
    # ``current_context`` still forces/validates context creation there.
    get_device = getattr(cuda, 'get_current_device', None)
    if get_device is not None:
        get_device()
    else:
        cuda.current_context()


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


def _has_active_coefficient(*coefficients):
    """Return whether any host coefficient array contains a non-zero value.

    Mixed-derivative coefficients determine the dependency graph of the 2-D
    stencil.  This inexpensive host-side check lets the common five-point
    case retain two-color red-black SOR while selecting four colors whenever
    diagonal neighbours are actually coupled.
    """
    return any(np.any(np.asarray(coef) != 0) for coef in coefficients)


def _validate_periodic_x_coloring(xc, BCx, n_color):
    """Reject periodic grids incompatible with the selected linear coloring.

    Wrapping must preserve the color assigned to a logical column.  Parity
    colorings therefore require an even x extent; the biharmonic five-color
    mapping requires an extent divisible by five.  Without this check the
    seam can connect two points updated by the same kernel launch.
    """
    if BCx != 'periodic':
        return
    period = 5 if n_color == 5 else 2
    if xc % period != 0:
        raise ValueError(
            f"GPU {n_color}-color SOR with periodic x requires the x grid "
            f"length to be divisible by {period}; got {xc}.  Use a compatible "
            "grid size or the CPU backend."
        )


# Max number of GPU solves running concurrently within one process.
# Concurrent solves gain nothing (all kernels serialise on the default
# stream, and the blocking convergence-check syncs create a convoy
# effect -- measured ~0.4x on small grids) while each in-flight solve
# holds ~5 device buffers, so VRAM grows linearly with concurrency.
# One at a time is both the fastest and the lightest option; the dask
# worker threads then pipeline disk I/O (e.g. to_netcdf) against the
# GPU solves.
_GPU_MAX_CONCURRENT = 1

_GPU_SEM = None
_GPU_SEM_INIT_LOCK = threading.Lock()  # only referenced inside _get_gpu_sem()


def _get_gpu_sem():
    """Return the per-process GPU concurrency semaphore (lazy singleton).

    Accessed only through this module-level *function* so the semaphore
    itself never travels through dask.distributed's serialisation -- each
    worker process lazily creates its own (per-process is the correct
    scope: every process has its own CUDA context anyway).
    """
    global _GPU_SEM
    if _GPU_SEM is None:
        with _GPU_SEM_INIT_LOCK:
            if _GPU_SEM is None:
                _GPU_SEM = threading.BoundedSemaphore(_GPU_MAX_CONCURRENT)
    return _GPU_SEM


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
        Relative change of the norm vs the previous check (1.0 on the
        first check, mirroring the CPU kernels -- see below).
    overflow : bool
        True if norm is nan or exceeds 1e100.
    """
    if norm_count > 0:
        norm = norm_sum / norm_count
    else:
        norm = np.nan

    overflow = (np.isnan(norm) or norm > 1e100)

    if need_convergence and norm == 0:
        error = 0.0
    elif need_convergence and norm_prev < np.finfo(np.float64).max:
        error = abs(norm - norm_prev) / norm_prev
    else:
        error = 1.0

    return norm, error, overflow


def _evaluate_update_metric(max_update, max_value):
    """Normalize the max SOR correction used as preconditioned residual."""
    if max_value > np.finfo(np.float64).tiny:
        return max_update / max_value
    return max_update


def _launch_config_2d(yc, xc, block_2d):
    """Compute 2D/1D launch configurations shared by all 2-D wrappers.

    Returns
    -------
    (blocks, threads, bcount_x, bs1d_x, bcount_y, bs1d_y)
        ``blocks``/``threads``: 2D grid for the SOR/norm kernels;
        ``bcount_*``/``bs1d_*``: 1D grid for the boundary kernels.
    """
    # Block shape: per-call override (iParams['gpu_block2d']) or the
    # built-in default (16, 16).
    #
    # Axis mapping (CUDA):
    #   j = cuda.grid(2)[0] = blockIdx.x*blockDim.x + threadIdx.x  (rows = yc)
    #   i = cuda.grid(2)[1] = blockIdx.y*blockDim.y + threadIdx.y  (cols = xc)
    # so gridDim.x (blocks[0]) pairs with blockDim.x (bx) to cover yc, and
    # gridDim.y (blocks[1]) pairs with blockDim.y (by) to cover xc.
    bx, by = block_2d if block_2d is not None else _DEFAULT_BLOCK_2D
    threads = (bx, by)
    blocks = (max((yc + bx - 1) // bx, 1), max((xc + by - 1) // by, 1))

    # 1D block sizes for boundary kernels, chosen adaptively by the extent
    # each kernel actually sweeps (xc for y-boundary, yc for x-boundary).
    bs1d_x = _auto_bsize_1d(xc)
    bs1d_y = _auto_bsize_1d(yc)
    bcount_x = max((xc + bs1d_x - 1) // bs1d_x, 1)
    bcount_y = max((yc + bs1d_y - 1) // bs1d_y, 1)
    return blocks, threads, bcount_x, bs1d_x, bcount_y, bs1d_y


def _launch_config_3d(zc, yc, xc):
    """Compute 3D launch configuration shared by the 3D wrappers.

    Thread block (tz, ty, tx) = (4, 8, 16); the z dimension of a block is
    limited to 64 on all CUDA devices, so keep tz small.
    """
    tz, ty, tx = 4, 8, 16
    threads = (tz, ty, tx)
    blocks = (max((zc + tz - 1) // tz, 1),
              max((yc + ty - 1) // ty, 1),
              max((xc + tx - 1) // tx, 1))
    return blocks, threads


def _run_sor_3d_loop(d_S, launch_color, blocks, threads,
                     bcx_periodic, bcx_extend, bcy_extend, undef,
                     d_norm, d_metric, mxLoop, tolerance, convergence):
    """Host-side Red-Black SOR iteration loop shared by 3D wrappers.

    Same structure as :func:`_run_sor_2d_loop`; note the z boundary is
    never updated (fixed values come in via icbc, as in the CPU kernel).

    Returns
    -------
    (overflow, error, loop)
    """
    check_interval = _compute_check_interval(mxLoop, tolerance)
    need_convergence = (tolerance > 0.0)

    norm_prev = np.finfo(np.float64).max
    loop = 0
    overflow = False
    error = 0.0

    zc, yc, xc = d_S.shape

    # 2D launch config for the 3D boundary kernels (sweeps (k,i) / (k,j))
    bt_z, bt_h = 4, 32
    b3d_y = (max((zc + bt_z - 1) // bt_z, 1), max((xc + bt_h - 1) // bt_h, 1))
    b3d_x = (max((zc + bt_z - 1) // bt_z, 1), max((yc + bt_h - 1) // bt_h, 1))
    t3d = (bt_z, bt_h)

    while True:
        next_loop = loop + 1
        track_residual = (need_convergence and convergence == 1 and
                          ((next_loop % check_interval == 0) or
                           next_loop >= mxLoop))
        if track_residual:
            d_metric[0] = 0.0
            d_metric[1] = 0.0

        # --- process boundaries (y / x extend; z is fixed) ---
        if bcy_extend:
            _extend_y_boundary_3d[b3d_y, t3d](
                d_S, zc, yc, xc, undef, bcx_periodic)
        if bcx_extend:
            _extend_x_boundary_3d[b3d_x, t3d](
                d_S, zc, yc, xc, undef, bcy_extend)

        launch_color(0, track_residual)
        launch_color(1, track_residual)

        loop += 1

        is_check_iter = (loop % check_interval == 0) or (loop >= mxLoop)

        if not is_check_iter:
            continue

        d_norm[0] = 0.0
        d_norm[1] = 0.0
        _abs_norm_3d[blocks, threads](d_S, undef, d_norm)
        norm_h = d_norm.copy_to_host()

        norm, norm_error, overflow = _evaluate_gpu_norm(
            norm_h[0], norm_h[1], norm_prev, need_convergence, tolerance)

        if convergence == 1 and need_convergence:
            metric_h = d_metric.copy_to_host()
            error = _evaluate_update_metric(metric_h[0], metric_h[1])
        else:
            error = norm_error

        if overflow:
            break

        if (need_convergence and error < tolerance) or loop >= mxLoop or norm == 0:
            break

        norm_prev = norm

    return overflow, error, loop


def _run_sor_2d_loop(d_S, launch_color, launch_boundary, blocks, threads,
                     undef, d_norm, d_metric, mxLoop, tolerance, convergence,
                     n_color=2):
    """Host-side multi-color SOR iteration loop shared by all 2-D wrappers.

    ``launch_color(color)`` must launch the SOR kernel for one color on the
    device arrays already bound inside the closure.  ``launch_boundary()``
    is called at the top of every iteration and must launch the correct
    extend kernels (1-row for the 5/9-point stencils, 2-row for the
    biharmonic stencil, gauge-anchor-aware; no-op unless BCy == 'extend').
    ``n_color`` is selected from the actual stencil dependency graph: two for
    five-point operators, four when a mixed derivative adds diagonal
    neighbours, and a separate scheme for the biharmonic operator.

    Returns
    -------
    (overflow, error, loop)
    """
    # Check convergence / overflow only every check_interval iterations to
    # amortise the host-device sync cost (see _compute_check_interval).
    check_interval = _compute_check_interval(mxLoop, tolerance)
    need_convergence = (tolerance > 0.0)

    norm_prev = np.finfo(np.float64).max
    loop = 0
    overflow = False
    error = 0.0

    while True:
        next_loop = loop + 1
        track_residual = (need_convergence and convergence == 1 and
                          ((next_loop % check_interval == 0) or
                           next_loop >= mxLoop))
        if track_residual:
            d_metric[0] = 0.0
            d_metric[1] = 0.0

        # --- process boundaries ---
        launch_boundary()

        # --- one globally synchronized launch per color ---
        for c in range(n_color):
            launch_color(c, track_residual)

        loop += 1

        # --- convergence / overflow check (only every check_interval iters) ---
        is_check_iter = (loop % check_interval == 0) or (loop >= mxLoop)

        if not is_check_iter:
            continue  # fast path: skip norm computation entirely

        d_norm[0] = 0.0
        d_norm[1] = 0.0
        _abs_norm_2d[blocks, threads](d_S, undef, d_norm)
        norm_h = d_norm.copy_to_host()

        norm, norm_error, overflow = _evaluate_gpu_norm(
            norm_h[0], norm_h[1], norm_prev, need_convergence, tolerance)

        if convergence == 1 and need_convergence:
            metric_h = d_metric.copy_to_host()
            error = _evaluate_update_metric(metric_h[0], metric_h[1])
        else:
            error = norm_error

        if overflow:
            break

        if (need_convergence and error < tolerance) or loop >= mxLoop or norm == 0:
            break

        norm_prev = norm

    return overflow, error, loop


# ---------------------------------------------------------------------------
# Python wrapper (same signature as numba kernel)
# ---------------------------------------------------------------------------

def invert_standard_2D_gpu(S, A, B, C, F, info,
                           yc, xc, BCy, BCx, delxSqr,
                           ratioQtr, ratioSqr, optArg, undef, flags,
                           mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_standard_2D`` using multi-color SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT``: when the input is dask-backed, each
    per-time-step task runs in a dask worker thread, and unbounded
    concurrency would hold ``num_workers`` x 5 device buffers in VRAM
    for zero throughput gain (kernels serialise on the default stream
    anyway).  See :func:`_solve_standard_2D_gpu` for the parameters.
    """
    with _get_gpu_sem():
        return _solve_standard_2D_gpu(S, A, B, C, F, info,
                                      yc, xc, BCy, BCx, delxSqr,
                                      ratioQtr, ratioSqr, optArg, undef,
                                      flags, mxLoop, tolerance, convergence,
                                      block_2d=block_2d)


def _solve_standard_2D_gpu(S, A, B, C, F, info,
                           yc, xc, BCy, BCx, delxSqr,
                           ratioQtr, ratioSqr, optArg, undef, flags,
                           mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_standard_2D`` using multi-color SOR.

    Same signature as :func:`xinvert.cpus.invert_standard_2D` so it can be
    used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.

    The host-side Python code controls the iteration loop.  Each iteration
    launches two color kernels for a five-point stencil or four for a
    mixed-derivative nine-point stencil, then computes the convergence norm
    via an atomic-reduction kernel.

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
    block_2d : tuple of int, optional
        2D thread-block shape ``(bx, by)`` for the SOR/norm kernels.
        None = built-in default ``(16, 16)``.  Set per call via
        ``iParams['gpu_block2d']``.
    """
    # A non-zero mixed derivative couples diagonal points.  Diagonals have
    # the same red-black parity, so they require four colors to avoid a data
    # race within a kernel launch.
    n_color = 4 if _has_active_coefficient(B) else 2
    _validate_periodic_x_coloring(xc, BCx, n_color)

    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_F = cuda.to_device(F)

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')
    bcy_extend = (BCy == 'extend')

    blocks, threads, bcount_x, bs1d_x, bcount_y, bs1d_y = \
        _launch_config_2d(yc, xc, block_2d)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    # boundary gauge anchor for singular extend systems (y-extend with
    # x-extend or x-periodic): ONE boundary point keeps its initial
    # (uploaded) value; every interior point updates normally.
    aside, aidx = -1, -1
    if BCy == 'extend' and BCx != 'fixed':
        aside, aidx = _find_boundary_anchor_2d_host(F, undef, BCx == 'extend')

    def launch_boundary():
        if bcy_extend:
            _extend_y_boundary[bcount_x, bs1d_x](
                d_S, yc, xc, undef, bcx_periodic, aside, aidx)
        if bcx_extend:
            _extend_x_boundary[bcount_y, bs1d_y](
                d_S, yc, xc, undef, bcy_extend, aside, aidx)

    def launch_color(color, track_residual):
        _sor_2d_rb[blocks, threads](
            d_S, d_A, d_B, d_C, d_F, yc, xc, bcx_periodic,
            delxSqr, ratioQtr, ratioSqr, optArg, undef, color, n_color,
            d_metric, track_residual)

    overflow, error, loop = _run_sor_2d_loop(
        d_S, launch_color, launch_boundary, blocks, threads,
        undef, d_norm, d_metric, mxLoop, tolerance, convergence,
        n_color=n_color)

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop


def invert_standard_2D_full_gpu(S, A, B, C, D, E, F, info,
                                yc, xc, BCy, BCx, delxSqr,
                                ratioQtr, ratioSqr, optArg, undef, flags,
                                mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_standard_2D_full`` using multi-color SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT`` (see :func:`invert_standard_2D_gpu`).
    """
    with _get_gpu_sem():
        return _solve_standard_2D_full_gpu(S, A, B, C, D, E, F, info,
                                           yc, xc, BCy, BCx, delxSqr,
                                           ratioQtr, ratioSqr, optArg, undef,
                                           flags, mxLoop, tolerance, convergence,
                                           block_2d=block_2d)


def _solve_standard_2D_full_gpu(S, A, B, C, D, E, F, info,
                                yc, xc, BCy, BCx, delxSqr,
                                ratioQtr, ratioSqr, optArg, undef, flags,
                                mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_standard_2D_full`` using multi-color SOR.

    Same signature as :func:`xinvert.cpus.invert_standard_2D_full` so it can
    be used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.

    Parameters are as in :func:`_solve_standard_2D_gpu`, plus:

    D, E : numpy.ndarray
        Coefficient arrays for the x-flux divergence and the Helmholtz
        (linear) term, same shape as S.
    """
    # B and C multiply diagonal-neighbour terms; either being active changes
    # the dependency graph from five-point red-black to four-color.
    n_color = 4 if _has_active_coefficient(B, C) else 2
    _validate_periodic_x_coloring(xc, BCx, n_color)

    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_D = cuda.to_device(D)
    d_E = cuda.to_device(E)
    d_F = cuda.to_device(F)

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')
    bcy_extend = (BCy == 'extend')

    blocks, threads, bcount_x, bs1d_x, bcount_y, bs1d_y = \
        _launch_config_2d(yc, xc, block_2d)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    # boundary gauge anchor for singular extend systems; value stays at
    # the uploaded initial S (see the standard_2D wrapper).
    aside, aidx = -1, -1
    if BCy == 'extend' and BCx != 'fixed':
        aside, aidx = _find_boundary_anchor_2d_host(F, undef, BCx == 'extend')

    def launch_boundary():
        if bcy_extend:
            _extend_y_boundary[bcount_x, bs1d_x](
                d_S, yc, xc, undef, bcx_periodic, aside, aidx)
        if bcx_extend:
            _extend_x_boundary[bcount_y, bs1d_y](
                d_S, yc, xc, undef, bcy_extend, aside, aidx)

    def launch_color(color, track_residual):
        _sor_2d_rb_full[blocks, threads](
            d_S, d_A, d_B, d_C, d_D, d_E, d_F, yc, xc, bcx_periodic,
            delxSqr, ratioQtr, ratioSqr, optArg, undef, color, n_color,
            d_metric, track_residual)

    overflow, error, loop = _run_sor_2d_loop(
        d_S, launch_color, launch_boundary, blocks, threads,
        undef, d_norm, d_metric, mxLoop, tolerance, convergence,
        n_color=n_color)

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop


def invert_general_bih_2D_gpu(S, A, B, C, D, E, F, G, H, I, J, info,
                              yc, xc, BCy, BCx, delxSSr, delxTr, delxSqr,
                              ratio, ratioSSr, ratioQtr, ratioSqr,
                              optArg, undef, flags,
                              mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_general_bih_2D`` using 5-color SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT`` (see :func:`invert_standard_2D_gpu`).

    The biharmonic stencil requires five colors instead of the classic
    red-black pair, so each iteration launches five synchronized passes.
    """
    with _get_gpu_sem():
        return _solve_general_bih_2D_gpu(S, A, B, C, D, E, F, G, H, I, J, info,
                                         yc, xc, BCy, BCx, delxSSr, delxTr,
                                         delxSqr, ratio, ratioSSr, ratioQtr,
                                         ratioSqr, optArg, undef, flags,
                                         mxLoop, tolerance, convergence,
                                         block_2d=block_2d)


def _solve_general_bih_2D_gpu(S, A, B, C, D, E, F, G, H, I, J, info,
                              yc, xc, BCy, BCx, delxSSr, delxTr, delxSqr,
                              ratio, ratioSSr, ratioQtr, ratioSqr,
                              optArg, undef, flags,
                              mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_general_bih_2D`` using 5-color SOR.

    Same signature as :func:`xinvert.cpus.invert_general_bih_2D` so it can
    be used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.
    """
    _validate_periodic_x_coloring(xc, BCx, 5)

    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_D = cuda.to_device(D)
    d_E = cuda.to_device(E)
    d_F = cuda.to_device(F)
    d_G = cuda.to_device(G)
    d_H = cuda.to_device(H)
    d_I = cuda.to_device(I)
    d_J = cuda.to_device(J)

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')
    bcy_extend = (BCy == 'extend')

    blocks, threads, bcount_x, bs1d_x, bcount_y, bs1d_y = \
        _launch_config_2d(yc, xc, block_2d)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    # NOTE: no gauge anchor for the biharmonic kernel -- the 4th-order
    # Neumann null space is multi-dimensional and a single pinned point
    # cannot remove it (see the corresponding note in cpus.py).
    aside, aidx = -1, -1

    def launch_boundary():
        # biharmonic extend copies TWO rows/cols; the anchor's outer
        # row/col copy is skipped so it keeps its initial value
        if bcy_extend:
            _extend_y_boundary_2d_bih[bcount_x, bs1d_x](
                d_S, yc, xc, undef, bcx_periodic, aside, aidx)
        if bcx_extend:
            _extend_x_boundary_2d_bih[bcount_y, bs1d_y](
                d_S, yc, xc, undef, bcy_extend, aside, aidx)

    def launch_color(color, track_residual):
        _sor_2d_mc_bih[blocks, threads](
            d_S, d_A, d_B, d_C, d_D, d_E, d_F, d_G, d_H, d_I, d_J,
            yc, xc, bcx_periodic,
            delxSSr, delxTr, delxSqr,
            ratio, ratioSSr, ratioQtr, ratioSqr,
            optArg, undef, color, d_metric, track_residual)

    overflow, error, loop = _run_sor_2d_loop(
        d_S, launch_color, launch_boundary, blocks, threads,
        undef, d_norm, d_metric, mxLoop, tolerance, convergence, n_color=5)

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop


def invert_standard_1D_gpu(S, A, B, F, info,
                           xc, BCx, delxSqr, optArg, undef, flags,
                           mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_standard_1D`` using Red-Black SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT`` (see :func:`invert_standard_2D_gpu`).

    Parameters
    ----------
    S : numpy.ndarray (modified in-place), shape (xc,)
        Solution array.
    A, B, F : numpy.ndarray, shape (xc,)
        Coefficients and forcing.
    info : any
        Information array (unused in computation, kept for signature compat).
    xc : int
        Grid count in x dimension.
    BCx : str
        Boundary condition ('fixed', 'extend', 'periodic').
    delxSqr : float
        Squared grid spacing.
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
    block_2d : any, optional
        Unused for the 1D kernel; accepted for signature compatibility with
        the ``iParams['gpu_block2d']`` injection in ``core._make_kernel``.
    """
    with _get_gpu_sem():
        return _solve_standard_1D_gpu(S, A, B, F, info,
                                      xc, BCx, delxSqr, optArg, undef, flags,
                                      mxLoop, tolerance, convergence)


def _solve_standard_1D_gpu(S, A, B, F, info,
                           xc, BCx, delxSqr, optArg, undef, flags,
                           mxLoop, tolerance, convergence=0):
    r"""GPU implementation of ``invert_standard_1D`` using Red-Black SOR.

    Same signature as :func:`xinvert.cpus.invert_standard_1D` so it can be
    used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.
    """
    _validate_periodic_x_coloring(xc, BCx, 2)

    # --- transfer to GPU ---
    d_S = cuda.to_device(np.ascontiguousarray(S))
    d_A = cuda.to_device(np.ascontiguousarray(A))
    d_B = cuda.to_device(np.ascontiguousarray(B))
    d_F = cuda.to_device(np.ascontiguousarray(F))

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')

    # 1D launch config; block size picked adaptively by extent
    bs = _auto_bsize_1d(xc)
    blocks = max((xc + bs - 1) // bs, 1)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    # boundary gauge anchor for the singular extend (Neumann at both
    # ends) system: ONE boundary point keeps its initial (uploaded)
    # value; all interior points update normally.
    aside = -1
    if BCx == 'extend':
        aside = _find_boundary_anchor_1d_host(F, undef)

    check_interval = _compute_check_interval(mxLoop, tolerance)
    need_convergence = (tolerance > 0.0)

    norm_prev = np.finfo(np.float64).max
    loop = 0
    overflow = False
    error = 0.0

    while True:
        next_loop = loop + 1
        track_residual = (need_convergence and convergence == 1 and
                          ((next_loop % check_interval == 0) or
                           next_loop >= mxLoop))
        if track_residual:
            d_metric[0] = 0.0
            d_metric[1] = 0.0

        # --- process boundaries ---
        if bcx_extend:
            _extend_boundary_1d[1, 1](d_S, xc, undef, aside)

        # --- Red + Black SOR update ---
        _sor_1d_rb[blocks, bs](
            d_S, d_A, d_B, d_F, xc, bcx_periodic,
            delxSqr, optArg, undef, 0, d_metric, track_residual)
        _sor_1d_rb[blocks, bs](
            d_S, d_A, d_B, d_F, xc, bcx_periodic,
            delxSqr, optArg, undef, 1, d_metric, track_residual)

        loop += 1

        # --- convergence / overflow check (only every check_interval iters) ---
        is_check_iter = (loop % check_interval == 0) or (loop >= mxLoop)

        if not is_check_iter:
            continue

        d_norm[0] = 0.0
        d_norm[1] = 0.0
        _abs_norm_1d[blocks, bs](d_S, undef, d_norm)
        norm_h = d_norm.copy_to_host()

        norm, norm_error, overflow = _evaluate_gpu_norm(
            norm_h[0], norm_h[1], norm_prev, need_convergence, tolerance)

        if convergence == 1 and need_convergence:
            metric_h = d_metric.copy_to_host()
            error = _evaluate_update_metric(metric_h[0], metric_h[1])
        else:
            error = norm_error

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


def invert_standard_3D_gpu(S, A, B, C, F, info,
                           zc, yc, xc, BCz, BCy, BCx, delxSqr,
                           ratio2Sqr, ratio1Sqr, optArg, undef, flags,
                           mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_standard_3D`` using Red-Black SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT`` (see :func:`invert_standard_2D_gpu`).

    Parameters
    ----------
    S : numpy.ndarray (modified in-place), shape (zc, yc, xc).
    A, B, C, F : numpy.ndarray
        Coefficients and forcing, same shape as S.
    info : any
        Information array (unused in computation, kept for signature compat).
    zc, yc, xc : int
        Grid counts in z, y and x dimensions.
    BCz, BCy, BCx : str
        Boundary conditions; z is never updated in the iteration (fixed
        values come in via ``icbc``), mirroring the CPU kernel.
    delxSqr, ratio2Sqr, ratio1Sqr : float
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
    block_2d : any, optional
        Unused for the 3D kernel (block shape is fixed at (4, 8, 16));
        accepted for signature compatibility with the
        ``iParams['gpu_block2d']`` injection in ``core._make_kernel``.
    """
    with _get_gpu_sem():
        return _solve_standard_3D_gpu(S, A, B, C, F, info,
                                      zc, yc, xc, BCz, BCy, BCx, delxSqr,
                                      ratio2Sqr, ratio1Sqr, optArg, undef,
                                      flags, mxLoop, tolerance, convergence)


def _solve_standard_3D_gpu(S, A, B, C, F, info,
                           zc, yc, xc, BCz, BCy, BCx, delxSqr,
                           ratio2Sqr, ratio1Sqr, optArg, undef, flags,
                           mxLoop, tolerance, convergence=0):
    r"""GPU implementation of ``invert_standard_3D`` using Red-Black SOR.

    Same signature as :func:`xinvert.cpus.invert_standard_3D` so it can be
    used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.
    """
    _validate_periodic_x_coloring(xc, BCx, 2)

    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_F = cuda.to_device(F)

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')
    bcy_extend = (BCy == 'extend')

    blocks, threads = _launch_config_3d(zc, yc, xc)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    def launch_color(color, track_residual):
        _sor_3d_rb[blocks, threads](
            d_S, d_A, d_B, d_C, d_F, zc, yc, xc, bcx_periodic,
            delxSqr, ratio2Sqr, ratio1Sqr, optArg, undef, color,
            d_metric, track_residual)

    overflow, error, loop = _run_sor_3d_loop(
        d_S, launch_color, blocks, threads,
        bcx_periodic, bcx_extend, bcy_extend, undef,
        d_norm, d_metric, mxLoop, tolerance, convergence)

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop


def invert_general_3D_gpu(S, A, B, C, D, E, F, G, H, info,
                          zc, yc, xc, delx, BCz, BCy, BCx, delxSqr,
                          ratio2, ratio1, ratio2Sqr, ratio1Sqr,
                          optArg, undef, flags,
                          mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_general_3D`` using Red-Black SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT`` (see :func:`invert_standard_2D_gpu`).
    """
    with _get_gpu_sem():
        return _solve_general_3D_gpu(S, A, B, C, D, E, F, G, H, info,
                                     zc, yc, xc, delx, BCz, BCy, BCx, delxSqr,
                                     ratio2, ratio1, ratio2Sqr, ratio1Sqr,
                                     optArg, undef, flags, mxLoop, tolerance,
                                     convergence)


def _solve_general_3D_gpu(S, A, B, C, D, E, F, G, H, info,
                          zc, yc, xc, delx, BCz, BCy, BCx, delxSqr,
                          ratio2, ratio1, ratio2Sqr, ratio1Sqr,
                          optArg, undef, flags,
                          mxLoop, tolerance, convergence=0):
    r"""GPU implementation of ``invert_general_3D`` using Red-Black SOR.

    Same signature as :func:`xinvert.cpus.invert_general_3D` so it can be
    used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.
    """
    _validate_periodic_x_coloring(xc, BCx, 2)

    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_D = cuda.to_device(D)
    d_E = cuda.to_device(E)
    d_F = cuda.to_device(F)
    d_G = cuda.to_device(G)
    d_H = cuda.to_device(H)

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')
    bcy_extend = (BCy == 'extend')

    blocks, threads = _launch_config_3d(zc, yc, xc)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    def launch_color(color, track_residual):
        _sor_3d_rb_general[blocks, threads](
            d_S, d_A, d_B, d_C, d_D, d_E, d_F, d_G, d_H,
            zc, yc, xc, bcx_periodic,
            delxSqr, delx, ratio2, ratio1, ratio2Sqr, ratio1Sqr,
            optArg, undef, color, d_metric, track_residual)

    overflow, error, loop = _run_sor_3d_loop(
        d_S, launch_color, blocks, threads,
        bcx_periodic, bcx_extend, bcy_extend, undef,
        d_norm, d_metric, mxLoop, tolerance, convergence)

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop


def invert_general_2D_gpu(S, A, B, C, D, E, F, G, info,
                          yc, xc, delx, BCy, BCx, delxSqr,
                          ratio, ratioQtr, ratioSqr, optArg, undef, flags,
                          mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_general_2D`` using multi-color SOR.

    Thin wrapper that bounds the number of concurrent GPU solves to
    ``_GPU_MAX_CONCURRENT`` (see :func:`invert_standard_2D_gpu`).
    """
    with _get_gpu_sem():
        return _solve_general_2D_gpu(S, A, B, C, D, E, F, G, info,
                                     yc, xc, delx, BCy, BCx, delxSqr,
                                     ratio, ratioQtr, ratioSqr, optArg, undef,
                                     flags, mxLoop, tolerance, convergence,
                                     block_2d=block_2d)


def _solve_general_2D_gpu(S, A, B, C, D, E, F, G, info,
                          yc, xc, delx, BCy, BCx, delxSqr,
                          ratio, ratioQtr, ratioSqr, optArg, undef, flags,
                          mxLoop, tolerance, convergence=0, block_2d=None):
    r"""GPU implementation of ``invert_general_2D`` using multi-color SOR.

    Same signature as :func:`xinvert.cpus.invert_general_2D` so it can be
    used as a drop-in replacement via ``iParams={'architect': 'gpu'}``.

    Parameters are as in :func:`_solve_standard_2D_gpu`, for the
    non-divergence (point) form with first-derivative terms.
    """
    n_color = 4 if _has_active_coefficient(B) else 2
    _validate_periodic_x_coloring(xc, BCx, n_color)

    # --- transfer to GPU ---
    d_S = cuda.to_device(S)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    d_D = cuda.to_device(D)
    d_E = cuda.to_device(E)
    d_F = cuda.to_device(F)
    d_G = cuda.to_device(G)

    bcx_periodic = (BCx == 'periodic')
    bcx_extend = (BCx == 'extend')
    bcy_extend = (BCy == 'extend')

    blocks, threads, bcount_x, bs1d_x, bcount_y, bs1d_y = \
        _launch_config_2d(yc, xc, block_2d)

    d_norm = cuda.device_array(2, dtype=np.float64)
    d_metric = cuda.device_array(2, dtype=np.float64)

    # boundary gauge anchor for singular extend systems; value stays at
    # the uploaded initial S (see the standard_2D wrapper).
    aside, aidx = -1, -1
    if BCy == 'extend' and BCx != 'fixed':
        aside, aidx = _find_boundary_anchor_2d_host(G, undef, BCx == 'extend')

    def launch_boundary():
        if bcy_extend:
            _extend_y_boundary[bcount_x, bs1d_x](
                d_S, yc, xc, undef, bcx_periodic, aside, aidx)
        if bcx_extend:
            _extend_x_boundary[bcount_y, bs1d_y](
                d_S, yc, xc, undef, bcy_extend, aside, aidx)

    def launch_color(color, track_residual):
        _sor_2d_rb_general[blocks, threads](
            d_S, d_A, d_B, d_C, d_D, d_E, d_F, d_G, yc, xc, bcx_periodic,
            delxSqr, delx, ratio, ratioQtr, ratioSqr, optArg, undef, color,
            n_color, d_metric, track_residual)

    overflow, error, loop = _run_sor_2d_loop(
        d_S, launch_color, launch_boundary, blocks, threads,
        undef, d_norm, d_metric, mxLoop, tolerance, convergence,
        n_color=n_color)

    # --- copy result back (in-place) ---
    d_S.copy_to_host(S)

    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
