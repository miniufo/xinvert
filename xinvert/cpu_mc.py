"""Multi-core, multi-color CPU kernels for xinvert.

The experimental ``cpu-mc`` backend parallelizes points within each color by
using :func:`numba.prange`. Color passes remain sequential because later
colors depend on values updated by earlier colors.

All seven elliptic templates are supported. Nearest-neighbor stencils use
two colors, 2-D mixed derivatives use four, and the radius-two biharmonic
stencil uses a three-by-three pattern of nine colors. Periodic x extents
must preserve the selected color period across the seam.
"""
import numba as nb
import numpy as np

from .cpus import (
    _accumulate_update_metric,
    _apply_extend_boundary_2d,
    _apply_extend_boundary_3d,
    _apply_extend_boundary_bih_2d,
    _check_interval,
    _find_boundary_anchor_1d,
    _find_boundary_anchor_2d,
    _relative_update_metric,
)


def invert_standard_2D_mc(S, A, B, C, F, info,
                          yc, xc, BCy, BCx, delxSqr,
                          ratioQtr, ratioSqr, optArg, undef, flags,
                          mxLoop, tolerance, check_interval=0):
    """Solve the standard 2-D equation with multi-core multi-color SOR.

    This function has the same call signature as
    :func:`xinvert.cpus.invert_standard_2D`. It selects two or four colors
    from the stencil dependency graph and calls the compiled parallel loop.

    Parameters
    ----------
    S : numpy.ndarray
        Solution array, updated in place.
    A, B, C, F : numpy.ndarray
        Standard-form coefficients and forcing.
    info : object
        Slice label retained for backend signature compatibility.
    yc, xc : int
        Grid sizes in the y and x directions.
    BCy, BCx : str
        Boundary conditions for y and x.
    delxSqr, ratioQtr, ratioSqr : float
        Grid-spacing factors used by the discrete operator.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Solver status and optional residual-history output.
    mxLoop : int
        Maximum number of SOR iterations.
    tolerance : float
        Relative residual tolerance.
    check_interval : int, optional
        Residual-check interval; zero selects the adaptive default.

    Returns
    -------
    numpy.ndarray
        The updated solution array ``S``.

    Raises
    ------
    ValueError
        If a periodic x grid has an odd number of points and therefore does
        not preserve the two/four-color parity across the seam.
    """
    n_color = 4 if np.any(np.asarray(B) != 0) else 2
    if BCx == 'periodic' and xc % 2:
        raise ValueError(
            f"CPU multi-color SOR with periodic x requires an even x grid "
            f"length; got {xc}")
    return _invert_standard_2D_mc_impl(
        S, A, B, C, F, yc, xc, BCy, BCx, delxSqr,
        ratioQtr, ratioSqr, optArg, undef, flags, mxLoop,
        tolerance, check_interval, n_color)


@nb.njit(cache=False, nogil=True, parallel=True)
def _invert_standard_2D_mc_impl(S, A, B, C, F,
                                yc, xc, BCy, BCx, delxSqr,
                                ratioQtr, ratioSqr, optArg, undef, flags,
                                mxLoop, tolerance, check_interval, n_color):
    """Compiled iteration loop underlying :func:`invert_standard_2D_mc`."""
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros(yc, dtype=np.float64)
    row_value = np.zeros(yc, dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0

    if BCy == 'extend' and BCx != 'fixed':
        aside, aidx = _find_boundary_anchor_2d(
            F, undef, BCx == 'extend')
    else:
        aside, aidx = -1, -1

    for loop in range(1, mxLoop + 1):
        track_residual = (loop % interval == 0) or loop >= mxLoop
        max_update = 0.0
        max_value = 0.0

        _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx)

        for color in range(n_color):
            for j in nb.prange(1, yc - 1):
                local_update = 0.0
                local_value = 0.0
                istart = 0 if BCx == 'periodic' else 1
                iend = xc if BCx == 'periodic' else xc - 1
                if n_color == 2:
                    target_i_parity = (color - j) & 1
                else:
                    if (j & 1) != color // 2:
                        row_update[j] = 0.0
                        row_value[j] = 0.0
                        continue
                    target_i_parity = color & 1

                for i in range(istart, iend):
                    if (i & 1) != target_i_parity:
                        continue

                    ip1 = i + 1
                    im1 = i - 1
                    if ip1 == xc:
                        ip1 = 0
                    if im1 < 0:
                        im1 = xc - 1

                    if (F[j, i] == undef or
                            A[j + 1, i] == undef or A[j, i] == undef or
                            B[j, ip1] == undef or B[j, im1] == undef or
                            B[j + 1, i] == undef or B[j - 1, i] == undef or
                            C[j, ip1] == undef or C[j, i] == undef):
                        continue

                    temp = (
                        (A[j + 1, i] * (S[j + 1, i] - S[j, i]) -
                         A[j, i] * (S[j, i] - S[j - 1, i])) * ratioSqr +
                        (B[j + 1, i] *
                         (S[j + 1, ip1] - S[j + 1, im1]) -
                         B[j - 1, i] *
                         (S[j - 1, ip1] - S[j - 1, im1])) * ratioQtr +
                        (B[j, ip1] *
                         (S[j + 1, ip1] - S[j - 1, ip1]) -
                         B[j, im1] *
                         (S[j + 1, im1] - S[j - 1, im1])) * ratioQtr +
                        (C[j, ip1] * (S[j, ip1] - S[j, i]) -
                         C[j, i] * (S[j, i] - S[j, im1]))
                    ) - F[j, i] * delxSqr

                    denom = ((A[j + 1, i] + A[j, i]) * ratioSqr +
                             C[j, ip1] + C[j, i])
                    if denom == 0.0:
                        continue
                    delta = temp * optArg / denom

                    if track_residual:
                        new_value = S[j, i] + delta
                        if (not np.isfinite(delta) or
                                not np.isfinite(new_value)):
                            local_update = np.inf
                            local_value = max(local_value, 1.0)
                        else:
                            local_update = max(local_update, abs(delta))
                            local_value = max(
                                local_value, abs(S[j, i]), abs(new_value))
                    S[j, i] += delta

                row_update[j] = local_update
                row_value[j] = local_value

            if track_residual:
                for j in range(1, yc - 1):
                    max_update = max(max_update, row_update[j])
                    max_value = max(max_value, row_value[j])

        if not track_residual:
            continue

        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop >= mxLoop:
            flags[2] = loop
            break

    flags[0] = overflow
    flags[1] = error
    return S


@nb.njit(cache=False, nogil=True, parallel=True)
def invert_general_2D_mc(S, A, B, C, D, E, F, G, info,
                      yc, xc, delx, BCy, BCx,
                      delxSqr, ratio, ratioQtr, ratioSqr, optArg, undef,
                      flags, mxLoop, tolerance, check_interval=0):
    r"""Inverting a 2D slice of elliptic equation in the general form.

    .. math::

        A \frac{\partial^2 \psi}{\partial y^2} +
        B \frac{\partial^2 \psi}{\partial y \partial x} +
        C \frac{\partial^2 \psi}{\partial x^2} +
        D \frac{\partial \psi}{\partial y} +
        E \frac{\partial \psi}{\partial x} + F \psi = G
    
    Parameters
    ----------
    S : numpy.ndarray
        Writable solution array, updated in place.
    A : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    B : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    C : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    D : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    E : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    F : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    G : numpy.ndarray
        Right-hand-side forcing.
    info : object
        Slice label retained for backend signature compatibility.
    yc : int
        Number of y grid points.
    xc : int
        Number of x grid points.
    delx : float
        x grid spacing.
    BCy : str
        y boundary mode: fixed or extend; follows existing solver constraints.
    BCx : str
        x boundary mode: fixed, extend, or periodic.
    delxSqr : float
        Squared x grid spacing.
    ratio : float
        x/y grid spacing ratio.
    ratioQtr : float
        x/y spacing ratio divided by four.
    ratioSqr : float
        Squared x/y spacing ratio.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Overflow flag, final convergence metric, iteration count, and optional history.
    mxLoop : int
        Maximum number of SOR sweeps.
    tolerance : float
        Stopping threshold for the relative update convergence metric.
    check_interval : int, optional
        Convergence/history sampling interval; zero selects the automatic interval.

    Returns
    -------
    numpy.ndarray
        Updated solution array S.

    Raises
    ------
    ValueError
        Periodic x extent is incompatible with the color period
        (2 grid points).
    """
    n_color = 4 if np.any(B != 0) else 2
    if BCx == 'periodic' and xc % 2:
        raise ValueError("CPU multi-color periodic x requires a length divisible by 2")
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros(yc, dtype=np.float64)
    row_value = np.zeros(yc, dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0
    if BCy == 'extend' and BCx != 'fixed':
        aside, aidx = _find_boundary_anchor_2d(G, undef, BCx == 'extend')
    else:
        aside, aidx = -1, -1
    for loop in range(1, mxLoop + 1):
        track_residual = loop % interval == 0 or loop == mxLoop
        max_update = 0.0
        max_value = 0.0
        _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx)
        for color in range(n_color):
            for j in nb.prange(1, yc - 1):
                row = j
                local_update = 0.0
                local_value = 0.0
                istart = 0 if BCx == 'periodic' else 1
                iend = xc if BCx == 'periodic' else xc - 1
                for i in range(istart, iend):
                    if ((j + i) % 2 if n_color == 2 else (j % 2) * 2 + i % 2) != color:
                        continue
                    ip1 = (i + 1) % xc
                    im1 = (i - 1) % xc
                    cond = (G[j, i] != undef and
                            A[j, i] != undef and B[j, i] != undef and
                            C[j, i] != undef and D[j, i] != undef and
                            E[j, i] != undef and F[j, i] != undef)
                    
                    if cond:
                        temp = (
                            A[j, i] * (
                                (S[j+1, i] - S[j, i])-(S[j, i] - S[j-1, i])
                            ) * ratioSqr +
                            B[j, i] * (
                                (S[j+1, ip1] - S[j-1, ip1])-(S[j+1, im1] - S[j-1, im1])
                            ) * ratioQtr +
                            C[j, i] * (
                                (S[j, ip1] - S[j, i])-(S[j, i] - S[j, im1])
                            ) + (
                            D[j, i] * (
                                S[j+1, i] - S[j-1, i]
                            ) * ratio +
                            E[j, i] * (
                                S[j, ip1] - S[j, im1]
                            )) * delx / 2.0 + (
                            F[j, i] * S[j, i] - G[j, i]) * delxSqr
                        )
                        
                        temp *= optArg / ((A[j, i]*ratioSqr + C[j, i]) * 2.0
                                          -F[j, i]*delxSqr)
                        if track_residual:
                            local_update, local_value = _accumulate_update_metric(
                                temp, S[j, i], local_update, local_value)
                        S[j, i] += temp
                row_update[row] = local_update
                row_value[row] = local_value
            if track_residual:
                for row in range(yc):
                    max_update = max(max_update, row_update[row])
                    max_value = max(max_value, row_value[row])
        if not track_residual:
            continue
        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop == mxLoop:
            break
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
    return S


@nb.njit(cache=False, nogil=True, parallel=True)
def invert_standard_3D_mc(S, A, B, C, F, info,
                       zc, yc, xc, BCz, BCy, BCx, delxSqr,
                       ratio2Sqr, ratio1Sqr, optArg, undef, flags,
                       mxLoop, tolerance, check_interval=0):
    r"""Inverting a 3D volume of elliptic equation in standard form.

    .. math::

        \frac{\partial}{\partial z}\left(A\frac{\partial \omega}{\partial z}\right)+
        \frac{\partial}{\partial y}\left(B\frac{\partial \omega}{\partial y}\right)+
        \frac{\partial}{\partial x}\left(C\frac{\partial \omega}{\partial x}\right)=F
    
    Parameters
    ----------
    S : numpy.ndarray
        Writable solution array, updated in place.
    A : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    B : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    C : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    F : numpy.ndarray
        Right-hand-side forcing.
    info : object
        Slice label retained for backend signature compatibility.
    zc : int
        Number of z grid points.
    yc : int
        Number of y grid points.
    xc : int
        Number of x grid points.
    BCz : str
        z boundary mode; the current solver keeps z boundaries fixed.
    BCy : str
        y boundary mode: fixed or extend; follows existing solver constraints.
    BCx : str
        x boundary mode: fixed, extend, or periodic.
    delxSqr : float
        Squared x grid spacing.
    ratio2Sqr : float
        Squared x/z grid spacing ratio.
    ratio1Sqr : float
        Squared x/y grid spacing ratio.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Overflow flag, final convergence metric, iteration count, and optional history.
    mxLoop : int
        Maximum number of SOR sweeps.
    tolerance : float
        Stopping threshold for the relative update convergence metric.
    check_interval : int, optional
        Convergence/history sampling interval; zero selects the automatic interval.

    Returns
    -------
    numpy.ndarray
        Updated solution array S.

    Raises
    ------
    ValueError
        Periodic x extent is incompatible with the color period
        (2 grid points).
    """
    n_color = 2
    if BCx == 'periodic' and xc % 2:
        raise ValueError("CPU multi-color periodic x requires a length divisible by 2")
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros((zc - 2) * (yc - 2), dtype=np.float64)
    row_value = np.zeros((zc - 2) * (yc - 2), dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0
    for loop in range(1, mxLoop + 1):
        track_residual = loop % interval == 0 or loop == mxLoop
        max_update = 0.0
        max_value = 0.0
        _apply_extend_boundary_3d(S, BCy, BCx, undef)
        for color in range(n_color):
            for row in nb.prange((zc - 2) * (yc - 2)):
                k = 1 + row // (yc - 2)
                j = 1 + row % (yc - 2)
                local_update = 0.0
                local_value = 0.0
                istart = 0 if BCx == 'periodic' else 1
                iend = xc if BCx == 'periodic' else xc - 1
                for i in range(istart, iend):
                    if (k + j + i) % 2 != color:
                        continue
                    ip1 = (i + 1) % xc
                    im1 = (i - 1) % xc
                    cond = (F[k, j, i] != undef and
                            A[k+1, j, i] != undef and A[k, j, i] != undef and
                            B[k, j+1, i] != undef and B[k, j, i] != undef and
                            C[k, j, ip1] != undef and C[k, j, i] != undef)
                    
                    if cond:
                        temp = (
                            (
                                A[k+1, j, i] * (S[k+1, j, i] - S[k, j, i])-
                                A[k, j, i] * (S[k, j, i] - S[k-1, j, i])
                            ) * ratio2Sqr + (
                                B[k, j+1, i] * (S[k, j+1, i] - S[k, j, i])-
                                B[k, j, i] * (S[k, j, i] - S[k, j-1, i])
                            ) * ratio1Sqr + (
                                C[k, j, ip1] * (S[k, j, ip1] - S[k, j, i])-
                                C[k, j, i] * (S[k, j, i] - S[k, j, im1])
                            )
                        ) - F[k, j, i] * delxSqr
                        
                        temp *= optArg / ((A[k+1, j, i] + A[k, j, i]) *ratio2Sqr +
                                          (B[k, j+1, i] + B[k, j, i]) *ratio1Sqr +
                                          (C[k, j, ip1] + C[k, j, i]))
                        if track_residual:
                            local_update, local_value = _accumulate_update_metric(
                                temp, S[k, j, i], local_update, local_value)
                        S[k, j, i] += temp
                row_update[row] = local_update
                row_value[row] = local_value
            if track_residual:
                for row in range((zc - 2) * (yc - 2)):
                    max_update = max(max_update, row_update[row])
                    max_value = max(max_value, row_value[row])
        if not track_residual:
            continue
        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop == mxLoop:
            break
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
    return S


@nb.njit(cache=False, nogil=True, parallel=True)
def invert_general_3D_mc(S, A, B, C, D, E, F, G, H, info,
                      zc, yc, xc, delx, BCz, BCy, BCx, delxSqr,
                      ratio2, ratio1, ratio2Sqr, ratio1Sqr, optArg, undef,
                      flags, mxLoop, tolerance, check_interval=0):
    r"""Inverting a 3D volume of elliptic equation in the general form.

    .. math::

        A \frac{\partial^2 \psi}{\partial z^2} +
        B \frac{\partial^2 \psi}{\partial y^2} +
        C \frac{\partial^2 \psi}{\partial x^2} +
        D \frac{\partial \psi}{\partial z} +
        E \frac{\partial \psi}{\partial y} +
        F \frac{\partial \psi}{\partial x} + G \psi = H
    
    Parameters
    ----------
    S : numpy.ndarray
        Writable solution array, updated in place.
    A : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    B : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    C : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    D : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    E : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    F : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    G : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    H : numpy.ndarray
        Right-hand-side forcing.
    info : object
        Slice label retained for backend signature compatibility.
    zc : int
        Number of z grid points.
    yc : int
        Number of y grid points.
    xc : int
        Number of x grid points.
    delx : float
        x grid spacing.
    BCz : str
        z boundary mode; the current solver keeps z boundaries fixed.
    BCy : str
        y boundary mode: fixed or extend; follows existing solver constraints.
    BCx : str
        x boundary mode: fixed, extend, or periodic.
    delxSqr : float
        Squared x grid spacing.
    ratio2 : float
        x/z grid spacing ratio.
    ratio1 : float
        x/y grid spacing ratio.
    ratio2Sqr : float
        Squared x/z grid spacing ratio.
    ratio1Sqr : float
        Squared x/y grid spacing ratio.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Overflow flag, final convergence metric, iteration count, and optional history.
    mxLoop : int
        Maximum number of SOR sweeps.
    tolerance : float
        Stopping threshold for the relative update convergence metric.
    check_interval : int, optional
        Convergence/history sampling interval; zero selects the automatic interval.

    Returns
    -------
    numpy.ndarray
        Updated solution array S.

    Raises
    ------
    ValueError
        Periodic x extent is incompatible with the color period
        (2 grid points).
    """
    n_color = 2
    if BCx == 'periodic' and xc % 2:
        raise ValueError("CPU multi-color periodic x requires a length divisible by 2")
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros((zc - 2) * (yc - 2), dtype=np.float64)
    row_value = np.zeros((zc - 2) * (yc - 2), dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0
    for loop in range(1, mxLoop + 1):
        track_residual = loop % interval == 0 or loop == mxLoop
        max_update = 0.0
        max_value = 0.0
        _apply_extend_boundary_3d(S, BCy, BCx, undef)
        for color in range(n_color):
            for row in nb.prange((zc - 2) * (yc - 2)):
                k = 1 + row // (yc - 2)
                j = 1 + row % (yc - 2)
                local_update = 0.0
                local_value = 0.0
                istart = 0 if BCx == 'periodic' else 1
                iend = xc if BCx == 'periodic' else xc - 1
                for i in range(istart, iend):
                    if (k + j + i) % 2 != color:
                        continue
                    ip1 = (i + 1) % xc
                    im1 = (i - 1) % xc
                    cond = (G[k, j, i] != undef and H[k, j, i] != undef and
                            A[k, j, i] != undef and B[k, j, i] != undef and
                            C[k, j, i] != undef and D[k, j, i] != undef and
                            E[k, j, i] != undef and F[k, j, i] != undef)
                    
                    if cond:
                        temp = (
                            A[k, j, i] * (
                                (S[k+1, j, i] - S[k, j, i])-(S[k, j, i] - S[k-1, j, i])
                            ) * ratio2Sqr +
                            B[k, j, i] * (
                                (S[k, j+1, i] - S[k, j, i])-(S[k, j, i] - S[k, j-1, i])
                            ) * ratio1Sqr +
                            C[k, j, i] * (
                                (S[k, j, ip1] - S[k, j, i])-(S[k, j, i] - S[k, j, im1])
                            ) + (
                            D[k, j, i] * (
                                S[k+1, j, i] - S[k-1, j, i]
                            ) * ratio2 +
                            E[k, j, i] * (
                                S[k, j+1, i] - S[k, j-1, i]
                            ) * ratio1 +
                            F[k, j, i] * (
                                S[k, j, ip1] - S[k, j, im1]
                            )) * delx / 2.0 + (
                            G[k, j, i] * S[k, j, i] - H[k, j, i]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k, j, i]*ratio2Sqr + B[k, j, i]*ratio1Sqr + C[k, j, i]
                        ) * 2.0 - G[k, j, i]*delxSqr)
                        
                        if track_residual:
                            local_update, local_value = _accumulate_update_metric(
                                temp, S[k, j, i], local_update, local_value)
                        S[k, j, i] += temp
                row_update[row] = local_update
                row_value[row] = local_value
            if track_residual:
                for row in range((zc - 2) * (yc - 2)):
                    max_update = max(max_update, row_update[row])
                    max_value = max(max_value, row_value[row])
        if not track_residual:
            continue
        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop == mxLoop:
            break
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
    return S


@nb.njit(cache=False, nogil=True, parallel=True)
def invert_general_bih_2D_mc(S, A, B, C, D, E, F, G, H, I, J, info,
                          yc, xc, BCy, BCx,
                          delxSSr, delxTr, delxSqr,
                          ratio, ratioSSr, ratioQtr, ratioSqr,
                          optArg, undef, flags,
                          mxLoop, tolerance, check_interval=0):
    r"""
    Inverting a 2D slice of biharmonic equation in the general form.

    .. math::

        A \frac{\partial^4 \psi}{\partial y^4} +
        B \frac{\partial^4 \psi}{\partial y^2 \partial x^2} +
        C \frac{\partial^4 \psi}{\partial x^4} +
        D \frac{\partial^2 \psi}{\partial y^2} +
        E \frac{\partial^2 \psi}{\partial y \partial x} +
        F \frac{\partial^2 \psi}{\partial x^2} +
        G \frac{\partial \psi}{\partial y} +
        H \frac{\partial \psi}{\partial x} + I \psi = J
    
    Parameters
    ----------
    S : numpy.ndarray
        Writable solution array, updated in place.
    A : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    B : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    C : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    D : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    E : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    F : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    G : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    H : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    I : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    J : numpy.ndarray
        Right-hand-side forcing.
    info : object
        Slice label retained for backend signature compatibility.
    yc : int
        Number of y grid points.
    xc : int
        Number of x grid points.
    BCy : str
        y boundary mode: fixed or extend; follows existing solver constraints.
    BCx : str
        x boundary mode: fixed, extend, or periodic.
    delxSSr : float
        Fourth power of x grid spacing.
    delxTr : float
        Cubed x grid spacing.
    delxSqr : float
        Squared x grid spacing.
    ratio : float
        x/y grid spacing ratio.
    ratioSSr : float
        Fourth power of the x/y spacing ratio.
    ratioQtr : float
        x/y spacing ratio divided by four.
    ratioSqr : float
        Squared x/y spacing ratio.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Overflow flag, final convergence metric, iteration count, and optional history.
    mxLoop : int
        Maximum number of SOR sweeps.
    tolerance : float
        Stopping threshold for the relative update convergence metric.
    check_interval : int, optional
        Convergence/history sampling interval; zero selects the automatic interval.

    Returns
    -------
    numpy.ndarray
        Updated solution array S.

    Raises
    ------
    ValueError
        Periodic x extent is incompatible with the color period
        (3 grid points).
    """
    n_color = 9
    if BCx == 'periodic' and xc % 3:
        raise ValueError("CPU multi-color periodic x requires a length divisible by 3")
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros(yc, dtype=np.float64)
    row_value = np.zeros(yc, dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0
    for loop in range(1, mxLoop + 1):
        track_residual = loop % interval == 0 or loop == mxLoop
        max_update = 0.0
        max_value = 0.0
        _apply_extend_boundary_bih_2d(S, BCy, BCx, undef)
        for color in range(n_color):
            for j in nb.prange(2, yc - 2):
                row = j
                local_update = 0.0
                local_value = 0.0
                istart = 0 if BCx == 'periodic' else 2
                iend = xc if BCx == 'periodic' else xc - 2
                for i in range(istart, iend):
                    if (j % 3) * 3 + i % 3 != color:
                        continue
                    ip1 = (i + 1) % xc
                    im1 = (i - 1) % xc
                    ip2 = (i + 2) % xc
                    im2 = (i - 2) % xc
                    cond = (A[j, i] != undef and B[j, i] != undef and
                            C[j, i] != undef and D[j, i] != undef and
                            E[j, i] != undef and F[j, i] != undef and
                            G[j, i] != undef and H[j, i] != undef and
                            I[j, i] != undef and J[j, i] != undef)
                    
                    if cond:
                        temp = (
                            A[j, i] * (
                                S[j+2, i] - 4.0*S[j+1, i] + 6.0*S[j, i] - 4.0*S[j-1, i] + S[j-2, i]
                            ) * ratioSSr +
                            B[j, i] * (
                                    S[j+2, ip2] - 2.0*S[j+2, i] +     S[j+2, im2] +
                                -2.0*S[j, ip2] + 4.0*S[j, i] - 2.0*S[j, im2] +
                                     S[j-2, ip2] - 2.0*S[j-2, i] +     S[j-2, im2]
                            ) * ratioSqr / 16.0 +
                            C[j, i] * (
                                S[j, ip2] - 4.0*S[j, ip1] + 6.0*S[j, i] - 4.0*S[j, im1] + S[j, im2]
                            ) +
                            D[j, i] * (
                                (S[j+1, i] - S[j, i])-(S[j, i] - S[j-1, i])
                            ) * ratioSqr * delxSqr +
                            E[j, i] * (
                                (S[j+1, ip1] - S[j-1, ip1])-(S[j+1, im1] - S[j-1, im1])
                            ) * ratioQtr * delxSqr +
                            F[j, i] * (
                                (S[j, ip1] - S[j, i])-(S[j, i] - S[j, im1])
                            ) * delxSqr +
                            G[j, i] * (
                                S[j+1, i] - S[j-1, i]
                            ) * delxTr * ratio / 2.0 +
                            H[j, i] * (
                                S[j, ip1] - S[j, im1]
                            ) * delxTr / 2.0 + (
                            I[j, i] * S[j, i] - J[j, i]) * delxSSr
                        )
                        
                        temp *= -optArg / ((A[j, i]*ratioSSr + C[j, i]) * 6.0 +
                                            B[j, i]*ratioSqr / 4.0 +
                                          -(D[j, i]*ratioSqr + F[j, i]) * 2.0 * delxSqr +
                                            I[j, i]*delxSSr)
                        if track_residual:
                            local_update, local_value = _accumulate_update_metric(
                                temp, S[j, i], local_update, local_value)
                        S[j, i] += temp
                row_update[row] = local_update
                row_value[row] = local_value
            if track_residual:
                for row in range(yc):
                    max_update = max(max_update, row_update[row])
                    max_value = max(max_value, row_value[row])
        if not track_residual:
            continue
        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop == mxLoop:
            break
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
    return S


@nb.njit(cache=False, nogil=True, parallel=True)
def invert_standard_1D_mc(S, A, B, F, info,
                       xc, BCx, delxSqr, optArg, undef, flags,
                       mxLoop, tolerance, check_interval=0):
    r"""Inverting a 1D series of elliptic equation in standard form.

    .. math::

        \frac{\partial}{\partial x}\left(
        A\frac{\partial \psi}{\partial x}\right) + B\psi = F
    
    Parameters
    ----------
    S : numpy.ndarray
        Writable solution array, updated in place.
    A : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    B : numpy.ndarray
        Coefficient of the correspondingly labelled term in the equation above.
    F : numpy.ndarray
        Right-hand-side forcing.
    info : object
        Slice label retained for backend signature compatibility.
    xc : int
        Number of x grid points.
    BCx : str
        x boundary mode: fixed, extend, or periodic.
    delxSqr : float
        Squared x grid spacing.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Overflow flag, final convergence metric, iteration count, and optional history.
    mxLoop : int
        Maximum number of SOR sweeps.
    tolerance : float
        Stopping threshold for the relative update convergence metric.
    check_interval : int, optional
        Convergence/history sampling interval; zero selects the automatic interval.

    Returns
    -------
    numpy.ndarray
        Updated solution array S.

    Raises
    ------
    ValueError
        Periodic x extent is incompatible with the color period
        (2 grid points).
    """
    n_color = 2
    if BCx == 'periodic' and xc % 2:
        raise ValueError("CPU multi-color periodic x requires a length divisible by 2")
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros(xc, dtype=np.float64)
    row_value = np.zeros(xc, dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0
    aside = _find_boundary_anchor_1d(F, undef) if BCx == 'extend' else -1
    for loop in range(1, mxLoop + 1):
        track_residual = loop % interval == 0 or loop == mxLoop
        max_update = 0.0
        max_value = 0.0
        if BCx == 'extend':
            if S[1] != undef and aside != 0:
                S[0] = S[1]
            if S[-2] != undef and aside != 1:
                S[-1] = S[-2]
        for color in range(n_color):
            for i in nb.prange(xc):
                local_update = 0.0
                local_value = 0.0
                if i % 2 == color and (BCx == 'periodic' or 0 < i < xc - 1):
                    ip1 = (i + 1) % xc
                    im1 = (i - 1) % xc
                    cond = (F[i]!=undef and A[i]!=undef and A[ip1]!=undef and B[i]!=undef)
                    
                    if cond:
                        temp = (
                            A[ip1] * (S[ip1] - S[i]) - A[i] * (S[i] - S[im1])
                        ) / delxSqr + (B[i] * S[i] - F[i])
                        
                        temp *= optArg / ((A[ip1] + A[i]) / delxSqr - B[i])
                        if track_residual:
                            local_update, local_value = _accumulate_update_metric(
                                temp, S[i], local_update, local_value)
                        S[i] += temp
                row_update[i] = local_update
                row_value[i] = local_value
            if track_residual:
                for row in range(xc):
                    max_update = max(max_update, row_update[row])
                    max_value = max(max_value, row_value[row])
        if not track_residual:
            continue
        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop == mxLoop:
            break
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
    return S


def invert_standard_2D_full_mc(S, A, B, C, D, E, F, info,
                               yc, xc, BCy, BCx, delxSqr,
                               ratioQtr, ratioSqr, optArg, undef, flags,
                               mxLoop, tolerance, check_interval=0):
    """Solve the full 2-D divergence-form equation with multi-color SOR.

    Parameters
    ----------
    S : numpy.ndarray
        Two-dimensional solution, updated in place.
    A, D : numpy.ndarray
        Coefficients of the y and x diagonal flux derivatives.
    B, C : numpy.ndarray
        Independent coefficients of the two cross flux derivatives.
    E : numpy.ndarray
        Coefficient of the local linear term.
    F : numpy.ndarray
        Right-hand-side forcing.
    info : object
        Slice label retained for backend signature compatibility.
    yc, xc : int
        Number of y and x grid points.
    BCy, BCx : str
        Boundary modes passed by the public solver.
    delxSqr : float
        Squared x spacing.
    ratioQtr, ratioSqr : float
        x/y spacing ratio divided by four and squared, respectively.
    optArg : float
        SOR relaxation factor.
    undef : float
        Internal undefined-value sentinel.
    flags : numpy.ndarray
        Overflow, final metric, iteration count, and optional history output.
    mxLoop : int
        Maximum number of iterations.
    tolerance : float
        Stopping threshold for the relative update metric.
    check_interval : int, optional
        Diagnostic interval; zero selects the automatic interval.

    Returns
    -------
    numpy.ndarray
        The updated solution array S.

    Raises
    ------
    ValueError
        If periodic x has an odd extent, incompatible with parity coloring.
    """
    n_color = 4 if (np.any(np.asarray(B) != 0) or
                    np.any(np.asarray(C) != 0)) else 2
    if BCx == 'periodic' and xc % 2:
        raise ValueError(
            f"CPU multi-color SOR with periodic x requires an even x grid "
            f"length; got {xc}")
    return _invert_standard_2D_full_mc_impl(
        S, A, B, C, D, E, F, yc, xc, BCy, BCx, delxSqr,
        ratioQtr, ratioSqr, optArg, undef, flags, mxLoop,
        tolerance, check_interval, n_color)

@nb.njit(cache=False, nogil=True, parallel=True)
def _invert_standard_2D_full_mc_impl(S, A, B, C, D, E, F,
                                yc, xc, BCy, BCx, delxSqr,
                                ratioQtr, ratioSqr, optArg, undef, flags,
                                mxLoop, tolerance, check_interval, n_color):
    """Compiled full divergence-form multi-color SOR loop."""
    interval = _check_interval(mxLoop, tolerance, check_interval)
    row_update = np.zeros(yc, dtype=np.float64)
    row_value = np.zeros(yc, dtype=np.float64)
    diagnostic_index = 0
    overflow = False
    error = 0.0

    if BCy == 'extend' and BCx != 'fixed':
        aside, aidx = _find_boundary_anchor_2d(
            F, undef, BCx == 'extend')
    else:
        aside, aidx = -1, -1

    for loop in range(1, mxLoop + 1):
        track_residual = (loop % interval == 0) or loop >= mxLoop
        max_update = 0.0
        max_value = 0.0

        _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx)

        for color in range(n_color):
            for j in nb.prange(1, yc - 1):
                local_update = 0.0
                local_value = 0.0
                istart = 0 if BCx == 'periodic' else 1
                iend = xc if BCx == 'periodic' else xc - 1
                if n_color == 2:
                    target_i_parity = (color - j) & 1
                else:
                    if (j & 1) != color // 2:
                        row_update[j] = 0.0
                        row_value[j] = 0.0
                        continue
                    target_i_parity = color & 1

                for i in range(istart, iend):
                    if (i & 1) != target_i_parity:
                        continue

                    ip1 = i + 1
                    im1 = i - 1
                    if ip1 == xc:
                        ip1 = 0
                    if im1 < 0:
                        im1 = xc - 1

                    if (F[j, i] == undef or
                            A[j + 1, i] == undef or A[j, i] == undef or
                            E[j, i] == undef or
                            B[j + 1, i] == undef or B[j - 1, i] == undef or
                            C[j, ip1] == undef or C[j, im1] == undef or
                            D[j, ip1] == undef or D[j, i] == undef):
                        continue

                    temp = (
                        (A[j + 1, i] * (S[j + 1, i] - S[j, i]) -
                         A[j, i] * (S[j, i] - S[j - 1, i])) * ratioSqr +
                        (B[j + 1, i] *
                         (S[j + 1, ip1] - S[j + 1, im1]) -
                         B[j - 1, i] *
                         (S[j - 1, ip1] - S[j - 1, im1])) * ratioQtr +
                        (C[j, ip1] *
                         (S[j + 1, ip1] - S[j - 1, ip1]) -
                         C[j, im1] *
                         (S[j + 1, im1] - S[j - 1, im1])) * ratioQtr +
                        (D[j, ip1] * (S[j, ip1] - S[j, i]) -
                         D[j, i] * (S[j, i] - S[j, im1]))
                    ) + (E[j, i] * S[j, i] - F[j, i]) * delxSqr

                    denom = ((A[j + 1, i] + A[j, i]) * ratioSqr +
                             D[j, ip1] + D[j, i] - E[j, i] * delxSqr)
                    if denom == 0.0:
                        continue
                    delta = temp * optArg / denom

                    if track_residual:
                        new_value = S[j, i] + delta
                        if (not np.isfinite(delta) or
                                not np.isfinite(new_value)):
                            local_update = np.inf
                            local_value = max(local_value, 1.0)
                        else:
                            local_update = max(local_update, abs(delta))
                            local_value = max(
                                local_value, abs(S[j, i]), abs(new_value))
                    S[j, i] += delta

                row_update[j] = local_update
                row_value[j] = local_value

            if track_residual:
                for j in range(1, yc - 1):
                    max_update = max(max_update, row_update[j])
                    max_value = max(max_value, row_value[j])

        if not track_residual:
            continue

        error = _relative_update_metric(max_update, max_value)
        overflow = not np.isfinite(error) or error > 1e100
        if flags.size > 3:
            flags[3 + diagnostic_index] = error
            diagnostic_index += 1
        if overflow or error < tolerance or loop >= mxLoop:
            flags[2] = loop
            break

    flags[0] = overflow
    flags[1] = error
    return S
