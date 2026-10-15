"""
CPUs module of xinvert: numba-jitted SOR iteration kernels.

Contains the ``@nb.jit`` decorated inner loops (``invert_standard_3D``,
``invert_general_3D``, ``invert_general_2D_bih``, etc.) for maximum iteration
speed, called by the high-level solvers in :mod:`xinvert.core`.
"""
import numba as nb
import numpy as np

"""
Below are the numba functions
"""

@nb.njit(cache=False, nogil=True)
def invert_standard_3D(S, A, B, C, F, info,
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
    S: numpy.array (output)
        Results of the SOR inversion.
    A: numpy.array
        Coefficient for the first dimensional derivative.
    B: numpy.array
        Coefficient for the cross derivatives.
    C: numpy.array
        Coefficient for the second dimensional derivative.
    F: numpy.array
        Forcing function.
    info: numpy.array
        Information array for logging purpose.
    zc: int
        Number of grid point in z-dimension (e.g., Z or lev).
    yc: int
        Number of grid point in y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    BCz: str
        Boundary condition for dimension z in ['fixed', 'extend'].
    BCy: str
        Boundary condition for dimension y in ['fixed', 'extend', 'periodic'].
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSqr: float
        Squared increment (interval) in dimension x (unit of m^2).
    ratio2Sqr: float
        Squared Ratio of delx to delz.
    ratio1Sqr: float
        Squared Ratio of delx to dely.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # process x/y boundaries independently; z remains fixed
        _apply_extend_boundary_3d(S, BCy, BCx, undef)
        
        for k in range(1, zc-1):
            for j in range(1, yc-1):
                # for the west boundary iteration (i==0)
                if BCx == 'periodic':
                    cond = (F[k,j  ,0] != undef and
                            A[k+1,j,0] != undef and A[k,j,0] != undef and
                            B[k,j+1,0] != undef and B[k,j,0] != undef and
                            C[k,j  ,1] != undef and C[k,j,0] != undef)
                    
                    if cond:
                        temp = (
                            (
                                A[k+1,j,0] * (S[k+1,j,0] - S[k,  j,0])-
                                A[k,j  ,0] * (S[k,j  ,0] - S[k-1,j,0])
                            ) * ratio2Sqr + (
                                B[k,j+1,0] * (S[k,j+1,0] - S[k,j  ,0])-
                                B[k,j  ,0] * (S[k,j  ,0] - S[k,j-1,0])
                            ) * ratio1Sqr + (
                                C[k,j  ,1] * (S[k,j  ,1] - S[k,j , 0])-
                                C[k,j  ,0] * (S[k,j  ,0] - S[k,j ,-1])
                            )
                        ) - F[k,j,0] * delxSqr
                        
                        temp *= optArg / ((A[k+1,j,0] + A[k,j,0]) *ratio2Sqr +
                                          (B[k,j+1,0] + B[k,j,0]) *ratio1Sqr +
                                          (C[k,j  ,1] + C[k,j,0]))
                        if track_residual:
                            maxUpdate, maxValue = _accumulate_update_metric(
                                temp, S[k,j,0], maxUpdate, maxValue)
                        S[k,j,0] += temp
                
                # inner loop
                for i in range(1, xc-1):
                    cond = (F[k  ,j,i] != undef and
                            A[k+1,j,i] != undef and A[k,j,i] != undef and
                            B[k,j+1,i] != undef and B[k,j,i] != undef and
                            C[k,j,i+1] != undef and C[k,j,i] != undef)
                    
                    if cond:
                        temp = (
                            (
                                A[k+1,j,i] * (S[k+1,j,i] - S[k  ,j,i])-
                                A[k  ,j,i] * (S[k  ,j,i] - S[k-1,j,i])
                            ) * ratio2Sqr + (
                                B[k,j+1,i] * (S[k,j+1,i] - S[k,j  ,i])-
                                B[k,j  ,i] * (S[k,j  ,i] - S[k,j-1,i])
                            ) * ratio1Sqr + (
                                C[k,j,i+1] * (S[k,j,i+1] - S[k,j,  i])-
                                C[k,j,i  ] * (S[k,j,i  ] - S[k,j,i-1])
                            )
                        ) - F[k,j,i] * delxSqr
                        
                        temp *= optArg / ((A[k+1,j,i] + A[k,j,i]) *ratio2Sqr +
                                          (B[k,j+1,i] + B[k,j,i]) *ratio1Sqr +
                                          (C[k,j,i+1] + C[k,j,i]))
                        if track_residual:
                            maxUpdate, maxValue = _accumulate_update_metric(
                                temp, S[k,j,i], maxUpdate, maxValue)
                        S[k,j,i] += temp
                
                # for the east boundary iteration (i==-1)
                if BCx == 'periodic':
                    cond = (F[k,j  ,-1] != undef and
                            A[k+1,j,-1] != undef and A[k,j,-1] != undef and
                            B[k,j+1,-1] != undef and B[k,j,-1] != undef and
                            C[k,j  , 0] != undef and C[k,j,-1] != undef)
                    
                    if cond:
                        temp = (
                            (
                                A[k+1,j,-1] * (S[k+1,j,-1] - S[k  ,j,-1])-
                                A[k,  j,-1] * (S[k,  j,-1] - S[k-1,j,-1])
                            ) * ratio2Sqr + (
                                B[k,j+1,-1] * (S[k,j+1,-1] - S[k,j , -1])-
                                B[k,j  ,-1] * (S[k,j  ,-1] - S[k,j-1,-1])
                            ) * ratio1Sqr + (
                                C[k,j  , 0] * (S[k,j  , 0] - S[k,j  ,-1])-
                                C[k,j  ,-1] * (S[k,j  ,-1] - S[k,j  ,-2])
                            )
                        ) - F[k,j,-1] * delxSqr
                        
                        temp *= optArg / ((A[k+1,j,-1] + A[k,j,-1]) *ratio2Sqr +
                                          (B[k,j+1,-1] + B[k,j,-1]) *ratio1Sqr+
                                          (C[k,j  , 0] + C[k,j,-1]))
                        if track_residual:
                            maxUpdate, maxValue = _accumulate_update_metric(
                                temp, S[k,j,-1], maxUpdate, maxValue)
                        S[k,j,-1] += temp
        
        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100

        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, F, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop

    return S


@nb.njit(cache=False, nogil=True)
def invert_standard_2D(S, A, B, C, F, info,
                       yc, xc, BCy, BCx, delxSqr,
                       ratioQtr, ratioSqr, optArg, undef, flags,
                       mxLoop, tolerance, check_interval=0):
    r"""Inverting a 2D slice of elliptic equation in standard form.

    .. math::

        \frac{\partial}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{\partial}{\partial x}\left(
        B\frac{\partial \psi}{\partial y} +
        C\frac{\partial \psi}{\partial x} \right) = F
    
    Parameters
    ----------
    S: numpy.array (output)
        Results of the SOR inversion.
    A: numpy.array
        Coefficient for the first dimensional derivative.
    B: numpy.array
        Coefficient for the cross derivatives.
    C: numpy.array
        Coefficient for the second dimensional derivative.
    F: numpy.array
        Forcing function.
    info: numpy.array
        Information array for logging purpose.
    yc: int
        Number of grid point in y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    BCy: str
        Boundary condition for dimension y in ['fixed', 'extend', 'periodic'].
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSqr: float
        Squared increment (interval) in dimension x (unit of m^2).
    ratioQtr: float
        Ratio of delx to dely, divided by 4.
    ratioSqr: float
        Squared Ratio of delx to dely.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # resolve the gauge anchor once, BEFORE the first sweep: pin ONE
        # BOUNDARY grid point (extend-copy skipped -> keeps its initial
        # value); every interior point keeps updating normally.  Active
        # whenever y is extend and x is NOT fixed (extend or periodic
        # both leave the constant null space unresolved).
        if aside == -2:
            if BCy == 'extend' and BCx != 'fixed':
                aside, aidx = _find_boundary_anchor_2d(
                    F, undef, BCx == 'extend')
            else:
                aside, aidx = -1, -1

        # process x/y boundaries independently after resolving the anchor
        _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx)

        for j in range(1, yc-1):
            # for the west boundary iteration (i==0)
            if BCx == 'periodic':
                cond = (F[j  ,0] != undef and
                        A[j+1,0] != undef and A[j  , 0] != undef and
                        B[j  ,1] != undef and B[j  ,-1] != undef and
                        B[j+1,0] != undef and B[j-1, 0] != undef and
                        C[j  ,1] != undef and C[j  , 0] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j+1,0] * (S[j+1,0] - S[j , 0])-
                            A[j  ,0] * (S[j  ,0] - S[j-1,0])
                        ) * ratioSqr + (
                            B[j+1,1] * (S[j+1,1] - S[j+1,-1])-
                            B[j-1,0] * (S[j-1,0] - S[j-1,-1])
                        ) * ratioQtr + (
                            B[j, 1] * (S[j+1, 1] - S[j-1, 1])-
                            B[j,-1] * (S[j+1,-1] - S[j-1,-1])
                        ) * ratioQtr + (
                            C[j,1] * (S[j,1] - S[j, 0])-
                            C[j,0] * (S[j,0] - S[j,-1])
                        )
                    ) - F[j,0] * delxSqr
                    
                    temp *= optArg / ((A[j+1,0] + A[j,0]) *ratioSqr +
                                      (C[j  ,1] + C[j,0]))
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,0], maxUpdate, maxValue)
                    S[j,0] += temp
            
            # inner loop
            for i in range(1, xc-1):
                cond = (F[j  ,i  ] != undef and
                        A[j+1,i  ] != undef and A[j  ,  i] != undef and
                        B[j  ,i+1] != undef and B[j  ,i-1] != undef and
                        B[j+1,i  ] != undef and B[j-1,  i] != undef and
                        C[j  ,i+1] != undef and C[j  ,  i] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j+1,i] * (S[j+1,i] - S[j  ,i])-
                            A[j  ,i] * (S[j  ,i] - S[j-1,i])
                        ) * ratioSqr + (
                            B[j+1,i] * (S[j+1,i+1] - S[j+1,i-1])-
                            B[j-1,i] * (S[j-1,i+1] - S[j-1,i-1])
                        ) * ratioQtr + (
                            B[j,i+1] * (S[j+1,i+1] - S[j-1,i+1])-
                            B[j,i-1] * (S[j+1,i-1] - S[j-1,i-1])
                        ) * ratioQtr + (
                            C[j,i+1] * (S[j,i+1] - S[j,  i])-
                            C[j,i  ] * (S[j,i  ] - S[j,i-1])
                        )
                    ) - F[j,i] * delxSqr
                    
                    temp *= optArg / ((A[j+1,i] + A[j,i]) *ratioSqr +
                                      (C[j,i+1] + C[j,i]))
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,i], maxUpdate, maxValue)
                    S[j,i] += temp
            
            
            # for the east boundary iteration (i==-1)
            if BCx == 'periodic':
                cond = (F[j  ,-1] != undef and
                        A[j+1,-1] != undef and A[j  ,-1] != undef and
                        B[j  , 0] != undef and B[j  ,-2] != undef and
                        B[j+1,-1] != undef and B[j-1,-1] != undef and
                        C[j  , 0] != undef and C[j  ,-1] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j+1,-1] * (S[j+1,-1] - S[j , -1])-
                            A[j  ,-1] * (S[j  ,-1] - S[j-1,-1])
                        ) * ratioSqr + (
                            B[j+1,-1] * (S[j+1,0] - S[j+1,-2])-
                            B[j-1,-1] * (S[j-1,0] - S[j-1,-2])
                        ) * ratioQtr + (
                            B[j, 0] * (S[j+1, 0] - S[j-1, 0])-
                            B[j,-2] * (S[j+1,-2] - S[j-1,-2])
                        ) * ratioQtr + (
                            C[j, 0] * (S[j, 0] - S[j,-1])-
                            C[j,-1] * (S[j,-1] - S[j,-2])
                        )
                    ) - F[j,-1] * delxSqr
                    
                    temp *= optArg / ((A[j+1,-1] + A[j,-1]) *ratioSqr +
                                      (C[j  , 0] + C[j,-1]))
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,-1], maxUpdate, maxValue)
                    S[j,-1] += temp

        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100

        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, F, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
        
    return S



@nb.njit(cache=False, nogil=True)
def invert_standard_2D_full(S, A, B, C, D, E, F, info,
                       yc, xc, BCy, BCx, delxSqr,
                       ratioQtr, ratioSqr, optArg, undef, flags,
                       mxLoop, tolerance, check_interval=0):
    r"""Inverting a 2D slice of elliptic equation in standard form.

    .. math::

        \frac{\partial}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{\partial}{\partial x}\left(
        C\frac{\partial \psi}{\partial y} +
        D\frac{\partial \psi}{\partial x} \right) + E\psi = F
    
    Parameters
    ----------
    S: numpy.array (output)
        Results of the SOR inversion.
        
    A: numpy.array
        Coefficient for the first dimensional derivative.
    B: numpy.array
        Coefficient for the cross derivatives.
    C: numpy.array
        Coefficient for the cross derivatives.
    D: numpy.array
        Coefficient for the second dimensional derivative.
    E: numpy.array
        Coefficient for the linear term.
    F: numpy.array
        Forcing function.
    info: numpy.array
        Information array for logging purpose.
    yc: int
        Number of grid point in y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    BCy: str
        Boundary condition for dimension y in ['fixed', 'extend', 'periodic'].
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSqr: float
        Squared increment (interval) in dimension x (unit of m^2).
    ratioQtr: float
        Ratio of delx to dely, divided by 4.
    ratioSqr: float
        Squared Ratio of delx to dely.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    S: numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # resolve the gauge anchor once, BEFORE the first sweep: pin ONE
        # BOUNDARY grid point (extend-copy skipped -> keeps its initial
        # value); every interior point keeps updating normally.  Active
        # whenever y is extend and x is NOT fixed (extend or periodic
        # both leave the constant null space unresolved).
        if aside == -2:
            if BCy == 'extend' and BCx != 'fixed':
                aside, aidx = _find_boundary_anchor_2d(
                    F, undef, BCx == 'extend')
            else:
                aside, aidx = -1, -1

        # process x/y boundaries independently after resolving the anchor
        _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx)

        for j in range(1, yc-1):
            # for the west boundary iteration (i==0)
            if BCx == 'periodic':
                cond = (F[j  ,0] != undef and
                        A[j+1,0] != undef and A[j  , 0] != undef and
                        B[j+1,0] != undef and B[j-1, 0] != undef and
                        C[j  ,1] != undef and C[j  ,-1] != undef and
                        D[j  ,1] != undef and D[j  , 0] != undef and
                        E[j  ,0] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j+1,0] * (S[j+1,0] - S[j , 0])-
                            A[j  ,0] * (S[j  ,0] - S[j-1,0])
                        ) * ratioSqr + (
                            B[j+1,1] * (S[j+1,1] - S[j+1,-1])-
                            B[j-1,0] * (S[j-1,0] - S[j-1,-1])
                        ) * ratioQtr + (
                            C[j, 1] * (S[j+1, 1] - S[j-1, 1])-
                            C[j,-1] * (S[j+1,-1] - S[j-1,-1])
                        ) * ratioQtr + (
                            D[j,1] * (S[j,1] - S[j, 0])-
                            D[j,0] * (S[j,0] - S[j,-1])
                        )
                    ) + (E[j,0] * S[j,0] - F[j,0]) * delxSqr
                    
                    temp *= optArg / ((A[j+1,0] + A[j,0]) *ratioSqr +
                                      (D[j  ,1] + D[j,0]) - E[j, 0]*delxSqr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,0], maxUpdate, maxValue)
                    S[j,0] += temp
            
            # inner loop
            for i in range(1, xc-1):
                cond = (F[j  ,i  ] != undef and
                        A[j+1,i  ] != undef and A[j  ,  i] != undef and
                        B[j+1,i  ] != undef and B[j-1,  i] != undef and
                        C[j  ,i+1] != undef and C[j  ,i-1] != undef and
                        D[j  ,i+1] != undef and D[j  ,  i] != undef and
                        E[j,  i  ] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j+1,i] * (S[j+1,i] - S[j  ,i])-
                            A[j  ,i] * (S[j  ,i] - S[j-1,i])
                        ) * ratioSqr + (
                            B[j+1,i] * (S[j+1,i+1] - S[j+1,i-1])-
                            B[j-1,i] * (S[j-1,i+1] - S[j-1,i-1])
                        ) * ratioQtr + (
                            C[j,i+1] * (S[j+1,i+1] - S[j-1,i+1])-
                            C[j,i-1] * (S[j+1,i-1] - S[j-1,i-1])
                        ) * ratioQtr + (
                            D[j,i+1] * (S[j,i+1] - S[j,  i])-
                            D[j,i  ] * (S[j,i  ] - S[j,i-1])
                        )
                    ) + (E[j,i] * S[j,i] - F[j,i]) * delxSqr
                    
                    temp *= optArg / ((A[j+1,i] + A[j,i]) *ratioSqr +
                                      (D[j,i+1] + D[j,i]) - E[j, i]*delxSqr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,i], maxUpdate, maxValue)
                    S[j,i] += temp
            
            
            # for the east boundary iteration (i==-1)
            if BCx == 'periodic':
                cond = (F[j  ,-1] != undef and
                        A[j+1,-1] != undef and A[j  ,-1] != undef and
                        B[j+1,-1] != undef and B[j-1,-1] != undef and
                        C[j  , 0] != undef and C[j  ,-2] != undef and
                        D[j  , 0] != undef and D[j  ,-1] != undef and
                        E[j  ,-1] != undef)
                
                if cond:
                    temp = (
                        (
                            A[j+1,-1] * (S[j+1,-1] - S[j , -1])-
                            A[j  ,-1] * (S[j  ,-1] - S[j-1,-1])
                        ) * ratioSqr + (
                            B[j+1,-1] * (S[j+1,0] - S[j+1,-2])-
                            B[j-1,-1] * (S[j-1,0] - S[j-1,-2])
                        ) * ratioQtr + (
                            C[j, 0] * (S[j+1, 0] - S[j-1, 0])-
                            C[j,-2] * (S[j+1,-2] - S[j-1,-2])
                        ) * ratioQtr + (
                            D[j, 0] * (S[j, 0] - S[j,-1])-
                            D[j,-1] * (S[j,-1] - S[j,-2])
                        )
                    ) + (E[j,-1] * S[j,-1] - F[j,-1]) * delxSqr
                    
                    temp *= optArg / ((A[j+1,-1] + A[j,-1]) *ratioSqr +
                                      (D[j  , 0] + D[j,-1]) - E[j, -1]*delxSqr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,-1], maxUpdate, maxValue)
                    S[j,-1] += temp

        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100

        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, F, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
        
    return S


@nb.njit(cache=False, nogil=True)
def invert_standard_1D(S, A, B, F, info,
                       xc, BCx, delxSqr, optArg, undef, flags,
                       mxLoop, tolerance, check_interval=0):
    r"""Inverting a 1D series of elliptic equation in standard form.

    .. math::

        \frac{\partial}{\partial x}\left(
        A\frac{\partial \psi}{\partial x}\right) + B\psi = F
    
    Parameters
    ----------
    S: numpy.array (output)
        Results of the SOR inversion.
        
    A: numpy.array
        Coefficient for the 2nd-order derivative.
    B: numpy.array
        Coefficient for the linear term.
    F: numpy.array
        Forcing function.
    info: numpy.array
        Information array for logging purpose.
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSqr: float
        Squared increment (interval) in dimension x (unit of m^2).
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    S: numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # process boundaries
        if BCx == 'extend':
            # gauge anchor (singular Neumann-at-both-ends system): the
            # pinned boundary point keeps its initial value
            if  S[ 1] != undef and aside != 0:
                S[ 0] = S[ 1]
            if  S[-2] != undef and aside != 1:
                S[-1] = S[-2]

        # resolve the gauge anchor once, BEFORE the first sweep: pin ONE
        # BOUNDARY grid point (extend-copy skipped); all interior points
        # keep updating normally.
        if aside == -2:
            if BCx == 'extend':
                aside = _find_boundary_anchor_1d(F, undef)
            else:
                aside = -1
        
        # for the west boundary iteration (i==0)
        if BCx == 'periodic':
            cond = (F[0]!=undef and A[0]!=undef and A[1]!=undef and B[0]!=undef)
            
            if cond:
                temp = (
                    A[1] * (S[1] - S[0]) - A[0] * (S[0] - S[-1])
                ) / delxSqr + (B[0] * S[0] - F[0])
                
                temp *= optArg / ((A[1] + A[0]) / delxSqr - B[0])
                if track_residual:
                    maxUpdate, maxValue = _accumulate_update_metric(
                        temp, S[0], maxUpdate, maxValue)
                S[0] += temp
        
        # inner loop
        for i in range(1, xc-1):
            cond = (F[i]!=undef and A[i]!=undef and A[i+1]!=undef and B[i]!=undef)
            
            if cond:
                temp = (
                    A[i+1] * (S[i+1] - S[i]) - A[i] * (S[i] - S[i-1])
                ) / delxSqr + (B[i] * S[i] - F[i])
                
                temp *= optArg / ((A[i+1] + A[i]) / delxSqr - B[i])
                if track_residual:
                    maxUpdate, maxValue = _accumulate_update_metric(
                        temp, S[i], maxUpdate, maxValue)
                S[i] += temp
        
        # for the west boundary iteration (i==-1)
        if BCx == 'periodic':
            cond = (F[-1]!=undef and A[-1]!=undef and A[0]!=undef and B[-1]!=undef)
            
            if cond:
                temp = (
                    A[0] * (S[0] - S[-1]) - A[-1] * (S[-1] - S[-2])
                ) / delxSqr + (B[-1] * S[-1] - F[-1])
                
                temp *= optArg / ((A[0] + A[-1]) / delxSqr - B[-1])
                if track_residual:
                    maxUpdate, maxValue = _accumulate_update_metric(
                        temp, S[-1], maxUpdate, maxValue)
                S[-1] += temp

        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100

        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, F, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
        
    return S


@nb.njit(cache=False, nogil=True)
def invert_general_3D(S, A, B, C, D, E, F, G, H, info,
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
    S: numpy.array (output)
        Results of the SOR inversion.
        
    A: numpy.array
        Coefficient for the first term.
    B: numpy.array
        Coefficient for the second term.
    C: numpy.array
        Coefficient for the third term.
    D: numpy.array
        Coefficient for the fourth term.
    E: numpy.array
        Coefficient for the fifth term.
    F: numpy.array
        Coefficient for the sixth term.
    G: numpy.array
        Coefficient for the seventh term.
    H: numpy.array
        A known forcing function.
    info: numpy.array
        Information array for logging purpose.
    zc: int
        Number of grid point in the z-dimension (e.g., Z or lev).
    yc: int
        Number of grid point in the y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in the x-dimension (e.g., X or lon).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
    BCz: str
        Boundary condition for dimension z in ['fixed', 'extend'].
    BCy: str
        Boundary condition for dimension y in ['fixed', 'extend'].
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSqr: float
        Squared increment (interval) in dimension y (unit of m^2).
    ratio2Sqr: float
        Squared Ratio of delx to delz.
    ratio1Sqr: float
        Squared Ratio of delx to dely.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # process x/y boundaries independently; z remains fixed
        _apply_extend_boundary_3d(S, BCy, BCx, undef)
        
        for k in range(1, zc-1):
            for j in range(1, yc-1):
                # for the west boundary iteration (i==0)
                if BCx == 'periodic':
                    cond = (G[k,j,0] != undef and H[k,j,0] != undef and
                            A[k,j,0] != undef and B[k,j,0] != undef and
                            C[k,j,0] != undef and D[k,j,0] != undef and
                            E[k,j,0] != undef and F[k,j,0] != undef)
                    
                    if cond:
                        temp = (
                            A[k,j,0] * (
                                (S[k+1,j,0] - S[k,j,0])-(S[k,j,0] - S[k-1,j,0])
                            ) * ratio2Sqr +
                            B[k,j,0] * (
                                (S[k,j+1,0] - S[k,j,0])-(S[k,j,0] - S[k,j-1,0])
                            ) * ratio1Sqr +
                            C[k,j,0] * (
                                (S[k,j,1] - S[k,j,0])-(S[k,j,0] - S[k,j,-1])
                            ) + (
                            D[k,j,0] * (
                                S[k+1,j,0] - S[k-1,j,0]
                            ) * ratio2 +
                            E[k,j,0] * (
                                S[k,j+1,0] - S[k,j-1,0]
                            ) * ratio1 +
                            F[k,j,0] * (
                                S[k,j,1] - S[k,j,-1]
                            )) * delx / 2.0 + (
                            G[k,j,0] * S[k,j,0] - H[k,j,0]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k,j,0]*ratio2Sqr + B[k,j,0]*ratio1Sqr + C[k,j,0]
                        ) * 2.0 - G[k,j,0]*delxSqr)
                        
                        if track_residual:
                            maxUpdate, maxValue = _accumulate_update_metric(
                                temp, S[k,j,0], maxUpdate, maxValue)
                        S[k,j,0] += temp
                
                # inner loop
                for i in range(1, xc-1):
                    cond = (G[k,j,i] != undef and H[k,j,i] != undef and
                            A[k,j,i] != undef and B[k,j,i] != undef and
                            C[k,j,i] != undef and D[k,j,i] != undef and
                            E[k,j,i] != undef and F[k,j,i] != undef)
                    
                    if cond:
                        temp = (
                            A[k,j,i] * (
                                (S[k+1,j,i] - S[k,j,i])-(S[k,j,i] - S[k-1,j,i])
                            ) * ratio2Sqr +
                            B[k,j,i] * (
                                (S[k,j+1,i] - S[k,j,i])-(S[k,j,i] - S[k,j-1,i])
                            ) * ratio1Sqr +
                            C[k,j,i] * (
                                (S[k,j,i+1] - S[k,j,i])-(S[k,j,i] - S[k,j,i-1])
                            ) + (
                            D[k,j,i] * (
                                S[k+1,j,i] - S[k-1,j,i]
                            ) * ratio2 +
                            E[k,j,i] * (
                                S[k,j+1,i] - S[k,j-1,i]
                            ) * ratio1 +
                            F[k,j,i] * (
                                S[k,j,i+1] - S[k,j,i-1]
                            )) * delx / 2.0 + (
                            G[k,j,i] * S[k,j,i] - H[k,j,i]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k,j,i]*ratio2Sqr + B[k,j,i]*ratio1Sqr + C[k,j,i]
                        ) * 2.0 - G[k,j,i]*delxSqr)
                        
                        if track_residual:
                            maxUpdate, maxValue = _accumulate_update_metric(
                                temp, S[k,j,i], maxUpdate, maxValue)
                        S[k,j,i] += temp
                
                # for the east boundary iteration (i==-1)
                if BCx == 'periodic':
                    cond = (G[k,j,-1] != undef and H[k,j,-1] != undef and
                            A[k,j,-1] != undef and B[k,j,-1] != undef and
                            C[k,j,-1] != undef and D[k,j,-1] != undef and
                            E[k,j,-1] != undef and F[k,j,-1] != undef)
                    
                    if cond:
                        temp = (
                            A[k,j,-1] * (
                                (S[k+1,j,-1] - S[k,j,-1])-(S[k,j,-1] - S[k-1,j,-1])
                            ) * ratio2Sqr +
                            B[k,j,-1] * (
                                (S[k,j+1,-1] - S[k,j,-1])-(S[k,j,-1] - S[k,j-1,-1])
                            ) * ratio1Sqr +
                            C[k,j,-1] * (
                                (S[k,j,0] - S[k,j,-1])-(S[k,j,-1] - S[k,j,-2])
                            ) + (
                            D[k,j,-1] * (
                                S[k+1,j,-1] - S[k-1,j,-1]
                            ) * ratio2 +
                            E[k,j,-1] * (
                                S[k,j+1,-1] - S[k,j-1,-1]
                            ) * ratio1 +
                            F[k,j,-1] * (
                                S[k,j,0] - S[k,j,-2]
                            )) * delx / 2.0 + (
                            G[k,j,-1] * S[k,j,-1] - H[k,j,-1]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k,j,-1]*ratio2Sqr + B[k,j,-1]*ratio1Sqr + C[k,j,-1]
                        ) * 2.0 - G[k,j,-1]*delxSqr)
                        
                        if track_residual:
                            maxUpdate, maxValue = _accumulate_update_metric(
                                temp, S[k,j,-1], maxUpdate, maxValue)
                        S[k,j,-1] += temp
        
        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100

        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, H, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
        
    return S


@nb.njit(cache=False, nogil=True)
def invert_general_2D(S, A, B, C, D, E, F, G, info,
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
    S: numpy.array (output)
        Results of the SOR inversion.
        
    A: numpy.array
        Coefficient for the first term.
    B: numpy.array
        Coefficient for the second term.
    C: numpy.array
        Coefficient for the third term.
    D: numpy.array
        Coefficient for the fourth term.
    E: numpy.array
        Coefficient for the fifth term.
    F: numpy.array
        Coefficient for the sixth term.
    G: numpy.array
        Known forcing function.
    info: numpy.array
        Information array for logging purpose.
    yc: int
        Number of grid point in the y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in the x-dimension (e.g., X or lon).
    BCy: str
        Boundary condition for dimension y in ['fixed', 'extend', 'periodic'].
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSqr: float
        Squared increment (interval) in dimension y (unit of m^2).
    ratio: float
        Ratio of delx to dely.
    ratioQtr: float
        Ratio of delx to dely, divided by 4.
    ratioSqr: float
        Squared Ratio of delx to dely.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    S: numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # resolve the gauge anchor once, BEFORE the first sweep: pin ONE
        # BOUNDARY grid point (extend-copy skipped -> keeps its initial
        # value); every interior point keeps updating normally.  Active
        # whenever y is extend and x is NOT fixed (extend or periodic
        # both leave the constant null space unresolved).
        if aside == -2:
            if BCy == 'extend' and BCx != 'fixed':
                aside, aidx = _find_boundary_anchor_2d(
                    G, undef, BCx == 'extend')
            else:
                aside, aidx = -1, -1

        # process x/y boundaries independently after resolving the anchor
        _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx)

        for j in range(1, yc-1):
            # for the west boundary iteration (i==0)
            if BCx == 'periodic':
                cond = (G[j,0] != undef and
                        A[j,0] != undef and B[j,0] != undef and
                        C[j,0] != undef and D[j,0] != undef and
                        E[j,0] != undef and F[j,0] != undef)
                
                if cond:
                    temp = (
                        A[j,0] * (
                            (S[j+1,0] - S[j,0])-(S[j,0] - S[j-1,0])
                        ) * ratioSqr +
                        B[j,0] * (
                            (S[j+1,1] - S[j-1,1])-(S[j+1,-1] - S[j-1,-1])
                        ) * ratioQtr +
                        C[j,0] * (
                            (S[j,1] - S[j,0])-(S[j,0] - S[j,-1])
                        ) + (
                        D[j,0] * (
                            S[j+1,0] - S[j-1,0]
                        ) * ratio +
                        E[j,0] * (
                            S[j,1] - S[j,-1]
                        )) * delx / 2.0 + (
                        F[j,0] * S[j,0] - G[j,0]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,0]*ratioSqr + C[j,0]) * 2.0
                                      -F[j,0]*delxSqr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,0], maxUpdate, maxValue)
                    S[j,0] += temp
            
            # inner loop
            for i in range(1, xc-1):
                cond = (G[j,i] != undef and
                        A[j,i] != undef and B[j,i] != undef and
                        C[j,i] != undef and D[j,i] != undef and
                        E[j,i] != undef and F[j,i] != undef)

                if cond:
                    temp = (
                        A[j,i] * (
                            (S[j+1,i] - S[j,i])-(S[j,i] - S[j-1,i])
                        ) * ratioSqr +
                        B[j,i] * (
                            (S[j+1,i+1] - S[j-1,i+1])-(S[j+1,i-1] - S[j-1,i-1])
                        ) * ratioQtr +
                        C[j,i] * (
                            (S[j,i+1] - S[j,i])-(S[j,i] - S[j,i-1])
                        ) + (
                        D[j,i] * (
                            S[j+1,i] - S[j-1,i]
                        ) * ratio +
                        E[j,i] * (
                            S[j,i+1] - S[j,i-1]
                        )) * delx / 2.0 + (
                        F[j,i] * S[j,i] - G[j,i]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,i]*ratioSqr + C[j,i]) * 2.0
                                      -F[j,i]*delxSqr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,i], maxUpdate, maxValue)
                    S[j,i] += temp
            
            # for the east boundary iteration (i==-1)
            if BCx == 'periodic':
                cond = (G[j,-1] != undef and
                        A[j,-1] != undef and B[j,-1] != undef and
                        C[j,-1] != undef and D[j,-1] != undef and
                        E[j,-1] != undef and F[j,-1] != undef)
                
                if cond:
                    temp = (
                        A[j,-1] * (
                            (S[j+1,-1] - S[j,-1])-(S[j,-1] - S[j-1,-1])
                        ) *ratioSqr +
                        B[j,-1] * (
                            (S[j+1,0] - S[j-1,0])-(S[j+1,-2] - S[j-1,-2])
                        ) * ratioQtr +
                        C[j,-1] * (
                            (S[j,0] - S[j,-1])-(S[j,-1] - S[j,-2])
                        ) + (
                        D[j,-1] * (
                            S[j+1,-1] - S[j-1,-1]
                        ) * ratio +
                        E[j,-1] * (
                            S[j,0] - S[j,-2]
                        )) * delx / 2.0 + (
                        F[j,-1] * S[j,-1] - G[j,-1]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,-1]*ratioSqr + C[j,-1]) * 2.0
                                      -F[j,-1]*delxSqr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,-1], maxUpdate, maxValue)
                    S[j,-1] += temp

        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100
        
        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, G, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
        
    return S


@nb.njit(cache=False, nogil=True)
def invert_general_bih_2D(S, A, B, C, D, E, F, G, H, I, J, info,
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
    S: numpy.array (output)
        Results of the SOR inversion.
        
    A: numpy.array
        Coefficient for the first term.
    B: numpy.array
        Coefficient for the second term.
    C: numpy.array
        Coefficient for the third term.
    D: numpy.array
        Coefficient for the fourth term.
    E: numpy.array
        Coefficient for the fifth term.
    F: numpy.array
        Coefficient for the sixth term.
    G: numpy.array
        Coefficient for the seventh term.
    H: numpy.array
        Coefficient for the eighth term.
    I: numpy.array
        Coefficient for the ninth term.
    J: numpy.array
        Known forcing function.
    info: numpy.array
        Information array for logging purpose.
    yc: int
        Number of grid point in the y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in the x-dimension (e.g., X or lon).
    BCy: str
        Boundary condition for dimension y in ['fixed', 'extend', 'periodic'].
    BCx: str
        Boundary condition for dimension x in ['fixed', 'extend', 'periodic'].
    delxSSr: float
        Increment (interval) in dimension y (unit of m^2) to the power of 4.
    delxTr: float
        cubed increment (interval) in dimension y (unit of m^2).
    delxSqr: float
        Squared increment (interval) in dimension y (unit of m^2).
    ratio: float
        Ratio of delx to dely.
    ratioSSr: float
        Ratio of delx to dely to the power of 4.
    ratioQtr: float
        Ratio of delx to dely, divided by 4.
    ratioSqr: float
        Squared Ratio of delx to dely.
    optArg: float
        Optimal argument 'omega' (relaxation factor between 1 and 2) for SOR.
    undef: float
        Undefined value.
    flags: numpy.array
        Length of 3 array, [0] is flag for overflow, [1] for converge speed and
        [2] for how many loops used for iteration.
    mxLoop: int
        Maximum loop count, larger than this will break the iteration.
    tolerance: float
        Tolerance for iteraction, smaller than this will break the iteraction.

    Returns
    -------
    numpy.array
        Results of the SOR inversion.
    """
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    overflow = False
    error = 0.0
    interval = _check_interval(mxLoop, tolerance, check_interval)
    diagnostic_index = 0
    # gauge anchor for singular extend systems (resolved before the first
    # sweep; -2 = unresolved sentinel, -1 = not applicable, 0..3 = side)
    aside, aidx = -2, -2
    
    while(True):
        next_loop = loop + 1
        track_residual = ((next_loop % interval == 0) or
                          next_loop >= mxLoop)
        maxUpdate = 0.0
        maxValue = 0.0
        # process the two-cell x/y boundaries independently
        _apply_extend_boundary_bih_2d(S, BCy, BCx, undef)

        # NOTE: the biharmonic kernel deliberately does NOT use a gauge
        # anchor.  The Neumann null space of the 4th-order operator is
        # MULTI-dimensional (all biharmonic polynomials x, y, x^2, xy...
        # satisfy laplacian^2 psi = 0), so a single pinned point cannot
        # remove it -- pinning one value actually destabilises the
        # iteration along the remaining null space (verified
        # empirically: extend+periodic diverges with an anchor).  The
        # 5/9-point (2nd-order) forms have a 1-D constant null space,
        # which is exactly what the single anchor removes.
        if aside == -2:
            aside, aidx = -1, -1

        for j in range(2, yc-2):
            # for the west boundary iteration (i==0)
            if BCx == 'periodic':
                cond = (A[j,0] != undef and B[j,0] != undef and
                        C[j,0] != undef and D[j,0] != undef and
                        E[j,0] != undef and F[j,0] != undef and
                        G[j,0] != undef and H[j,0] != undef and
                        I[j,0] != undef and J[j,0] != undef)
                
                if cond:
                    temp = (
                        A[j,0] * (
                            S[j+2,0] - 4.0*S[j+1,0] + 6.0*S[j,0]- 4.0*S[j-1,0] + S[j-2,0]
                        ) * ratioSSr +
                        B[j,0] * (
                                S[j+2,2] - 2.0*S[j+2,0] +     S[j+2,-2] +
                            -2.0*S[j  ,2] + 4.0*S[j  ,0] - 2.0*S[j  ,-2] +
                                 S[j-2,2] - 2.0*S[j-2,0] +     S[j-2,-2]
                        ) * ratioSqr / 16.0 +
                        C[j,0] * (
                            S[j,2] - 4.0*S[j,1] + 6.0*S[j,0] - 4.0*S[j,-1] + S[j,-2]
                        ) +
                        D[j,0] * (
                            (S[j+1,0] - S[j,0])-(S[j,0] - S[j-1,0])
                        ) * ratioSqr * delxSqr +
                        E[j,0] * (
                            (S[j+1,1] - S[j-1,1])-(S[j+1,-1] - S[j-1,-1])
                        ) * ratioQtr * delxSqr +
                        F[j,0] * (
                            (S[j,1] - S[j,0])-(S[j,0] - S[j,-1])
                        ) * delxSqr +
                        G[j,0] * (
                            S[j+1,0] - S[j-1,0]
                        ) * delxTr / 2.0 * ratio +
                        H[j,0] * (
                            S[j,1] - S[j,-1]
                        ) * delxTr / 2.0 + (
                        I[j,0] * S[j,0] - J[j,0]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,0]*ratioSSr + C[j,0]) * 6.0 +
                                        B[j,0]*ratioSqr/4.0 +
                                      -(D[j,0]*ratioSqr + F[j,0]) * 2.0 * delxSqr +
                                        I[j,0]*delxSSr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,0], maxUpdate, maxValue)
                    S[j,0] += temp
            
            # for the west boundary iteration (i==1)
            if BCx == 'periodic':
                cond = (A[j,1] != undef and B[j,1] != undef and
                        C[j,1] != undef and D[j,1] != undef and
                        E[j,1] != undef and F[j,1] != undef and
                        G[j,1] != undef and H[j,1] != undef and
                        I[j,1] != undef and J[j,1] != undef)
                
                if cond:
                    temp = (
                        A[j,1] * (
                            S[j+2,1] - 4.0*S[j+1,1] + 6.0*S[j,1]- 4.0*S[j-1,1] + S[j-2,1]
                        ) * ratioSSr +
                        B[j,1] * (
                                S[j+2,3] - 2.0*S[j+2,1] +     S[j+2,-1] +
                            -2.0*S[j  ,3] + 4.0*S[j  ,1] - 2.0*S[j  ,-1] +
                                 S[j-2,3] - 2.0*S[j-2,1] +     S[j-2,-1]
                        ) * ratioSqr / 16.0 +
                        C[j,1] * (
                            S[j,3] - 4.0*S[j,2] + 6.0*S[j,1] - 4.0*S[j,0] + S[j,-1]
                        ) +
                        D[j,1] * (
                            (S[j+1,1] - S[j,1])-(S[j,1] - S[j-1,1])
                        ) * ratioSqr * delxSqr +
                        E[j,1] * (
                            (S[j+1,2] - S[j-1,2])-(S[j+1,0] - S[j-1,0])
                        ) * ratioQtr * delxSqr +
                        F[j,1] * (
                            (S[j,2] - S[j,1])-(S[j,1] - S[j,0])
                        ) * delxSqr +
                        G[j,1] * (
                            S[j+1,1] - S[j-1,1]
                        ) * delxTr / 2.0 * ratio +
                        H[j,1] * (
                            S[j,2] - S[j,0]
                        ) * delxTr / 2.0 + (
                        I[j,1] * S[j,1] - J[j,1]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,1]*ratioSSr + C[j,1]) * 6.0 +
                                        B[j,1]*ratioSqr/4.0 +
                                      -(D[j,1]*ratioSqr + F[j,1]) * 2.0 * delxSqr +
                                        I[j,1]*delxSSr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,1], maxUpdate, maxValue)
                    S[j,1] += temp
            
            # inner loop
            for i in range(2, xc-2):
                cond = (A[j,i] != undef and B[j,i] != undef and
                        C[j,i] != undef and D[j,i] != undef and
                        E[j,i] != undef and F[j,i] != undef and
                        G[j,i] != undef and H[j,i] != undef and
                        I[j,i] != undef and J[j,i] != undef)
                
                if cond:
                    temp = (
                        A[j,i] * (
                            S[j+2,i] - 4.0*S[j+1,i] + 6.0*S[j,i] - 4.0*S[j-1,i] + S[j-2,i]
                        ) * ratioSSr +
                        B[j,i] * (
                                S[j+2,i+2] - 2.0*S[j+2,i] +     S[j+2,i-2] +
                            -2.0*S[j  ,i+2] + 4.0*S[j  ,i] - 2.0*S[j  ,i-2] +
                                 S[j-2,i+2] - 2.0*S[j-2,i] +     S[j-2,i-2]
                        ) * ratioSqr / 16.0 +
                        C[j,i] * (
                            S[j,i+2] - 4.0*S[j,i+1] + 6.0*S[j,i] - 4.0*S[j,i-1] + S[j,i-2]
                        ) +
                        D[j,i] * (
                            (S[j+1,i] - S[j,i])-(S[j,i] - S[j-1,i])
                        ) * ratioSqr * delxSqr +
                        E[j,i] * (
                            (S[j+1,i+1] - S[j-1,i+1])-(S[j+1,i-1] - S[j-1,i-1])
                        ) * ratioQtr * delxSqr +
                        F[j,i] * (
                            (S[j,i+1] - S[j,i])-(S[j,i] - S[j,i-1])
                        ) * delxSqr +
                        G[j,i] * (
                            S[j+1,i] - S[j-1,i]
                        ) * delxTr * ratio / 2.0 +
                        H[j,i] * (
                            S[j,i+1] - S[j,i-1]
                        ) * delxTr / 2.0 + (
                        I[j,i] * S[j,i] - J[j,i]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,i]*ratioSSr + C[j,i]) * 6.0 +
                                        B[j,i]*ratioSqr / 4.0 +
                                      -(D[j,i]*ratioSqr + F[j,i]) * 2.0 * delxSqr +
                                        I[j,i]*delxSSr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,i], maxUpdate, maxValue)
                    S[j,i] += temp
            
            # for the east boundary iteration (i==-2)
            if BCx == 'periodic':
                cond = (A[j,-2] != undef and B[j,-2] != undef and
                        C[j,-2] != undef and D[j,-2] != undef and
                        E[j,-2] != undef and F[j,-2] != undef and
                        G[j,-2] != undef and H[j,-2] != undef and
                        I[j,-2] != undef and J[j,-2] != undef)
                
                if cond:
                    temp = (
                        A[j,-2] * (
                            S[j+2,-2] - 4.0*S[j+1,-2] + 6.0*S[j,-2]- 4.0*S[j-1,-2] + S[j-2,-2]
                        ) * ratioSSr +
                        B[j,-2] * (
                                S[j+2,0] - 2.0*S[j+2,-2] +     S[j+2,-4] +
                            -2.0*S[j  ,0] + 4.0*S[j  ,-2] - 2.0*S[j  ,-4] +
                                 S[j-2,0] - 2.0*S[j-2,-2] +     S[j-2,-4]
                        ) * ratioSqr / 16.0 +
                        C[j,-2] * (
                            S[j,0] - 4.0*S[j,-1] + 6.0*S[j,-2] - 4.0*S[j,-3] + S[j,-4]
                        ) +
                        D[j,-2] * (
                            (S[j+1,-2] - S[j,-2])-(S[j,-2] - S[j-1,-2])
                        ) * ratioSqr * delxSqr +
                        E[j,-2] * (
                            (S[j+1,-1] - S[j-1,-1])-(S[j+1,-3] - S[j-1,-3])
                        ) * ratioQtr * delxSqr +
                        F[j,-2] * (
                            (S[j,-1] - S[j,-2])-(S[j,-2] - S[j,-3])
                        ) * delxSqr +
                        G[j,-2] * (
                            S[j+1,-2] - S[j-1,-2]
                        ) * delxTr / 2.0 * ratio +
                        H[j,-2] * (
                            S[j,-1] - S[j,-3]
                        ) * delxTr / 2.0 + (
                        I[j,-2] * S[j,-2] - J[j,-2]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,-2]*ratioSSr + C[j,-2]) * 6.0 +
                                        B[j,-2]*ratioSqr/4.0
                                      -(D[j,-2]*ratioSqr + F[j,-2]) * 2.0 * delxSqr +
                                        I[j,-2]*delxSSr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,-2], maxUpdate, maxValue)
                    S[j,-2] += temp
            
            # for the east boundary iteration (i==-1)
            if BCx == 'periodic':
                cond = (A[j,-1] != undef and B[j,-1] != undef and
                        C[j,-1] != undef and D[j,-1] != undef and
                        E[j,-1] != undef and F[j,-1] != undef and
                        G[j,-1] != undef and H[j,-1] != undef and
                        I[j,-1] != undef and J[j,-1] != undef)
                
                if cond:
                    temp = (
                        A[j,-1] * (
                            S[j+2,-1] - 4.0*S[j+1,-1] + 6.0*S[j,-1]- 4.0*S[j-1,-1] + S[j-2,-1]
                        ) * ratioSSr +
                        B[j,-1] * (
                                S[j+2,1] - 2.0*S[j+2,-1] +     S[j+2,-3] +
                            -2.0*S[j  ,1] + 4.0*S[j  ,-1] - 2.0*S[j  ,-3] +
                                 S[j-2,1] - 2.0*S[j-2,-1] +     S[j-2,-3]
                        ) * ratioSqr / 16.0 +
                        C[j,-1] * (
                            S[j,1] - 4.0*S[j,0] + 6.0*S[j,-1] - 4.0*S[j,-2] + S[j,-3]
                        ) +
                        D[j,-1] * (
                            (S[j+1,-1] - S[j,-1])-(S[j,-1] - S[j-1,-1])
                        ) * ratioSqr * delxSqr +
                        E[j,-1] * (
                            (S[j+1,0] - S[j-1,0])-(S[j+1,-2] - S[j-1,-2])
                        ) * ratioQtr * delxSqr +
                        F[j,-1] * (
                            (S[j,0] - S[j,-1])-(S[j,-1] - S[j,-2])
                        ) * delxSqr +
                        G[j,-1] * (
                            S[j+1,-1] - S[j-1,-1]
                        ) * delxTr / 2.0 * ratio +
                        H[j,-1] * (
                            S[j,0] - S[j,-2]
                        ) * delxTr / 2.0 + (
                        I[j,-1] * S[j,-1] - J[j,-1]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,-1]*ratioSSr + C[j,-1]) * 6.0 +
                                        B[j,-1]*ratioSqr/4.0
                                      -(D[j,-1]*ratioSqr + F[j,-1]) * 2.0 * delxSqr +
                                        I[j,-1]*delxSSr)
                    if track_residual:
                        maxUpdate, maxValue = _accumulate_update_metric(
                            temp, S[j,-1], maxUpdate, maxValue)
                    S[j,-1] += temp

        loop += 1

        # Evaluate the residual only at the requested sparse checkpoints.
        if (loop % interval != 0) and (loop < mxLoop):
            continue

        error = _relative_update_metric(maxUpdate, maxValue)
        overflow = not np.isfinite(error) or error > 1e100
        
        if flags.size > 3:
            _record_solution_diagnostics(
                flags, diagnostic_index, error, S, J, undef, mxLoop, interval)
            diagnostic_index += 1

        if overflow or error < tolerance or loop >= mxLoop:
            break
        
    flags[0] = overflow
    flags[1] = error
    flags[2] = loop
        
    return S


@nb.njit(cache=False, nogil=True)
def trace(a, b, c, d):
    r"""
    Trace method for solving tri-diagonal equation set.
    
    Parameters
    ----------
    a: numpy.array
        Lower coefficients of the matrix (N-1).
    b: numpy.array
        Diagonal coefficients of the matrix (N).
    c: numpy.array
        Upper coefficients of the matrix (N-1).
    d: numpy.array
        Vector on the right-hand side of the equation (N).
    
    Returns
    -------
    numpy.array
        Results of the unknown (N).
    """
    N = len(b)
    
    if len(a) != N-1 or len(d) != N or len(c) != N-1:
        raise ValueError('lengths of given arrays are not satisfied')
        
    buf0 = np.zeros_like(b) # N
    buf1 = np.zeros_like(a) # N - 1
    res  = np.zeros_like(b) # N
    
    buf1[0] = c[0] / b[0]
    buf0[0] = b[0]
    
    for i in range(1, N-1):
        buf0[i] = b[i] - a[i-1] * buf1[i-1]
        buf1[i] = c[i] / buf0[i]
    
    buf0[N-1] = b[N-1] - a[N-2] * buf1[N-2]
    
    res[0] = d[0] / buf0[0]
    
    for i in range(1, N):
        res[i] = (d[i] - a[i-1] * res[i-1]) / buf0[i]
    
    for i in range(N-2, -1, -1):
        res[i] -= buf1[i] * res[i+1]
    
    return res


@nb.njit(cache=False, nogil=True)
def traceCyclic(a, b, c, d, a0, cn):
    r"""
    Trace method for solving tri-diagonal equation set with periodic BCs.
    
    Parameters
    ----------
    a: numpy.array
        Lower coefficients of the matrix (N-1).
    b: numpy.array
        Diagonal coefficients of the matrix (N).
    c: numpy.array
        Upper coefficients of the matrix (N-1).
    d: numpy.array
        Vector on the right-hand side of the equation (N).
    a0: float
        Cyclic coefficient for a.
    cn: float
        Cyclic coefficient for c.
    
    Returns
    -------
    numpy.array
        Results of the unknown (N).
    """
    N = len(b)
    
    buf4 = np.zeros_like(b) # N
    res  = np.zeros_like(b) # N
    
    buf4[N-1], buf4[0] = cn, 0
    buf1 = trace(a, b, c, buf4)
    
    buf4[N-1], buf4[0] = 0, a0
    buf2 = trace(a, b, c, buf4)
    
    buf4[N-1], buf4[0] = 0, a0
    buf3 = trace(a, b, c, d)
    
    res[N-1] = ((1.0 + buf1[0]) / buf1[N-1] * buf3[N-1] - buf3[0]) / \
               ((1.0 + buf1[0]) * (1.0 + buf2[N-1]) / buf1[N-1] - buf2[0]);
    res[ 0 ] = (buf3[0] - buf2[0] * res[N-1]) / (1 + buf1[0])
    
    for i in range(1, N-1):
        res[i] = buf3[i] - buf1[i] * res[0]-buf2[i] * res[N-1];
    
    return res



"""
Below are the helper methods
"""
@nb.njit(inline='always')
def _apply_extend_boundary_2d(S, BCy, BCx, undef, aside, aidx):
    """Apply independent x/y extend boundaries to a 2-D solution in place.

    Parameters
    ----------
    S : numpy.ndarray
        Two-dimensional solution array, modified in place.
    BCy, BCx : str
        Boundary names for the y and x directions.
    undef : float
        Internal sentinel marking invalid cells.
    aside : int
        Gauge-anchor side: 0 north, 1 south, 2 west, 3 east, or -1.
    aidx : int
        Coordinate along the anchored side, or -1 when no anchor is used.

    Returns
    -------
    None
    """
    yc, xc = S.shape

    if BCy == 'extend':
        istart = 0 if BCx == 'periodic' else 1
        iend = xc if BCx == 'periodic' else xc - 1
        for i in range(istart, iend):
            if S[1, i] != undef and not (aside == 0 and i == aidx):
                S[0, i] = S[1, i]
            if S[yc - 2, i] != undef and not (aside == 1 and i == aidx):
                S[yc - 1, i] = S[yc - 2, i]

    if BCx == 'extend':
        for j in range(1, yc - 1):
            if S[j, 1] != undef and not (aside == 2 and j == aidx):
                S[j, 0] = S[j, 1]
            if S[j, xc - 2] != undef and not (aside == 3 and j == aidx):
                S[j, xc - 1] = S[j, xc - 2]

    # A corner belongs to both dimensions: extend it only when both
    # dimensions are extend.  If either dimension is fixed, its prescribed
    # corner value must remain untouched.
    if BCy == 'extend' and BCx == 'extend':
        if S[1, 1] != undef:
            S[0, 0] = S[1, 1]
        if S[1, xc - 2] != undef:
            S[0, xc - 1] = S[1, xc - 2]
        if S[yc - 2, 1] != undef:
            S[yc - 1, 0] = S[yc - 2, 1]
        if S[yc - 2, xc - 2] != undef:
            S[yc - 1, xc - 1] = S[yc - 2, xc - 2]


@nb.njit(inline='always')
def _apply_extend_boundary_3d(S, BCy, BCx, undef):
    """Apply x/y extend boundaries to a 3-D solution in place.

    Parameters
    ----------
    S : numpy.ndarray
        Solution array in ``(z, y, x)`` order, modified in place.  The z
        boundary remains fixed.
    BCy, BCx : str
        Boundary names for the y and x directions.
    undef : float
        Internal sentinel marking invalid cells.

    Returns
    -------
    None
    """
    zc, yc, xc = S.shape
    for k in range(1, zc - 1):
        if BCy == 'extend':
            istart = 0 if BCx == 'periodic' else 1
            iend = xc if BCx == 'periodic' else xc - 1
            for i in range(istart, iend):
                if S[k, 1, i] != undef:
                    S[k, 0, i] = S[k, 1, i]
                if S[k, yc - 2, i] != undef:
                    S[k, yc - 1, i] = S[k, yc - 2, i]

        if BCx == 'extend':
            for j in range(1, yc - 1):
                if S[k, j, 1] != undef:
                    S[k, j, 0] = S[k, j, 1]
                if S[k, j, xc - 2] != undef:
                    S[k, j, xc - 1] = S[k, j, xc - 2]

        if BCy == 'extend' and BCx == 'extend':
            if S[k, 1, 1] != undef:
                S[k, 0, 0] = S[k, 1, 1]
            if S[k, 1, xc - 2] != undef:
                S[k, 0, xc - 1] = S[k, 1, xc - 2]
            if S[k, yc - 2, 1] != undef:
                S[k, yc - 1, 0] = S[k, yc - 2, 1]
            if S[k, yc - 2, xc - 2] != undef:
                S[k, yc - 1, xc - 1] = S[k, yc - 2, xc - 2]


@nb.njit(inline='always')
def _apply_extend_boundary_bih_2d(S, BCy, BCx, undef):
    """Apply the two-cell biharmonic extend boundary in place.

    Parameters
    ----------
    S : numpy.ndarray
        Two-dimensional solution array with a two-cell boundary halo.
    BCy, BCx : str
        Boundary names for the y and x directions.
    undef : float
        Internal sentinel marking invalid cells.

    Returns
    -------
    None
    """
    yc, xc = S.shape
    if BCy == 'extend':
        istart = 0 if BCx == 'periodic' else 2
        iend = xc if BCx == 'periodic' else xc - 2
        for i in range(istart, iend):
            if S[2, i] != undef:
                S[0, i] = S[1, i] if BCx == 'periodic' else S[2, i]
                S[1, i] = S[2, i]
            if S[yc - 3, i] != undef:
                S[yc - 1, i] = S[yc - 3, i]
                S[yc - 2, i] = S[yc - 3, i]

    if BCx == 'extend':
        for j in range(2, yc - 2):
            if S[j, 2] != undef:
                S[j, 0] = S[j, 2]
                S[j, 1] = S[j, 2]
            if S[j, xc - 3] != undef:
                S[j, xc - 1] = S[j, xc - 3]
                S[j, xc - 2] = S[j, xc - 3]

    if BCy == 'extend' and BCx == 'extend':
        for jj in range(2):
            for ii in range(2):
                if S[2, 2] != undef:
                    S[jj, ii] = S[2, 2]
                if S[2, xc - 3] != undef:
                    S[jj, xc - 1 - ii] = S[2, xc - 3]
                if S[yc - 3, 2] != undef:
                    S[yc - 1 - jj, ii] = S[yc - 3, 2]
                if S[yc - 3, xc - 3] != undef:
                    S[yc - 1 - jj, xc - 1 - ii] = S[yc - 3, xc - 3]

@nb.njit(cache=False, nogil=True)
def _check_interval(mxLoop, tolerance, requested=0):
    r"""Resolve an explicit or adaptive convergence-check interval.

    A positive ``requested`` value is used directly, capped at ``mxLoop`` so
    the reported effective interval matches the mandatory final check.
    ``requested == 0`` preserves the historical adaptive calculation and
    mirrors :func:`gpus._compute_check_interval`.

    Residual checks are performed every *interval* iterations:
    tolerance > 0  ->  clamp(mxLoop/50, 10, 100);
    tolerance <= 0 (fixed-iteration runs)  ->  clamp(mxLoop/20, 50, 500).

    Parameters
    ----------
    mxLoop : int
        Maximum number of SOR sweeps.
    tolerance : float
        Positive values enable early stopping; non-positive values request a
        fixed-iteration run.
    requested : int, optional
        Explicit interval.  Zero selects the adaptive schedule.

    Returns
    -------
    int
        Effective interval, capped at ``mxLoop``.
    """
    if requested > 0:
        return min(requested, mxLoop)
    if tolerance > 0.0:
        return max(10, min(100, mxLoop // 50))
    else:
        return max(50, min(500, mxLoop // 20))


# NOTE: no _find_anchor_3d is needed -- in the 3D kernels the z-boundary
# rows are never updated (values come in fixed via icbc), which acts as a
# Dirichlet condition that already removes the constant null space.


@nb.njit(cache=False, nogil=True)
def _find_boundary_anchor_2d(F, undef, allow_we):
    """Locate a BOUNDARY grid point to pin as the gauge anchor for
    singular extend systems, scanning the boundary ring for the first
    point whose adjacent interior equation is valid.

    For 'extend' (Neumann-type) boundaries the discrete system has a
    constant null space and the iteration drifts.  Pinning ONE boundary
    grid point (its extend-copy is skipped, so it keeps its initial
    value) is equivalent to a Dirichlet gauge there: the system becomes
    non-singular and converges to a unique solution, while EVERY
    interior point keeps updating normally.

    Search order: north -> south, then (only when the x boundary is
    also 'extend', i.e. a real copyable boundary exists) west -> east.

    Parameters
    ----------
    F: numpy.array
        Forcing array (undef marks points whose equation is invalid);
        the anchor must be adjacent to a point where the equation holds.
    undef: float
        Undefined value.
    allow_we: bool
        Whether the west/east boundaries may host the anchor (True only
        when BCx == 'extend'; under BCx == 'periodic' columns 0/xc-1 are
        ordinary interior-periodic points, not boundaries).

    Returns
    -------
    (side, idx) : tuple of int
        side 0: anchor is S[0, idx]      (north row)
        side 1: anchor is S[yc-1, idx]   (south row)
        side 2: anchor is S[idx, 0]      (west column)
        side 3: anchor is S[idx, xc-1]   (east column)
        (-1, -1) if no valid boundary point exists.
    """
    yc, xc = F.shape
    for i in range(1, xc - 1):
        if F[1, i] != undef:
            return 0, i
    for i in range(1, xc - 1):
        if F[yc - 2, i] != undef:
            return 1, i
    if allow_we:
        for j in range(1, yc - 1):
            if F[j, 1] != undef:
                return 2, j
        for j in range(1, yc - 1):
            if F[j, xc - 2] != undef:
                return 3, j
    return -1, -1


@nb.njit(cache=False, nogil=True)
def _find_boundary_anchor_1d(F, undef):
    """1D counterpart of :func:`_find_boundary_anchor_2d`.

    Parameters
    ----------
    F : numpy.ndarray
        One-dimensional forcing array.
    undef : float
        Internal sentinel marking invalid equations.

    Returns
    -------
    int
        0 to pin the west endpoint, 1 to pin the east endpoint, or -1 if
        neither endpoint is adjacent to a valid equation.
    """
    if F[1] != undef:
        return 0
    if F[F.shape[0] - 2] != undef:
        return 1
    return -1


@nb.njit(inline='always')
def _accumulate_update_metric(delta, value, max_update, max_value):
    """Accumulate a max-norm diagonally preconditioned residual metric.

    For SOR, ``delta = omega * D**-1 * residual``.  Tracking the largest
    point correction therefore measures the equation residual after diagonal
    preconditioning.  Both the old and new point magnitudes contribute to the
    normalization so the first non-zero sweep remains well defined.

    Parameters
    ----------
    delta : float
        SOR correction for the current grid point.
    value : float
        Solution value before applying ``delta``.
    max_update : float
        Largest absolute correction accumulated so far.
    max_value : float
        Largest solution magnitude accumulated so far.

    Returns
    -------
    max_update, max_value : tuple of float
        Updated accumulators.  A non-finite correction makes ``max_update``
        infinite so the outer loop reports overflow.
    """
    new_point = value + delta
    if not np.isfinite(delta) or not np.isfinite(new_point):
        return np.inf, max(max_value, 1.0)

    update = abs(delta)
    max_update = max(max_update, update)

    old_value = abs(value)
    new_value = abs(new_point)
    value_scale = max(new_value, old_value)
    max_value = max(max_value, value_scale)

    return max_update, max_value


@nb.njit(inline='always')
def _relative_update_metric(max_update, max_value):
    """Return the normalized maximum SOR correction.

    Parameters
    ----------
    max_update : float
        Maximum absolute point correction in the checked sweep.
    max_value : float
        Maximum old/new solution magnitude in the checked sweep.

    Returns
    -------
    float
        Relative residual when ``max_value`` is non-zero, otherwise the
        absolute residual.
    """
    if max_value > np.finfo(np.float64).tiny:
        return max_update / max_value
    return max_update



@nb.njit
def _record_solution_diagnostics(flags, index, error, solution, forcing,
                                 undef, mxLoop, interval):
    """Store a sampled error and signed extrema of the valid solution domain.

    Parameters
    ----------
    flags : numpy.ndarray
        Status followed by error, minimum and maximum history blocks. Legacy
        error-only buffers remain supported.
    index : int
        Zero-based diagnostic checkpoint.
    error : float
        Existing convergence metric; its definition is unchanged.
    solution, forcing : numpy.ndarray
        Current complete slice and forcing used to identify valid cells.
        Fixed boundary cells are included; undefined and non-finite cells
        are excluded from the extrema.
    undef : float
        Internal missing-data sentinel.
    mxLoop, interval : int
        Iteration limit and effective diagnostic sampling interval.

    Returns
    -------
    None
        Updates flags in place. An empty valid domain yields NaN extrema.
    """
    count = (mxLoop + interval - 1) // interval
    flags[3 + index] = error
    if flags.size < 3 + 3*count:
        return
    low, high = np.inf, -np.inf
    values, rhs = solution.ravel(), forcing.ravel()
    for i in range(values.size):
        value = values[i]
        if (rhs[i] != undef and np.isfinite(rhs[i]) and
                value != undef and np.isfinite(value)):
            low = min(low, value)
            high = max(high, value)
    if low == np.inf:
        low, high = np.nan, np.nan
    flags[3 + count + index] = low
    flags[3 + 2*count + index] = high
