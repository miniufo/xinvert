# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import numba as nb


"""
Below are the numba functions
"""
@nb.jit(nopython=True, cache=False)
def invert_standard_3D(S, A, B, C, F,
                       zc, yc, xc, delz, dely, delx, BCz, BCy, BCx, delxSqr,
                       ratio2Sqr, ratio1Sqr, optArg, undef, flags,
                       mxLoop, tolerance):
    r"""Inverting a 3D volume of elliptic equation in standard form.

    .. math::

        \frac{1}{\partial z}\left(A\frac{\partial \omega}{\partial z}\right)+
        \frac{1}{\partial y}\left(B\frac{\partial \omega}{\partial y}\right)+
        \frac{1}{\partial x}\left(C\frac{\partial \omega}{\partial x}\right)=F
    
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
    zc: int
        Number of grid point in z-dimension (e.g., Z or lev).
    yc: int
        Number of grid point in y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    delz: float
        Increment (interval) in dimension z (unit of m or Pa).
    dely: float
        Increment (interval) in dimension y (unit of m, not degree).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCy == 'extend':
            if BCx == 'periodic':
                for k in range(1, zc-1):
                    for i in range(xc):
                        if  S[k, 1,i] != undef:
                            S[k, 0,i]  = S[k, 1,i]
                        if  S[k,-2,i] != undef:
                            S[k,-1,i]  = S[k,-2,i]
            else:
                for k in range(1, zc-1):
                    for i in range(1, xc-1):
                        if  S[k, 1,i] != undef:
                            S[k, 0,i]  = S[k, 1,i]
                        if  S[k,-2,i] != undef:
                            S[k,-1,i]  = S[k,-2,i]
                    for i in range(1, xc-1):
                        if  S[k, 1,i] != undef:
                            S[k, 0,i]  = S[k, 1,i]
                        if  S[k,-2,i] != undef:
                            S[k,-1,i]  = S[k,-2,i]
                    
                    if  S[k, 1, 1] != undef:
                        S[k, 0, 0] = S[k, 1, 1]
                    if  S[k, 1,-2] != undef:
                        S[k, 0,-1] = S[k, 1,-2]
                    if  S[k,-2, 1] != undef:
                        S[k,-1, 0] = S[k,-2, 1]
                    if  S[k,-2,-2] != undef:
                        S[k,-1,-1] = S[k,-2,-2]
        
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
                        S[k,j,-1] += temp
        
        norm = absNorm3D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True, cache=False)
def invert_standard_2D(S, A, B, C, F,
                       yc, xc, dely, delx, BCy, BCx, delxSqr,
                       ratioQtr, ratioSqr, optArg, undef, flags,
                       mxLoop, tolerance):
    r"""Inverting a 2D slice of elliptic equation in standard form.

    .. math::

        \frac{1}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{1}{\partial x}\left(
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
    yc: int
        Number of grid point in y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    dely: float
        Increment (interval) in dimension y (unit of m, not degree).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCy == 'extend':
            if BCx == 'periodic':
                for i in range(xc):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
            else:
                for i in range(1, xc-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                for i in range(1, yc-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                
                if  S[ 1, 1] != undef:
                    S[ 0, 0] = S[ 1, 1]
                if  S[ 1,-2] != undef:
                    S[ 0,-1] = S[ 1,-2]
                if  S[-2, 1] != undef:
                    S[-1, 0] = S[-2, 1]
                if  S[-2,-2] != undef:
                    S[-1,-1] = S[-2,-2]
        
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
                    S[j,-1] += temp
        
        norm = absNorm2D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop or norm == 0:
            break
        
        normPrev = norm
        loop += 1
        
    return S



@nb.jit(nopython=True, cache=False)
def invert_standard_2D_test(S, A, B, C, D, E, F,
                       yc, xc, dely, delx, BCy, BCx, delxSqr,
                       ratioQtr, ratioSqr, optArg, undef, flags,
                       mxLoop, tolerance):
    r"""Inverting a 2D slice of elliptic equation in standard form.

    .. math::

        \frac{1}{\partial y}\left(
        A\frac{\partial \psi}{\partial y} + 
        B\frac{\partial \psi}{\partial x} \right) +
        \frac{1}{\partial x}\left(
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
    yc: int
        Number of grid point in y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    dely: float
        Increment (interval) in dimension y (unit of m, not degree).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCy == 'extend':
            if BCx == 'periodic':
                for i in range(xc):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
            else:
                for i in range(1, xc-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                for i in range(1, yc-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                
                if  S[ 1, 1] != undef:
                    S[ 0, 0] = S[ 1, 1]
                if  S[ 1,-2] != undef:
                    S[ 0,-1] = S[ 1,-2]
                if  S[-2, 1] != undef:
                    S[-1, 0] = S[-2, 1]
                if  S[-2,-2] != undef:
                    S[-1,-1] = S[-2,-2]
        
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
                    S[j,-1] += temp
        
        norm = absNorm2D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop or norm == 0:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True, cache=False)
def invert_standard_1D(S, A, B, F,
                       xc, delx, BCx, delxSqr, optArg, undef, flags,
                       mxLoop, tolerance):
    r"""Inverting a 1D series of elliptic equation in standard form.

    .. math::

        \frac{1}{\partial x}\left(
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
    xc: int
        Number of grid point in x-dimension (e.g., X or lon).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCx == 'extend':
            if  S[ 1] != undef:
                S[ 0] = S[ 1]
            if  S[-2] != undef:
                S[-1] = S[-2]
        
        # for the west boundary iteration (i==0)
        if BCx == 'periodic':
            cond = (F[0]!=undef and A[0]!=undef and A[1]!=undef and B[0]!=undef)
            
            if cond:
                temp = (
                    A[1] * (S[1] - S[0]) - A[0] * (S[0] - S[-1])
                ) / delxSqr + (B[0] * S[0] - F[0])
                
                temp *= optArg / ((A[1] + A[0]) / delxSqr - B[0])
                S[0] += temp
        
        # inner loop
        for i in range(1, xc-1):
            cond = (F[i]!=undef and A[i]!=undef and A[i+1]!=undef and B[i]!=undef)
            
            if cond:
                temp = (
                    A[i+1] * (S[i+1] - S[i]) - A[i] * (S[i] - S[i-1])
                ) / delxSqr + (B[i] * S[i] - F[i])
                
                temp *= optArg / ((A[i+1] + A[i]) / delxSqr - B[i])
                S[i] += temp
        
        # for the west boundary iteration (i==-1)
        if BCx == 'periodic':
            cond = (F[-1]!=undef and A[-1]!=undef and A[0]!=undef and B[-1]!=undef)
            
            if cond:
                temp = (
                    A[0] * (S[0] - S[-1]) - A[-1] * (S[-1] - S[-2])
                ) / delxSqr + (B[-1] * S[-1] - F[-1])
                
                temp *= optArg / ((A[0] + A[-1]) / delxSqr - B[-1])
                S[-1] += temp
        
        norm = absNorm1D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop or norm == 0:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True, cache=False)
def invert_general_3D(S, A, B, C, D, E, F, G, H,
                      zc, yc, xc, delz, dely, delx, BCz, BCy, BCx, delxSqr,
                      ratio2, ratio1, ratio2Sqr, ratio1Sqr, optArg, undef, flags,
                      mxLoop, tolerance):
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
    zc: int
        Number of grid point in the z-dimension (e.g., Z or lev).
    yc: int
        Number of grid point in the y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in the x-dimension (e.g., X or lon).
    delz: float
        Increment (interval) in dimension z (unit of m, not degree).
    dely: float
        Increment (interval) in dimension y (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCy == 'extend':
            if BCx == 'periodic':
                for k in range(1, zc-1):
                    for i in range(xc):
                        if  S[k, 1,i] != undef:
                            S[k, 0,i]  = S[k, 1,i]
                        if  S[k,-2,i] != undef:
                            S[k,-1,i]  = S[k,-2,i]
            else:
                for k in range(1, zc-1):
                    for i in range(1, xc-1):
                        if  S[k, 1,i] != undef:
                            S[k, 0,i]  = S[k, 1,i]
                        if  S[k,-2,i] != undef:
                            S[k,-1,i]  = S[k,-2,i]
                    for i in range(1, yc-1):
                        if  S[k, 1,i] != undef:
                            S[k, 0,i]  = S[k, 1,i]
                        if  S[k,-2,i] != undef:
                            S[k,-1,i]  = S[k,-2,i]
                    
                    if  S[k, 1, 1] != undef:
                        S[k, 0, 0] = S[k, 1, 1]
                    if  S[k, 1,-2] != undef:
                        S[k, 0,-1] = S[k, 1,-2]
                    if  S[k,-2, 1] != undef:
                        S[k,-1, 0] = S[k,-2, 1]
                    if  S[k,-2,-2] != undef:
                        S[k,-1,-1] = S[k,-2,-2]
        
        for k in range(1, zc-1):
            for j in range(1, yc-1):
                # for the west boundary iteration (i==0)
                if BCx == 'periodic':
                    cond = (G[k,j,0] != undef and G[k,j,0] != undef and
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
                                (S[k+1,j,0] - S[k-1,j,0])
                            ) * ratio2 +
                            E[k,j,0] * (
                                (S[k,j+1,0] - S[k,j-1,0])
                            ) * ratio1 +
                            F[k,j,0] * (
                                (S[k,j,1] - S[k,j,-1])
                            )) * delx / 2.0 + (
                            G[k,j,0] * S[k,j,0] - H[k,j,0]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k,j,0]*ratio2Sqr + B[k,j,0]*ratio1Sqr + C[k,j,0]
                        ) * 2.0 - G[k,j,0]*delxSqr)
                        
                        S[k,j,0] += temp
                
                # inner loop
                for i in range(1, xc-1):
                    cond = (H[k,j,i] != undef and G[k,j,i] != undef and
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
                                (S[k+1,j,i] - S[k-1,j,i])
                            ) * ratio2 +
                            E[k,j,i] * (
                                (S[k,j+1,i] - S[k,j-1,i])
                            ) * ratio1 +
                            F[k,j,i] * (
                                (S[k,j,i+1] - S[k,j,i-1])
                            )) * delx / 2.0 + (
                            G[k,j,i] * S[k,j,i] - H[k,j,i]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k,j,i]*ratio2Sqr + B[k,j,i]*ratio1Sqr + C[k,j,i]
                        ) * 2.0 - G[k,j,i]*delxSqr)
                        
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
                                (S[k+1,j,-1] - S[k-1,j,-1])
                            ) * ratio2 +
                            E[k,j,-1] * (
                                (S[k,j+1,-1] - S[k,j-1,-1])
                            ) * ratio1 +
                            F[k,j,-1] * (
                                (S[k,j,0] - S[k,j,-2])
                            )) * delx / 2.0 + (
                            G[k,j,-1] * S[k,j,-1] - H[k,j,-1]) * delxSqr
                        )
                        
                        temp *= optArg / ((
                            A[k,j,-1]*ratio2Sqr + B[k,j,-1]*ratio1Sqr + C[k,j,-1]
                        ) * 2.0 - G[k,j,-1]*delxSqr)
                        
                        S[k,j,-1] += temp
        
        norm = absNorm3D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True, cache=False)
def invert_general_2D(S, A, B, C, D, E, F, G,
                      yc, xc, dely, delx, BCy, BCx,
                      delxSqr, ratio, ratioQtr, ratioSqr, optArg, undef, flags,
                      mxLoop, tolerance):
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
    yc: int
        Number of grid point in the y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in the x-dimension (e.g., X or lon).
    dely: float
        Increment (interval) in dimension y (unit of m, not degree).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCy == 'extend':
            if BCx == 'periodic':
                for i in range(xc):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
            else:
                for i in range(1, xc-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                for i in range(1, yc-1):
                    if  S[ 1,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                    if  S[-2,i] != undef:
                        S[-1,i]  = S[-2,i]
                
                if  S[ 1, 1] != undef:
                    S[ 0, 0] = S[ 1, 1]
                if  S[ 1,-2] != undef:
                    S[ 0,-1] = S[ 1,-2]
                if  S[-2, 1] != undef:
                    S[-1, 0] = S[-2, 1]
                if  S[-2,-2] != undef:
                    S[-1,-1] = S[-2,-2]
        
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
                            (S[j+1,0] - S[j-1,0])
                        ) * ratio +
                        E[j,0] * (
                            (S[j,1] - S[j,-1])
                        )) * delx / 2.0 + (
                        F[j,0] * S[j,0] - G[j,0]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,0]*ratioSqr + C[j,0]) * 2.0
                                      -F[j,0]*delxSqr)
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
                            (S[j+1,i] - S[j-1,i])
                        ) * ratio +
                        E[j,i] * (
                            (S[j,i+1] - S[j,i-1])
                        )) * delx / 2.0 + (
                        F[j,i] * S[j,i] - G[j,i]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,i]*ratioSqr + C[j,i]) * 2.0
                                      -F[j,i]*delxSqr)
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
                            (S[j+1,-1] - S[j-1,-1])
                        ) * ratio +
                        E[j,-1] * (
                            (S[j,0] - S[j,-2])
                        )) * delx / 2.0 + (
                        F[j,-1] * S[j,-1] - G[j,-1]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,-1]*ratioSqr + C[j,-1]) * 2.0
                                      -F[j,-1]*delxSqr)
                    S[j,-1] += temp
        
        norm = absNorm2D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True, cache=False)
def invert_general_bih_2D(S, A, B, C, D, E, F, G, H, I, J,
                          yc, xc, dely, delx, BCy, BCx,
                          delxSSr, delxTr, delxSqr,
                          ratio, ratioSSr, ratioQtr, ratioSqr,
                          optArg, undef, flags,
                          mxLoop, tolerance):
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
    yc: int
        Number of grid point in the y-dimension (e.g., Y or lat).
    xc: int
        Number of grid point in the x-dimension (e.g., X or lon).
    dely: float
        Increment (interval) in dimension y (unit of m, not degree).
    delx: float
        Increment (interval) in dimension x (unit of m, not degree).
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
    
    while(True):
        # process boundaries
        if BCy == 'extend':
            if BCx == 'periodic':
                for i in range(xc):
                    if  S[ 2,i] != undef:
                        S[ 0,i]  = S[ 1,i]
                        S[ 1,i]  = S[ 2,i]
                    if  S[-3,i] != undef:
                        S[-1,i]  = S[-3,i]
                        S[-2,i]  = S[-3,i]
            else:
                for i in range(1, xc-1):
                    if  S[ 2,i] != undef:
                        S[ 0,i]  = S[ 2,i]
                        S[ 1,i]  = S[ 2,i]
                    if  S[-3,i] != undef:
                        S[-1,i]  = S[-3,i]
                        S[-2,i]  = S[-3,i]
                for i in range(1, yc-1):
                    if  S[ 2,i] != undef:
                        S[ 0,i]  = S[ 2,i]
                        S[ 1,i]  = S[ 2,i]
                    if  S[-3,i] != undef:
                        S[-1,i]  = S[-3,i]
                        S[-2,i]  = S[-3,i]
                
                if  S[ 2, 2] != undef:
                    S[ 0, 0] = S[ 2, 2]
                    S[ 0, 1] = S[ 2, 2]
                    S[ 1, 0] = S[ 2, 2]
                    S[ 1, 1] = S[ 2, 2]
                if  S[ 2,-3] != undef:
                    S[ 0,-1] = S[ 2,-3]
                    S[ 0,-2] = S[ 2,-3]
                    S[ 1,-1] = S[ 2,-3]
                    S[ 1,-2] = S[ 2,-3]
                if  S[-3, 2] != undef:
                    S[-1, 0] = S[-3, 2]
                    S[-2, 0] = S[-3, 2]
                    S[-1, 1] = S[-3, 2]
                    S[-2, 1] = S[-3, 2]
                if  S[-3,-3] != undef:
                    S[-1,-1] = S[-3,-3]
                    S[-1,-2] = S[-3,-3]
                    S[-2,-1] = S[-3,-3]
                    S[-2,-2] = S[-3,-3]
        
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
                            (S[j+2,0] - 4.0*S[j+1,0] + 6.0*S[j,0]- 4.0*S[j-1,0] + S[j-2,0])
                        ) * ratioSSr +
                        B[j,0] * (
                            (    S[j+2,2] - 2.0*S[j+2,0] +     S[j+2,-2] +
                            -2.0*S[j  ,2] + 4.0*S[j  ,0] - 2.0*S[j  ,-2] +
                                 S[j-2,2] - 2.0*S[j-2,0] +     S[j-2,-2])
                        ) * ratioSqr / 16.0 +
                        C[j,0] * (
                            (S[j,2] - 4.0*S[j,1] + 6.0*S[j,0] - 4.0*S[j,-1] + S[j,-2])
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
                            (S[j+1,0] - S[j-1,0])
                        ) * delxTr / 2.0 * ratio +
                        H[j,0] * (
                            (S[j,1] - S[j,-1])
                        ) * delxTr / 2.0 + (
                        I[j,0] * S[j,0] - J[j,0]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,0]*ratioSSr + C[j,0]) * 6.0 +
                                        B[j,0]*ratioSqr/4.0 +
                                      -(D[j,0]*ratioSqr + F[j,0]) * 2.0 * delxSqr +
                                        I[j,0]*delxSSr)
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
                            (S[j+2,1] - 4.0*S[j+1,1] + 6.0*S[j,1]- 4.0*S[j-1,1] + S[j-2,1])
                        ) * ratioSSr +
                        B[j,1] * (
                            (    S[j+2,3] - 2.0*S[j+2,1] +     S[j+2,-1] +
                            -2.0*S[j  ,3] + 4.0*S[j  ,1] - 2.0*S[j  ,-1] +
                                 S[j-2,3] - 2.0*S[j-2,1] +     S[j-2,-1])
                        ) * ratioSqr / 16.0 +
                        C[j,1] * (
                            (S[j,3] - 4.0*S[j,2] + 6.0*S[j,1] - 4.0*S[j,0] + S[j,-1])
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
                            (S[j+1,1] - S[j-1,1])
                        ) * delxTr / 2.0 * ratio +
                        H[j,1] * (
                            (S[j,2] - S[j,0])
                        ) * delxTr / 2.0 + (
                        I[j,1] * S[j,1] - J[j,1]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,1]*ratioSSr + C[j,1]) * 6.0 +
                                        B[j,1]*ratioSqr/4.0 +
                                      -(D[j,1]*ratioSqr + F[j,1]) * 2.0 * delxSqr +
                                        I[j,1]*delxSSr)
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
                            (S[j+2,i] - 4.0*S[j+1,i] + 6.0*S[j,i] - 4.0*S[j-1,i] + S[j-2,i])
                        ) * ratioSSr +
                        B[j,i] * (
                            (    S[j+2,i+2] - 2.0*S[j+2,i] +     S[j+2,i-2] +
                            -2.0*S[j  ,i+2] + 4.0*S[j  ,i] - 2.0*S[j  ,i-2] +
                                 S[j-2,i+2] - 2.0*S[j-2,i] +     S[j-2,i-2])
                        ) * ratioSqr / 16.0 +
                        C[j,i] * (
                            (S[j,i+2] - 4.0*S[j,i+1] + 6.0*S[j,i] - 4.0*S[j,i-1] + S[j,i-2])
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
                            (S[j+1,i] - S[j-1,i])
                        ) * delxTr * ratio / 2.0 +
                        H[j,i] * (
                            (S[j,i+1] - S[j,i-1])
                        ) * delxTr / 2.0 + (
                        I[j,i] * S[j,i] - J[j,i]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,i]*ratioSSr + C[j,i]) * 6.0 +
                                        B[j,i]*ratioSqr / 4.0 +
                                      -(D[j,i]*ratioSqr + F[j,i]) * 2.0 * delxSqr +
                                        I[j,i]*delxSSr)
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
                            (S[j+2,-2] - 4.0*S[j+1,-2] + 6.0*S[j,-2]- 4.0*S[j-1,-2] + S[j-2,-2])
                        ) * ratioSSr +
                        B[j,-2] * (
                            (    S[j+2,0] - 2.0*S[j+2,-2] +     S[j+2,i-4] +
                            -2.0*S[j  ,0] + 4.0*S[j  ,-2] - 2.0*S[j  ,i-4] +
                                 S[j-2,0] - 2.0*S[j-2,-2] +     S[j-2,i-4])
                        ) * ratioSqr / 16.0 +
                        C[j,-2] * (
                            (S[j,0] - 4.0*S[j,-1] + 6.0*S[j,-2] - 4.0*S[j,-3] + S[j,-4])
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
                            (S[j+1,-2] - S[j-1,-2])
                        ) * delxTr / 2.0 * ratio +
                        H[j,-2] * (
                            (S[j,-1] - S[j,-3])
                        ) * delxTr / 2.0 + (
                        I[j,-2] * S[j,-2] - J[j,-2]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,-2]*ratioSSr + C[j,-2]) * 6.0 +
                                        B[j,-2]*ratioSqr/4.0
                                      -(D[j,-2]*ratioSqr + F[j,-2]) * 2.0 * delxSqr +
                                        I[j,-2]*delxSSr)
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
                            (S[j+2,-1] - 4.0*S[j+1,-1] + 6.0*S[j,-1]- 4.0*S[j-1,-1] + S[j-2,-1])
                        ) * ratioSSr +
                        B[j,-1] * (
                            (    S[j+2,1] - 2.0*S[j+2,-1] +     S[j+2,i-3] +
                            -2.0*S[j  ,1] + 4.0*S[j  ,-1] - 2.0*S[j  ,i-3] +
                                 S[j-2,1] - 2.0*S[j-2,-1] +     S[j-2,i-3])
                        ) * ratioSqr / 16.0 +
                        C[j,-1] * (
                            (S[j,1] - 4.0*S[j,0] + 6.0*S[j,-1] - 4.0*S[j,-2] + S[j,-3])
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
                            (S[j+1,-1] - S[j-1,-1])
                        ) * delxTr / 2.0 * ratio +
                        H[j,-1] * (
                            (S[j,0] - S[j,-2])
                        ) * delxTr / 2.0 + (
                        I[j,-1] * S[j,-1] - J[j,-1]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,-1]*ratioSSr + C[j,-1]) * 6.0 +
                                        B[j,-1]*ratioSqr/4.0
                                      -(D[j,-1]*ratioSqr + F[j,-1]) * 2.0 * delxSqr +
                                        I[j,-1]*delxSSr)
                    S[j,-1] += temp
        
        norm = absNorm2D(S, undef)
        
        if np.isnan(norm) or norm > 1e100:
            flags[0] = True
            break
        
        flags[1] = abs(norm - normPrev) / normPrev
        flags[2] = loop
        
        if flags[1] < tolerance or loop >= mxLoop:
            break
        
        normPrev = norm
        loop += 1
        
    return S


@nb.jit(nopython=True, cache=False)
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
        raise Exception('lengths of given arrays are not satisfied')
        
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


@nb.jit(nopython=True, cache=False)
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



@nb.jit(nopython=True, cache=False)
def absNorm3D(S, undef):
    r"""Sum up 3D absolute value S"""
    norm = 0.0
    
    K, J, I = S.shape
    count = 0
    for k in range(K):
        for j in range(J):
            for i in range(I):
                if S[k,j,i] != undef:
                    norm += abs(S[k,j,i])
                    count += 1
    
    if count != 0:
        norm /= count
    else:
        norm = np.nan
    
    return norm

@nb.jit(nopython=True, cache=False)
def absNorm2D(S, undef):
    r"""Sum up 2D absolute value S"""
    norm = 0.0
    
    J, I = S.shape
    count = 0
    for j in range(J):
        for i in range(I):
            if S[j,i] != undef:
                norm += abs(S[j,i])
                count += 1
    
    if count != 0:
        norm /= count
    else:
        norm = np.nan
    
    return norm

@nb.jit(nopython=True, cache=False)
def absNorm1D(S, undef):
    r"""Sum up 1D absolute value S"""
    norm = 0.0
    
    I = S.shape[0]
    count = 0
    for i in range(I):
        if S[i] != undef:
            norm += abs(S[i])
            count += 1
    
    if count != 0:
        norm /= count
    else:
        norm = np.nan
    
    return norm
    
    