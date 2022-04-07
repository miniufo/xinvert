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
                       mxLoop=5000, tolerance=1e-7):
    """
    Inverting a 3D volume of elliptic equation in standard form as:
    
        d ┌  dS ┐   d ┌  dS ┐   d ┌  dS ┐
        --│A(--)│ + --│B(--)│ + --│C(--)│ = F
        dz└  dz ┘   dy└  dy ┘   dx└  dx ┘
    
    using SOR iteration. If F = F['time', 'lev', 'lat', 'lon'] and we invert
    for the 3D spatial distribution, then 3rd dim is 'lev', 2nd dim is 'lat'
    and 1st dim is 'lon'.
    
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
    ----------
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
        
        if np.isnan(norm) or norm > 1e17:
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
                       mxLoop=5000, tolerance=1e-7):
    """
    Inverting a 2D slice of elliptic equation in standard form as:
    
        d ┌  dS      dS ┐   d ┌  dS      dS ┐
        --│A(--) + B(--)│ + --│B(--) + C(--)│ = F
        dy└  dy      dx ┘   dx└  dy      dx ┘
    
    using SOR iteration. If F = F['time', 'lat', 'lon'] and we invert
    for the horizontal slice, then 2nd dim is 'lat' and 1st dim is 'lon'.
    
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
    ----------
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
        
        if np.isnan(norm) or norm > 1e17:
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
                      mxLoop=5000, tolerance=1e-7):
    """
    Inverting a 2D slice of elliptic equation in general form as:
    
          d^2S     d^2S     d^2S     dS     dS 
        A ---- + B ---- + C ---- + D -- + E -- + F*S + G = 0
          dy^2     dydx     dx^2     dy     dx 
    
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
    ----------
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
                        ) +
                        D[j,0] * (
                            (S[j+1,0] - S[j-1,0])
                        ) * delx / 2.0 * ratio +
                        E[j,0] * (
                            (S[j,1] - S[j,-1])
                        ) * delx / 2.0 + (
                        F[j,0] * S[j,0] + G[j,0]) * delxSqr
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
                        ) +
                        D[j,i] * (
                            (S[j+1,i] - S[j-1,i])
                        ) * delx / 2.0 * ratio +
                        E[j,i] * (
                            (S[j,i+1] - S[j,i-1])
                        ) * delx / 2.0 + (
                        F[j,i] * S[j,i] + G[j,i]) * delxSqr
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
                        ) +
                        D[j,-1] * (
                            (S[j+1,-1] - S[j-1,-1])
                        ) * delx / 2.0 * ratio +
                        E[j,-1] * (
                            (S[j,0] - S[j,-2])
                        ) * delx / 2.0+ (
                        F[j,-1] * S[j,-1] + G[j,-1]) * delxSqr
                    )
                    
                    temp *= optArg / ((A[j,-1]*ratioSqr + C[j,-1]) * 2.0
                                      -F[j,-1]*delxSqr)
                    S[j,-1] += temp
        
        norm = absNorm2D(S, undef)
        
        if np.isnan(norm) or norm > 1e17:
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
                          mxLoop=5000, tolerance=1e-7):
    """
    Inverting a 2D slice of biharmonic equation in general form as:
    
      d^4S       d^4S       d^4S     d^2S     d^2S     d^2S     dS     dS 
    A ---- + B -------- + C ---- + D ---- + E ---- + F ---- + G -- + H -- + I*S + J = 0
      dy^4     dy^2dx^2     dx^4     dy^2     dydx     dx^2     dy     dx 
    
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
        Coefficient for the tenth term.
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
    ----------
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
                        I[j,0] * S[j,0] + J[j,0]) * delxSSr
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
                        I[j,1] * S[j,1] + J[j,1]) * delxSSr
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
                        I[j,i] * S[j,i] + J[j,i]) * delxSSr
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
                        I[j,-2] * S[j,-2] + J[j,-2]) * delxSSr
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
                        I[j,-1] * S[j,-1] + J[j,-1]) * delxSSr
                    )
                    
                    temp *= -optArg / ((A[j,-1]*ratioSSr + C[j,-1]) * 6.0 +
                                        B[j,-1]*ratioSqr/4.0
                                      -(D[j,-1]*ratioSqr + F[j,-1]) * 2.0 * delxSqr +
                                        I[j,-1]*delxSSr)
                    S[j,-1] += temp
        
        norm = absNorm2D(S, undef)
        
        if np.isnan(norm) or norm > 1e17:
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
def absNorm3D(S, undef):
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
    
    