# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
import xarray as xr
import numpy as np
import numba as nb


@nb.jit(nopython=True)
def solve(S, A, F, dx, flags, undef, optArg=1,
                      mxLoop=5000, tolerance=1e-7, BC='fixed'):
    loop = 0
    temp = 0.0
    normPrev = np.finfo(np.float64).max
    xc = len(S)
    
    while(True):
        # process boundaries
        if BC == 'extend':
            if  S[ 2] != undef:
                S[ 0] = S[ 2]
                S[ 1] = S[ 2]
            if  S[-3] != undef:
                S[-1] = S[-3]
                S[-2] = S[-3]
        
        # inner loop
        for i in range(2, xc-2):
            cond = (A[i] != undef and F[i] != undef)
            
            if cond:
                temp = (
                    A[i] * (
                        (S[i+2] - 4.0*S[i+1] + 6.0*S[i]- 4.0*S[i-1] + S[i-2])
                    ) / dx ** 4.0 - F[i]
                )
                
            temp *= optArg / (A[i]/dx**4.0 * 6.0)
            S[i] -= temp
        
        norm = absNorm(S, undef)
        
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



@nb.jit(nopython=True)
def absNorm(S, undef):
    norm = 0.0
    
    I = len(S)
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


nx = 1000
a  = 5
undef = np.nan

xdef = xr.DataArray(np.linspace(0,10,nx), dims=['xdef'],
                    coords={'xdef':np.linspace(0,10,nx)})

F = np.sin(a * xdef)

psi   =  1.0/a**4 * np.sin(a * xdef)
psiD1 =  1.0/a**3 * np.cos(a * xdef)
psiD2 = -1.0/a**2 * np.sin(a * xdef)
psiD3 = -1.0/a**1 * np.cos(a * xdef)
psiD4 =  1.0/a**0 * np.sin(a * xdef)

zero = F - F

S = zero
A = zero + 1

dx = xdef.diff('xdef').values[0]
flags = np.array([0.0, 1.0, 0.0])

solve(S.values, A.values, F.values, dx,
      flags, undef, optArg=1.3, mxLoop=100, tolerance=1e-7)

if flags[0]:
    print('loops {0:4.0f} and tolerance is {1:e} (overflows!)'
          .format(flags[2], flags[1]))
else:
    print('loops {0:4.0f} and tolerance is {1:e}'
          .format(flags[2], flags[1]))


