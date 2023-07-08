# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases for deep ocean circulation (TODO)
import numpy as np
import xarray as xr
from xinvert import invert_RefStateSWM


def test_RefStateSWM():
    ds1 = xr.open_dataset('./Data/Barotropic2D.nc', chunks={'time':1})
    ds2 = xr.open_dataset('./Data/contour.nc', chunks={'time':1})
    
    # load Q, M(Q), and C(Q) from a pre-calculated dataset
    ctr  = ds2.PV
    Mass = ds2.Mass
    Circ = ds2. Circ
    
    def getQandC(M):
        # get Qref and Cref given Mref, using tabulated M(Q) and C(Q)
        Q = xr.apply_ufunc(np.interp, M, Mass, ctr,
                           dask='parallelized',
                           input_core_dims=[['lat'], ['contour'], ['contour']],
                           vectorize=True,
                           output_core_dims=[['lat']])
        
        Q = xr.where(Q.lat==90, ctr.max('contour'), Q).transpose() # ensure the north-pole consistency
        
        C = xr.apply_ufunc(np.interp, Q, ctr, Circ,
                           dask='parallelized',
                           input_core_dims=[['lat'], ['contour'], ['contour']],
                           vectorize=True,
                           output_core_dims=[['lat']])
        
        assert (M[:,  0] == Mass[:,  0]).all()
        assert (M[:, -1] == Mass[:, -1]).all()
        
        assert (Q[:,  0] == ctr[:,  0]).all()
        assert (Q[:, -1] == ctr[:, -1]).all()
        
        assert (C[:,  0] == Circ[:,  0]).all()
        assert (C[:, -1] == Circ[:, -1]).all()
        
        return Q, C


    ################### invert using Stommel-Munk model ##################
    iParams = {
        'BCs'      : ['fixed'],
        'mxLoop'   : 5000,
        'tolerance': 1e-18,
        'undef'    : np.nan,
        'debug'    : False,
        'printInfo': False}
    
    # initial guess of Mref
    Mref = Mass.max('contour') * (np.sin(np.deg2rad(ds1.lat)) + 1.0) / 2.0
    
    for i in range(5): # nonlinear iterations
        print(i)
        
        Qref, Cref = getQandC(Mref)
        mParams = {'M0':Mref, 'C0':Cref}
        
        dM = invert_RefStateSWM(Qref, dims=['lat'], iParams=iParams, mParams=mParams)
        Mref = Mref + dM
    
    
    r = 6371200. * np.cos(np.deg2rad(ds1.lat))
    # uref = Cref / (2.0 * np.pi * r) - 7.292e-5 * r
    href = Mref.differentiate('lat') / (2.0 * np.pi * r) / (6371200. * np.deg2rad(1.))
    
    assert np.isclose(href[:, 2:-7], ds1.href[:, 2:-7], rtol=3e-2).all()


