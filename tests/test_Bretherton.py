# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert import invert_BrethertonHaidvogel, cal_flow

def test_BrethertonHaidvogel():
    topo = xr.open_dataset('./Data/topo.nc').topo
    topo = topo - topo.mean()
    
    assert topo.dims == ('y', 'x')
    assert topo.shape == (201, 301)
    
    iParams = {
        # Preserve the historical snapshot precision explicitly.
        'dtype': 'float32',
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 3000,
        # Fixed-work historical snapshot, not a converged reference.
        'tolerance': 0.0,
        'undef'    : np.nan,
    }
    
    mParams1 = {'f0': 1e-4, 'D': 1000, 'lambda': 1e-15}
    
    S1 = invert_BrethertonHaidvogel(topo,
                                    dims=['y','x'], coords='cartesian',
                                    mParams=mParams1, iParams=iParams)
    
    u1, v1 = cal_flow(S1, dims=['y','x'], coords='cartesian')
    
    assert topo.dims == S1.dims
    assert topo.shape == S1.shape
    assert topo.dims == u1.dims
    assert topo.shape == u1.shape
    
    KE = (u1**2 + v1**2).sum() / 2
    
    assert np.isclose(KE.values, 0.0812731, rtol=1e-5, atol=1e-8)


