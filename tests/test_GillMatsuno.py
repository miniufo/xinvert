# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xinvert import invert_GillMatsuno, cal_flow


def test_GillMatsuno_idealized():
    lon = xr.DataArray(np.linspace(0, 360, 144), dims='lon',
                       coords={'lon':np.linspace(0, 360, 144)})
    lat = xr.DataArray(np.linspace(-90, 90, 73), dims='lat',
                       coords={'lat':np.linspace(-90, 90, 73)})
    
    lat, lon = xr.broadcast(lat, lon)
    
    Q1 = 0.05*np.exp(-((lat-0)**2+(lon-120)**2)/100.0)
    Q2 = 0.05*np.exp(-((lat-10)**2+(lon-120)**2)/100.0) \
       - 0.05*np.exp(-((lat+10)**2+(lon-120)**2)/100.0)
    Q3 = 0.05*np.exp(-((lat-10)**2+(lon-120)**2)/100.0)
    
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'periodic'],
        'mxLoop'   : 2000,
        'tolerance': 1e-8,
        'optArg'   : 1.4,
    }
    
    mParams = {
        'epsilon': 1e-5,
        'Phi'    : 5000,
    }
    
    h1 = invert_GillMatsuno(Q1, dims=['lat','lon'], iParams=iParams, mParams=mParams)
    h2 = invert_GillMatsuno(Q2, dims=['lat','lon'], iParams=iParams, mParams=mParams)
    h3 = invert_GillMatsuno(Q3, dims=['lat','lon'], iParams=iParams, mParams=mParams)
    
    u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['fixed','periodic'],
                      mParams=mParams, vtype='GillMatsuno')
    u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['fixed','periodic'],
                      mParams=mParams, vtype='GillMatsuno')
    u3, v3 = cal_flow(h3, dims=['lat','lon'], BCs=['fixed','periodic'],
                      mParams=mParams, vtype='GillMatsuno')
    
    assert (h1 <= 0).all()
    assert (np.abs(h2) <= 370).all()
    assert (h3 <= 0).all()
    assert np.isclose(((u1**2+v1**2)/2).sum(), 4351.62244687)
    assert np.isclose(((u2**2+v2**2)/2).sum(), 5833.33192343)
    assert np.isclose(((u3**2+v3**2)/2).sum(), 5100.85325027)


def test_GillMatsuno_real():
    ds = xr.open_dataset('./Data/MJO.nc')
    
    Q  = (ds.ol*-0.0015).where(np.abs(ds.lat)<60, 0)
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'periodic'],
        'mxLoop'   : 2000,
        'tolerance': 1e-12,
        'optArg'   : 1.4,
    }
    
    mParams1 = {'epsilon': 1e-5, 'Phi': 5000}
    mParams2 = {'epsilon': 7e-6, 'Phi': 8000}
    mParams3 = {'epsilon': 7e-6, 'Phi': 10000}
    
    h1 = invert_GillMatsuno(Q, dims=['lat','lon'], iParams=iParams, mParams=mParams1)
    h2 = invert_GillMatsuno(Q, dims=['lat','lon'], iParams=iParams, mParams=mParams2)
    h3 = invert_GillMatsuno(Q, dims=['lat','lon'], iParams=iParams, mParams=mParams3)
    
    u1, v1 = cal_flow(h1, dims=['lat','lon'], BCs=['fixed','periodic'],
                      mParams=mParams1, vtype='GillMatsuno')
    u2, v2 = cal_flow(h2, dims=['lat','lon'], BCs=['fixed','periodic'],
                      mParams=mParams2, vtype='GillMatsuno')
    u3, v3 = cal_flow(h3, dims=['lat','lon'], BCs=['fixed','periodic'],
                      mParams=mParams3, vtype='GillMatsuno')
    
    assert (np.abs(h1) <= 1200).all()
    assert (np.abs(h2) <= 1200).all()
    assert (np.abs(h3) <= 1000).all()
    assert np.isclose(((u1**2+v1**2)/2).sum(), 137039.11)
    assert np.isclose(((u2**2+v2**2)/2).sum(), 110614.41)
    assert np.isclose(((u3**2+v3**2)/2).sum(),  77191.09)


