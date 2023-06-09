# -*- coding: utf-8 -*-
"""
Created on 2022.08.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""

#%% prepare the ZonalMean.nc
import xarray as xr
import numpy as np
from xinvert import deriv, FiniteDiff, invert_Eliassen


def test_Hadley():
    ds = xr.open_dataset('./Data/atmos3D.nc')\
        .isel(lat=np.arange(0, 72))
        # .isel(latitude=np.arange(115, -1, -1))
    
    dset = ds.interp({'LEV':np.linspace(1000, 100, 37)}).astype('f4')
    dset['LEV'] = dset['LEV'] * 100 # hPa changed to Pa
    
    assert dset.dims == {'LEV': 37, 'lat': 72, 'lon': 288}
    
    Re = 6371200
    Rd = 287.04
    Cp = 1004.88
    P0 = 100000
    Om = 7.292e-5
    
    f  = 2 * Om * np.sin(np.deg2rad(dset['lat']))
    r  = Re * np.cos(np.deg2rad(dset['lat']))
    pi = (dset['LEV'] / P0) ** (Rd / Cp)
    
    u  = dset.U
    v  = dset.V
    w  = dset.Omega
    T  = dset.T
    z  = dset.hgt
    Th = (T / pi).astype('f4')
    Ag = ((u + Om * r) * r).astype('f4')
    
    assert u.dims == v.dims == w.dims == T.dims == z.dims == Th.dims == Ag.dims
    assert u.shape== v.shape== w.shape== T.shape== z.shape== Th.shape== Ag.shape
    
    um = u.mean('lon').rename('um')
    vm = v.mean('lon').rename('vm')
    wm = w.mean('lon').rename('wm')
    Am =Ag.mean('lon').rename('Am')
    Tm = T.mean('lon').rename('Tm')
    Thm=Th.mean('lon').rename('Thm')
    zm = z.mean('lon').rename('zm')
    
    assert um.dims == vm.dims == wm.dims == Tm.dims == zm.dims == Thm.dims == Am.dims
    assert um.shape== vm.shape== wm.shape== Tm.shape== zm.shape== Thm.shape== Am.shape
    
    ua = (u - um).rename('ua')
    va = (v - vm).rename('va')
    wa = (w - wm).rename('wa')
    Aa =(Ag - Am).rename('Aa')
    Ta = (T - Tm).rename('Ta')
    Tha=(Th -Thm).rename('Tha')
    za = (z - zm).rename('za')
    
    assert ua.dims == va.dims == wa.dims == Ta.dims == za.dims == Tha.dims == Aa.dims
    assert ua.shape== va.shape== wa.shape== Ta.shape== za.shape== Tha.shape== Aa.shape
    
    Aava = (Aa * va).mean('lon').rename('Aava')
    Aawa = (Aa * wa).mean('lon').rename('Aawa')
    Tava = (Tha* va).mean('lon').rename('Tava')
    Tawa = (Tha* wa).mean('lon').rename('Tawa')
    
    assert Aava.dims == Aawa.dims == Tava.dims == Tawa.dims
    assert Aava.shape== Aawa.shape== Tava.shape== Tawa.shape
    
    
    # calculate A, B, C
    fd = FiniteDiff(dim_mapping={'Z':'LEV', 'Y':'lat', 'X':'lon'},
                    BCs={'Z':['extend', 'extend'],
                         'Y':['extend', 'extend'],
                         'X':['periodic','periodic']},
                    coords='lat-lon', fill=0)
    
    
    deg2m = Re * 3.1416 / 180.0
    cosL  = np.cos(np.deg2rad(dset['lat']))
    sinL  = np.sin(np.deg2rad(dset['lat']))
    
    # defined on half grid
    Ccoef =-deriv(Thm, dim='LEV',
                  scheme='backward') * Rd * pi / dset['LEV'] /r
    # defined on half grid
    Bcoef = deriv(Thm, dim='lat',
                  scale=deg2m, scheme='center') * Rd * pi / dset['LEV'] /r
    Bcoef2= deriv(Am, dim='LEV',
                  scheme='center') * f / r /r
    # defined on half grid
    Acoef =-deriv(Am, dim='lat',
                  scale=deg2m, scheme='backward') * f / r /r
    Acoef2=-deriv(Am*Am, dim='lat',
                  scale=deg2m, scheme='backward') * sinL / (r**3) /r
    
    assert np.isnan(Ccoef [0  ]).all()
    assert np.isnan(Acoef [:,0]).all()
    assert np.isnan(Acoef2[:,0]).all()
    
    EHFC =  fd.divg((Tava, Tawa), dims=['Y', 'Z'])
    EAFC = -fd.divg((Aava, Aawa), dims=['Y', 'Z'])
    
    EHFC[:, :1] = 0 # maskout polar forcing to 0
    EAFC[:, :1] = 0 # maskout polar forcing to 0
    
    F_EHF = (fd.grad(EHFC, dims='Y') * Rd * pi / dset['LEV']).rename('EHF')
    F_AHF =(-fd.grad(EAFC, dims='Z') * f / r).rename('EAF')
    
    
    F = (Thm - Thm).load().rename('F_ideal')
    F[27, 42:50] = 1e-11
    F[27, 43] = 1e-11
    F[10, 42:50] =-1e-11
    F[10, 43] =-1e-11
    
    Acoef = Acoef.rename('Acoef')
    Bcoef = Bcoef.rename('Bcoef')
    Ccoef = Ccoef.rename('Ccoef')
    
    dsnew = xr.merge([um, vm, wm, Am, Tm, Thm, zm, Aava, Aawa, Tava, Tawa,
                      Acoef, Bcoef, Ccoef, F_EHF, F_AHF, F])

    # dsnew.to_netcdf('./xinvert/Data/ZonalMean.nc')
    
    assert dsnew.dims == {'lat': 72, 'LEV':37}
    
    
    # inversion here
    iParams = {
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 600,
        'tolerance': 1e-10,
    }
    
    mParams = {'A': Acoef, 'B': Bcoef, 'C': Ccoef}

    sf = invert_Eliassen(F_EHF+F_AHF, dims=['LEV','lat'],
                         coords='z-lat', iParams=iParams, mParams=mParams)
    
    assert sf.dims  == um.dims
    assert sf.shape == um.shape
    

def test_ideal_TC():
    # load a snapshot as a radius-vertical 2D structure
    ds = xr.open_dataset('./Data/TC2D.nc')
    
    assert ds.dims == {'lev': 37, 'lat': 50}
    
    undef = 9.99e20
    
    A = ds.Aa.where(ds.Aa!=undef).load()
    B = ds.Bb.where(ds.Bb!=undef).load()
    C = ds.Cc.where(ds.Cc!=undef).load()
    
    A[:,0] = 2 * A[:, 1] - A[:, 2]
    # B[:,0] = np.nan
    C[:,0] = 2 * C[:, 1] - C[:, 2]
    
    F = (ds.faf.where(ds.faf!=undef) - ds.faf.where(ds.faf!=undef)).load()
    F[27, 22] = 1e-11
    F[27, 23] =-1e-11
    F[10, 22] = 1e-11
    F[10, 23] =-1e-11
    
    F = F.where(A)
    
    # interpolate to a uniform y-z coords.
    lat = xr.DataArray(np.linspace(-90, -75.3, 99), dims='lat',
                       coords={'lat':np.linspace(-90, -75.3, 99)})
    lev = xr.DataArray(np.linspace(100000, 10000, 73), dims='lev',
                       coords={'lev':np.linspace(100000, 10000, 73)})
    
    A = A.interp({'lev':lev, 'lat':lat})
    B = B.interp({'lev':lev, 'lat':lat})
    C = C.interp({'lev':lev, 'lat':lat})
    F = F.interp({'lev':lev, 'lat':lat})
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 600,
        'tolerance': 1e-12,
        'optArg'   : 1.14,
    }
    
    mParams = {'A': A, 'B': B, 'C': C}
    
    # use south pole as the center and latlon to fake the cylindrical coordinate
    sf = invert_Eliassen(F, dims=['lev','lat'], coords='z-lat',
                         iParams=iParams, mParams=mParams)
    
    assert sf.dims  == F.dims
    assert sf.shape == F.shape


def test_real_TC():
    ds = xr.open_dataset('./Data/TC2D.nc')
    
    assert ds.dims == {'lev': 37, 'lat': 50}
    
    undef = 9.99e20
    
    A = ds.Aa.where(ds.Aa!=undef).load()
    B = ds.Bb.where(ds.Bb!=undef).load()
    C = ds.Cc.where(ds.Cc!=undef).load()
    F = ds.faf.where(ds.faf!=undef).load()
    
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'fixed'],
        'mxLoop'   : 600,
        'tolerance': 1e-12,
        'optArg'   : 1.4,
    }
    
    mParams = {'A': A, 'B': B, 'C': C}
    
    # use south pole as the center and latlon to fake the cylindrical coordinate
    sf = invert_Eliassen(F, dims=['lev','lat'], coords='z-lat',
                         iParams=iParams, mParams=mParams)
    
    assert sf.dims  == F.dims
    assert sf.shape == F.shape



