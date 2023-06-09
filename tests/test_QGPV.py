# -*- coding: utf-8 -*-
"""
Created on 2022.04.12

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% idealized test case (geostrophic balance in z-coord.)
import xarray as xr
import numpy as np
from xinvert import invert_PV2D


def test_invert_PV2D():
    ds = xr.open_dataset('./Data/atmos3D.nc', decode_times=False)
    
    # select mid-latitude range as background state
    dset = ds.sel({'lon':slice(150, 200), 'lat':slice(50, 40)})
    dset['LEV'] = dset['LEV'] * 100 # convert hPa to Pa
    
    omega = 7.292e-5
    Rd = 287.04
    Cp = 1004.88
    
    P  = dset.LEV.interp(LEV=np.linspace(100000, 10000, 73))
    H  = dset.hgt.mean(['lon','lat']).interp(LEV=np.linspace(100000, 10000, 73)) * 9.81
    T  = dset.T.mean(['lon','lat']).interp(LEV=np.linspace(100000, 10000, 73))
    TH   = T * (100000.0 / P)**(Rd/Cp)
    TH2 = TH.copy()
    for k in range(56, 73):
        TH2[k] = TH2[k] + (k-55)*5
    f  = 2 * omega * np.sin(np.deg2rad(40))  # a constant at 40N
    RPiP = (Rd * T / P / TH)
    S    = - RPiP * TH2.differentiate('LEV')
    
    zc, yc = len(P), 201
    ys  = np.linspace(-1000000, 1000000, yc)
    
    zdef = P
    ydef = xr.DataArray(ys , dims='Y', coords={'Y':ys })
    S    = S.rolling(LEV=5, center=True, min_periods=1).mean()
    
    # case 2
    S[:56] = 1E-5
    S[56:] = 6E-5
    
    # specifying PV
    q = zdef * ydef - zdef * ydef
    # q = xr.where(q.LEV<(20000*np.exp(-ydef**2/1.5e11)+25000), 5, 0)
    # q = xr.where(q.LEV>(10000*np.exp(-ydef**2/1.5e11)+25000), q, 0)
    amplit = 1*np.exp(-ydef**2/1e11)
    zscale = 5e7+np.exp(-ydef**2/2e11)*3e7
    q = np.exp(-(zdef-(30000+np.exp(-ydef**2/1e11)*10000))**2/zscale) * amplit
    
    # invert
    iParams = {
        'BCs'      : ['fixed', 'extend'],
        'tolerance': 1e-12,
    }
    
    mParams = {'f0':f, 'beta':0, 'N2':S}
    
    Ha = invert_PV2D(q, dims=['LEV','Y'], coords='cartesian',
                     iParams=iParams, mParams=mParams)
    
    # verification
    Ua  = -Ha.differentiate('Y') / f
    THa =  Ha.differentiate('LEV') / -RPiP
    Ta  = THa * (P / 100000.0)**(Rd/Cp)
    
    THo = TH2 + THa
    To  = T  + Ta
    Ho  = H  + Ha
    
    assert np.isclose(Ha.max(), 0)
    assert np.abs(Ha.min()) <= 9.464e+10

#%% plot
# import proplot as pplt

# fontsize = 16

# fig, axes = pplt.subplots(figsize=(9,6))

# ax = axes[0]

# tmp = q.copy()
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# m=ax.contourf(tmp.where(tmp>0.2), levels=np.linspace(0,2.4,13), cmap='gray_r')
# ax.colorbar(m, loc='b', label='')

# tmp = THo.copy()
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# ax.contour(tmp, levels=31, colors='k', lw=1)

# tmp = Ua.copy()
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# ax.contour(tmp, levels=15, cmap='RdYlBu_r', lw=2.6)

# tmp = To.rolling(Y=51, center=True, min_periods=1).mean()
# tmp = tmp.where(tmp.LEV>=(5000*np.exp(-(ydef)**2/1.5e11)+24000))
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# ax.contour(tmp, levels=[221.9], lw=5)

# ax.set_yscale('log')
# ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
# ax.set_xticks(np.linspace(-750, 750, 9))
# ax.set_ylim([1000, 100])
# ax.set_xlim([-900, 900])
# ax.set_title('PV inversion for a symmetric vortex', fontsize=fontsize)

# ax.dualy('height', label='height (km)', fontsize=fontsize-2, grid=False)

# axes.format(abc='(a)')





#%% idealized test case 2 (gradient balance in theta-coord.)
# from xgrads.xgrads import open_CtlDataset
# import numpy as np

# dset, ctl = open_CtlDataset('D:/Data/ERAInterim/BKGState/OriginalData/DataPT.ctl',
#                             returnctl=True)

# dsetInt = dset.interp(lev=np.linspace(265, 850, 118))

# dsetInt.to_netcdf('D:/Data/ERAInterim/BKGState/OriginalData/theta.nc')

# #%%
# import xarray as xr
# import numpy as np
# from xinvert.xinvert import FiniteDiff
# from GeoApps.ConstUtils import Rd, Cp

# # path = 'D:/'
# path = 'D:/Data/ERAInterim/BKGState/OriginalData/'
# ds = xr.open_dataset(path + 'theta.nc').sel(lev=slice(265, 600))

# yc, zc = 201, len(ds.lev)

# zdef = ds.lev
# ydef = xr.DataArray(np.linspace(0, 2000000, yc) , dims='Y',
#                     coords={'Y':np.linspace(0, 2000000, yc) })

# fd = FiniteDiff({'X':'lon', 'Y':'lat', 'Z':'lev', 'T':'time'}, BCs='extend', fill=0)

# Gamma = (Rd/ds.pres*(ds.pres/100000)**(Rd/Cp))[0].isel({'lon':100, 'lat':50})

# f0 = 2E-4

# # specifying PV
# q = zdef * ydef - zdef * ydef

# amplit = 1E-4*np.exp(-ydef**2/1e11)
# zscale = 5e2+np.exp(-ydef**2/2e11)*2e2
# q = np.exp(-(zdef-(450-np.exp(-ydef**2/1e11)*50))**2/zscale) * amplit + f0
# Gamma = q-q+Gamma


# #%% invert
# from xinvert.xinvert.apps import invert_Vortex_2D

# AngM = q-q+f0*ydef**2.0/2.0

# params = {
#     'BCs'      : ['fixed', 'fixed'],
#     'tolerance': 1e-19,
#     'mxLoop'   : 600,
# }

# Ang = invert_Vortex_2D(q, AngM, Gamma, dims=['lev','Y'],
#                        coords='cartesian', params=params, out=AngM)

# u = (Ang-f0*ydef**2.0/2.0) / ydef

# #%% infer other variables
# h = Ang.differentiate('Y') / q / ydef
# p = (-h*9.8).cumsum('lev') * 5 + 100000
# M = (h*Gamma*9.81).cumsum('lev').cumsum('lev') * 5 * 5 + Cp*300
# Pai = Cp * (p/100000)**(Rd/Cp)
# M2 = Pai.cumsum('lev') * 5 + Cp*300
# T = zdef*(p/100000)**(Rd/Cp)
# z = (M - Cp *T) / 9.81

#%% plot
# import proplot as pplt

# fontsize = 16

# fig, axes = pplt.subplots(figsize=(9,6))

# ax = axes[0]

# tmp = q.copy()
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# m=ax.contourf(tmp.where(tmp>0.2), levels=np.linspace(0,2.4,13), cmap='gray_r')
# ax.colorbar(m, loc='b', label='')
# tmp = THo.copy()
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# ax.contour(tmp, levels=31, colors='k', lw=1)
# tmp = Ua.copy()
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# ax.contour(tmp, levels=15, cmap='RdYlBu_r', lw=2.6)
# tmp = To.rolling(Y=51, center=True, min_periods=1).mean()
# tmp = tmp.where(tmp.LEV>=(5000*np.exp(-(ydef)**2/1.5e11)+24000))
# tmp['LEV'] = tmp['LEV'] / 100
# tmp['Y'] = tmp['Y'] / 1000
# ax.contour(tmp, levels=[221.9], lw=5)
# ax.set_yscale('log')
# ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
# ax.set_xticks(np.linspace(-750, 750, 9))
# ax.set_ylim([1000, 100])
# ax.set_xlim([-900, 900])
# ax.set_title('PV inversion for a symmetric vortex', fontsize=fontsize)

# ax.dualy('height', label='height (km)', fontsize=fontsize-2, grid=False)

# axes.format(abc='(a)')



