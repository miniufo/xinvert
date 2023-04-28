# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load sample data
import xarray as xr


dset = xr.open_dataset('./xinvert/Data/Helmholtz_atmos.nc')

T = dset.sf[0]

print(T)

#%% test padding
from xinvert.xinvert.finitediffs import padBCs

T_Px = padBCs(T, dim='lon', BCs=('fixed','fixed'), fill=(1,1))
T_Py = padBCs(T, dim='lat', BCs=('extend','fixed'), fill=(2,2))
T_Py2= padBCs(T, dim='lat', BCs=('periodic','periodic'))
T_Py3= padBCs(T, dim='lat', BCs=('reflect','extend'), fill=(3,3))


#%% test derivatives
from xinvert.xinvert.finitediffs import deriv, deriv2

Tx1 = deriv(T, dim='lon', scheme='center')
Tx2 = deriv(T, dim='lon', scheme='forward')
Tx3 = deriv(T, dim='lon', scheme='backward')

Txx = deriv2(T, dim='lon')

#%% test grad, curl, vort, Laplacian
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff(dim_mapping={'T':'time', 'Y':'lat', 'X':'lon'},
                BCs={'Y':'reflect', 'X':'periodic'},
                coords='lat-lon')

Ty, Tx = fd.grad(T, dims=['Y', 'X'])
Tcurl  = fd.curl(Tx, Ty)
Tdivg  = fd.divg([Tx, Ty], dims=['X', 'Y'])
TLap   = fd.Laplacian(T, dims=['Y', 'X'])

#%%
from GeoApps.GridUtils import add_latlon_metrics
from GeoApps.DiagnosticMethods import Dynamics


ds, grid = add_latlon_metrics(dset, {'Y':'lat', 'X':'lon'},
                              boundary={'X':'periodic', 'Y':'extend'})

dyn = Dynamics(ds, grid, arakawa='A')

Ty2, Tx2= dyn.grad(T, dims=['Y', 'X'])
Tcurl2  = dyn.vort(u=Tx2, v=Ty2, components='k')
Tdivg2  = dyn.divg([Tx2, Ty2], dims=['X', 'Y'])
TLap2   = dyn.Laplacian(T, dims=['X', 'Y'])


#%%
import proplot as pplt
import numpy as np

fig, axes = pplt.subplots(nrows=5, ncols=3, figsize=(11, 12))

fontsize = 14

ax = axes[0,0]
m=ax.contourf(Tx, levels=np.linspace(-35, 35, 15))
ax.set_title('Tx by FiniteDiff', fontsize=fontsize)
ax = axes[0,1]
m=ax.contourf(Tx2, levels=np.linspace(-35, 35, 15))
ax.colorbar(m, loc='r')
ax.set_title('Tx2 by xgcm', fontsize=fontsize)
ax = axes[0,2]
m=ax.contourf(Tx-Tx2, levels=21)
ax.colorbar(m, loc='r')
ax.set_title('diff of Tx', fontsize=fontsize)

ax = axes[1,0]
m=ax.contourf(Ty, levels=np.linspace(-35, 35, 15))
ax.set_title('Ty by FiniteDiff', fontsize=fontsize)
ax = axes[1,1]
m=ax.contourf(Ty2, levels=np.linspace(-35, 35, 15))
ax.colorbar(m, loc='r')
ax.set_title('Ty2 by xgcm', fontsize=fontsize)
ax = axes[1,2]
m=ax.contourf(Ty-Ty2, levels=21)
ax.colorbar(m, loc='r')
ax.set_title('diff of Ty', fontsize=fontsize)

ax = axes[2,0]
m=ax.contourf(Tcurl*1e6, levels=np.linspace(-4,4,17))
ax.set_title('Tcurl by FiniteDiff', fontsize=fontsize)
ax = axes[2,1]
m=ax.contourf(Tcurl2*1e6, levels=np.linspace(-4,4,17))
ax.colorbar(m, loc='r')
ax.set_title('Tcurl2 by xgcm', fontsize=fontsize)
ax = axes[2,2]
m=ax.contourf(Tcurl-Tcurl2, levels=21)
ax.colorbar(m, loc='r')
ax.set_title('diff of Tcurl', fontsize=fontsize)

ax = axes[3,0]
m=ax.contourf(Tdivg*1e5, levels=np.linspace(-8,8,17))
ax.set_title('Tdivg by FiniteDiff', fontsize=fontsize)
ax = axes[3,1]
m=ax.contourf(Tdivg2*1e5, levels=np.linspace(-8,8,17))
ax.colorbar(m, loc='r')
ax.set_title('Tdivg2 by xgcm', fontsize=fontsize)
ax = axes[3,2]
m=ax.contourf(Tdivg-Tdivg2, levels=21)
ax.colorbar(m, loc='r')
ax.set_title('diff of Tdivg', fontsize=fontsize)

ax = axes[4,0]
m=ax.contourf(TLap*1e5, levels=np.linspace(-10,10,21))
ax.set_title('TLap by FiniteDiff', fontsize=fontsize)
ax = axes[4,1]
m=ax.contourf(TLap2*1e5, levels=np.linspace(-10,10,21))
ax.colorbar(m, loc='r')
ax.set_title('TLap2 by xgcm', fontsize=fontsize)
ax = axes[4,2]
m=ax.contourf(TLap-TLap2, levels=21)
ax.colorbar(m, loc='r')
ax.set_title('diff of TLap', fontsize=fontsize)

axes.format(abc='(a)', ylabel='', xlabel='')





#%%
from GeoApps.GridUtils import add_latlon_metrics
from GeoApps.DiagnosticMethods import Dynamics
from xgrads.xgrads import open_CtlDataset
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff(dim_mapping={'T':'time', 'Z':'lev', 'Y':'lat', 'X':'lon'},
                coords='lat-lon', BCs={'Y':'extend', 'X':'periodic'})

ds = open_CtlDataset('D:/Data/SODA/2.2.6/SODA226Clim_1993_2003.ctl')
dset, grid = add_latlon_metrics(dset, {'Y':'lat', 'X':'lon'},
                                boundary={'X':'periodic', 'Y':'extend'})
dyn = Dynamics(ds, grid, arakawa='A')

taux = ds.taux.where(ds.taux!=ds.undef).isel(time=[0, 6])
tauy = ds.tauy.where(ds.taux!=ds.undef).isel(time=[0, 6])
curl = fd.curl(taux, tauy).rename('curl')
curl2= dyn.curl(taux, tauy, dims=['Y', 'X'])

# xr.merge([taux, tauy, curl]).to_netcdf('E:/OneDrive/Python/MyPack/xinvert/Data/SODA.nc')


