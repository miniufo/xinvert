# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global data from Dr. Yuan Zhao
import xarray as xr
import numpy as np
from GeoApps.GridUtils import add_latlon_metrics


dset = xr.open_dataset('./xinvert/Data/atmos3D.nc', decode_times=False)

ds, grid = add_latlon_metrics(dset, dims={'Z':'LEV', 'Y':'lat', 'X':'lon'},
                              boundary={'X':'periodic', 'Y':'extend', 'Z':'extend'})

ds['LEV'] = ds['LEV'] * 100

#%% zonal running smooth of polar grid
def smooth(v, gridpoint=13, lat=80):
    rolled = v.pad({'lon':(gridpoint,gridpoint)},mode='wrap')\
              .rolling(lon=gridpoint, center=True, min_periods=1).mean()\
              .isel(lon=slice(gridpoint, -gridpoint))
    return xr.where(np.abs(v-v+v.lat)>lat, rolled, v)


#%% calculate forcings using Dynamics (depend on GeoApps)
from GeoApps.ConstUtils import Rd, Cp, omega
from GeoApps.DiagnosticMethods import Dynamics

dyn = Dynamics(ds, grid, arakawa='A')

p   = ds.LEV
Psfc= ds.psfc
f   = 2*omega*np.sin(np.deg2rad(ds.lat))
cos = np.cos(np.deg2rad(ds.lat))

T = smooth(ds.T, lat=80)
U = smooth(ds.U, lat=80)
V = smooth(ds.V, lat=80)
W = smooth(ds.Omega, lat=80)
H = smooth(ds.hgt, lat=80)

lap = dyn.Laplacian(H)

th   = T * (100000 / p)**(Rd/Cp)
TH   = grid.average(th, ['Y','X'])
dTHdp= TH.differentiate('LEV')
RPiP = (Rd * T / p / TH)
S    = - RPiP * dTHdp
vor  = dyn.curl(U,V)
zeta = vor + f

########## traditional form of forcings ##########
grdthx, grdthy = dyn.grad(th)
grdvrx, grdvry = dyn.grad(vor)

F1 = dyn.Laplacian((U * grdthx + V * grdthy) * RPiP)
F2 = ((U * grdvrx + V * grdvry) * f).differentiate('LEV')

FAll = (F1 + F2)
FAll = xr.where(np.isinf(FAll), np.nan, FAll)

########### Q-vector form of forcings ###########
ux, uy = dyn.grad(U, ['X', 'Y'])
vx, vy = dyn.grad(V, ['X', 'Y'])

Qx = - RPiP * (ux * grdthx + vx * grdthy)
Qy = - RPiP * (uy * grdthx + vy * grdthy)

FQvec = -2 * dyn.divg((Qx, Qy), dims=['X', 'Y'])
FQvec = xr.where(np.isinf(FQvec), np.nan, FQvec)

#%% calculate forcings using FiniteDiff (which is integrated in xinvert)
from xinvert.xinvert import FiniteDiff

fd = FiniteDiff({'X':'lon', 'Y':'lat', 'Z':'LEV'},
                BCs={'X':('periodic','periodic'),
                     'Y':('reflect','reflect'),
                     'Z':('extend','extend')}, fill=0, coords='lat-lon')

vor2  = fd.curl(U,V).load()
_, tmp = xr.broadcast(vor2, vor2.mean('lon'))
vor2[:,0,:] = tmp[:,0,:]
vor2[:,-1,:] = tmp[:,-1,:]
zeta2 = vor2 + f

lap2 = fd.Laplacian(H)

########## traditional form of forcings ##########
grdthx2, grdthy2 = fd.grad(th)
grdvrx2, grdvry2 = fd.grad(vor2)

F12 = fd.Laplacian((U * grdthx2 + V * grdthy2) * RPiP)
F22 = ((U * grdvrx2 + V * grdvry2) * f).differentiate('LEV')

FAll2 = (F12 + F22)
FAll2 = smooth(xr.where(np.isinf(FAll2), np.nan, FAll2), lat=83)

########### Q-vector form of forcings ###########
ux2, uy2 = fd.grad(U)
vx2, vy2 = fd.grad(V)

Qx2 = - RPiP * (ux2 * grdthx2 + vx2 * grdthy2)
Qy2 = - RPiP * (uy2 * grdthx2 + vy2 * grdthy2)

FQvec2 = -2 * fd.divg((Qx2, Qy2), dims=['X', 'Y'])
FQvec2 = smooth(xr.where(np.isinf(FQvec2), np.nan, FQvec2), gridpoint=17, lat=85)

#%% prepare lower boundary for inversion
p3D = T-T+p

FAll2 = FAll.where(p<=Psfc)
FQvec2 = FQvec.where(p<=Psfc)
WBC = xr.where(p3D<=Psfc, 0, W).load()

#%% invert
from xinvert.xinvert.apps import invert_omega

iParams = {
    'BCs'      : ['fixed', 'fixed', 'periodic'],
    'tolerance': 1e-16,
}

mParams = {'N2': S}

WQG  = invert_omega(FAll, dims=['LEV', 'lat', 'lon'],
                    iParams=iParams, mParams=mParams)
WQG2 = invert_omega(FAll2, dims=['LEV', 'lat', 'lon'],
                    iParams=iParams, mParams=mParams, icbc=WBC)
WQvec = invert_omega(FQvec, dims=['LEV', 'lat', 'lon'],
                     iParams=iParams, mParams=mParams)
WQvec2 = invert_omega(FQvec2, dims=['LEV', 'lat', 'lon'],
                      iParams=iParams, mParams=mParams, icbc=WBC)

#%% plot cross section
import proplot as pplt

x = 80

fontsize = 16

fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(11, 11))

ax = axes[0, 0]
m=ax.pcolormesh(WQG[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (traditional)')

ax = axes[1, 0]
m=ax.pcolormesh(WQG2[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (trad. with topo)')

ax = axes[2, 0]
m=ax.pcolormesh(W[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('observed omega')

ax = axes[0, 1]
m=ax.pcolormesh(WQvec[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (Q-vector)')

ax = axes[1, 1]
m=ax.pcolormesh(WQvec2[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (Q-vec. with topo)')

ax = axes[2, 1]
m=ax.pcolormesh(W[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('observed omega')
fig.colorbar(m, loc='b', cols=(1,2), length=1)

axes.format(abc='(a)', ylim=[100000, 10000])


#%% plot horizontal plane
import proplot as pplt

fontsize = 16
z = 25

fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(11, 11))

ax = axes[0,0]
m=ax.pcolormesh(WQG[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (traditional)')

ax = axes[1,0]
m=ax.pcolormesh(WQG2[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (trad. with topo)')

ax = axes[2,0]
m=ax.pcolormesh(W[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('observed omega')

ax = axes[0,1]
m=ax.pcolormesh(WQvec[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (Q-vector)')

ax = axes[1,1]
m=ax.pcolormesh(WQvec2[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (Q-vec. with topo)')

ax = axes[2,1]
m=ax.pcolormesh(W[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
ax.set_title('observed omega')
fig.colorbar(m, loc='b', cols=(1,2), length=1, label='')

axes.format(abc='(a)', ylim=[-90, 90])


#%% test oceanic case from Dr. Lei Liu
# convert data into netcdf
import numpy as np
import xarray as xr
from xgrads.xgrads import CtlDescriptor

ctl = CtlDescriptor(file='I:/Omega/OFES30_20011206_Qian/salt01.ctl')

xdef = np.linspace(142.1, 152.066546, 300)
ydef = np.linspace( 30.0,  39.966567, 300)
zdef = ctl.zdef.samples

coords = {'lat':ydef, 'lon':xdef, 'lev':zdef}

s   = xr.DataArray(S    , dims=['lat', 'lon', 'lev'], coords=coords, name='s')
t   = xr.DataArray(T    , dims=['lat', 'lon', 'lev'], coords=coords, name='t')
u   = xr.DataArray(U    , dims=['lat', 'lon', 'lev'], coords=coords, name='u')
v   = xr.DataArray(V    , dims=['lat', 'lon', 'lev'], coords=coords, name='v')
w   = xr.DataArray(W    , dims=['lat', 'lon', 'lev'], coords=coords, name='w')
rho = xr.DataArray(poden, dims=['lat', 'lon', 'lev'], coords=coords, name='rho')

QVec= xr.DataArray(Qvec[::-1] , dims=['lev', 'lat', 'lon'], name='Qvec' ,
                   coords={'lev':zdef[:91], 'lat':ydef, 'lon':xdef})
QGw = xr.DataArray(Diagnosed_w , dims=['lev', 'lat', 'lon'], name='QGw' ,
                   coords={'lev':zdef[:93], 'lat':ydef, 'lon':xdef})
N2  = xr.DataArray(n2[:,0], dims=['lev'], coords={'lev':zdef[:90]}, name='N2')

N22, _ = xr.broadcast(N2, s.lev)
N22 = N22.interpolate_na('lev', fill_value='extrapolate')

QVec2, _ = xr.broadcast(QVec, s.lev)
QVec2 = QVec2.interpolate_na('lev', fill_value='extrapolate')

ds = xr.merge([s.transpose('lev','lat','lon').astype('f4'),
               t.transpose('lev','lat','lon').astype('f4'),
               u.transpose('lev','lat','lon').astype('f4'),
               v.transpose('lev','lat','lon').astype('f4'),
               w.transpose('lev','lat','lon').astype('f4'),
               rho.transpose('lev','lat','lon').astype('f4'),
               QVec2.transpose('lev','lat','lon').astype('f4'),
               N22.astype('f4')])

zNew = np.linspace(2.5, 3002.5, 601)

ds.interp(lev=zNew).astype('f4').to_netcdf('I:/Omega/OFES30_20011206_Qian/data.nc')


#%% load converted netcdf
import numpy as np
import xarray as xr
from GeoApps.GridUtils import add_latlon_metrics
from GeoApps.DiagnosticMethods import Dynamics
from xinvert.xinvert.apps import invert_OmegaEquation

ds = xr.open_dataset('I:/Omega/OFES30_20011206_Qian/data.nc',
                     chunks={'lev':6}).astype('f4')

# dset = ds.sel({'lon':slice(143.1, 150.9), 'lat':slice(31.1, 38.9)})
ds['lev'] = -ds['lev'] # Reverse the z-coord. positive direction
                       # This is important for taking vertical derivatives.

#%% calculate QG forcings
from GeoApps.ConstUtils import omega

dset, grid = add_latlon_metrics(ds, dims={'Y':'lat', 'X':'lon'})

dyn = Dynamics(dset, grid=grid, arakawa='A')

u = ds.u / 100 # change unit from cm/s to m/s
v = ds.v / 100 # change unit from cm/s to m/s
w = ds.w / 100 # change unit from cm/s to m/s

dyn = Dynamics(ds, grid, arakawa='A')

b  = ds.rho * (-9.81/1023)
f  = 2*omega*np.sin(np.deg2rad(ds.lat))
N2 = b.mean(['lat','lon']).load().differentiate('lev').load()

########## traditional form of forcings ##########
bx, by = dyn.grad(b)
zx, zy = dyn.grad(dyn.curl(u, v))

adv_b = u*bx + v*by
adv_z = u*zx + v*zy

Ftrad = dyn.Laplacian(-adv_b) + adv_z.load().differentiate('lev')*f
Ftrad = (xr.where(np.isfinite(Ftrad), Ftrad, np.nan)).load()

############ Q-vector form of forcings ############
ux, uy = dyn.grad(u)
vx, vy = dyn.grad(v)

Qx = ux*bx + vx*by
Qy = uy*bx + vy*by

divQ  = -2 * dyn.divg((Qx, Qy), ['X', 'Y'])
FQvec = xr.where(np.isfinite(divQ), divQ, np.nan).load()

#%% maskout
WBC1 = xr.where(np.isnan(Ftrad), 0, w)
WBC2 = xr.where(np.isnan(FQvec), 0, w)

#%% invert
import time

params = {
    'BCs'      : ['fixed', 'fixed', 'extend'],
    'tolerance': 1e-9,
    'mxLoop'   : 500,
}

start = time.time()
W1 = invert_OmegaEquation(Ftrad, N2, dims=['lev', 'lat', 'lon'], params=params).load()
W2 = invert_OmegaEquation(FQvec, N2, dims=['lev', 'lat', 'lon'], params=params).load()
W1t= invert_OmegaEquation(Ftrad, N2, dims=['lev', 'lat', 'lon'], out=WBC1,
                          params=params).load()
W2t= invert_OmegaEquation(FQvec, N2, dims=['lev', 'lat', 'lon'], out=WBC2,
                          params=params).load()
elapsed = time.time() - start
print('time used: ', elapsed)


#%% plot and compare
import proplot as pplt

fontsize = 13
z = 10

fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(11, 11))

ax = axes[0,0]
m=ax.pcolormesh(W1[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (traditional)')

ax = axes[1,0]
m=ax.pcolormesh(W2[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (Q-vector)')

ax = axes[2,0]
m=ax.pcolormesh(w[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
ax.set_title('observed omega')

ax = axes[0,1]
m=ax.pcolormesh(W1t[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (traditional topo.)')

ax = axes[1,1]
m=ax.pcolormesh(W2t[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
ax.set_title('inverted QG omega (Q-vector topo.)')

ax = axes[2,1]
m=ax.pcolormesh(w[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
ax.set_title('observed omega')

fig.colorbar(m, loc='b', cols=(1,2), length=1, label='')

axes.format(abc='(a)')

#%% multi-grids
from xinvert.xinvert.core import invert_Omega_MG

start = time.time()
omegaMG, fs, os = invert_Omega_MG(force, N2,
                          dims=['lev', 'lat', 'lon'],
                          BCs=['fixed', 'fixed', 'extend'],
                          printInfo=True, debug=False, tolerance=1e-16)
elapsed = time.time() - start
print('time used: ', elapsed)







