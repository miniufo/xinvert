# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global idealized
import xarray as xr
import numpy as np


ds = xr.open_dataset('D:/Data/ERAInterim/BKGState/OriginalData/Prs/T.nc',
                     chunks={'time':1, 'level':29})

levnew = np.linspace(1000, 100, 37)

T = ds.t[0,:, :121,:].interp(level=levnew)


#%% invert
import numpy as np
from xinvert.xinvert.core import invert_OmegaEquation
from GeoApps.ConstUtils import Rd, Cp, omega


xdef = T.longitude
ydef = T.latitude
zdef = T.level

zgrid, ygrid, xgrid = xr.broadcast(zdef, ydef, xdef)

F = 0.05*np.exp( -(zdef-300)**2/10000 -((ydef-30)**2+(xdef-120)**2)/100.0)
T = T - T + (5*np.exp(T.level/500)-35)
th = (T+273.15) * (1000/T.level)**(Rd/Cp)

dTHdz = np.log(th).differentiate('level') * Rd * T / th.level


omega = invert_OmegaEquation(F, dTHdz, dims=['level', 'latitude','longitude'],
                     BCs=['fixed', 'fixed', 'fixed'],
                     printInfo=True, debug=False, tolerance=1e-8)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np


fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=3, sharey=3)

fontsize = 16

axes.format(abc='(a)')

ax = axes[0]
p = ax.contourf(F[:,40,:], cmap='seismic',
                levels=np.linspace(-1e-4, 1e-4, 21))
ax.set_title('Forcing', fontsize=fontsize)
ax.colorbar(p, loc='b', label='', ticks=1e-5, length=0.895)

ax = axes[1]
p = ax.contourf(omega[:,40,:], levels=31, cmap='jet')
ax.set_title('QG omega', fontsize=fontsize)
ax.colorbar(p, loc='b', label='', length=0.895)
# ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360],
#                     fontsize=fontsize)



#%% test global data from Dr. Yuan Zhao
import xarray as xr
import numpy as np
from GeoApps.GridUtils import add_latlon_metrics


dset = xr.open_dataset('I:/Omega/var2.nc', decode_times=False,
                       chunks={'time':1})

ds, grid = add_latlon_metrics(dset, dims={'lat':'lat', 'lon':'lon'},
                              boundary={'lat':'extend','lon':'periodic'})

ds['lev'] = ds['lev'] * 100

#%%
from GeoApps.ConstUtils import Rd, Cp, omega
from GeoApps.DiagnosticMethods import Dynamics

dyn = Dynamics(ds, grid, arakawa='A')

p   = ds.lev
# Psfc= ds.psfc
f   = 2*omega*np.sin(np.deg2rad(ds.lat))
cos = np.cos(np.deg2rad(ds.lat))

T   = ds.T
U   = ds.U
V   = ds.V

th   = T * (100000 / p)**(Rd/Cp)
TH   = grid.average(th, ['Y','X'])
vor  = dyn.curl(U,V)
dTHdp= TH.differentiate('lev')
a    = Rd * T / p
S    = - (a/TH) * dTHdp

# vor3 = (V.differentiate('lon') - U.differentiate('lat')*cos)
#           )/(np.pi/180*6371200*cos)
# TH3 = (th2 * cos).mean(['lat','lon']) / cos.mean('lat')

Um   =   U.mean('lon');   Ua =   U - Um
Vm   =   V.mean('lon');   Va =   V - Vm
thm  =  th.mean('lon');  tha =  th - thm
vorm = vor.mean('lon'); vora = vor - vorm

grdthx, grdthy = dyn.grad(tha)
grdthmx,grdthmy= dyn.grad(th-th+thm)
grdvax, grdvay = dyn.grad(vora)
grdvmx, grdvmy = dyn.grad(vor-vor+vorm + f)


tmp1 = (Um * grdthx + Vm * grdthy + Va*grdthmy) / (-dTHdp)
tmp2 = (Um * grdvax + Vm * grdvay) + (Ua * grdvmx + Va * grdvmy)

term1 = dyn.Laplacian(tmp1)
term2 = tmp2.differentiate('lev') * f / S

F = (term1 + term2)

#%%
p3D = T-T+p

F2 = F.where(p<=Psfc)
O2 = xr.where(p3D<=Psfc, 0, ds.Omega).load()

#%% invert
from xinvert.xinvert.core import invert_OmegaEquation

omega = invert_OmegaEquation(F, S,
                              dims=['lev', 'lat', 'lon'],
                              BCs=['fixed', 'fixed', 'extend'],
                              printInfo=True, debug=False, tolerance=1e-16)

# omega2 = invert_OmegaEquation(F2, S,
#                              dims=['lev', 'lat', 'lon'],
#                              BCs=['fixed', 'fixed', 'extend'],
#                              printInfo=True, debug=False, tolerance=1e-16,
#                              icbc=O2)
omega3 = ds.Omega - ds.Omega.mean('lon')

#%%
dsRe = xr.merge([omega.rename('QGomega'), omega2.rename('QGomegaTopo'),
                 omega3.rename('omegaAnom')])

dsRe.to_netcdf('d:/QGOmega.nc')

#%%
import proplot as pplt

x = 80

fontsize = 16

fig, axes = pplt.subplots(nrows=3, ncols=1, figsize=(11, 10))

ax = axes[0]
m=ax.pcolormesh(omega[:, :, x], levels=np.linspace(-0.1, 0.1, 21))
ax.set_title('inverted QG omega')
ax.colorbar(m, loc='r', length=1)

ax = axes[1]
m=ax.pcolormesh(omega2[:, :, x], levels=np.linspace(-0.1, 0.1, 21))
ax.set_title('inverted QG omega (with topo)')
ax.colorbar(m, loc='r', length=1)

ax = axes[2]
m=ax.pcolormesh(omega3[:, :, x], levels=np.linspace(-0.1, 0.1, 21))
ax.set_title('observed omega')
ax.colorbar(m, loc='r', length=1)

axes.format(abc='(a)', ylim=[100000, 10000])


#%%
import proplot as pplt

fontsize = 16
z = 10

fig, axes = pplt.subplots(nrows=3, ncols=1, figsize=(9, 10))

ax = axes[0]
m=ax.pcolormesh(omega[z], levels=np.linspace(-0.1, 0.1, 21))
ax.set_title('inverted QG omega')
ax.colorbar(m, loc='r', length=1)

ax = axes[1]
m=ax.pcolormesh(omega2[z], levels=np.linspace(-0.1, 0.1, 21))
ax.set_title('inverted QG omega (with topo)')
ax.colorbar(m, loc='r', length=1)

ax = axes[2]
m=ax.pcolormesh(omega3[z], levels=np.linspace(-0.1, 0.1, 21))
ax.set_title('observed omega')
ax.colorbar(m, loc='r', length=1)

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

#%%

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


#%% invert
import numpy as np
import xarray as xr
from GeoApps.EOSMethods import EOSMethods
from GeoApps.GridUtils import add_latlon_metrics
from GeoApps.DiagnosticMethods import Dynamics
from xinvert.xinvert.core import invert_OmegaEquation

ds = xr.open_dataset('I:/Omega/OFES30_20011206_Qian/data.nc',
                     chunks={'lev':5}).astype('f4')

# dset = ds.sel({'lon':slice(143.1, 150.9), 'lat':slice(31.1, 38.9)})

#%% calculate Q vector forcing
dset, grid = add_latlon_metrics(ds, dims={'lat':'lat', 'lon':'lon'},
                                boundary={'Y':'fill', 'X':'fill'})

dyn = Dynamics(dset, grid=grid, arakawa='A')
eos = EOSMethods(dset, grid=grid, arakawa='A')

u = ds.u
v = ds.v

b = eos.cal_linear_buoyancy(ds.rho, rhoRef=1023)

ux, uy = dyn.grad(u)
vx, vy = dyn.grad(v)
bx, by = dyn.grad(b)

Qx = ux*bx + vx*by
Qy = uy*bx + vy*by

divQ = dyn.divg((Qx, Qy), ['X', 'Y'])
force = (xr.where(np.isfinite(divQ), divQ, np.nan) * 2).load()

N2 = b.mean(['lat','lon']).load().differentiate('lev')

#%%
import time

start = time.time()
omega = invert_OmegaEquation(force, N2,
                              dims=['lev', 'lat', 'lon'],
                              BCs=['fixed', 'fixed', 'extend'],
                              printInfo=True, debug=False, tolerance=1e-16).load()
elapsed = time.time() - start
print('time used: ', elapsed)

#%% multi-grids
from xinvert.xinvert.core import invert_Omega_MG

start = time.time()
omegaMG, fs, os = invert_Omega_MG(force, N2,
                          dims=['lev', 'lat', 'lon'],
                          BCs=['fixed', 'fixed', 'extend'],
                          printInfo=True, debug=False, tolerance=1e-16)
elapsed = time.time() - start
print('time used: ', elapsed)







