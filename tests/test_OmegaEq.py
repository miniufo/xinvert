# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test global data from Dr. Yuan Zhao
import xarray as xr
import numpy as np
from xinvert import invert_omega, FiniteDiff

def test_omega_atmos():
    ds = xr.open_dataset('./Data/atmos3D.nc', decode_times=False)
    
    # ds, grid = add_latlon_metrics(dset, dims={'Z':'LEV', 'Y':'lat', 'X':'lon'},
    #                               boundary={'X':'periodic', 'Y':'extend', 'Z':'extend'})
    
    ds['LEV'] = ds['LEV'] * 100
    
    # zonal running smooth of polar grid
    def smooth(v, gridpoint=13, lat=80):
        rolled = v.pad({'lon':(gridpoint,gridpoint)},mode='wrap')\
                  .rolling(lon=gridpoint, center=True, min_periods=1).mean()\
                  .isel(lon=slice(gridpoint, -gridpoint))
        return xr.where(np.abs(v-v+v.lat)>lat, rolled, v)
    
    omega = 7.292e-5
    Rd = 287.04
    Cp = 1004.88
    p   = ds.LEV
    Psfc= ds.psfc
    f   = 2*omega*np.sin(np.deg2rad(ds.lat))
    
    T = smooth(ds.T, lat=80)
    U = smooth(ds.U, lat=80)
    V = smooth(ds.V, lat=80)
    W = smooth(ds.Omega, lat=80)
    
    th   = T * (100000 / p)**(Rd/Cp)
    TH   = th.mean(['lat', 'lon'])
    dTHdp= TH.differentiate('LEV')
    RPiP = (Rd * T / p / TH)
    S    = - RPiP * dTHdp
    
    
    #### calculate forcings using FiniteDiff (which is part of xinvert) ###
    fd = FiniteDiff({'X':'lon', 'Y':'lat', 'Z':'LEV'},
                    BCs={'X':('periodic','periodic'),
                         'Y':('reflect','reflect'),
                         'Z':('extend','extend')}, fill=0, coords='lat-lon')
    
    vor  = fd.curl(U,V).load()
    _, tmp = xr.broadcast(vor, vor.mean('lon'))
    vor[:,0,:] = tmp[:,0,:]
    vor[:,-1,:] = tmp[:,-1,:]
    
    ########## traditional form of forcings ##########
    grdthx, grdthy = fd.grad(th)
    grdvrx, grdvry = fd.grad(vor)
    
    F1 = fd.Laplacian((U * grdthx + V * grdthy) * RPiP)
    F2 = ((U * grdvrx + V * grdvry) * f).differentiate('LEV')
    
    FAll = (F1 + F2)
    FAll = smooth(xr.where(np.isinf(FAll), np.nan, FAll), lat=83)
    
    ########### Q-vector form of forcings ###########
    ux, uy = fd.grad(U)
    vx, vy = fd.grad(V)
    
    Qx = - RPiP * (ux * grdthx + vx * grdthy)
    Qy = - RPiP * (uy * grdthx + vy * grdthy)
    
    FQvec = -2 * fd.divg((Qx, Qy), dims=['X', 'Y'])
    FQvec = smooth(xr.where(np.isinf(FQvec), np.nan, FQvec), gridpoint=17, lat=85)
    
    ##### prepare lower boundary for inversion ######
    p3D = T-T+p
    
    FAll  = FAll.where(p<=Psfc)
    FQvec = FQvec.where(p<=Psfc)
    WBC = xr.where(p3D<=Psfc, 0, W).load()
    
    ##### invert #####
    iParams = {
        'BCs'      : ['fixed', 'fixed', 'periodic'],
        'tolerance': 1e-16,
    }
    
    mParams = {'N2': S}
    
    WQG   = invert_omega(FAll, dims=['LEV', 'lat', 'lon'],
                         iParams=iParams, mParams=mParams)
    WQvec = invert_omega(FQvec, dims=['LEV', 'lat', 'lon'],
                         iParams=iParams, mParams=mParams)
    WQvTp = invert_omega(FQvec, dims=['LEV', 'lat', 'lon'], icbc=WBC,
                         iParams=iParams, mParams=mParams)
    
    assert np.isclose(  WQG.max(),  0.28392872)
    assert np.isclose(  WQG.min(), -0.32804008)
    assert np.isclose(WQvec.max(),  0.11733621)
    assert np.isclose(WQvec.min(), -0.37005570)
    assert np.isclose(WQvTp.max(),  2.10157466)
    assert np.isclose(WQvTp.min(), -0.69490683)

#%% plot cross section
# import proplot as pplt

# x = 80

# fontsize = 16

# fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(11, 11))

# ax = axes[0, 0]
# m=ax.pcolormesh(WQG[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (traditional)')

# ax = axes[1, 0]
# m=ax.pcolormesh(WQG2[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (trad. with topo)')

# ax = axes[2, 0]
# m=ax.pcolormesh(W[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('observed omega')

# ax = axes[0, 1]
# m=ax.pcolormesh(WQvec[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (Q-vector)')

# ax = axes[1, 1]
# m=ax.pcolormesh(WQvec2[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (Q-vec. with topo)')

# ax = axes[2, 1]
# m=ax.pcolormesh(W[:, :, x], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('observed omega')
# fig.colorbar(m, loc='b', cols=(1,2), length=1)

# axes.format(abc='(a)', ylim=[100000, 10000])


#%% plot horizontal plane
# import proplot as pplt

# fontsize = 16
# z = 25

# fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(11, 11))

# ax = axes[0,0]
# m=ax.pcolormesh(WQG[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (traditional)')

# ax = axes[1,0]
# m=ax.pcolormesh(WQG2[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (trad. with topo)')

# ax = axes[2,0]
# m=ax.pcolormesh(W[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('observed omega')

# ax = axes[0,1]
# m=ax.pcolormesh(WQvec[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (Q-vector)')

# ax = axes[1,1]
# m=ax.pcolormesh(WQvec2[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (Q-vec. with topo)')

# ax = axes[2,1]
# m=ax.pcolormesh(W[z], levels=np.linspace(-0.1, 0.1, 21), cmap='RdBu_r')
# ax.set_title('observed omega')
# fig.colorbar(m, loc='b', cols=(1,2), length=1, label='')

# axes.format(abc='(a)', ylim=[-90, 90])


#%% test oceanic case from Dr. Lei Liu
# convert data into netcdf
# import numpy as np
# import xarray as xr
# from xgrads.xgrads import CtlDescriptor

# ctl = CtlDescriptor(file='I:/Omega/OFES30_20011206_Qian/salt01.ctl')

# xdef = np.linspace(142.1, 152.066546, 300)
# ydef = np.linspace( 30.0,  39.966567, 300)
# zdef = ctl.zdef.samples

# coords = {'lat':ydef, 'lon':xdef, 'lev':zdef}

# s   = xr.DataArray(S    , dims=['lat', 'lon', 'lev'], coords=coords, name='s')
# t   = xr.DataArray(T    , dims=['lat', 'lon', 'lev'], coords=coords, name='t')
# u   = xr.DataArray(U    , dims=['lat', 'lon', 'lev'], coords=coords, name='u')
# v   = xr.DataArray(V    , dims=['lat', 'lon', 'lev'], coords=coords, name='v')
# w   = xr.DataArray(W    , dims=['lat', 'lon', 'lev'], coords=coords, name='w')
# rho = xr.DataArray(poden, dims=['lat', 'lon', 'lev'], coords=coords, name='rho')

# QVec= xr.DataArray(Qvec[::-1] , dims=['lev', 'lat', 'lon'], name='Qvec' ,
#                    coords={'lev':zdef[:91], 'lat':ydef, 'lon':xdef})
# QGw = xr.DataArray(Diagnosed_w , dims=['lev', 'lat', 'lon'], name='QGw' ,
#                    coords={'lev':zdef[:93], 'lat':ydef, 'lon':xdef})
# N2  = xr.DataArray(n2[:,0], dims=['lev'], coords={'lev':zdef[:90]}, name='N2')

# N22, _ = xr.broadcast(N2, s.lev)
# N22 = N22.interpolate_na('lev', fill_value='extrapolate')

# QVec2, _ = xr.broadcast(QVec, s.lev)
# QVec2 = QVec2.interpolate_na('lev', fill_value='extrapolate')

# ds = xr.merge([s.transpose('lev','lat','lon').astype('f4'),
#                t.transpose('lev','lat','lon').astype('f4'),
#                u.transpose('lev','lat','lon').astype('f4'),
#                v.transpose('lev','lat','lon').astype('f4'),
#                w.transpose('lev','lat','lon').astype('f4'),
#                rho.transpose('lev','lat','lon').astype('f4'),
#                QVec2.transpose('lev','lat','lon').astype('f4'),
#                N22.astype('f4')])

# zNew = np.linspace(2.5, 3002.5, 601)

# ds.interp(lev=zNew).astype('f4').to_netcdf('I:/Omega/OFES30_20011206_Qian/data.nc')



# to large dataset for this tests
# def test_omega_ocean():
#     ds = xr.open_dataset('I:/Omega/OFES30_20011206_Qian/data.nc',
#                          chunks={'lev':6}).astype('f4')
    
#     # dset = ds.sel({'lon':slice(143.1, 150.9), 'lat':slice(31.1, 38.9)})
#     ds['lev'] = -ds['lev'] # Reverse the z-coord. positive direction
#                            # This is important for taking vertical derivatives.
    
#     omega = 7.292e-5
    
#     ###### calculate QG forcings ######
#     u = ds.u / 100 # change unit from cm/s to m/s
#     v = ds.v / 100 # change unit from cm/s to m/s
#     w = ds.w / 100 # change unit from cm/s to m/s
    
#     b  = ds.rho * (-9.81/1023)
#     f  = 2*omega*np.sin(np.deg2rad(ds.lat))
#     N2 = b.mean(['lat','lon']).load().differentiate('lev').load()
    
    
#     #### calculate forcings using FiniteDiff (which is part of xinvert) ###
#     fd = FiniteDiff({'X':'lon', 'Y':'lat', 'Z':'lev'},
#                     BCs={'X':('extend','extend'),
#                          'Y':('extend','extend'),
#                          'Z':('extend','extend')}, fill=0, coords='lat-lon')
    
#     ########## traditional form of forcings ##########
#     bx, by = fd.grad(b)
#     zx, zy = fd.grad(fd.curl(u, v))
    
#     adv_b = u*bx + v*by
#     adv_z = u*zx + v*zy
    
#     Ftrad = fd.Laplacian(-adv_b) + adv_z.load().differentiate('lev')*f
#     Ftrad = (xr.where(np.isfinite(Ftrad), Ftrad, np.nan)).load()
    
#     ############ Q-vector form of forcings ############
#     ux, uy = fd.grad(u)
#     vx, vy = fd.grad(v)
    
#     Qx = ux*bx + vx*by
#     Qy = uy*bx + vy*by
    
#     divQ  = -2 * fd.divg((Qx, Qy), ['X', 'Y'])
#     FQvec = xr.where(np.isfinite(divQ), divQ, np.nan).load()
    
#     ###### maskout ######
#     WBC1 = xr.where(np.isnan(Ftrad), 0, w)
#     WBC2 = xr.where(np.isnan(FQvec), 0, w)

#     # invert
#     iParams = {
#         'BCs'      : ['fixed', 'fixed', 'extend'],
#         'tolerance': 1e-9,
#         'mxLoop'   : 500,
#     }
    
#     mParams = {'N2': N2}
    
#     W1 = invert_omega(Ftrad, dims=['lev', 'lat', 'lon'],
#                       iParams=iParams, mParams=mParams).load()
#     W2 = invert_omega(FQvec, dims=['lev', 'lat', 'lon'],
#                       iParams=iParams, mParams=mParams).load()
#     W1t= invert_omega(Ftrad, dims=['lev', 'lat', 'lon'],
#                       iParams=iParams, mParams=mParams, icbc=WBC1).load()
#     W2t= invert_omega(FQvec, dims=['lev', 'lat', 'lon'],
#                       iParams=iParams, mParams=mParams, icbc=WBC2).load()
    
#     assert np.isclose(W1 , W2 ).all()
#     assert np.isclose(W1t, W2t).all()


#%% plot and compare
# import proplot as pplt

# fontsize = 13
# z = 10

# fig, axes = pplt.subplots(nrows=3, ncols=2, figsize=(11, 11))

# ax = axes[0,0]
# m=ax.pcolormesh(W1[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (traditional)')

# ax = axes[1,0]
# m=ax.pcolormesh(W2[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (Q-vector)')

# ax = axes[2,0]
# m=ax.pcolormesh(w[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
# ax.set_title('observed omega')

# ax = axes[0,1]
# m=ax.pcolormesh(W1t[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (traditional topo.)')

# ax = axes[1,1]
# m=ax.pcolormesh(W2t[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
# ax.set_title('inverted QG omega (Q-vector topo.)')

# ax = axes[2,1]
# m=ax.pcolormesh(w[z]*1e4, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
# ax.set_title('observed omega')

# fig.colorbar(m, loc='b', cols=(1,2), length=1, label='')

# axes.format(abc='(a)')


#%% multi-grids
# from xinvert.xinvert.core import invert_Omega_MG

# start = time.time()
# omegaMG, fs, os = invert_Omega_MG(force, N2,
#                           dims=['lev', 'lat', 'lon'],
#                           BCs=['fixed', 'fixed', 'extend'],
#                           printInfo=True, debug=False, tolerance=1e-16)
# elapsed = time.time() - start
# print('time used: ', elapsed)







