# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load sample data
import xarray as xr
import numpy as np
from xinvert import padBCs, FiniteDiff, deriv


def test_FD():
    dset = xr.open_dataset('./Data/Helmholtz_atmos.nc')
    
    assert dset.dims == {'time':2, 'lat':73, 'lon':144}
    
    T = dset.sf[0]
    
    # test padding
    T_Px = padBCs(T, dim='lon', BCs=('fixed','fixed'), fill=(1,1))
    T_Py = padBCs(T, dim='lat', BCs=('extend','fixed'), fill=(2,2))
    T_Py2= padBCs(T, dim='lat', BCs=('periodic','periodic'))
    T_Py3= padBCs(T, dim='lat', BCs=('reflect','extend'), fill=(3,3))
    
    assert (T_Px.isel({'lon':  0}) == 1).all()
    assert (T_Px.isel({'lon': -1}) == 1).all()
    
    assert (T_Py.isel({'lat':  0}) == T_Py.isel({'lat':  1})).all()
    assert (T_Py.isel({'lat': -1}) == 2).all()
    
    assert (T_Py2.isel({'lat':  1}) == T_Py2.isel({'lat': -1})).all()
    assert (T_Py2.isel({'lat': -2}) == T_Py2.isel({'lat':  0})).all()
    
    assert (T_Py3.isel({'lat':  0}) == T_Py3.isel({'lat':  2})).all()
    assert (T_Py3.isel({'lat': -1}) == T_Py3.isel({'lat': -2})).all()
    
    # test derivatives
    Tx1 = deriv(T, dim='lon', scheme='center')
    Tx2 = deriv(T, dim='lon', scheme='forward')
    Tx3 = deriv(T, dim='lon', scheme='backward')
    
    assert np.isclose(Tx1[1:-1, 1:-1], (Tx2 + Tx3)[1:-1, 1:-1]/2,
                      rtol=5e-5).all()
    
    # Txx = deriv2(T, dim='lon') not sure how to assert
    
    
    # test grad, curl, vort, Laplacian
    fd = FiniteDiff(dim_mapping={'T':'time', 'Y':'lat', 'X':'lon'},
                    BCs={'Y':'reflect', 'X':'periodic'},
                    coords='lat-lon')
    
    Ty, Tx = fd.grad(T, dims=['Y', 'X'])
    Tcurl  = fd.curl(Tx, Ty)
    Tdivg  = fd.divg([Tx, Ty], dims=['X', 'Y'])
    TLap   = fd.Laplacian(T, dims=['Y', 'X'])
    
    assert (np.abs(Tcurl) < 5e-11).all()
    
    # should look similar but not close as different schemes
    # assert np.isclose(Tdivg, TLap).all()


#%%
# import proplot as pplt
# import numpy as np

# fig, axes = pplt.subplots(nrows=5, ncols=3, figsize=(11, 12))

# fontsize = 14

# ax = axes[0,0]
# m=ax.contourf(Tx, levels=np.linspace(-35, 35, 15))
# ax.set_title('Tx by FiniteDiff', fontsize=fontsize)
# ax = axes[0,1]
# m=ax.contourf(Tx2, levels=np.linspace(-35, 35, 15))
# ax.colorbar(m, loc='r')
# ax.set_title('Tx2 by xgcm', fontsize=fontsize)
# ax = axes[0,2]
# m=ax.contourf(Tx-Tx2, levels=21)
# ax.colorbar(m, loc='r')
# ax.set_title('diff of Tx', fontsize=fontsize)

# ax = axes[1,0]
# m=ax.contourf(Ty, levels=np.linspace(-35, 35, 15))
# ax.set_title('Ty by FiniteDiff', fontsize=fontsize)
# ax = axes[1,1]
# m=ax.contourf(Ty2, levels=np.linspace(-35, 35, 15))
# ax.colorbar(m, loc='r')
# ax.set_title('Ty2 by xgcm', fontsize=fontsize)
# ax = axes[1,2]
# m=ax.contourf(Ty-Ty2, levels=21)
# ax.colorbar(m, loc='r')
# ax.set_title('diff of Ty', fontsize=fontsize)

# ax = axes[2,0]
# m=ax.contourf(Tcurl*1e6, levels=np.linspace(-4,4,17))
# ax.set_title('Tcurl by FiniteDiff', fontsize=fontsize)
# ax = axes[2,1]
# m=ax.contourf(Tcurl2*1e6, levels=np.linspace(-4,4,17))
# ax.colorbar(m, loc='r')
# ax.set_title('Tcurl2 by xgcm', fontsize=fontsize)
# ax = axes[2,2]
# m=ax.contourf(Tcurl-Tcurl2, levels=21)
# ax.colorbar(m, loc='r')
# ax.set_title('diff of Tcurl', fontsize=fontsize)

# ax = axes[3,0]
# m=ax.contourf(Tdivg*1e5, levels=np.linspace(-8,8,17))
# ax.set_title('Tdivg by FiniteDiff', fontsize=fontsize)
# ax = axes[3,1]
# m=ax.contourf(Tdivg2*1e5, levels=np.linspace(-8,8,17))
# ax.colorbar(m, loc='r')
# ax.set_title('Tdivg2 by xgcm', fontsize=fontsize)
# ax = axes[3,2]
# m=ax.contourf(Tdivg-Tdivg2, levels=21)
# ax.colorbar(m, loc='r')
# ax.set_title('diff of Tdivg', fontsize=fontsize)

# ax = axes[4,0]
# m=ax.contourf(TLap*1e5, levels=np.linspace(-10,10,21))
# ax.set_title('TLap by FiniteDiff', fontsize=fontsize)
# ax = axes[4,1]
# m=ax.contourf(TLap2*1e5, levels=np.linspace(-10,10,21))
# ax.colorbar(m, loc='r')
# ax.set_title('TLap2 by xgcm', fontsize=fontsize)
# ax = axes[4,2]
# m=ax.contourf(TLap-TLap2, levels=21)
# ax.colorbar(m, loc='r')
# ax.set_title('diff of TLap', fontsize=fontsize)

# axes.format(abc='(a)', ylabel='', xlabel='')



