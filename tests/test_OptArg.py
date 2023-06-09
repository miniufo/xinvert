# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
import xarray as xr
import numpy as np

def test_optArg():
    nx, ny = 100, 100
    
    gridx = xr.DataArray(np.arange(nx), dims=['X'], coords={'X': np.arange(nx)})
    gridy = xr.DataArray(np.arange(ny), dims=['Y'], coords={'Y': np.arange(ny)})
    
    gy, gx = xr.broadcast(gridy, gridx)
    
    epsilon = np.sin(np.pi/(2.*gx+2.))**2. + np.sin(np.pi/(2.*gy+2.))**2.
    
    optArg = 2./(1.+np.sqrt(epsilon*(2.-epsilon)))

    assert np.logical_and(optArg >=1, optArg <=2).all()


#%% plot wind and streamfunction
# import proplot as pplt
# import xarray as xr
# import numpy as np

# fig, axes = pplt.subplots(nrows=1, ncols=2, figsize=(11, 5), sharex=3, sharey=3)

# fontsize = 16

# axes.format(abc='(a)', grid=False)

# ax = axes[0]
# p = ax.contourf(epsilon, cmap='jet')
# ax.set_title('epsilon', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='', ticks=0.2)

# ax = axes[1]
# p = ax.contourf(optArg, cmap='jet')
# ax.set_title('optArg', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='', ticks=0.2)

