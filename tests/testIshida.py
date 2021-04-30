# -*- coding: utf-8 -*-
"""
Created on 2021.04.23

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% classical cases
import numpy as np
import xarray as xr
from xgrads.xgrads import open_CtlDataset

xnum = 251
ynum = 151
Lx = 1e7 # 10,000 km
Ly = 2 * np.pi * 1e6 # 6249 km
R = 0.0009 # Rayleigh friction (default 0.02)
depth = 200 # fluid depth 200
beta = 2.2e-11 # gradient of f 1e-13
undef = -9999

xdef = xr.DataArray(np.linspace(0, Lx, xnum), dims='xdef',
                    coords={'xdef':np.linspace(0, Lx, xnum)})
ydef = xr.DataArray(np.linspace(0, Ly, ynum), dims='ydef',
                    coords={'ydef':np.linspace(0, Ly, ynum)})

ygrid, xgrid = xr.broadcast(ydef, xdef)

tau_ideal = xr.DataArray((1.-np.cos(2.*np.pi * ygrid / Ly))/2.,
                         dims=['ydef','xdef'],
                         coords={'ydef':ydef, 'xdef':xdef})
curl_tau  = xr.DataArray(-np.pi * np.sin(2.*np.pi * ygrid / Ly)/Ly,
                         dims=['ydef','xdef'],
                         coords={'ydef':ydef, 'xdef':xdef})


# add topography
curl_tau[65:, 100:104] = undef
curl_tau[:75, 130:134] = undef


#%% invert
from xinvert.xinvert.core import invert_StommelWBC,invert_MunkWBC


h1, u1, v1 = invert_MunkWBC(curl_tau, dims=['ydef','xdef'],
                               BCs=['fixed', 'periodic'],
                               optArg=1.9, mxLoop=400,
                               cal_flow=True,
                               coords='cartesian',
                               beta=beta,
                               R=R,
                               depth=depth,
                               undef=undef,
                               debug=False)
h2, u2, v2 = invert_MunkWBC(curl_tau, dims=['ydef','xdef'],
                                BCs=['fixed', 'periodic'],
                                optArg=1.9, mxLoop=1000, tolerance=1e-14,
                                cal_flow=True,
                                coords='cartesian',
                                beta=beta,
                                R=R*20.,
                                depth=depth,
                                undef=undef,
                                debug=False)


#%% plot wind and streamfunction
import proplot as pplt
import xarray as xr
import numpy as np
from utils.PlotUtils import plot


fig, axes = pplt.subplots(nrows=2, ncols=1, figsize=(12,11),
                          sharex=3, sharey=3)

skip = 2
fontsize = 16

axes.format(abc=True, abcloc='l', abcstyle='(a)', grid=False)

ax = axes[0]
plot(h1.where(h1!=undef)/1e6*depth, ax=ax, ptype='both', cmap='greens',
     fmt='%1.0f', ylabel='y-coordinate (m)', cbarpos='horizontal',
     clevs=np.linspace(-90, 90, 37), xlabel='x-coordinate (m)')
ax.set_title('R = R0', fontsize=fontsize)
# p=ax.quiver(xgrid.values[::skip,::skip+2], ygrid.values[::skip,::skip+2],
#               u1.where(u1!=undef).values[::skip,::skip+2],
#               v1.where(v1!=undef).values[::skip,::skip+2],
#               width=0.0014, headwidth=8., headlength=12., scale=15)
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])

ax = axes[1]
plot(h2.where(h2!=undef)/1e6*depth, ax=ax, ptype='both', cmap='greens',
     fmt='%1.0f', ylabel='y-coordinate (m)', cbarpos='horizontal',
     clevs=np.linspace(-6, 6, 25), xlabel='x-coordinate (m)')
ax.set_title('R = 20*R0', fontsize=fontsize)
# ax.quiver(xgrid.values[::skip,::skip+2], ygrid.values[::skip,::skip+2],
#               u2.where(u2!=undef).values[::skip,::skip+2],
#               v2.where(v2!=undef).values[::skip,::skip+2],
#               width=0.0014, headwidth=8., headlength=12., scale=12)
              # headwidth=1, headlength=3, width=0.002)
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])



