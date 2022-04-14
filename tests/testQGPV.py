# -*- coding: utf-8 -*-
"""
Created on 2022.04.12

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% idealized test case
import xarray as xr
import numpy as np
from GeoApps.ConstUtils import Rd, Cp, omega


ds = xr.open_dataset('xinvert/Data/atmos3D.nc', decode_times=False)

# select mid-latitude range as background state
dset = ds.sel({'lon':slice(150, 200), 'lat':slice(50, 40)})
dset['LEV'] = dset['LEV'] * 100 # convert hPa to Pa

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

xc, zc = len(P), 201

ys  = np.linspace(-1000000, 1000000, zc)

zdef = P
ydef = xr.DataArray(ys , dims='Y', coords={'Y':ys })
S    = S.rolling(LEV=5, center=True, min_periods=1).mean()

#%% case 2
S[:56] = 1E-5
S[56:] = 6E-5
# TH = S.cumsum('LEV') * zdef.diff('LEV')

#%% specifying PV
# q  = np.exp(-(zdef-40000)**2/4e7) * np.exp(-(ydef)**2/7e10) * 1.2
# q[:, 0] = np.nan
# q[:, -1] = np.nan
# q[0, :] = np.nan
# q[1, :] = np.nan

q = zdef * ydef - zdef * ydef

# q = xr.where(q.LEV<(20000*np.exp(-ydef**2/1.5e11)+25000), 5, 0)
# q = xr.where(q.LEV>(10000*np.exp(-ydef**2/1.5e11)+25000), q, 0)
amplit = 1*np.exp(-ydef**2/1e11)
zscale = 5e7+np.exp(-ydef**2/2e11)*3e7
q = np.exp(-(zdef-(30000+np.exp(-ydef**2/1e11)*10000))**2/zscale) * amplit
# q = xr.where(q>1, 1, q)


#%% invert
from xinvert.xinvert.apps import invert_QGPV_2D

Ha = invert_QGPV_2D(q, S, f0=f, dims=['LEV','Y'], BCs=['fixed', 'extend'],
                    coords='Cartesian', tolerance=1e-12)

#%%
Ua  = -Ha.differentiate('Y') / f
THa =  Ha.differentiate('LEV') / -RPiP
Ta  = THa * (P / 100000.0)**(Rd/Cp)

THo = TH2 + THa
To  = T  + Ta
Ho  = H  + Ha

#%% plot
import proplot as pplt

fontsize = 16

fig, axes = pplt.subplots(figsize=(9,6))

ax = axes[0]

tmp = q.copy()
tmp['LEV'] = tmp['LEV'] / 100
tmp['Y'] = tmp['Y'] / 1000
m=ax.contourf(tmp.where(tmp>0.2), levels=np.linspace(0,2.4,13), cmap='gray_r')
ax.colorbar(m, loc='b', label='')
tmp = THo.copy()
tmp['LEV'] = tmp['LEV'] / 100
tmp['Y'] = tmp['Y'] / 1000
ax.contour(tmp, levels=31, colors='k', lw=1)
tmp = Ua.copy()
tmp['LEV'] = tmp['LEV'] / 100
tmp['Y'] = tmp['Y'] / 1000
ax.contour(tmp, levels=15, cmap='RdYlBu_r', lw=2.6)
tmp = To.rolling(Y=51, center=True, min_periods=1).mean()
tmp = tmp.where(tmp.LEV>=(5000*np.exp(-(ydef)**2/1.5e11)+24000))
tmp['LEV'] = tmp['LEV'] / 100
tmp['Y'] = tmp['Y'] / 1000
ax.contour(tmp, levels=[221.9], lw=5)
ax.set_yscale('log')
ax.set_yticks([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
ax.set_xticks(np.linspace(-750, 750, 9))
ax.set_ylim([1000, 100])
ax.set_xlim([-900, 900])
ax.set_title('PV inversion for a symmetric vortex', fontsize=fontsize)

ax.dualy('height', label='height (km)', fontsize=fontsize-2, grid=False)

axes.format(abc='(a)')





