# -*- coding: utf-8 -*-
"""
Created on 2022.11.02

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load sample data
import xarray as xr
import numpy as np

# not available now (TODO)

# R     = 287.04
# Cp    = 1004.88
# p0    = 1e5
# pT    = 1e4
# g     = 9.81
# ka    = 1.0/40.0/86400.0
# ks    = 1.0/ 4.0/86400.0
# kf    = 1.0
# sigb  = 0.7
# omega = 7.292e-5
# Re    = 6371200
# delTy = 60
# delThz= 10
# const = Re * 180 / np.pi


# lev = xr.DataArray(np.linspace(p0, pT, 37), dims='lev',
#                    coords={'lev':np.linspace(p0,pT,37)})
# lat = xr.DataArray(np.linspace(-90, 90, 181), dims='lat',
#                    coords={'lat':np.linspace(-90, 90, 181)})
# # lon = xr.DataArray(np.linspace(0, 359, 360), dims='lon',
# #                    coords={'lon':np.linspace(0, 359, 360)})

# latR = np.deg2rad(lat)
# sigma = lev/p0
# ratio = (sigma - sigb) / (1.0 - sigb)
# ratio = xr.where(ratio > 0, ratio, 0)
# Kv = kf * ratio
# KT = ka + (ks - ka) * ratio * np.cos(latR)**4.0
# f = 2 * omega * np.sin(latR)

# Teq = (315.0+sigma-sigma - delTy * np.sin(latR)**2.0 -
#        delThz * np.log(sigma) * np.cos(latR)**2.0) * sigma**(R/Cp)

# Teq = xr.where(Teq > 200, 200, Teq)

# T = Teq / KT

# dphidp = -R*T/lev
# phi = dphidp.cumsum('lev') * 1e4
# u = - phi.differentiate('lat') / const / f


