# -*- coding: utf-8 -*-
"""
Created on 2020.12.10

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% combine data and grids (TODO)
# import xarray as xr
# import numpy as np

# ds  = xr.open_dataset('D:/Data/LLC4320/Face_4.nc',
#                       chunks={'time':1})
# dsG = xr.open_dataset('D:/Data/LLC4320/Face_4_grid.nc',
#                       chunks={'time':1})

# ds = ds.rename({'i':'XC', 'j':'YC'}).astype('float32')

# ds['XC'] = dsG['XC'][0].values
# ds['YC'] = dsG['YC'][:,0].values

# ds.to_netcdf('D:/Data/LLC4320/Face_4_all.nc')

# #%% test global 1
# import xarray as xr
# import numpy as np
# from xinvert.xinvert import invert_Poisson, FiniteDiff


# ds = xr.open_dataset('D:/Data/LLC4320/Face_4_all.nc', chunks={'time':1})

# fd = FiniteDiff({'X':'XC', 'Y':'YC', 'T':'time'},
#                 BCs={'X':'fixed', 'Y':'fixed'}, fill=0, coords='lat-lon')

# vor = fd.curl(ds.U[0], ds.V[0] , ['X', 'Y'])

# #%%
# iParams = {
#     'BCs'      : ['fixed', 'fixed'],
#     'undef'    : np.nan,
#     'mxLoop'   : 5000,
#     'tolerance': 1e-16,
#     'optArg'   : None,
# }

# sf = invert_Poisson(vor, dims=['YC','XC'], coords='lat-lon', iParams=iParams)

# iParams = {
#     'BCs'      : ['fixed', 'fixed'],
#     'undef'    : np.nan,
#     'mxLoop'   : 30000,
#     'tolerance': 1e-16,
#     'optArg'   : None,
# }

# sf2 = invert_Poisson(vor, dims=['YC','XC'], coords='lat-lon', iParams=iParams)


# #%% plot wind and streamfunction
# import proplot as pplt
# import xarray as xr
# import numpy as np

# fig, axes = pplt.subplots(nrows=2, ncols=2, figsize=(13, 10), sharex=3,
#                           sharey=3, proj='cyl', proj_kw={'central_longitude':180})

# fontsize = 14

# ax = axes[0, 0]
# p = ax.pcolormesh(vor, cmap='seismic', levels=np.linspace(-7e-5, 7e-5, 28),
#                   rasterized=True)
# ax.set_title('vorticity', fontsize=fontsize)
# ax.colorbar(p, loc='r', label='', length=0.9, ticks=1e-5)

# ax = axes[0, 1]
# p = ax.pcolormesh(sf, levels=np.linspace(-100000, 100000, 30),
#                   cmap='jet', rasterized=True)
# ax.set_title('streamfunction iter5000', fontsize=fontsize)
# ax.colorbar(p, loc='r', label='', length=0.9, ticks=20000)

# ax = axes[1, 1]
# p = ax.pcolormesh(sf2, cmap='jet', levels=np.linspace(-100000, 100000, 30),
#                   rasterized=True)
# ax.set_title('streamfunction iter30000', fontsize=fontsize)
# ax.colorbar(p, loc='r', label='', length=0.9, ticks=20000)

# ax = axes[1, 0]
# p = ax.pcolormesh((sf-sf2), levels=np.linspace(-500, 500, 30),
#                   cmap='seismic', rasterized=True)
# ax.set_title('diff. = iter5000 - iter30000', fontsize=fontsize)
# ax.colorbar(p, loc='r', label='', length=0.9, ticks=100)

# axes.format(abc='(a)', coast=True, lonlines=10, latlines=10,
#             lonlim=[52, 142], latlim=[-57, 14], lonlabels='b', latlabels='l')



# #%% test global 1
# import xarray as xr
# import numpy as np
# from xinvert.xinvert import invert_Poisson





# #%%
# import xarray as xr
# import xrft
# import numpy as np
# from xinvert.xinvert import FiniteDiff


# def vor2sf(vor, dims):
#     J, I = vor.shape
    
#     JHalf = int(np.floor(J/2))
#     IHalf = int(np.floor(I/2))
    
#     delJ = vor[dims[0]].diff(dims[0])[0] * 6371200
#     delI = vor[dims[1]].diff(dims[1])[0] * 6371200 * np.cos(np.deg2rad(vor[dims[0]]))
    
#     freq_dims = ['freq_' + dim for dim in dims]
    
#     spec = xrft.fft(vor, dim=dims, true_phase=True, true_amplitude=True)
    
#     k2 = ((np.sin(spec['freq_'+dims[0]] * 2.5)**2) / 2.5**2 +
#           (np.sin(spec['freq_'+dims[1]] * 2.5)**2) / 2.5**2)
    
#     k22 = -2 * ((np.cos(spec['freq_'+dims[0]] * 2.5) - 1) / 2.5**2 +
#                 (np.cos(spec['freq_'+dims[1]] * 2.5) - 1) / 2.5**2)
    
#     k222 = (spec['freq_'+dims[0]]/2.5) ** 2 + (spec['freq_'+dims[1]]/2.5) ** 2
    
#     assert(  k2[JHalf, IHalf] == 0)
#     assert( k22[JHalf, IHalf] == 0)
#     assert(k222[JHalf, IHalf] == 0)
    
#     oldDC = spec[JHalf, IHalf] # keep mean from zero-denominator problem
#     spec = - spec / k222 # solve Poisson equation in spectral space
#     spec[JHalf, IHalf] = oldDC # assign mean
    
#     sf = xrft.ifft(spec, dim=freq_dims, true_phase=True, true_amplitude=True)
    
#     for dim in dims:
#         sf[dim] = vor[dim].values # asign original coordinates
    
#     return sf.real

# def vor_sf(vor, k2):
#     # forward FFT
#     hat = np.fft.fft2(vor)
    
#     # modify 0-wavenumber (mean) to avoid zero-denominator problem
#     oldDC = hat[0,0]
#     hat = hat / k2
#     hat[0,0] = oldDC
    
#     # inverse FFT
#     sf = np.real(np.fft.ifft2(hat))
    
#     return sf


# ### spherical coordinates
# def spher_poisson_fft_v0(f, r=6371200, lat=1, lon=1):
#     """
#         solve Poisson equation using FFT-based method on the spherical coordinate
#         f is the rhs,such as vorticity (oemga_r) or divergence (divXY) on the spherical surface
#         lon and lat are the spherical coordiante (e.g., longitude and latitude)     
#        The method proposed by Sunaina, Mansi Butola and Kedar Khare in "Calculating Numerical Derivatives using Fourier Transform: some pitfalls
#         and how to avoid them" (Sunaina et al 2018 Eur. J. Phys. 39 065806, doi:10.1088/1361-6404/aadda6) is used. 
#        The algorithm is the second-order accurracy. 
#     """
#     if len(lat) != np.shape(f)[0]:
#         raise Exception("The dimension of latitude is not correct!")
#     if len(lon) != np.shape(f)[1]:
#         raise Exception("The dimension of longitude is not correct!")

#     lon = np.deg2rad(lon)
#     lat = np.deg2rad(lat)

#     dlon = (lon[1] - lon[0])
#     dlat = (lat[1] - lat[0])
    
#     N_lon = len(lon)
#     N_lat = len(lat)
#     # get the wavenumbers -- we need these to be physical, so divide by dx
#     klon = 2 * np.pi * np.fft.fftfreq(N_lon, d=1)
#     klat = 2 * np.pi * np.fft.fftfreq(N_lat, d=1)
    
#     # k2 = np.sin(klat[:, None])**2 / dlat**2 + np.sin(
#     #     klon[None, :])**2 / dlon**2 / np.cos(lat[:, None])**2 + 1j * np.sin(
#     #         klat[:, None]) / dlat * np.tan(lat)[:, None]
    
#     # k2 = klat[:, None]**2 + klon[None, :]**2 / np.cos(
#     #      lat[:, None])**2 - 1j * klon[None, :] * np.tan(lat)[:, None]
#     k2 = -2*((np.cos(klat[:, None]) - 1.0) / dlat**2 +
#              (np.cos(klon[None, :]) - 1.0) / dlon**2 / np.cos(lat[:, None])**2)+\
#              2.0*1j * np.sin(klon[None, :]/2.0) / dlat * np.tan(lat)[:, None]
#     #     k2 = np.sin( kx[None,:])**2 / dx**2 +np.sin(ky[:,None])**2  / dy**2

#     k2 = np.where(np.abs(k2)<=1e-20, 1e-10, k2)
    
#     maskF = ~np.isnan(f)
#     f = np.where(maskF, f, 0)
#     F = r**2 * np.fft.fft2(f)
#     oldDC = F[0, 0]

#     F = -F / k2
#     F = np.where(k2 > 1e-20, F, 0)  # set values for k2 close to 0
#     F[0, 0] = oldDC  #
#     # # transform back to real space
#     phi = np.real(np.fft.ifft2(F))
#     #     print(np.shape(F))
#     phi = np.where(maskF, phi, np.nan)
#     return phi


# #%% spherical earth
# import xarray as xr
# import numpy as np
# from xinvert.xinvert import invert_Poisson, FiniteDiff

# ds = xr.open_dataset('./xinvert/Data/Helmholtz_atmos.nc')

# vor = ds.vor[0,1:-1].rename('vorticity')

# iParams = {
#     'BCs'      : ['fixed', 'periodic'],
#     'undef'    : np.nan,
#     'mxLoop'   : 100000,
#     'tolerance': 1e-16,
#     'optArg'   : None,
# }

# sf = invert_Poisson(vor, dims=['lat','lon'], iParams=iParams)


# #%%
# # J, I = vor.shape
# # R = 6371200 # Earth radius
# # dx = (np.deg2rad(vor.lon.diff('lon')[0]) * np.cos(np.deg2rad(vor.lat))).values * R # unit m
# # dy = np.deg2rad(vor.lat.diff('lat')[0]).values * R # unit m

# # # set polar dx to small values
# # dx = xr.where(dx<=0, 1e-3, dx)

# # # get the wavenumbers k2
# # kx = 2 * np.pi * np.fft.fftfreq(I)
# # ky = 2 * np.pi * np.fft.fftfreq(J)

# # k2 = - (np.sin(ky[:, None])**2 / dy**2 + \
# #         np.sin(kx[None, :])**2 / dx[:, None]**2) #+ \
# #         #1j * np.sin(ky[:, None]) / dy * np.tan(np.deg2rad(vor.lat.values))[:, None] / R)

# # # k2 = -(ky[:, None]**2 + kx[None, :]**2)

# # # k2 = np.where(np.abs(k2)<=1e-2, -1e-2, k2)
# # k2[0,0] = -1e-20

# # k2 = xr.DataArray(k2, dims=['freq_lat', 'freq_lon'],
# #                   coords={'freq_lat':ky, 'freq_lon':kx})

# # sf2 = xr.apply_ufunc(vor_sf, vor, k2,
# #                      input_core_dims=[['lat', 'lon'], ['freq_lat', 'freq_lon']],
# #                      output_core_dims=[['lat', 'lon']])


# sf3 = spher_poisson_fft_v0(vor, r=6371200, lat=vor.lat.values, lon=vor.lon.values)

# sf3 = xr.DataArray(sf3, dims=['lat', 'lon'], coords={'lat':vor.lat, 'lon':vor.lon})

# #%%
# fd = FiniteDiff({'X':'lon', 'Y':'lat', 'T':'time'},
#                 BCs={'X':'periodic', 'Y':'extend'}, fill=0, coords='lat-lon')

# vorR = fd.Laplacian(sf, dims=['Y','X'])
# vor2 = fd.Laplacian(sf2, dims=['Y', 'X'])


# #%% cartesian
# J, I = vor.shape

# vor2 = vor.copy()
# vor2['lon'] = np.arange(I) * 100
# vor2['lat'] = np.arange(J) * 100

# iParams = {
#     'BCs'      : ['fixed', 'periodic'],
#     'undef'    : np.nan,
#     'mxLoop'   : 100000,
#     'tolerance': 1e-16,
#     'optArg'   : None,
# }

# sfc = invert_Poisson(vor2, dims=['lat', 'lon'], coords='cartesian', iParams=iParams)

# def cartesian_poisson_fft_v0(f, dx=100, dy=100):
#     """
#         phi=cartesian_poisson_fft_v0(f,dx,dy)
        
#         solve Poisson equation in cartesian coordinates using FFT-based method
#         f is the rhs 
#         x and y is the Cartesian coordinate (regular grid)
        
#        The method proposed by Sunaina, Mansi Butola and Kedar Khare in "Calculating Numerical Derivatives using Fourier Transform: some pitfalls
#     and how to avoid them" (Sunaina et al 2018 Eur. J. Phys. 39 065806, doi:10.1088/1361-6404/aadda6) is used. 
#     The algorithm is the second-order accurracy. 
#     """
#     Ny, Nx = np.shape(f)
#     maskF = ~np.isnan(f)
#     f = np.where(maskF, f, 0)
#     F = np.fft.fft2(f)
#     oldDC = F[0, 0]
#     # get the wavenumbers -- we need these to be physical, so divide by dx
#     kx = 2 * np.pi * np.fft.fftfreq(Nx, d=1)
#     ky = 2 * np.pi * np.fft.fftfreq(Ny, d=1)
#     k2 = -2* ((np.cos(kx[None,:]) - 1.0) / dx**2 +
#               (np.cos(ky[:,None]) - 1.0) / dy**2)
#     # k2 = kx[None, :]**2 + ky[:,None]**2
#     F = -F / k2
#     F = np.where(k2 > 1e-20, F, 0)  # set values for k2 close to 0
#     F[0, 0] = oldDC  #
#     # # transform back to real space
#     phi = np.real(np.fft.ifft2(F))
#     #     print(np.shape(F))
#     phi = np.where(maskF, phi, np.nan)
#     return phi

# sfc2 = cartesian_poisson_fft_v0(vor2)
# sfc2 = xr.DataArray(sfc2, dims=['lat', 'lon'], coords={'lat':vor2['lat'],
#                                                        'lon':vor2['lon']})

# fd = FiniteDiff({'X':'lon', 'Y':'lat', 'T':'time'},
#                 BCs={'X':'periodic', 'Y':'extend'}, fill=0, coords='cartesian')

# vorR = fd.Laplacian(sfc, dims=['Y','X'])
# vor2 = fd.Laplacian(sfc2, dims=['Y', 'X'])

# #%%
# sfR = xr.open_dataset('d:/sfRef.nc').sf2

# ds = xr.open_dataset('D:/Face_4.nc', chunks={'time':1})

# fd = FiniteDiff({'X':'i', 'Y':'j', 'T':'time'},
#                 BCs={'X':'extend', 'Y':'extend'}, fill=0, coords='cartesian')

# vor = fd.curl(ds.U[0], ds.V[0] , ['X', 'Y'])


# sf_bias = vor2sf(vor.fillna(0), dims=['j', 'i'])

# sf_land = sf_bias.where(np.isnan(vor))

# vor_land = fd.Laplacian(sf_land.fillna(0), dims=['Y', 'X'], BCs='extend').chunk(vor.chunks)

# sf_corr = vor2sf(vor_land, dims=['j','i'])

# sf = sf_bias - sf_corr
# sf = sf - sf.mean()



