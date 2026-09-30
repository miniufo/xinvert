import sys
import xarray as xr

sys.path.append('/home/qianyk/OneDrive/Python/MyPack/xinvert/')
from xinvert import invert_Poisson

iParams = {
    'BCs'      : ['extend', 'periodic'],
    'mxLoop'   : 100000,
    'tolerance': 1e-15,
}

ds  = xr.open_dataset('/home/qianyk/OneDrive/Python/MyPack/xinvert/Data/Helmholtz_atmos.nc')
vor = ds.vor

sf = invert_Poisson(vor, dims=['lat','lon'], iParams=iParams)
