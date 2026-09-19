# -*- coding: utf-8 -*-
"""
xinvert: invert geophysical fluid dynamics problems using SOR iteration.

Built on xarray and numba, this package solves classical elliptic PDEs
(Poisson, Gill-Matsuno, Stommel-Munk, QG-omega, Eliassen, PV inversion,
reference-state, etc.) via successive over-relaxation with spatially-varying
coefficients and dask-enabled parallel computation.
"""
from .core import inv_standard3D, \
                  inv_standard2D, inv_standard2D_test, \
                  inv_general3D, \
                  inv_general2D,\
                  inv_general2D_bih
                  
from .apps import invert_Poisson, \
                  invert_GillMatsuno, invert_GillMatsuno_test, \
                  invert_geostrophic, \
                  invert_Stommel, invert_Stommel_test, \
                  invert_StommelMunk, \
                  invert_StommelArons, \
                  invert_Eliassen, \
                  invert_BrethertonHaidvogel, \
                  invert_Fofonoff, \
                  invert_omega, \
                  invert_PV2D, \
                  invert_RefStateSWM, \
                  invert_GeoAdjustment, \
                  invert_RefState, \
                  invert_3DOcean, \
                  animate_iteration, cal_flow
                  
from .utils import loop_noncore

from .finitediffs import FiniteDiff, deriv, deriv2, padBCs

__version__ = "0.2.2"
