# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
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
                  invert_RefState, \
                  invert_3DOcean, \
                  animate_iteration, cal_flow
                  
from .utils import loop_noncore

from .finitediffs import FiniteDiff, deriv, deriv2, padBCs

__version__ = "0.1.7"
