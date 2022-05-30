# -*- coding: utf-8 -*-
"""
Created on 2020.12.09

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
from .core import inv_standard3D, inv_standard2D, inv_general2D,\
                  inv_general2D_bih
                  
from .apps import invert_Poisson, invert_GillMatsuno,\
                  invert_geostreamfunction, invert_Stommel, \
                  invert_StommelMunk, invert_Eliassen, invert_Omega_MG, \
                  invert_MultiGrid, invert_OmegaEquation, invert_Vortex_2D
                  
from .utils import loop_noncore

from .finitediffs import FiniteDiff

__version__ = "0.1.0"
