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
                  invert_MultiGrid, invert_OmegaEquation
                  
from .utils import Laplacian, loop_noncore

__version__ = "0.1.0"
