# -*- coding: utf-8 -*-
from .core import invert_Poisson, invert_Poisson_animated, invert_GillMatsuno,\
                  invert_geostreamfunction, invert_Stommel, \
                  invert_StommelMunk, invert_Eliassen, invert_Omega_MG, \
                  invert_MultiGrid
from .utils import Laplacian, loop_noncore
__version__ = "0.1.0"
