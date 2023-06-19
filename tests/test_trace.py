# -*- coding: utf-8 -*-
"""
Created on 2020.12.25

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% test trace methods
import numpy as np
from xinvert.numbas import trace, traceCyclic

def test_trace():
    a = np.array([2., 2., 0.])
    b = np.array([3., 3., 3., 3.])
    c = np.array([0., 1., 1.])
    d = np.array([5., 9., 9., 8.])
    
    res = trace(a, b, c, d)
    
    expect = np.array([
        1.6666666666666667,
        1.5238095238095233,
        1.0952380952380958,
        2.6666666666666665])
    
    assert np.isclose(res,  expect).all()
    
    res = traceCyclic(a, b, c, d, 5.2, 3.9)
    
    expect = np.array([
        2.35815602836879370,
        0.49316109422492393,
        2.80420466058763960,
       -0.39893617021276560])
   
    assert np.isclose(res,  expect).all()


