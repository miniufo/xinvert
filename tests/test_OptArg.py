# -*- coding: utf-8 -*-
"""
Created on 2020.12.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
import xarray as xr
import numpy as np

import xinvert.apps as apps

def test_optArg():
    omega = apps.__optimal_sor_parameter(
        (80, 100), (2.0, 1.0), ('fixed', 'fixed'))
    weights = np.array([1.0 / 4.0, 1.0])
    rho = np.sum(weights * [np.cos(np.pi / 79.0),
                            np.cos(np.pi / 99.0)]) / weights.sum()
    expected = 2.0 / (1.0 + np.sqrt(1.0 - rho*rho))

    assert omega == expected
    assert 1.0 <= omega < 2.0


def test_optarg_changes_with_grid_anisotropy():
    isotropic = apps.__optimal_sor_parameter(
        (30, 60), (1.0, 1.0), ('fixed', 'fixed'))
    anisotropic = apps.__optimal_sor_parameter(
        (30, 60), (4.0, 1.0), ('fixed', 'fixed'))

    assert anisotropic != isotropic


def test_optarg_respects_periodic_lowest_mode():
    omega = apps.__optimal_sor_parameter(
        (30, 60), (1.0, 1.0), ('fixed', 'periodic'))
    rho = (np.cos(np.pi / 29.0) + 1.0) / 2.0
    expected = 2.0 / (1.0 + np.sqrt(1.0 - rho*rho))
    assert omega == expected


#%% plot wind and streamfunction
# import proplot as pplt
# import xarray as xr
# import numpy as np

# fig, axes = pplt.subplots(nrows=1, ncols=2, figsize=(11, 5), sharex=3, sharey=3)

# fontsize = 16

# axes.format(abc='(a)', grid=False)

# ax = axes[0]
# p = ax.contourf(epsilon, cmap='jet')
# ax.set_title('epsilon', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='', ticks=0.2)

# ax = axes[1]
# p = ax.contourf(optArg, cmap='jet')
# ax.set_title('optArg', fontsize=fontsize)
# ax.colorbar(p, loc='b', label='', ticks=0.2)

