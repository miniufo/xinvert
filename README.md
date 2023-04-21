# xinvert

[![DOI](https://zenodo.org/badge/323045845.svg)](https://zenodo.org/badge/latestdoi/323045845)
![GitHub](https://img.shields.io/github/license/miniufo/xinvert)
[![Documentation Status](https://readthedocs.org/projects/xinvert/badge/?version=latest)](https://xinvert.readthedocs.io/en/latest/?badge=latest)

![animate plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/animateConverge.gif)


## 1. Introduction
Researches on meteorology and oceanography usually encounter [inversion problems](https://doi.org/10.1017/CBO9780511629570) that need to be solved numerically.  One of the classical inversion problem is to solve Poisson equation for a streamfunction $\psi$ given the vertical component of vorticity $\zeta$ and proper boundary conditions.

> $$\nabla^2\psi=\zeta$$

Nowadays [`xarray`](http://xarray.pydata.org/en/stable/) becomes a popular data structure commonly used in [Big Data Geoscience](https://pangeo.io/).  Since the whole 4D data, as well as the coordinate information, are all combined into [`xarray`](http://xarray.pydata.org/en/stable/), solving the inversion problem become quite straightforward and the only input would be just one [`xarray.DataArray`](http://xarray.pydata.org/en/stable/) of vorticity.  Inversion on the spherical earth, like some meteorological problems, could utilize the spherical harmonics like [windspharm](https://github.com/ajdawson/windspharm), which would be more efficient using FFT than SOR used here.  However, in the case of ocean, SOR method is definitely a better choice in the presence of irregular land/sea mask.

More importantly, this could be generalized into a numerical solver for elliptical equation using [SOR](https://mathworld.wolfram.com/SuccessiveOverrelaxationMethod.html) method, with spatially-varying coefficients.  Various popular inversion problems in geofluid dynamics will be illustrated as examples.

One problem with SOR is that the speed of iteration using **explicit loops in Python** will be **e-x-t-r-e-m-e-l-y ... s-l-o-w**!  A very suitable solution here is to use [`numba`](https://numba.pydata.org/).  We may try our best to speed things up using more hardwares (possibly GPU).

Classical problems include Gill-Matsuno model, Stommel-Munk model, QG omega model, PV inversion model, Swayer-Eliassen balance model...  A complete list of the classical inversion problems can be found at [this notebook](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/Introduction.ipynb).

Why `xinvert`?

- **Thinking and coding in equations:** User APIs are very close to the equations: unknowns are on the LHS of `=`, whereas the known forcings are on its RHS;
- **Genearlize all the steady-state problems:** All the known steady-state problems in geophysical fluid dynamics can be easily adapted to fit the solvers;
- **Very short parameter list:** Passing a single `xarray` forcing is enough for the inversion.  Coordinates information is already encapsulated.
- **Flexible model parameters:** Model paramters can be either a constant, or varying with a specific dimension (like Coriolis $f$), or fully varying with space and time, due to the use of `xarray`'s broadcasting capability;
- **Parallel inverting:** The use of `xarray`, and thus `dask` allow parallel inverting, which is almost transparent to the user;
- **Pure Python code for C-code speed:** The use of `numba` allow pure python code in this package but native speed;

---
## 2. How to install
**Requirements**
`xinvert` is developed under the environment with `xarray` (=version 0.15.0), `dask` (=version 2.11.0), `numpy` (=version 1.15.4), and `numba` (=version 0.51.2).  Older versions of these packages are not well tested.

**Install via pip** (not yet)
```
pip install xinvert
```

**Install from github**
```
git clone https://github.com/miniufo/xinvert.git
cd xinvert
python setup.py install
```


---
## 3. Example: Helmholtz decomposition
This is a classical problem in both meteorology and oceanography that a vector flow field can be deomposed into rotational and divergent parts, where rotational and divergent parts are represented by the streamfunction and velocity potential.  Given vorticity (vor) and divergence (div) as the forcing functions, one can invert the streamfunction and velocity potential directly.

### 3.1 Atmospheric demonstration
Here is an atmospheric demonstration with no lateral boundaries:
```python
import xarray as xr
from xinvert import invert_Poisson

dset = xr.open_dataset('data.nc')

vor = dset.vor

# specify boundary conditions in invert parameters
# 'extend' for lat, 'periodic' for lon
iParams = {'BCs': ['extend', 'periodic']}

# Invert within lat/lon plane, with extend and periodic boundary
# conditions in lat and lon respectively
psi = invert_Poisson(vor, dims=['lat','lon'], iParams=iParams)
```
![atmospheric plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/atmosExample.png)


### 3.2 Oceanic demonstration
Here is a oceanic demonstration with complex lateral boundaries of land/sea:
```python
import xarray as xr
from xinvert import invert_Poisson

dset = xr.open_dataset('mitgcm.nc')

vor = dset.vor

# specify boundary conditions in invert parameters
# 'fixed' for lat, 'periodic' for lon; undefined value is 0
iParams = {'BCs':['fixed', 'periodic'], 'undef':0}

# Invert within YG/XGplane, with fixed and periodic boundary respectively.
# Kwarg undef is used as a mask for land value.
psi = invert_Poisson(vor, dims=['YG','XG'], iParams=iParams)
```
![oceanic plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/oceanExample.png)

### 3.3 Animate the convergence of iteration
One can see the whole convergence process of SOR iteration as:
```python
from xinvert import invert_Poisson_animated

# input of vor need to be two dimensional only;
# psi has one more dimension than vor as iteration, which could be animated over.
# Here psi has 40 frames and loop 1 per frame (final state is after 40 iterations)
psi = animate_iteration(invert_Poisson, vor, iParams=iParams,
                              loop_per_frame=1, max_frames=40)
```
![animate plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/animateConverge.gif)

More examples can be found in these notebooks:
1.  [Poisson equation for streamfunction/velocity potential](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/01_Poisson_equation_horizontal.ipynb);
2.  [Poisson equation for meridional overturning and zonal Walker circulations](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/02_Poisson_equation_vertical.ipynb);
3.  [Geopotential model for balanced mass field](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/03_Balanced_mass_and_flow.ipynb);
4.  [Eliassen model for the meridional overturning circulation](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/04_Eliassen_model.ipynb);
5.  PV inversion for 2D reference state (TODO);
6.  PV inversion for 2D QGPV (TODO);
7.  [Matsuno-Gill model for heat-induced tropical circulation](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/07_Gill_Matsuno_model.ipynb)
8.  [Stommel-Munk model for wind-driven ocean circulation](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/08_Stommel_Munk_model.ipynb)
9.  [Omega equation for quasi-geostrophic vertical motion](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/09_Omega_equation.ipynb);
10. [3D oceanic flow](https://github.com/miniufo/xinvert/blob/master/docs/source/notebooks/10_3D_Ocean_flow.ipynb);

more to be added...
