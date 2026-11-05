# xinvert

[![DOI](https://zenodo.org/badge/323045845.svg)](https://doi.org/10.5281/zenodo.7801500)
![GitHub](https://img.shields.io/github/license/miniufo/xinvert)
[![Documentation Status](https://readthedocs.org/projects/xinvert/badge/?version=latest)](https://xinvert.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/xinvert.svg)](https://badge.fury.io/py/xinvert)
![Workflow](https://github.com/miniufo/xinvert/actions/workflows/python-publish.yml/badge.svg)
[![pytest](https://github.com/miniufo/xinvert/actions/workflows/tests.yml/badge.svg)](https://github.com/miniufo/xinvert/actions/workflows/tests.yml)
[![status](https://joss.theoj.org/papers/1fc4ac8f98c0778516971880727a3a94/status.svg)](https://joss.theoj.org/papers/1fc4ac8f98c0778516971880727a3a94)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e6f6733ded33461993c1a9180826ce53)](https://app.codacy.com/gh/miniufo/xinvert/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

![animate plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/animateConverge.gif)


## 1. Introduction
Researches on meteorology and oceanography usually encounter [inversion problems](https://doi.org/10.1017/CBO9780511629570) that need to be solved numerically.  One of the classical inversion problem is to solve Poisson equation for a streamfunction $\psi$ given the vertical component of vorticity $\zeta$ and proper boundary conditions.

> $$\nabla^2\psi=\zeta$$

Nowadays [`xarray`](http://xarray.pydata.org/en/stable/) becomes a popular data structure commonly used in [Big Data Geoscience](https://pangeo.io/).  Since the whole 4D data, as well as the coordinate information, are all combined into [`xarray`](http://xarray.pydata.org/en/stable/), solving the inversion problem become quite straightforward and the only input would be just one [`xarray.DataArray`](http://xarray.pydata.org/en/stable/) of vorticity.  Inversion on the spherical earth, like some meteorological problems, could utilize the spherical harmonics like [windspharm](https://github.com/ajdawson/windspharm), which would be more efficient using FFT than SOR used here.  However, in the case of ocean, xinvert supports irregular land/sea masks directly with SOR.

More importantly, this could be generalized into a numerical solver for elliptic equations using [SOR](https://mathworld.wolfram.com/SuccessiveOverrelaxationMethod.html) method, with spatially-varying coefficients.  Various popular inversion problems in geofluid dynamics will be illustrated as examples.

One problem with SOR is that the speed of iteration using **explicit loops in Python** will be **e-x-t-r-e-m-e-l-y ... s-l-o-w**!  A very suitable solution here is to use [`numba`](https://numba.pydata.org/).  In addition, through using [`xarray`](http://xarray.pydata.org/en/stable/)'s [`dask`](https://docs.dask.org/en/latest/) backend for parallel computing, the speed of the inversion could be further improved using multiple CPU cores.  Now, we could also make use of CUDA GPU for further acceleration, with a Red-Black SOR kernel achieving up to **~26× speedup** over the CPU for 1,000 fixed sweeps in the archived RTX 3090 benchmark (see the [benchmark docs](https://xinvert.readthedocs.io/en/latest/Benchmark.html)).  See this [notebook](./docs/source/notebooks/Parallel_inversions.ipynb) for more details of CPU/GPU parallel computing, and this [docs](https://xinvert.readthedocs.io/en/latest/Benchmark.html) for benchmark of the inversion speed.

Classical problems include Gill-Matsuno model, Stommel-Munk model, QG omega model, PV inversion model, Swayer-Eliassen balance model...  A complete list of the classical inversion problems can be found at [this notebook](./docs/source/notebooks/00_Introduction.ipynb).

Why `xinvert`?

- **Thinking and coding in equations:** User APIs are very close to the equations: unknowns are on the LHS of `=`, whereas the known forcings are on its RHS;
- **Generalize steady-state problems:** Many steady-state problems in geophysical fluid dynamics can be formulated using the supported elliptic equations;
- **Very short parameter list:** Passing a single `xarray` forcing is enough for the inversion.  Coordinates information is already encapsulated.
- **Flexible model parameters:** Model parameters can be either a constant, or varying with a specific dimension (like Coriolis $f$), or fully varying with space and time, due to the use of `xarray`'s broadcasting capability;
- **Parallel inverting:** The use of `xarray`, and thus `dask` allow multi-core CPU parallel inverting, as well as GPU inverting.  One only needs to change the setting `iParams['architect']='cpu'/'gpu'/'cpu-mc'`, and the internal implementation is almost transparent to the user;
- **Pure Python code for C-code speed:** The use of `numba` allow pure python code in this package but native speed;

---
## 2. How to install
**Requirements**
`xinvert` supports Python 3.9 and newer and requires `xarray`, `dask`, `numpy`, `numba`, SciPy, netCDF4, Matplotlib, imageio-ffmpeg, and cloudpickle. The GPU extra installs Numba CUDA, CuPy and CUDA 12 runtime dependencies for both SOR and Li2006; an NVIDIA GPU with a compatible driver is required.


**Install via pip**
```bash
pip install xinvert
```

**Install GPU support**
```bash
pip install "xinvert[gpu]"
```

**Install via conda**
```bash
conda install -c conda-forge xinvert
```

**Install from github**
```bash
git clone https://github.com/miniufo/xinvert.git
cd xinvert
python -m pip install .
```


---
## 3. Examples:
The classic Poisson equation for streamfunction can be inverted by:
```python
from xinvert import invert_Poisson

iParams = {
    'BCs'      : ['fixed', 'periodic'], # fixed north/south,  periodic east/west BCs
    'mxLoop'   : 1000,                  # maximum loop count
    'tolerance': 1e-6,                  # define convergence if error is smaller than this
    'architect': 'GPU',                 # use GPU backend
}

psi = invert_Poisson(vor, dims=['lat','lon'], iParams=iParams)
```

For vector fields, `xinvert.helmholtz` provides SOR and Li2006
methods through one interface:

```python
from xinvert.helmholtz import decompose

components = decompose(u, v, dims=['lat', 'lon'], method='li2006')
```

See [Helmholtz decomposition](https://xinvert.readthedocs.io/en/latest/Helmholtz.html)
for the method comparison and parameters.

This is a list of the problems that can be solved by `xinvert`:

|    Gallery    |    Gallery    |
| :-----------: | :-----------: |
| <img src="./pics/Gallery_Streamfunction.png" width="380"><br/>[invert Poisson equation for<br/> horizontal streamfunction](./docs/source/notebooks/01_Poisson_equation_horizontal.ipynb) | <img src="./pics/Gallery_Overturning.png" width="380"><br/>[invert Poisson equation for<br/> overturning streamfunction](./docs/source/notebooks/02_Poisson_equation_vertical.ipynb) |
| <img src="./pics/Gallery_balanceMass.png" width="380"><br/>[invert geostrophic equation for<br/> balanced mass](./docs/source/notebooks/03_Balanced_mass_and_flow.ipynb) | <img src="./pics/Gallery_Eliassen.png" width="380"><br/>[invert Eliassen model for<br/> overturning streamfunction](./docs/source/notebooks/04_Eliassen_model.ipynb) |
| <img src="./pics/Gallery_SWMReference.png" width="380"><br/>[invert PV balance equation for<br/> steady reference state](./docs/source/notebooks/05_reference_SWM.ipynb)| <img src="./pics/Gallery_GillMatsuno.png" width="380"><br/>[invert Gill-Matsuno model for<br/> wind and mass fields](./docs/source/notebooks/07_Gill_Matsuno_model.ipynb) |
| <img src="./pics/Gallery_StommelMunk.png" width="380"><br/>[invert Stommel-Munk model for<br/> wind-driven ocean circulation](./docs/source/notebooks/08_Stommel_Munk_model.ipynb) | <img src="./pics/Gallery_Fofonoff.png" width="380"><br/>[invert Fofonoff model for<br/> inviscid/adiabatic steady state](./docs/source/notebooks/09_Fofonoff_flow.ipynb) |
| <img src="./pics/Gallery_Bretherton.png" width="380"><br/>[invert Bretherton model for<br/> steady flow over topography](./docs/source/notebooks/10_Bretherton_flow_over_topography.ipynb) | <img src="./pics/Gallery_Omega.png" width="380"><br/>[invert Omega equation for<br/> QG vertical velocity](./docs/source/notebooks/11_Omega_equation.ipynb) |
| <img src="./pics/Gallery_GeoAdjust.png" width="380"><br/>[invert geostrophic-adjustment problem](./docs/source/notebooks/12_GeostrophicAdjustment.ipynb) | ... add more examples ... |



## 4 Record iterations and create animations
The animation at the top was made using `record_iteration`. Pass one spatial
slice; singleton dimensions are removed and spatial dimensions are inferred.
Either an inversion function or its name (such as `'Poisson'`) is accepted:

```python
from xinvert import record_iteration, invert_Poisson

frames = record_iteration(
    invert_Poisson, vor,  # vor is one 2-D spatial slice
    iParams={'BCs': ['fixed', 'periodic'], 'architect': 'cpu'},
    iter_per_frame=1, max_frames=40,
)
# frames is a DataArray with an additional iter dimension.
```

For file output, add `output='iteration.nc'` to write frames without keeping
all of them in memory. Use `create_animation` to render the NetCDF frames as GIF or MP4. See [Record iteration and create animation](https://xinvert.readthedocs.io/en/latest/Animation.html)
for plotting examples and optional convergence diagnostics.


## 5 Diagnose the error
`xinvert` uses a normalized, diagonally preconditioned residual for stopping.
Its history can optionally be returned for analysis:

```python
iParams = {
    'tolerance': 1e-8,
    'diag_interval': 20,  # diagnose every 20 SOR iterations
}
psi, diags = invert_Poisson(vor, dims=['lat', 'lon'], iParams=iParams)

diags.error.plot(x='iteration', yscale='log')
```

Through plotting the error, one can observe the convergence rate at regular
interation interval.  The default ``diag_interval=0`` returns only the solution.  See
[diagnostic analysis documentation](https://xinvert.readthedocs.io/en/latest/Diagnostics.html)
for more details.



## 6 Cite

If you use the package in research, teaching, or other activities, we would be grateful
if you mention `xinvert` and cite our paper in JOSS:

```bibtex
@article{Qian2023,
    doi = {10.21105/joss.05510},
    url = {https://doi.org/10.21105/joss.05510},
    year = {2023},
    publisher = {Journal of Open Source Software},
    volume = {8},
    number = {89},
    pages = {5510},
    author = {Yu-Kun Qian},
    title = {xinvert: A Python package for inversion problems in geophysical fluid dynamics}, journal = {Journal of Open Source Software}
}
```
