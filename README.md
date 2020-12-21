# xinvert


## 1. Introduction
Researches on meteorology and oceanography usually encounter [inversion problems](https://doi.org/10.1017/CBO9780511629570) that need to be solved numerically.  One of the classical inversion problem is to solve for a streamfunction $\psi$ given the vertical component of vorticity $\zeta$ and proper boundary conditions:
>$$
\nabla^2 \psi = \frac{\partial^2 \psi}{\partial x^2} +\frac{\partial^2 \psi}{\partial y^2} =\zeta
$$

Nowadays [`xarray`](http://xarray.pydata.org/en/stable/) becomes a popular data structure commonly used in [Big Data Geoscience](https://pangeo.io/).  Since the whole 4D data, as well as the coordinate information, are all combined into [`xarray`](http://xarray.pydata.org/en/stable/), solving the inversion problem become quite straightforward and the only input would be just one [`xarray.DataArray`](http://xarray.pydata.org/en/stable/) of $\zeta$.  Inversion on the spherical earth, like some meteorological problems, could utilize the spherical harmonics like [windspharm](https://github.com/ajdawson/windspharm), which would be more efficient using FFT than SOR used here.  However, in the case of ocean, SOR method is definitely a better choice in the presence of land/sea mask.

More importantly, this could be generalized into a numerical solver for elliptical equation using [SOR](https://mathworld.wolfram.com/SuccessiveOverrelaxationMethod.html) method, with varying coefficients $A$, $B$ and $C$ as:
>$$
\nabla^2 \psi \equiv \frac{\partial}{\partial x}\left(A\frac{\partial \psi}{\partial x}+B\frac{\partial \psi}{\partial y}\right) +\frac{\partial}{\partial y}\left(B\frac{\partial \psi}{\partial x}+C\frac{\partial \psi}{\partial y}\right) =F
$$

where $AC-B^2>0$ ensures the ellipticness of the equation and convergence of the SOR iteration.  Various popular inversion problems in geofluid science will be illustrated as examples.

One problem with SOR is that the speed of iteration using explicit loops will be **e-x-t-r-e-m-e-l-y ... s-l-o-w**!  A very suitable solution here is to use [`numba`](https://numba.pydata.org/).  We may try our best to speed things up using more hardwares (possibly GPU).

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
This is a classical problem in both meteorology and oceanography that a vector flow field can be deomposed into rotational and divergent parts, where rotational and divergent parts are represented by the streamfunction $\psi$ and velocity potential $\chi$ as:
>$$
\begin{align}
\frac{\partial^2 \psi}{\partial x^2} +\frac{\partial^2 \psi}{\partial y^2} =\frac{\partial v}{\partial x} -\frac{\partial u}{\partial y}=vor \tag{1}\\
\frac{\partial^2 \chi}{\partial x^2} +\frac{\partial^2 \chi}{\partial y^2} =\frac{\partial u}{\partial x} +\frac{\partial v}{\partial y}=div \tag{2}
\end{align}
$$

Given vorticity (vor) and divergence (div) as the forcing functions respectively in Eqs. (1) and (2), one can invert the streamfunction $\psi$ and velocity potential $\chi$ as:
```python
from xinvert import invert_Poisson

psi = invert_Poisson(vor, dims=['lat','lon'], BCs=['extend', 'periodic'])
chi = invert_Poisson(div, dims=['lat','lon'], BCs=['extend', 'periodic'])
```
### 3.1 Atmospheric demonstration
Here is an atmospheric demonstration with no lateral boundaries:
```python
import xarray as xr
from xinvert import invert_Poisson

dset = xr.open_dataset('data.nc')

vor = dset.vor

# Invert within lat/lon plane, with extend and periodic boundary
# conditions in lat and lon respectively
psi = invert_Poisson(vor, dims=['lat','lon'], BCs=['extend', 'periodic'])
```
![atmospheric plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/atmosExample.png)


### 3.2 Oceanic demonstration
Here is a oceanic demonstration with complex lateral boundaries of land/sea:
```python
import xarray as xr
from xinvert import invert_Poisson

dset = xr.open_dataset('mitgcm.nc')

vor = dset.vor

# Invert within YG/XGplane, with fixed and periodic boundary respectively.
# Kwarg undef is used as a mask for land value.
psi = invert_Poisson(vor, dims=['YG','XG'], BCs=['fixed', 'periodic'], undef=0)
```
![oceanic plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/oceanExample.png)

### 3.3 Animate the convergence of iteration
One can see the whole convergence process of SOR iteration as:
```python
from xinvert import invert_Poisson_animated

# input of vor need to be two dimensional only;
# psi has one more dimension than vor as iteration, which could be animated over.
# Here psi has 40 frames and loop 1 per frame (final state is after 40 iterations)
psi = invert_Poisson_animated(vor[0,0] BCs=['extend', 'periodic'],
                              loop_per_frame=1, max_loop=40)
```
![animate plot](https://raw.githubusercontent.com/miniufo/xinvert/master/pics/animateConverge.gif)

More examples can be found at this [notebook](https://github.com/miniufo/xinvert/blob/master/notebooks/1.%20Invert%20Poisson%20equation.ipynb)
