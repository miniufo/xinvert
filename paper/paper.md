---
title: 'xinvert: A Python package for inversion problems in geophysical fluid dynamics'
tags:
  - Python
  - geophysics
  - atmosphere
  - ocean
  - geophysical fluid dynamics
  - steady state problem
  - second-order partial differential equation
  - successive over relaxation
authors:
  - name: Yu-Kun Qian
    orcid: 0000-0001-5660-7619
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Shiqiu Peng
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: State Key Laboratory of Tropical Oceanography, South China Sea Institute of Oceanology, Chinese Academy of Sciences, Guangzhou, China
   index: 1
date: 13 August 2017
bibliography: paper.bib

---


# Statement of need

Many problems in meteorology and oceanography can be cast into balanced models in the form of partial differential equations (PDEs).  The most classical one is the relation between the streamfunction $\psi$ and the vertical vorticity $\zeta$ (also known as Poisson equation):

>$$\begin{equation}
\nabla^2 \psi = \frac{\partial^2 \psi}{\partial y^2}+\frac{\partial^2 \psi}{\partial y^2} = \zeta
\end{equation}$$

Once $\zeta$ is given as a known, one may want to get the unknown $\psi$, which is essentially an inversion problem (\autoref{fig:1}).  These geophysical fluid dynamical (GFD) models are generally of second (or fourth) order in spatial derivatives and do not depend explicitly on time, which are therefore balanced models or steady-state models [@Wunch:1996].  Early scientists tried to find analytical solutions by simplified the parameters of these models (e.g., assuming constant coefficients).  Nowadays, with the new developments in linear algrebra algorithms and parallel-computing programming, one may need a modern solver, written in the popular programming language `Python`, to invert all these models in a numerical fashion.  More specifically, the following needs should be satisfied:

- **A unified numerical solver:** It can solve all the classical balanced GFD models, even in a domain with irregular boundaries like the ocean.  New models can also be easily adapted to fit the solver;
- **Thinking and coding in equations:** Users focus naturally on the key inputs and outputs of the GFD models, just like thinking of the knowns and unknowns of the PDEs;
- **Flexible parameter specification:** Coefficients of the models can be either constant, 1D vector, or ND array.  This allows an easy reproduce of early simplified results and also an extension to more general/realistic results;
- **Fast and efficient:** The algorithm should be fast and efficient.  Also, the codes can be compiled first and then executed as fast as C or FORTRAN, instead of executed in the slow pure-Python interpreter.  In addition, it can utilize multi-core and out-of-core computations capabilities of modern computers;

`xinvert` is then designed to satisfy all the above needs, based on the ecosystem of Python.

![(left) Vertical vorticity $\zeta$ and (right) inverted streamfunction $\psi$ (shading) with current vector superimposed over global oceans.\label{fig:1}](streamfunction.png){width=100%}

# Mathematics

This package, `xinvert`, is designed to invert or solve the following PDE in an abstract form:

>$$\begin{equation}L\left(\psi \right) = F  \label{eq:1} \tag{1}\end{equation}$$

where $L$ is a second-order partial differential operator, $\psi$ is the unknown to be inverted for, and $F$ a prescribed forcing function.  There could be also some coefficients or parameters in the definition of $L$, which should be specified before inverting $\psi$.

For the **2D case**, the **general form** of Eq. (\autoref{eq:1}) is:

>$$\begin{equation}L\left(\psi\right) \equiv A\frac{\partial^2 \psi}{\partial y^2}+B\frac{\partial^2 \psi}{\partial y \partial x}+C\frac{\partial^2 \psi}{\partial x^2}+D\frac{\partial \psi}{\partial y}+E\frac{\partial \psi}{\partial x}+F\psi = G \label{eq:2} \tag{2}\end{equation}$$

where coefficients $A-G$ are all known variables.  When the condition $4AC-B^2>0$ is met everywhere in the domain, the above equation is an elliptic-type equation.  In this case, one can invert $\psi$ using the [successive over relaxation (SOR)](https://mathworld.wolfram.com/SuccessiveOverrelaxationMethod.html) iteration method.  When $4AC-B^2=0$ or $4AC-B^2<0$, it is a parabolic or hyperbolic equation.  In either case, SOR would *fail* to converge to the solution.

Sometimes the **general form** of Eq. (\autoref{eq:2}) can be transformed into the **standard form** (i.e., standarization):

>$$\begin{equation}L\left(\psi\right) \equiv \frac{\partial}{\partial y}\left(A\frac{\partial \psi}{\partial y}+B\frac{\partial \psi}{\partial x}\right)+\frac{\partial}{\partial x}\left(C\frac{\partial \psi}{\partial y}+D\frac{\partial \psi}{\partial x}\right) + E\psi =F \label{eq:3} \tag{3}\end{equation}$$

In this case, $AD-BC>0$ should be met to insure its ellipticity.  The elliptic condition has its own physical meaning in the problems of interest.  That is, the system is in steady (or balanced) states that are stable to any small perturbation.

Many problems in meteorology and oceanography can be cast into the forms of either Eq. (\autoref{eq:2}) or Eq. (\autoref{eq:3}).  However, some of them are formulated in **3D case** (like the QG-omega equation):

>$$\begin{equation}L\left(\psi\right) \equiv \frac{\partial}{\partial z}\left(A\frac{\partial \psi}{\partial z}\right) +\frac{\partial}{\partial y}\left(B\frac{\partial \psi}{\partial y}\right) +\frac{\partial}{\partial x}\left(C\frac{\partial \psi}{\partial x}\right) =F \label{eq:4} \tag{4}\end{equation}$$

or in **fourth-order** case (Munk model):

>$$\begin{align}L\left(\psi\right) &\equiv A\frac{\partial^4 \psi}{\partial y^4}+B\frac{\partial^4 \psi}{\partial y^2 \partial x^2}+C\frac{\partial^4 \psi}{\partial x^4}\notag\\&+D\frac{\partial^2 \psi}{\partial y^2}+E\frac{\partial^2 \psi}{\partial y \partial x}+F\frac{\partial^2 \psi}{\partial x^2}+G\frac{\partial \psi}{\partial y}+H\frac{\partial \psi}{\partial x}+I\psi = J \label{eq:5} \tag{5}\end{align}\end{equation}$$

So we implements four basic solvers to take into account the above four Eqs. (\autoref{eq:2}-\autoref{eq:5}) or cases.  If a problem do not fit into one of these four types, we are going to add one solver for this type of problem.  We hope *NOT* so because we want to keep the solvers as minimum and general as possible.  It is also *NOT* clear which form, the genral form Eq. (\autoref{eq:2}) or the standard form Eq. (\autoref{eq:3}), is preferred for the inversion if a problem can be cast into either form.


# Summary

`xinvert` is an open-source and uesr-friendly Python package that enables GFD scientists or interested amateurs to solve all possible GFD problems in a numerical fashion.  With the ecosystem of open-source python packages, in particular `xarray` [@Hoyer:2017], `dask` [@Rocklin:2015], and `numba` [@Lam:2015], it is able to satisfy the above requirements:
- Top user APIs (\ref{table1}) are very close to the equations: unknowns are on the left-hand side of the equal sign `=`, whereas the known forcing functions are on its right-hand side (other known coefficients are also on the left-hand side and are passed in through `mParams`);
- Passing a single `xarray.DataArray` is usually enough for the inversion. Coordinates information is already encapsulated and thus reducing the length of the parameter list.  In addition, paramters in `mParams` can be either a constant, or varying with a specific dimension (like Coriolis parameter $f$), or fully varying with space and time, due to the use of `xarray`'s [@Hoyer:2017] broadcasting capability;
- This package leverages `numba` [@Lam:2015], `xarray` [@Hoyer:2017], and `dask` [@Rocklin:2015] to support Just-In-Time (JIT) compilation, multi-core, and out-of-core computations, and therefore greatly increases the speed and efficiency of the inversion.

Here we summarize the inversion problems in meteorology and oceanography into the following Table (\ref{table1}).  The table can be extended further if one finds more problems that fit the abstract form of Eq. (\autoref{eq:1}).

| Types | Names | Comments | Equations | API calls |
| :----- | -----: | :---------: | :---------: | :---------- |
| 2D standard | Poisson model | Horizontal streamfunction | $\displaystyle{\nabla^2\psi=\frac{\partial^2 \psi}{\partial y^2}+\frac{\partial^2 \psi}{\partial x^2}=\zeta_k}$ | `sf = invert_Poisson(vork,`<br>`dims=['Y','X'], mParams=None)` |
| 2D standard | Poisson model | MOC streamfunction | $\displaystyle{\nabla^2\psi=\frac{\partial^2 \psi}{\partial z^2}+\frac{\partial^2 \psi}{\partial y^2}=\zeta_i}$ | `sf = invert_Poisson(vori,`<br>`dims=['Z','Y'], mParams=None)` |
| 2D standard | Poisson model | Walker streamfunction | $\displaystyle{\nabla^2\psi=\frac{\partial^2 \psi}{\partial z^2}+\frac{\partial^2 \psi}{\partial x^2}=\zeta_j}$ | `sf = invert_Poisson(vorj,`<br>`dims=['Z','X'], mParams=None)` |
| 2D standard | Poisson model | balanced mass field [@Yuan:2008] | $\displaystyle{\nabla^2\Phi=\frac{\partial^2 \Phi}{\partial y^2}+\frac{\partial^2 \Phi}{\partial x^2}=F}$ | `sf = invert_Poisson(F,`<br>`dims=['Y','X'], mParams=None)` |
| 2D standard | Geostrophic model | balanced flow field | $\displaystyle{\frac{\partial}{\partial y}\left(f\frac{\partial \psi}{\partial y}\right)+\frac{\partial}{\partial x}\left(f\frac{\partial \psi}{\partial x}\right)=\nabla^2 \Phi}$ | `sf = invert_geostrophic(LapPhi,`<br>`dims=['Y','X'], mParams={f})` |
| 2D standard | Eliassen model [@Eliassen:1952] | Reduce to Poisson if<br>$B=0$ and $A=C=1$ | $\displaystyle{\frac{\partial}{\partial p}\left(A\frac{\partial \psi}{\partial p}+B\frac{\partial \psi}{\partial y}\right)+\frac{\partial}{\partial y}\left(B\frac{\partial \psi}{\partial p}+C\frac{\partial \psi}{\partial y}\right)=F}$ | `sf = invert_Eliassen(F,`<br>`dims=['Z','Y'], mParams={Angm, Thm})` |
| 2D standard | PV inversion | 2D reference state [@Hoskins:1985] | $\displaystyle{\frac{\partial}{\partial \theta}\left(\frac{2\Lambda_0}{r^2}\frac{\partial\Lambda}{\partial \theta}\right)+\frac{\partial}{\partial r}\left(\frac{\Gamma g}{Qr}\frac{\partial\Lambda}{\partial r}\right)=0}$ | `angM = invert_RefState(PV,`<br>`dims=['Z','Y'], mParams={ang0, Gamma})` |
| 2D standard | PV inversion | 2D QGPV inversion | $\displaystyle{\frac{\partial}{\partial p}\left(\frac{f^2}{N^2}\frac{\partial \psi}{\partial p}\right)+\frac{\partial^2 \psi}{\partial y^2}=q}$ | `sf = invert_PV2D(PV,`<br>`dims=['Z','Y'], mParams={f, N2})` |
| 2D general | Gill-Matsuno model [@Matsuno:1966; @Gill:1980] | Reduce to geostrophic<br>model if $\epsilon=Q=0$ | $\displaystyle{A\frac{\partial^2 \phi}{\partial y^2}+B\frac{\partial^2 \phi}{\partial x^2}+C\frac{\partial \phi}{\partial y}+D\frac{\partial \phi}{\partial x}+E\phi=Q}$ | `h = invert_GillMatsuno(Q,`<br>`dims=['Y','X'], mParams={f, epsilon, Phi})` |
| 2D general | Stommel-Munk model [@Stommel:1948; @Munk:1950] | $A=0$ for Stommel model<br>$R=0$ for Munk model  | $\displaystyle{A\nabla^4\psi-\frac{R}{D}\nabla^2\psi-\beta\frac{\partial \psi}{\partial x}=-\frac{\mathbf k\cdot\nabla\times\mathbf{\tau}}{\rho_0 D}}$ | `sf = invert_StommelMunk(curl,`<br>`dims=['Y','X'], mParams={A, R, D, beta, rho})` |
| 3D standard | QG-Omega equation [@Hoskins:1978] | similar to 2D PV inversion | $\displaystyle{\frac{\partial}{\partial p}\left(f^2\frac{\partial \omega}{\partial p}\right)+\nabla\cdot\left(S\nabla\omega\right)=F}$ | `w = invert_Omega(F,`<br>`dims=['Z','Y','X'], mParams={f, S})` |
| 3D standard | 3D ocean flow | similar to omega equation | $\displaystyle{\frac{\partial}{\partial p}\left(c_3\frac{\partial \phi}{\partial p}\right)+\nabla\cdot\left(c_1\nabla\phi-c_2\hat\nabla\phi\right)=F}$ | `w = invert_3DFlow(F,`<br>`dims=['Z','Y','X'], mParams={f, N2, epsilon})` |
|  |  |  | **...** more problems **...** |  |

  : Classical inversion problems in GFD.  The model names, references, equations and APIs are listed here \label{table:1}


# Usage

`xinvert` is programmed in a functional style.  Users only need to import the core function they are interested in and call it to get the inverted results (Table \autoref{table:1}).  Note that several keyward arguments need to be specified:

- `dims`: Some `xarray.DataArray` has different dimension names like [`latitude`, `longitude`] or [`lat`, `lon`].  One needs to specify the right dimension names;
- `coords`: For Poisson Equation (\autoref{eq:1}), it can be inverted in the horizontal plane (`lat-lon`), or the vertical plane (`z-lat`).  One need to specify coordinate-combination in `coords` in accordance to the dimension names.  Note that `lat-lon` is for the horizontal plane on the spherical coordinate whereas `cartesian` is for the same plane on the Cartesian coordinate;
- `iParams`: This contains all the parameters for the iteration control like maximum loop and tolerance for the iteration to stop;
- `mParams`: This contains all the parameters for the models themselves like the Coriolis parameter $f$ or stratification $N^2$.

Note that the calculation of the forcing function $F$ (e.g., calculating the vorticity using velocity vector) on the right-hand side of the equation is **NOT** the core part of this package.  But there is a `FiniteDiff` module with which finite difference calculus can be readily performed.


# Acknowledgements

This work is jointly supported by the National Natural Science Foundation of China (42227901, 41931182, 41976023, 42106008), and the support of the Independent Research Project Program of State Key Laboratory of Tropical Oceanography (LTOZZ2102). The author gratefully acknowledge the use of the HPCC at the South China Sea Institute of ceanology, Chinese Academy of Sciences.


# References

