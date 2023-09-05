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
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: State Key Laboratory of Tropical Oceanography, South China Sea Institute of Oceanology, Chinese Academy of Sciences, Guangzhou, China
   index: 1
date: 23 April 2023
bibliography: paper.bib

---


# Statement of need

Many problems in meteorology and oceanography can be cast into a balanced model in the form of a partial differential equation (PDE).  The most well-known one is the relation between the stream function $\psi$ and the vertical vorticity $\zeta$ (also known as Poisson equation):

$$\nabla^2 \psi = \frac{\partial^2 \psi}{\partial y^2}+\frac{\partial^2 \psi}{\partial y^2} = \zeta$$

Once $\zeta$ is given as a known, one needs to get the unknown $\psi$ with proper boundary conditions, which is essentially an inversion problem (\autoref{fig:1}).  Many geophysical fluid dynamical (GFD) problems can be described by such a balanced model, which is generally of second (or fourth) order in spatial derivatives and does not depend explicitly on time.  Therefore, it is also known as a steady-state model.  Early scientists tried to find analytical solutions by simplifying the parameters of these models (e.g., assuming constant coefficients).  Nowadays, with the new developments in numerical algorithms and parallel-computing programming, one may need a modern solver, written in a popular programming language like `Python`, to invert all these models in a numerical fashion.  More specifically, the following needs should be satisfied:

- **A unified numerical solver:** It can solve all of the classical balanced GFD models, even in a domain with irregular boundaries like the ocean.  New models can also be adapted straightforwardly to fit the solver;
- **Thinking and coding in equations:** Users focus naturally on the key inputs and outputs of the GFD models, just like thinking of the knowns and unknowns of the PDEs;
- **Flexible parameter specification:** Coefficients of the models can be either constant, 1D vectors, or ND arrays.  This allows an easy reproduction of early simplified results and also an extension to more general/realistic results;
- **Fast and efficient:** The algorithm should be fast and efficient.  Also, the code can be compiled first and then executed as fast as `C` or `FORTRAN`, instead of executed in the slow pure `Python` interpreter.  In addition, it can leverage the multi-core and out-of-core computational capabilities of modern computers;

`xinvert` is designed to satisfy all of the above needs, based on the `Python` ecosystem.

![(a) Vertical relative vorticity $\zeta$ and (b) the inverted streamfunction $\psi$ (shading) with current vector superimposed.  Note the irregular boundaries over the global ocean. \label{fig:1}](streamfunction.png){width=100%}

# State of the field
There are also several PDE solvers written in Python, like **windspharm** [@Dawson:2016] and **Dedalus** [@Burns:2020].  While they are efficient and accurate in double-periodic or global domains using the spectral method, they may not be suitable for arbitrary domains or boundaries like the ocean (Gibbs phenomenon will arise near ocean boundaries).  Also, it is not easy to apply these solvers to more general elliptic equations with arbitrary coefficients (sometimes given in a numerical or discretized fashion).  In these cases, the successive over relaxation (SOR) method should be a better choice.

# Mathematics

This package, `xinvert`, is designed to invert or solve the following PDE in an abstract form:

$$L\left(\psi \right) = F  \label{eq:1} \tag{1}$$

where $L$ is a second-order partial differential operator, $\psi$ the unknown to be inverted for, and $F$ a prescribed forcing function.  There could be also some coefficients or parameters in the definition of $L$, which should be specified before inverting $\psi$.

For the 2D case, the **general form** of Eq. \eqref{eq:1} is:

$$L\left(\psi\right) \equiv A_1\frac{\partial^2 \psi}{\partial y^2}+A_2\frac{\partial^2 \psi}{\partial y \partial x}+A_3\frac{\partial^2 \psi}{\partial x^2}+A_4\frac{\partial \psi}{\partial y}+A_5\frac{\partial \psi}{\partial x}+A_6\psi = F \label{eq:2} \tag{2}$$

where coefficients $A_1-A_6$ are all known.  When the condition $4A_1 A_3-A_2^2>0$ is met everywhere in the domain, the above equation is an elliptic-type equation.  In this case, one can invert $\psi$ using the **SOR** iteration method.  When $4A_1 A_3-A_2^2=0$ or $4A_1 A_3-A_2^2<0$, it is a parabolic or hyperbolic equation.  In either case, SOR would *fail* to converge to the solution.

Sometimes the **general form** of Eq. \eqref{eq:2} can be transformed into the **standard form** (i.e., standardization):

$$L\left(\psi\right) \equiv \frac{\partial}{\partial y}\left(A_1\frac{\partial \psi}{\partial y}+A_2\frac{\partial \psi}{\partial x}\right)+\frac{\partial}{\partial x}\left(A_3\frac{\partial \psi}{\partial y}+A_4\frac{\partial \psi}{\partial x}\right) + A_5\psi = F \label{eq:3} \tag{3}$$

In this case, $A_1 A_4-A_2 A_3>0$ should be met to ensure its ellipticity.  The elliptic condition has its own physical meaning in the problems of interest.  That is, the system is in a steady (or balanced) state that is stable to any small perturbation.

Many problems in meteorology and oceanography can be cast into the forms of either Eq. \eqref{eq:2} or Eq. \eqref{eq:3}.  However, some of them are formulated in a 3D form (like the QG-omega equation):

$$L\left(\psi\right) \equiv \frac{\partial}{\partial z}\left(A_1\frac{\partial \psi}{\partial z}\right) +\frac{\partial}{\partial y}\left(A_2\frac{\partial \psi}{\partial y}\right) +\frac{\partial}{\partial x}\left(A_3\frac{\partial \psi}{\partial x}\right) = F \label{eq:4} \tag{4}$$

or in **fourth-order** form (the Munk model):

$$
\begin{aligned}
L\left(\psi\right) &\equiv A_1\frac{\partial^4 \psi}{\partial y^4}+A_2\frac{\partial^4 \psi}{\partial y^2 \partial x^2}+A_3\frac{\partial^4 \psi}{\partial x^4} \notag\\
&+A_4\frac{\partial^2 \psi}{\partial y^2}+A_5\frac{\partial^2 \psi}{\partial y \partial x}+A_6\frac{\partial^2 \psi}{\partial x^2}+A_7\frac{\partial \psi}{\partial y}+A_8\frac{\partial \psi}{\partial x}+A_9\psi = F \label{eq:5}
\end{aligned} \tag{5}
$$

So we implement four basic solvers to take into account the above equations \eqref{eq:2} to \eqref{eq:5}.
Most of the problems fit one of these four types of solver.  However, it is not clear which form, the general form Eq. \eqref{eq:2} or the standard form Eq. \eqref{eq:3}, is preferred for the inversion if a problem can be cast into either one.


# Summary

`xinvert` is an open-source and user-friendly `Python` package that enables GFD scientists or interested amateurs to solve all possible GFD problems in a numerical fashion.  With the ecosystem of open-source `Python` packages, in particular `xarray` [@Hoyer:2017], `dask` [@Rocklin:2015], and `numba` [@Lam:2015], it is able to satisfy the above requirements:

- All the classical balanced GFD models can be inverted by this unified numerical solver;
- User APIs (\autoref{table:1}) are very close to the equations: unknowns are on the left-hand side of the equal sign `=`, whereas the known forcing functions are on its right-hand side (other known coefficients are also on the left-hand side but are passed in through `mParams`);
- Passing a single `xarray.DataArray` is usually enough for the inversion. The fact that coordinate information is already encapsulated reduces the length of the parameter list.  In addition, parameters in `mParams` can be either a constant, or varying with a specific dimension (like the Coriolis parameter $f$), or fully varying with space and time, due to the use of `xarray`'s [@Hoyer:2017] broadcasting capability;
- This package leverages `numba` [@Lam:2015] and `dask` [@Rocklin:2015] to support Just-In-Time (JIT) compilation, multi-core, and out-of-core computations, and therefore greatly increases the speed and efficiency of the inversion.

Here we summarize some inversion problems in meteorology and oceanography into \autoref{table:1}.  The table can be extended further if one finds more problems that fit the abstract form of Eq. \eqref{eq:1}.

| Problem names and equations | Function calls |
| :--------- | ---------: |
| Streamfunction \newline $\displaystyle{\nabla^2\psi=\frac{\partial^2 \psi}{\partial y^2}+\frac{\partial^2 \psi}{\partial x^2}=\zeta_k}$ | `psi = invert_Poisson(vork,`\newline`dims=['Y','X'],`\newline`mParams=None)` \newline|
| Balanced mass field [@Yuan:2008] \newline $\displaystyle{\nabla^2\Phi=\frac{\partial^2 \Phi}{\partial y^2}+\frac{\partial^2 \Phi}{\partial x^2}=F}$ | `Phi = invert_Poisson(F,`\newline`dims=['Y','X'],`\newline`mParams=None)` \newline|
| Geostrophic streamfunction \newline $\displaystyle{\frac{\partial}{\partial y}\left(f\frac{\partial \psi}{\partial y}\right)+\frac{\partial}{\partial x}\left(f\frac{\partial \psi}{\partial x}\right)=\nabla^2 \Phi}$ | `psi = invert_geostrophic(LapPhi,`\newline`dims=['Y','X'],`\newline`mParams={f})` \newline|
| Eliassen model [@Eliassen:1952] \newline $\displaystyle{\frac{\partial}{\partial p}\left(A\frac{\partial \psi}{\partial p}+B\frac{\partial \psi}{\partial y}\right)+\frac{\partial}{\partial y}\left(B\frac{\partial \psi}{\partial p}+C\frac{\partial \psi}{\partial y}\right)=F}$ | `psi = invert_Eliassen(F,`\newline`dims=['Z','Y'],`\newline`mParams={Angm, Thm})` \newline|
| PV inversion for vortex [@Hoskins:1985] \newline $\displaystyle{\frac{\partial}{\partial \theta}\left(\frac{2\Lambda_0}{r^2}\frac{\partial\Lambda}{\partial \theta}\right)+\frac{\partial}{\partial r}\left(\frac{\Gamma g}{Qr}\frac{\partial\Lambda}{\partial r}\right)=0}$ | `angM = invert_RefState(PV,`\newline`dims=['Z','Y'],`\newline`mParams={ang0, Gamma})` \newline|
| PV inversion for QG flow \newline $\displaystyle{\frac{\partial}{\partial p}\left(\frac{f^2}{N^2}\frac{\partial \psi}{\partial p}\right)+\frac{\partial^2 \psi}{\partial y^2}=q}$ | `psi = invert_PV2D(PV,`\newline`dims=['Z','Y'],`\newline`mParams={f, N2})` \newline |
| Gill-Matsuno model [@Matsuno:1966; @Gill:1980] \newline $\displaystyle{A\frac{\partial^2 \phi}{\partial y^2}+B\frac{\partial^2 \phi}{\partial x^2}+C\frac{\partial \phi}{\partial y}+D\frac{\partial \phi}{\partial x}+E\phi=Q}$ | `phi = invert_GillMatsuno(Q,`\newline`dims=['Y','X'],`\newline`mParams={f, epsilon, Phi})` \newline|
| Stommel-Munk model [@Stommel:1948; @Munk:1950] \newline $\displaystyle{A\nabla^4\psi-\frac{R}{D}\nabla^2\psi-\beta\frac{\partial \psi}{\partial x}=-\frac{\hat\nabla \cdot \vec{\tau}  }{\rho_0 D} }$ | `psi = invert_StommelMunk(curl,`\newline`dims=['Y','X'],`\newline`mParams={A, R, D, beta, rho})` \newline |
| Fofonoff flow [@Fofonoff:1954] \newline $\displaystyle{\nabla^2\psi-c_0\psi=c_1-f}$ | `psi = invert_Fofonoff(f,`\newline`dims=['Y','X'],`\newline`mParams={f, c0, c1})` \newline |
| Bretherton flow [@Bretherton:1976] \newline $\displaystyle{\nabla^2\psi-\lambda D\psi=-\frac{f_0}{D}\eta_B}$ | `psi = invert_BrethertonHaidvogel(topo,`\newline`dims=['Y','X'],`\newline`mParams={f, D, lambda})` \newline |
| QG-Omega equation [@Hoskins:1978] \newline $\displaystyle{\frac{\partial}{\partial p}\left(f^2\frac{\partial \omega}{\partial p}\right)+\nabla\cdot\left(S\nabla\omega\right)=F}$ | `w = invert_Omega(F,`\newline`dims=['Z','Y','X'],`\newline`mParams={f, S})` \newline|


  : Classical inversion problems in GFD.  The model names, equations, typical references and function calls are listed \label{table:1}


# Usage

`xinvert` provides a set of functions with a prefix `invert_`.  Users only need to import the function they are interested in and call it to get the inverted results (\autoref{table:1}).  Note that the calculation of the forcing function $F$ (e.g., calculating the vorticity using the velocity vector) on the right-hand side of the equation is not the core part of this package.  But there is a `FiniteDiff` utility module with which finite difference calculus can be readily performed.


# Acknowledgements

This work is jointly supported by the National Natural Science Foundation of China (42376028, 41931182, 41976023), and the support of the Independent Research Project Program of State Key Laboratory of Tropical Oceanography (LTOZZ2102). The author gratefully acknowledges the use of the HPCC at the South China Sea Institute of oceanology, Chinese Academy of Sciences.


# References

