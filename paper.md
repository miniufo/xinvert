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


Once $\zeta$ is given as a known, one may want to get the unknown $\psi$, which is essentially an inversion problem (\autoref{fig:1}).  These geophysical fluid dynamical (GFD) models are generally of second (or fourth) order in spatial derivatives and do not depend explicitly on time, which are therefore balanced models or steady-state models [@Wunsch:1996].  Early scientists tried to find analytical solutions by simplified the parameters of these models (e.g., assuming constant coefficients).  Nowadays, with the new developments in linear algrebra algorithms and parallel-computing programming, one may need a modern solver, written in the popular programming language `Python`, to invert all these models in a numerical fashion.  More specifically, the following needs should be satisfied:

- **A unified numerical solver:** It can solve all the classical balanced GFD models, even in a domain with irregular boundaries like the ocean.  New models can also be easily adapted to fit the solver;
- **Thinking and coding in equations:** Users focus naturally on the key inputs and outputs of the GFD models, just like thinking of the knowns and unknowns of the PDEs;
- **Flexible parameter specification:** Coefficients of the models can be either constant, 1D vector, or ND array.  This allows an easy reproduce of early simplified results and also an extension to more general/realistic results;
- **Fast and efficient:** The algorithm should be fast and efficient.  Also, the codes can be compiled first and then executed as fast as C or FORTRAN, instead of executed in the slow pure-Python interpreter.  In addition, it can utilize multi-core and out-of-core computations capabilities of modern computers;

`xinvert` is then designed to satisfy all the above needs, based on the ecosystem of Python.

![(left) Vertical vorticity $\zeta$ and (right) inverted streamfunction $\psi$ (shading) with current vector superimposed over global oceans.\label{fig:1}](streamfunction.png){width=100%}


# Acknowledgements

This work is jointly supported by the National Natural Science Foundation of China (42227901, 41931182, 41976023, 42106008), and the support of the Independent Research Project Program of State Key Laboratory of Tropical Oceanography (LTOZZ2102). The author gratefully acknowledge the use of the HPCC at the South China Sea Institute of ceanology, Chinese Academy of Sciences.


# References

