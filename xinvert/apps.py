# -*- coding: utf-8 -*-
"""
Created on 2022.04.13

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import xarray as xr
import copy
from .utils import loop_noncore
from .core import inv_standard3D, inv_standard2D, inv_standard1D,\
                  inv_general3D, inv_general2D,\
                  inv_general2D_bih, inv_standard2D_test


# default undefined value
_undeftmp = -9.99e8

###### default invert parameters. ######
default_iParams = copy.deepcopy({
    # boundary conditions for the 2D slice
    # if 3D, should be ['fixed', 'fixed', 'fixed']
    'BCs'      : ['fixed', 'fixed'],
    # undefined value in the array
    'undef'    : np.nan,
    # max loop count, exceed which the iteration is stopped
    'mxLoop'   : 5000,
    # tolerance, smaller than which the iteration is stopped
    'tolerance': 1e-8,
    # optimal argument for SOR, 1 stand for G-S iteration.
    # This argument will be automatically updated according to grids
    'optArg'   : None,
    # Whether or not print the information of the iteration
    'printInfo': True,
    # Whether or not print out debug info.
    'debug'    : False,
})


###### default model parameters. ######
default_mParams = copy.deepcopy({
    'f0'     : 1e-5 , # Coriolis parameter at south BC on beta plane
    'beta'   : 2e-11, # meridional derivative of f
    'Phi'    : 1e4  , # background geopotential in Gill-Matsuno model
    'epsilon': 7e-6 , # linear damping coefficient in Gill-Matsuno model
    'N2'     : 2e-4 , # stratification or buoyancy frequency
    'A'      : 1e5  , # Laplacian viscosity of momentum in Munk model
    'R'      : 5e-5 , # linear drag coefficient in Stommel-Munk model
    'depth'  : 100  , # depth of the ocean or mixed layer in Stommel-Munk model
    'rho0'   : 1027 , # constant density of seawater in Stommel-Munk model
    'ang0'   : 2e5  , # background angular momentum
    'lambda' : 1e-8 , # used in Bretherton-Haidvogel model
    'c0'     : 8e-9 , # for Fofonoff model
    'c1'     : 8e-5 , # for Fofonoff model
    
    'Rearth' : 6371200.0, # Radius of Earth
    'Omega'  : 7.292e-5 , # angular speed of Earth's rotation
    'g'      : 9.80665  , # gravitational acceleration
})



"""
Application functions
"""
def invert_Poisson(F, dims, coords='lat-lon', icbc=None,
                   mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the Poisson equation.

    The Poisson equation is given as:

    .. math::
        
        \frac{\partial^2 \psi}{\partial y^2} + \frac{\partial^2 \psi}{\partial x^2} = F
    
    Invert the Poisson equation for :math:`\psi` given :math:`F`.
    
    Parameters
    ----------
    F: xarray.DataArray
        Forcing function (e.g., vorticity).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'z-lat', 'z-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribed inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Model parameters.  None for the Poisson model.
    iParams: dict, optional
        Iteration parameters.
    
    Returns
    -------
    xarray.DataArray
        Results of the SOR inversion.
    """
    return __template(__coeffs_Poisson, inv_standard2D, 2, F, dims, coords,
                      icbc, ['g', 'Omega', 'Rearth'], mParams, iParams)



def invert_RefState(PV, dims, coords='z-lat', icbc=None,
                    mParams=default_mParams, iParams=default_iParams):
    r"""PV inversion for a balanced symmetric vortex.

    The balanced symmetric vortex equation is given as:

    .. math::

         \frac{\partial}{\partial \theta}\left(\frac{2\Lambda_0}{r^3}
         \frac{\partial \Lambda}{\partial \theta}\right) +
         \frac{\partial}{\partial r}\left(\frac{\Gamma g}{Q r}
         \frac{\partial \Lambda}{\partial r}\right) = 0
    
    Invert this equation for absolute angular momentum :math:`\Lambda` given
    the PV distribution :math:`Q`.
    
    Parameters
    ----------
    PV: xarray.DataArray
        2D distribution of PV.
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat'].
    coords: {'z-lat', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict
        Parameters required for this model are:

		* Ang0: Angular momentum Λ0 as the known coefficient.
		* Gamma: vertical function defined as Γ = Rd/p * (p/p0)^κ = κ * Π/p.
        
    iParams: dict, optional
        Iteration parameters.
    
    Returns
    -------
    xarray.DataArray
        Results (angular momentum Λ) of the SOR inversion.
    """
    return __template(__coeffs_RefState, inv_standard2D, 2, PV, dims, coords,
                      icbc, ['Ang0', 'Gamma', 'g', 'Omega', 'Rearth'], mParams, iParams)


def invert_RefStateSWM(Q, dims, coords='lat', icbc=None,
                       mParams=default_mParams, iParams=default_iParams):
    r"""(PV) inversion for a steady state of shallow-water model.

    The balanced symmetric vortex equation is given as:

    .. math::

         \frac{\partial}{\partial y}\left(A\frac{\partial\Delta M}{\partial y}\right)
         -B\Delta M &=F
    
    where
    
    .. math::
         
         A = 1 / r
         B = \frac{2C_0 Q_0 \sin\phi}{2\pi g r^3}
         F = \frac{C_0^2\sin\phi }{2\pi g r^3}-\frac{2\pi\Omega^2r\sin\phi}{g}-\frac{\partial }{\partial y}\left(\frac{1}{r}\frac{\partial M_0}{\partial y}\right)
    
    Invert this equation for mass correction :math:`\Delta M` given
    the PV distribution :math:`Q`.
    
    Parameters
    ----------
    Q: xarray.DataArray
        A set of PV contours.
    C0: xarray.DataArray
        Initial circulation profile along the meridional direction.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict
        Parameters required for this model are:

        * M0: initial guess of meridional mass profile.
		* c0: Eulerian zonal-mean circulation.
        
    iParams: dict, optional
        Iteration parameters.
    
    Returns
    -------
    xarray.DataArray
        Results (angular momentum Λ) of the SOR inversion.
    """
    return __template(__coeffs_RefStateSWM, inv_standard1D, 1, Q, dims, coords,
                      icbc, ['M0', 'C0', 'g', 'Rearth', 'Omega'], mParams, iParams)


def invert_PV2D(PV, dims, coords='z-lat', icbc=None,
                mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the QG PV equation.

    The QG PV equation is given as:

    .. math::

        \frac{\partial}{\partial p}\left(\frac{f_0}{N^2}
        \frac{\partial \psi}{\partial p}\right) +
        \frac{1}{f_0}\frac{\partial^2 \psi}{\partial y^2} + f = q
    
    This is slightly changed to:

    .. math::

        L(\psi) = \frac{\partial}{\partial p}\left(\frac{f_0^2}{N^2}
        \frac{\partial \psi}{\partial p}\right) +
        \frac{\partial^2 \psi}{\partial y^2} = (q - f)f_0
    
    Invert this equation for QG streamfunction :math:`\psi` given
    the PV distribution :math:`q`.
    

    Parameters
    ----------
    PV: xarray.DataArray
        2D distribution of QGPV.
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat'].
    coords: {'z-lat', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter.
		* beta: Coriolis parameter.
		* N2: buoyancy frequency.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (QG streamfunction) of the SOR inversion.
    """
    return __template(__coeffs_PV2D, inv_standard2D, 2, PV, dims, coords,
                      icbc, ['f0', 'beta', 'N2', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_Eliassen(F, dims, coords='z-lat', icbc=None,
                    mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the Eliassen balanced vortex model.

    The Eliassen model is given as:

    .. math::

        \frac{\partial}{\partial z}\left(
        A\frac{\partial \psi}{\partial z} +
        B\frac{\partial \psi}{\partial y} \right) +
        \frac{\partial}{\partial y}\left(
        B\frac{\partial \psi}{\partial z} +
        C\frac{\partial \psi}{\partial y} \right) = F
    
    Invert this equation for the overturning streamfunction :math:`\psi` given
    the forcing :math:`F`.

    
    Parameters
    ----------
    F: xarray.DataArray
        Forcing function.
    dims: list
        Dimension combination for the inversion
        e.g., ['lev', 'lat'] or ['z','y'].
    coords: {'z-lat', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict
        Parameters required for this model are:

		* A: Inertial stability.
		* B: baroclinic stability.
		* C: static stability.

    iParams: dict, optional
        Iteration parameters.
    
    Returns
    -------
    xarray.DataArray
        Results of the SOR inversion.
    """
    return __template(__coeffs_Eliassen, inv_standard2D, 2, F, dims, coords,
                      icbc, ['A', 'B', 'C', 'g', 'Omega', 'Rearth'], mParams, iParams)


def invert_GillMatsuno(Q, dims, coords='lat-lon', icbc=None, 
                       mParams=default_mParams, iParams=default_iParams):
    r"""Inverting Gill-Matsuno model.

    The Gill-Matsuno model is given as:

    .. math::

        \epsilon   u  =  fv - \frac{\partial \phi}{\partial x}\\\\
        \epsilon   v  = -fu - \frac{\partial \phi}{\partial y}\\\\
        \epsilon \phi + \Phi\left(\frac{\partial u}{\partial x}
        +\frac{\partial v}{\partial y}\right) = -Q
    
    Invert this equation for the mass distribution :math:`\phi` given
    the diabatic heating function :math:`Q`.

    
    Parameters
    ----------
    Q: xarray.DataArray
        Diabatic heating function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter at south BC if on beta plane.
		* beta: Meridional derivative of f.
		* epsilon: Linear damping coefficient.
		* Phi: Background geopotential.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (mass distribution) of the SOR inversion.
    """
    return __template(__coeffs_GillMatsuno, inv_general2D, 2, Q, dims, coords,
                      icbc, ['f0', 'beta', 'epsilon', 'Phi', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_GillMatsuno_test(Q, dims, coords='lat-lon', icbc=None, 
                       mParams=default_mParams, iParams=default_iParams):
    r"""Inverting Gill-Matsuno model (test use only).

    The Gill-Matsuno model is given as:

    .. math::

        \epsilon   u  &=  fv - \frac{\partial \phi}{\partial x}\\\\
        \epsilon   v  &= -fu - \frac{\partial \phi}{\partial y}\\\\
        \epsilon \phi + \Phi\left(\frac{\partial u}{\partial x}
        +\frac{\partial v}{\partial y}\right) &= -Q
    
    Invert this equation for the mass distribution :math:`\phi` given
    the diabatic heating function :math:`Q`.

    
    Parameters
    ----------
    Q: xarray.DataArray
        Diabatic heating function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter at south BC if on beta plane.
		* beta: Meridional derivative of f.
		* epsilon: Linear damping coefficient.
		* Phi: Background geopotential.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (mass distribution) of the SOR inversion.
    """
    return __template(__coeffs_GillMatsuno_test, inv_standard2D_test, 2, Q, dims, coords,
                      icbc, ['f0', 'beta', 'epsilon', 'Phi', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_Stommel(curl, dims, coords='lat-lon', icbc=None,
                   mParams=default_mParams, iParams=default_iParams):
    r"""Inverting Stommel model.

    The Stommel model is given as:

    .. math::

        - \frac{R}{D}\nabla^2 \psi - \beta\frac{\partial \psi}{\partial x} =
        - \frac{\hat\nabla \cdot \vec\tau}{\rho_0 D}
    
    Invert this equation for the streamfunction :math:`\psi` given wind-stress
    curl :math:`\hat\nabla \cdot \vec\tau`.

    
    Parameters
    ----------
    curl: xarray.DataArray
        Wind stress curl.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* beta: Meridional derivative of Coriolis parameter.
		* R: Laplacian viscosity.
		* D: Depth of the ocean or mixed layer depth.
		* rho0: Density of the fluid.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (streamfunction) of the SOR inversion.
    """
    return __template(__coeffs_Stommel, inv_general2D, 2, curl, dims, coords,
                      icbc, ['beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_Stommel_test(curl, dims, coords='lat-lon', icbc=None,
                   mParams=default_mParams, iParams=default_iParams):
    r"""Inverting Stommel model (test used only).

    The Stommel model is given as:

    .. math::

        - \frac{R}{D}\nabla^2 \psi - \beta\frac{\partial \psi}{\partial x} =
        - \frac{\hat\nabla \cdot \vec\tau}{\rho_0 D}
    
    Invert this equation for the streamfunction :math:`\psi` given wind-stress
    curl :math:`\hat\nabla \cdot \vec\tau`.

    
    Parameters
    ----------
    curl: xarray.DataArray
        Wind stress curl.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* beta: Meridional derivative of Coriolis parameter.
		* R: Laplacian viscosity.
		* D: Depth of the ocean or mixed layer depth.
		* rho0: Density of the fluid.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (streamfunction) of the SOR inversion.
    """
    return __template(__coeffs_Stommel_test, inv_standard2D_test, 2, curl, dims, coords,
                      icbc, ['beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_StommelMunk(curl, dims, coords='lat-lon', icbc=None,
                       mParams=default_mParams, iParams=default_iParams):
    r"""Inverting Stommel-Munk model.

    The Stommel-Munk model is given as:

    .. math::

        A_4\nabla^4\psi - \frac{R}{D}\nabla^2 \psi
        - \beta\frac{\partial \psi}{\partial x} =
        - \frac{\hat\nabla \cdot \vec\tau}{\rho_0 D}
    
    Invert this equation for the streamfunction :math:`\psi` given wind-stress
    curl :math:`\hat\nabla \cdot \vec\tau`.

    
    Parameters
    ----------
    curl: float or xarray.DataArray
        Wind-stress curl (N/m^2).
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* beta: Meridional derivative of Coriolis parameter.
		* A4: Hyperviscosity.
		* R: Laplacian viscosity.
		* D: Depth of the ocean or mixed layer depth.
		* rho0: Density of the fluid.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results of the SOR inversion.
    """
    return __template(__coeffs_StommelMunk, inv_general2D_bih, 2, curl, dims, coords,
                      icbc, ['A4', 'beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_StommelArons(Q, dims, coords='lat-lon', icbc=None, 
                        mParams=default_mParams, iParams=default_iParams):
    r"""Inverting Stommel-Arons model.

    The Stommel-Arons model is given as:

    .. math::

        \epsilon   u  =  fv - \frac{\partial \phi}{\partial x}\\\\
        \epsilon   v  = -fu - \frac{\partial \phi}{\partial y}\\\\
        \left(\frac{\partial u}{\partial x}
        +\frac{\partial v}{\partial y}\right) = -Q
    
    Invert this equation for the mass distribution :math:`\phi` given
    the diabatic heating function :math:`Q`.

    
    Parameters
    ----------
    Q: xarray.DataArray
        Diabatic heating function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter at south BC if on beta plane.
		* beta: Meridional derivative of f.
		* epsilon: Linear damping coefficient for velocity only.

    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (mass distribution) of the SOR inversion.
    """
    return __template(__coeffs_StommelArons, inv_general2D, 2, Q, dims, coords,
                      icbc, ['f0', 'beta', 'epsilon', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_geostrophic(lapPhi, dims, coords='lat-lon', icbc=None,
                       mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the geostrophic balance model.

    The geostrophic balance model is given as:

    .. math::

        \frac{1}{\partial y}\left(f\frac{\partial \psi}{\partial y}\right)+
        \frac{1}{\partial x}\left(f\frac{\partial \psi}{\partial x}\right)=\nabla^2 \Phi
    
    Invert this equation for the geostrophic streamfunction :math:`\psi` given
    the Laplacian of geopotential field :math:`\nabla^2\Phi`.

    
    Parameters
    ----------
    lapPhi: xarray.DataArray
        Laplacian of geopotential.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter.
		* beta: Meridional derivative of Coriolis parameter.

    iParams: dict
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (geostrophic streamfunction) of the SOR inversion.
    """
    return __template(__coeffs_geostrophic, inv_standard2D, 2, lapPhi, dims, coords,
                      icbc, ['f0', 'beta', 'Omega', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_BrethertonHaidvogel(h, dims, coords='cartesian', icbc=None,
                               mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the Bretherton-Haiduogel model.

    The Bretherton-Haiduogel model is given as:

    .. math::

        \nabla^2\psi - \lambda D \psi=-f_0-\beta y-\frac{f_0}{D}h \Phi
    
    Invert this equation for the geostrophic streamfunction :math:`\psi` given
    the topography :math:`h`.

    
    Parameters
    ----------
    lapPhi: xarray.DataArray
        Laplacian of geopotential.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter.
		* beta: Meridional derivative of Coriolis parameter.
		* D: Constant total depth of the fluid (>> h).
		* lambda: Lagrangian multiplier, determined by the total energy.

    iParams: dict
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results (geostrophic streamfunction) of the SOR inversion.
    """
    return __template(__coeffs_Bretherton, inv_standard2D_test, 2, h, dims, coords,
                      icbc, ['f0', 'beta', 'D', 'lambda', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_Fofonoff(F, dims, coords='cartesian', icbc=None,
                    mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the Fofonoff (1954) model.

    The equation is given as:

    .. math::
        
        \nabla^2 \psi - c_0 \psi = c_1 - f
    
    Invert the equation for :math:`\psi` given :math:`f`.
    
    Parameters
    ----------
    F: xarray.DataArray
        Forcing function.  Note that Forcing is irrelevant, only coordinates
        are used here.  The true forcing (Coriolis parameter f) will be
        calculated automatically depending on the geometry of the domain.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribed inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* c0: Linear coefficient (>0).
		* c1: A constant.
  		* f0: Coriolis parameter.
  		* beta: Meridional derivative of Coriolis parameter.
        
    iParams: dict, optional
        Iteration parameters.
    
    Returns
    -------
    xarray.DataArray
        Results of the SOR inversion.
    """
    return __template(__coeffs_Fofonoff, inv_standard2D_test, 2, F, dims, coords,
                      icbc, ['c0', 'c1', 'f0', 'beta', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_omega(F, dims, coords='lat-lon', icbc=None,
                 mParams=default_mParams, iParams=default_iParams):
    r"""Inverting the omega equation.

    The omega equation is given as:

    .. math::

        \frac{f^2}{N^2}\frac{\partial^2 \omega}{\partial z^2} +
        \frac{\partial^2 \omega}{\partial y^2} +
        \frac{\partial^2 \omega}{\partial x^2} = \frac{F}{N^2}

    This is slightly changed to:

    .. math::

        \frac{1}{\partial z}\left(f^2\frac{\partial \omega}{\partial z}\right)+
        \frac{1}{\partial y}\left(N^2\frac{\partial \omega}{\partial y}\right)+
        \frac{1}{\partial x}\left(N^2\frac{\partial \omega}{\partial x}\right)=F

    Invert this equation for the vertical velocity :math:`\omega` given
    the forcing function :math:`F`.


    Parameters
    ----------
    F: xarray.DataArray
        A forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter at south BC on beta plane.
		* beta: Meridional derivative of Coriolis parameter.
        * N2: Buoyancy frequency = g/theta0 * dtheta/dz = -R*pi/p * dtheta/dp.

    iParams: dict, optional
        Iteration parameters.

    Returns
    -------
    xarray.DataArray
        Results (vertical velocity) of the SOR inversion.
    """
    if isinstance(mParams['N2'], xr.DataArray):
        if not np.isfinite(mParams['N2'][1:]).all():
            raise Exception('inifinite stratification coefficient A')
        
        if np.isnan(mParams['N2'][1:]).any():
            raise Exception('nan in coefficient A')
        
        if (mParams['N2'][1:]<=0).any():
            raise Exception('unstable stratification in coefficient A')
    
    return __template(__coeffs_omega, inv_standard3D, 3, F, dims, coords,
                      icbc, ['f0', 'beta', 'N2', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)


def invert_3DOcean(F, dims, coords='lat-lon', icbc=None,
                   mParams=default_mParams, iParams=default_iParams):
    r"""Inverting 3D ocean flow.

    The 3D ocean flow equation is given as:

    .. math::

        c_3 \frac{\partial^2 \psi}{\partial z^2} +
        c_1 \frac{\partial^2 \psi}{\partial y^2} +
        c_1 \frac{\partial^2 \psi}{\partial x^2} +
        c_3 \frac{\partial \psi}{\partial z} +
        c_1 \frac{\partial \psi}{\partial y} +
        c_1 \frac{\partial \psi}{\partial x} = F

    Invert this equation for the streamfunction :math:`\psi` given
    the forcing function :math:`F`.


    Parameters
    ----------
    F: xarray.DataArray
        A forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat', 'lon'].
    coords: {'lat-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Parameters required for this model are:

		* f0: Coriolis parameter at south BC on beta plane.
		* beta: Meridional derivative of Coriolis parameter.
		* epsilon: Linear damping coefficient for momentum.
        * N2: Buoyancy frequency = g/theta0 * dtheta/dz = -R*pi/p * dtheta/dp.
        * k:linear damping coefficient for buoyancy
        
    iParams: dict, optional
        Iteration parameters.

    Returns
    -------
    xarray.DataArray
        Results (streamfunction) of the SOR inversion.
    """
    if isinstance(mParams['N2'], xr.DataArray):
        if not np.isfinite(mParams['N2'][1:]).all():
            raise Exception('inifinite stratification coefficient A')
        
        if np.isnan(mParams['N2'][1:]).any():
            raise Exception('nan in coefficient A')
        
        if (mParams['N2'][1:]<=0).any():
            raise Exception('unstable stratification in coefficient A')
    
    return __template(__coeffs_3DOcean, inv_general3D, 3, F, dims, coords,
                      icbc, ['f0', 'beta', 'epsilon', 'N2', 'k', 'g', 'Omega', 'Rearth'],
                      mParams, iParams)



"""
Some high-level functions are based on application functions
"""
def animate_iteration(app_name, F, dims, coords='lat-lon', icbc=None,
                      mParams=default_mParams, iParams=default_iParams,
                      loop_per_frame=5, max_frames=30):
    r"""Animate the iteration process.

    All the `invert_xxx` function can be animated here.  This function will add
    a `iter` dimension similar to a `time` dimension that shows the results at
    different iteration steps.
    
    Parameters
    ----------
    app_name: {'Poisson', 'PV2D', 'GillMatsuno', 'Eliassen', 'geostrophic', 'StommelMunk', 'RefState', 'Omega'}
        Application name as a suffix of `invert_xxx`.
    F: xarray.DataArray
        A forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: str, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Model parameters.  None for the Poisson model.
    iParams: dict, optional
        Iteration parameters.
    loop_per_frame: int, optional
        Iteration loop count per frame.
    max_frames: int, optional
        Max frames beyond which loop is stopped.
    
    Returns
    -------
    xarray.DataArray
        The result, in which an extra dimension called `iter` will be added.
    """
    len_nc = 0
    
    for sel in loop_noncore(F, dims):
        len_nc = len_nc + 1
    
    if len_nc != 1:
        raise Exception('For 2D case, only 2D slice  F is allowed;\n'+
                        'For 3D case, only 3D volume F is allowed.')
    
    app_name = app_name.lower()
    dimLen = len(dims)
    
    iParams = __update(default_iParams, iParams)
    
    if   app_name == 'poisson':
        coef_func = __coeffs_Poisson
        invt_func = inv_standard2D
        validMPs  = ['g', 'Omega', 'Rearth']
        
    elif app_name == 'pv2d':
        coef_func = __coeffs_PV2D
        invt_func = inv_standard2D
        validMPs  = ['f0', 'beta', 'N2', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'geostrophic':
        coef_func = __coeffs_geostrophic
        invt_func = inv_standard2D
        validMPs  = ['f0', 'beta', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'gillmatsuno':
        coef_func = __coeffs_GillMatsuno
        invt_func = inv_general2D
        validMPs  = ['f0', 'beta', 'epsilon', 'Phi', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'eliassen':
        coef_func = __coeffs_Eliassen
        invt_func = inv_standard2D
        validMPs  = ['A', 'B', 'C', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'stommel':
        coef_func = __coeffs_Stommel
        invt_func = inv_general2D_bih
        validMPs  = ['beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'stommelmunk':
        coef_func = __coeffs_StommelMunk
        invt_func = inv_general2D_bih
        validMPs  = ['A4', 'beta', 'R', 'D', 'rho0', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'refstate':
        coef_func = __coeffs_RefState
        invt_func = inv_standard2D
        validMPs  = ['Ang0', 'Gamma', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'brethertonhaidvogel':
        coef_func = __coeffs_Bretherton
        invt_func = inv_standard2D_test
        validMPs  = ['f0', 'beta', 'D', 'lambda', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'fofonoff':
        coef_func = __coeffs_Fofonoff
        invt_func = inv_standard2D_test
        validMPs  = ['c0', 'c1', 'f0', 'beta', 'g', 'Omega', 'Rearth']
        
    elif app_name == 'omega':
        coef_func = __coeffs_omega
        invt_func = inv_standard3D
        validMPs  = ['f0', 'beta', 'N2', 'g', 'Omega', 'Rearth']
        
    elif app_name == '3Docean':
        coef_func = __coeffs_omega
        invt_func = inv_standard3D
        validMPs  = ['f0', 'beta', 'N2', 'epsilon', 'k', 'g', 'Omega', 'Rearth']
    else:
        raise Exception('unsupported problem: '+app_name+', should be one of:\n'+
                        "'Poisson'\n'PV2D'\n'GillMatsuno'\n'Eliassen'\n"+
                        "'geostrophic'\n'StommelMunk'\n'RefState'\n'omega'")
    
    mParams = __update(default_mParams, mParams, validMPs)
    
    ######  1. calculating the coefficients  ######
    maskF, initS, coeffs = coef_func(F, dims, coords, mParams, iParams, icbc)
    
    ######  2. calculating the parameters  ######
    if dimLen == 2:
        ps = __cal_params2D(maskF[dims[0]], maskF[dims[1]], coords,
                            Rearth=mParams['Rearth'])
    elif dimLen == 3:
        ps = __cal_params3D(maskF[dims[0]], maskF[dims[1]], maskF[dims[2]], coords,
                            Rearth=mParams['Rearth'])
    else:
        raise Exception('dimension length should be one of [2, 3]')
    
    iParams = __update(ps, iParams)
    
    if iParams['debug']:
        __print_params(iParams)
    
    ######  3. inverting the solution  ######
    S = initS
    
    iParams['mxLoop'   ] = loop_per_frame
    iParams['printInfo'] = False
    
    lst = []
    frame = 0
    while True:
        frame += 1
        
        S = invt_func(*coeffs, maskF, initS, dims, iParams)
        
        lst.append(S.copy())
        
        if frame >= max_frames:
            break
    
    re = xr.concat(lst, dim='iter').rename('inverted')
    re['iter'] = xr.DataArray(np.arange(loop_per_frame,
                                        loop_per_frame*(max_frames+1),
                                        loop_per_frame),
                              dims=['iter'])
    
    ######  4. properly de-masking  ######
    if icbc is None:
        re = re.where(maskF!=_undeftmp, other=iParams['undef']).rename('inverted')
    else:
        re = re.rename('inverted')
    
    return re


def invert_MultiGrid(invert_func, *args, ratio=3, gridNo=3, **kwargs):
    r"""Using multi-grid method to do the inversion (test only now).

    All the `invert_xxx` function can be solved using multi-grid method here.
    
    Parameters
    ----------
    app_name: {'Poisson', 'PV2D', 'GillMatsuno', 'Eliassen', 'geostrophic', 'StommelMunk', 'RefState', 'Omega'}
        Application name as a suffix of `invert_xxx`.
    F: xarray.DataArray
        A forcing function.
    dims: list
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: str, optional
        Coordinate combinations in which inversion is performed.
    icbc: xarray.DataArray, optional
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    mParams: dict, optional
        Model parameters.  None for the Poisson model.
    iParams: dict, optional
        Iteration parameters.
    ratio: int, optional
        Ratio of multi grids.
    gridNo: int, optional
        Number of multi grids.
    
    Returns
    -------
    xarray.DataArray
        The result, in which an extra dimension called `iter` will be added.
    """
    from utils.XarrayUtils import coarsen
    
    ratios = [10, 6, 3, 1]
    
    mxLoop = kwargs['mxLoop'] if 'mxLoop' in kwargs else 5000
    
    loops  = [mxLoop*ratio/10 for ratio in ratios]
    
    if 'dims' not in kwargs:
        raise Exception('kwarg dims= should be provided')
    
    dims = kwargs['dims']
        
    if 'BCs' in kwargs:
        BCs = kwargs['BCs']
    else:
        BCs = ['fixed', 'fixed']
    
    periodics = [dim for dim, BC in zip(dims, BCs) if 'periodic' in BC]
    
    fs = [[coarsen(arg, kwargs['dims'], ratio=ratio,
                   periodic=periodics) for arg in args] for ratio in ratios]
    
    o_guess = fs[0][0] - fs[0][0]
    
    o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    os = []
    
    for i, (ratio, frs, loop) in enumerate(zip(ratios, fs, loops)):
        kwargs['mxLoop'] = loop
        
        # iteration over coarse grid
        o_guess = invert_func(*frs, **kwargs)
        os.append(o_guess)
        
        print('finish grid of ratio =', ratio, ', loop =', loop)
        
        if ratio == 1:
            break
        
        o_guess = o_guess.interp_like(frs[0])
        o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    
    return o_guess, fs, os


def _invert_omega_MG(force, S, dims, BCs=['fixed', 'fixed', 'fixed'],
                    coords='latlon', f0=None, beta=None,
                    undef=np.nan, mxLoop=5000, tolerance=1e-6,
                    optArg=None, printInfo=True, debug=False,
                    icbc=None, ratio=4, gridNo=3):
    from utils.XarrayUtils import coarsen
    
    ratios = [10, 6, 3, 1]
    loops  = [mxLoop*ratio/30 for ratio in ratios]
    
    fs = [coarsen(force, dims, ratio=ratio) for ratio in ratios]
    Ss = [coarsen(S    , dims, ratio=ratio) for ratio in ratios]
    
    o_guess = fs[0] - fs[0]
    o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
    os = []
    
    for i, (ratio, frc, stab, loop) in enumerate(zip(ratios, fs, Ss, loops)):
        # iteration over coarse grid
        o_guess = invert_omega(frc, stab, dims=dims, BCs=BCs, f0=f0,
                               coords=coords, beta=beta, undef=undef,
                               mxLoop=loop, tolerance=tolerance,
                               optArg=optArg, debug=debug, icbc=o_guess,
                               printInfo=printInfo)
        os.append(o_guess)
        
        print('finish grid of ratio =', ratio, ', loop =', loop)
        
        if ratio == 1:
            break
        
        o_guess = o_guess.interp_like(fs[i+1])
        o_guess = xr.where(np.isnan(o_guess), 0, o_guess) # maskout nan
        
        # # iteration over fine grid from the coarse initial guess
        # o = invert_omegaEquation(force, S, dims=dims, BCs=BCs, coords=coords,
        #                          f0=f0, beta=beta, undef=undef, mxLoop=mxLoop/50,
        #                          tolerance=tolerance, optArg=optArg, debug=debug,
        #                          printInfo=printInfo, icbc=o_guess)
    
    return o_guess, fs, os


def cal_flow(S, dims, coords='lat-lon', BCs=['fixed', 'fixed'],
             vtype='streamfunction', mParams=default_mParams):
    r"""Calculate flow vector using streamfunction or velocity potential.

    Parameters
    ----------
    S: xarray.DataArray
        Streamfunction or velocity potential.
    dims: list of str
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    coords: {'lat-lon', 'z-lat', 'z-lon', 'cartesian'}, optional
        Coordinate combinations in which inversion is performed.
    BCs: dict
        Boundary conditions e.g., ['fixed', 'periodic'] for 2D case.
    vtype: {'streamfunction', 'velocitypotential', 'GillMatsuno'}, optional
        Type of the given variable, which determins the returns.

    Returns
    -------
    tuple
        Flow vector components.
    """
    if vtype.lower() not in ['streamfunction', 'velocitypotential', 'gillmatsuno']:
        raise Exception('unsupported vtype: ' + vtype + ', should be one of:\n'+
                        "['streamfunction', 'velocitypotential', 'gillmatsuno']")
    
    if vtype != 'GillMatsuno': # Poisson case
        if vtype == 'streamfunction':
            sf = True
        else:
            sf = False
        
        from .finitediffs import FiniteDiff
        
        if coords.lower() == 'lat-lon':
            dmap = {'Y': dims[0], 'X': dims[1]}
            
            fd = FiniteDiff(dmap, {'Y': (BCs[0], BCs[0]),
                                   'X': (BCs[1], BCs[1])}, coords='lat-lon')
            
            grdy, grdx = fd.grad(S, ['Y', 'X'])
            
            if sf:
                return -grdy, grdx
            else:
                return grdx, grdy
        
        elif coords.lower() == 'z-lat':
            dmap = {'Z': dims[0], 'Y': dims[1]}
            
            fd = FiniteDiff(dmap, {'Z': (BCs[0], BCs[0]),
                                   'Y': (BCs[1], BCs[1])}, coords='lat-lon')
            
            grdz, grdy = fd.grad(S, ['Z', 'Y'])
            
            cos = np.cos(np.deg2rad(S[dims[1]]))
            
            grdz, grdy = grdz / cos, grdy / cos
            
            grdy = grdy.where(np.abs(grdy[dims[1]])!=90, other=0)
            
            if sf:
                return -grdz, grdy
            else:
                return grdy, grdz
            
        elif coords.lower() == 'z-lon':
            dmap = {'Z': dims[0], 'X': dims[1]}
            
            fd = FiniteDiff(dmap, {'Z': (BCs[0], BCs[0]),
                                   'X': (BCs[1], BCs[1])}, coords='lat-lon')
            
            grdz, grdx = fd.grad(S, ['Z', 'X'])
            
            if sf:
                return grdz, -grdx
            else:
                return grdx, grdz
            
        elif coords.lower() == 'cartesian':
            dmap = {'Y': dims[0], 'X': dims[1]}
            
            fd = FiniteDiff(dmap, {'Y': (BCs[0], BCs[0]),
                                   'X': (BCs[1], BCs[1])}, coords=coords)
            
            grdy, grdx = fd.grad(S, ['Y', 'X'])
            
            if sf:
                return -grdy, grdx
            else:
                return grdx, grdy
            
        else:
            raise Exception('unsupported coords ' + coords +
                            ', should be [lat-lon, z-lat, z-lon, cartesian]')
    
    else: # GillMatsuno case
        mParams = __update(default_mParams, mParams,
                           ['f0', 'beta', 'epsilon', 'Phi', 'Omega', 'Rearth'])
        
        eps    = mParams['epsilon']
        f0     = mParams['f0']
        beta   = mParams['beta']
        Omega  = mParams['Omega']
        Rearth = mParams['Rearth']
        
        if coords.lower() == 'lat-lon':
            lats = np.deg2rad(S[dims[0]])
            cosLat = np.cos(lats)
            sinLat = np.sin(lats)
            
            f = 2.0 * Omega * sinLat
            deg2m = np.deg2rad(1.0) * Rearth
            
            coef1 = eps / (eps**2.0 + f**2.0)
            coef2 = f   / (eps**2.0 + f**2.0)
            
            c1 = - coef1 * S.differentiate(dims[1]) / deg2m / cosLat \
                 - coef2 * S.differentiate(dims[0]) / deg2m
            c2 = - coef1 * S.differentiate(dims[0]) / deg2m \
                 + coef2 * S.differentiate(dims[1]) / deg2m / cosLat
        elif coords.lower() == 'cartesian':
            ydef = S[dims[0]]
            f = f0 + beta * ydef
            
            coef1 = eps / (eps**2.0 + f**2.0)
            coef2 = f   / (eps**2.0 + f**2.0)
            
            c1 = - coef1 * S.differentiate(dims[1]) \
                 - coef2 * S.differentiate(dims[0])
            c2 = - coef1 * S.differentiate(dims[0]) \
                 + coef2 * S.differentiate(dims[1])
        else:
            raise Exception('unsupported coords ' + coords +
                            ', should be [lat-lon, cartesian]')
        
        return c1, c2



"""
Below are the helper methods of these applications
"""
def __template(coef_func, inv_func, dimLen,
               F, dims, coords='lat-lon', icbc=None, validParams=[],
               mParams=default_mParams, iParams=default_iParams):
    r"""Template for the whole inverting process.
    
    Parameters
    ----------
    coef_func: function
        Function for calculating the coefficients of the equations.
    inv_func: function
        Function for the inversion.
    dimLen: int
        Problems of 2D or 3D, should be one of [2, 3].
    F: xarray.DataArray
        A given forcing.
    dims: list
        Dimension combination for the inversion e.g., ['lev', 'lat', 'lon']
        for 3D case and ['lat', 'lon'] for 2D case.
    coords: str, optional
        Coordinates in ['lat-lon', 'cartesian'] are supported.
    icbc: xarray.DataArray
        Prescribe inital condition/guess (IC) and boundary conditions (BC).
    validParams: list of str
        Valid mParams for a specific model.
    mParams: dict, optional
		Parameters that depends on a specific model.
    iParams: dict, optional
        Iteration parameters.
        
    Returns
    -------
    xarray.DataArray
        Results of the SOR inversion.
    """
    if len(dims) != dimLen:
        raise Exception('{0:2d} dimensional forcing are needed'.format(dimLen))
    
    iParams = __update(default_iParams, iParams)
    mParams = __update(default_mParams, mParams, validParams)
    
    ######  1. calculating the coefficients  ######
    maskF, initS, coeffs = coef_func(F, dims, coords, mParams, iParams, icbc)
    
    ######  2. calculating the parameters  ######
    if dimLen == 1:
        ps = __cal_params1D(maskF[dims[0]], coords,
                            Rearth=mParams['Rearth'])
    elif dimLen == 2:
        ps = __cal_params2D(maskF[dims[0]], maskF[dims[1]], coords,
                            Rearth=mParams['Rearth'])
    elif dimLen == 3:
        ps = __cal_params3D(maskF[dims[0]], maskF[dims[1]], maskF[dims[2]], coords,
                            Rearth=mParams['Rearth'])
    else:
        raise Exception('dimension length should be one of [2, 3]')
        
    iParams = __update(ps, iParams)
    
    if iParams['debug']:
        __print_params(iParams)
    
    ######  3. inverting the solution  ######
    S = inv_func(*coeffs, maskF, initS, dims, iParams)
    
    ######  4. properly de-masking  ######
    if icbc is None:
        S = S.where(maskF!=_undeftmp, other=iParams['undef']).rename('inverted')
    else:
        S = S.rename('inverted')
    
    return S


def __coeffs_Poisson(force, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Poisson equation."""
    maskF, initS, zero = __mask_FS(force, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(maskF[dims[0]])
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.0) # shift half grid
        
        A = zero + cosH
        B = zero
        C = zero + 1.0 / cosG
        F = (maskF * cosG).where(maskF!=_undeftmp, _undeftmp)

    elif coords.lower() == 'z-lat': # dims[0] is z, dims[1] is lat
        cosG = np.cos(np.deg2rad(maskF[dims[1]]))
        
        A = zero + 1.0
        B = zero
        C = zero + 1.0
        F = (maskF * cosG).where(maskF!=_undeftmp, _undeftmp)

    elif coords.lower() == 'z-lon': # dims[0] is z, dims[1] is lon
        # assuming at the equator and in this case cosLat = 1.0
        # which is exactly the same as cartesian case
        A = zero + 1.0
        B = zero
        C = zero + 1.0
        F = maskF.where(maskF!=_undeftmp, _undeftmp)
        
    elif coords.lower() == 'cartesian':
        A = zero + 1.0
        B = zero
        C = zero + 1.0
        F = maskF.where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, z-lat, z-lon, cartesian]')
    
    return F, initS, (A, B, C)


def __coeffs_RefState(Q, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for reference state."""
    ang0  = mParams['ang0']
    Gamma = mParams['Gamma']
    g     = mParams['g']
    
    maskF, initS, zero = __mask_FS(Q, dims, iParams, icbc)

    if coords.lower() == 'z-lat': # dims[0] is θ, dims[1] is lat
        lats = np.deg2rad(maskF[dims[1]])
        sinL = np.sin(lats)
        
        A = zero + sinL
        B = zero
        C = zero + Gamma * g / Q / maskF[dims[1]]
        F = maskF.where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is θ, dims[1] is r
        A = zero + 2.0 * ang0 / maskF[dims[1]]**3.0
        B = zero
        C = zero + Gamma * g / Q / maskF[dims[1]]
        F = maskF.where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [z-lat, cartesian]')
    
    return F, initS, (A, B, C)


def __coeffs_RefStateSWM(Q, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for reference state of a shallow-water model."""
    M0     = mParams['M0']
    C0     = mParams['C0']
    g      = mParams['g']
    Rearth = mParams['Rearth']
    Omega  = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(Q, dims, iParams, icbc)
    
    import numba as nb
    
    @nb.jit(nopython=True, cache=False)
    def diff_2nd(M, cosH, delY):
        re = np.zeros_like(M)
        
        J = len(re)
        # print('inside: ', re.shape, cosH.shape, delY)
        
        for j in range(1, J-1):
            re[j] = (((M[j+1] - M[j  ]) / cosH[j+1]) - 
                     ((M[j  ] - M[j-1]) / cosH[j  ])) / (delY ** 2)
        
        return re
    
    if coords.lower() == 'lat': # dims[0] is θ, dims[1] is lat
        lats = np.deg2rad(maskF[dims[0]])
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.0) # shift half grid
        sinG = np.sin(lats)
        asin = Rearth * sinG
        acos = Rearth * cosG
        
        acos = xr.where(acos<0, -acos*0.1, acos) # ensure positive 0 at poles
        
        diff = xr.apply_ufunc(diff_2nd, M0, cosH,
                              np.abs(lats[0]-lats[1]) * Rearth,
                              dask='parallelized',
                              vectorize=True,
                              input_core_dims=[[dims[0]], [dims[0]], []],
                              output_core_dims=[[dims[0]]])
        
        A = zero + 1.0 / cosH
        B = zero - C0 * maskF * asin  / (np.pi * g* acos**3.0)
        F = zero - (asin * C0**2.0 / (2.0 * np.pi * g* acos**3.0)) + \
                   (2.0 * np.pi * Omega**2.0 * asin * acos) / g - diff
    
    elif coords.lower() == 'cartesian': # dims[0] is θ, dims[1] is r
        raise Exception('not supported for cartesian coordinates')

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [z-lat, cartesian]')
    
    return F, initS, (A, B)


def __coeffs_PV2D(PV, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for QG PV equation."""
    f0 = mParams['f0']
    N2 = mParams['N2']
    
    maskF, initS, zero = __mask_FS(PV, dims, iParams, icbc)

    if coords.lower() == 'z-lat': # dims[0] is p, dims[1] is lat
        A = zero + f0**2 / N2  # should use f(lat)
        B = zero
        C = zero + 1
        F = maskF.where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is p, dims[1] is r
        A = zero + f0**2 / N2
        B = zero
        C = zero + 1
        F = maskF.where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [z-lat, cartesian]')
    
    return F, initS, (A, B, C)


def __coeffs_Eliassen(force, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Eliassen model."""
    Am = mParams['A']
    Bm = mParams['B']
    Cm = mParams['C']
    
    maskF, initS, zero = __mask_FS(force, dims, iParams, icbc)

    if coords.lower() == 'z-lat': # dims[0] is θ, dims[1] is lat
        A = zero + Am
        B = zero + Bm
        C = zero + Cm
        F = maskF.where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is θ, dims[1] is r
        A = zero + Am
        B = zero + Bm
        C = zero + Cm
        F = maskF.where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [z-lat, cartesian]')
    
    return F, initS, (A, B, C)


def __coeffs_GillMatsuno(Q, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Gill-Matsuno model."""
    Phi     = mParams['Phi' ]
    epsilon = mParams['epsilon']
    f0      = mParams['f0'  ]
    beta    = mParams['beta']
    Omega   = mParams['Omega']
    Rearth  = mParams['Rearth']
    
    maskF, initS, zero = __mask_FS(Q, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(Q[dims[0]])
        cosL = np.cos(lats)
        
        f = 2.0 * Omega * np.sin(lats)
        
        c1 = epsilon / (epsilon**2. + f**2.)
        c2 = f       / (epsilon**2. + f**2.)
        deg2m = Rearth / 180. * np.pi
        
        A = zero + c1 * Phi
        B = zero
        C = zero + c1 * Phi / cosL**2.
        D = zero + Phi *(c1.differentiate(dims[0]) / deg2m + c1*np.tan(lats)/Rearth)
        E = zero - Phi * c2.differentiate(dims[0]) / deg2m / cosL
        F = zero - epsilon
        G = maskF.where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is y, dims[1] is x
        ydef = Q[dims[0]]
        f    = f0 + beta * ydef
        
        c1 = epsilon / (epsilon**2. + f**2.)
        c2 = f       / (epsilon**2. + f**2.)
        
        A = zero + c1 * Phi
        B = zero
        C = zero + c1 * Phi
        D = zero + Phi * c1.differentiate(dims[0])
        E = zero - Phi * c2.differentiate(dims[0])
        F = zero - epsilon
        G = maskF.where(maskF!=_undeftmp, _undeftmp)
        
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return G, initS, (A, B, C, D, E, F)


def __coeffs_GillMatsuno_test(Q, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Gill-Matsuno model."""
    Phi     = mParams['Phi' ]
    epsilon = mParams['epsilon']
    f0      = mParams['f0'  ]
    beta    = mParams['beta']
    Omega   = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(Q, dims, iParams, icbc)

    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(Q[dims[0]])
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.)
        
        fG = 2. * Omega * np.sin(lats)
        fH = 2. * Omega * np.sin((lats+lats.shift({dims[0]:1}))/2.)
        
        c1G = epsilon / (epsilon**2. + fG**2.)
        c1H = epsilon / (epsilon**2. + fH**2.)
        c2G = fG      / (epsilon**2. + fG**2.)
        
        A = zero + c1H * Phi * cosH
        B = zero - c2G * Phi
        C = zero + c2G * Phi
        D = zero + c1G * Phi / cosG
        E = zero - epsilon * cosG
        F = (maskF * cosG).where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is y, dims[1] is x
        ydef = Q[dims[0]]
        fG = f0 + beta * ydef
        fH = f0 + beta * (ydef+ydef.shift({dims[0]:1}))/2.
        
        c1G = epsilon / (epsilon**2. + fG**2.)
        c1H = epsilon / (epsilon**2. + fH**2.)
        c2G = fG      / (epsilon**2. + fG**2.)
        
        A = zero + c1H * Phi
        B = zero - c2G * Phi
        C = zero + c2G * Phi
        D = zero + c1G * Phi
        E = zero - epsilon
        F = (maskF).where(maskF!=_undeftmp, _undeftmp)
        
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return F, initS, (A, B, C, D, E)


def __coeffs_Stommel(curl, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Stommel model."""
    beta   = mParams['beta']
    R      = mParams['R'   ]
    depth  = mParams['D'   ]
    rho0   = mParams['rho0']
    Rearth = mParams['Rearth']
    Omega  = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(curl, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(curl[dims[0]])
        cosL = np.cos(lats)
        
        A = zero - R / depth
        B = zero
        C = zero - R / depth / cosL**2.
        D = zero
        E = zero - 2. * Omega / Rearth
        F = zero
        G = (-maskF / depth / rho0).where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is θ, dims[1] is r
        A = zero - R / depth
        B = zero
        C = zero - R / depth
        D = zero
        E = zero - beta
        F = zero
        G = (-maskF / depth / rho0).where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, z-lat, z-lon, cartesian]')
    
    return G, initS, (A, B, C, D, E, F)


def __coeffs_Stommel_test(curl, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Stommel model."""
    f0    = mParams['f0'  ]
    beta  = mParams['beta']
    R     = mParams['R'   ]
    depth = mParams['D'   ]
    rho0  = mParams['rho0']
    Omega = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(curl, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(curl[dims[0]])
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.)
        f    = 2. * Omega * np.sin(lats)
        
        A = zero - R / depth * cosH
        B = zero - f
        C = zero + f
        D = zero - R / depth / cosG
        E = zero
        F = (-maskF / depth / rho0 * cosG).where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is θ, dims[1] is r
        ydef = curl[dims[0]]
        f = f0 + beta * ydef
        
        A = zero - R / depth
        B = zero - f
        C = zero + f
        D = zero - R / depth
        E = zero
        F = (-maskF / depth / rho0).where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, z-lat, z-lon, cartesian]')
    
    return F, initS, (A, B, C, D, E)


def __coeffs_StommelMunk(curl, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Stommel-Munk model."""
    beta   = mParams['beta']
    A4     = mParams['A4'  ]
    R      = mParams['R'   ]
    depth  = mParams['D'   ]
    rho0   = mParams['rho0']
    Omega  = mParams['Omega']
    Rearth = mParams['Rearth']
    
    maskF, initS, zero = __mask_FS(curl, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(curl[dims[0]])
        cosL = np.cos(lats)
        
        A = zero + A4
        B = zero
        C = zero + A4 / cosL**2.
        D = zero - R / depth
        E = zero
        F = zero - R / depth / cosL**2.
        G = zero
        H = zero - 2. * Omega / Rearth
        I = zero
        J = (-maskF / depth / rho0).where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is y, dims[1] is x
        A = zero + A4
        B = zero
        C = zero + A4
        D = zero - R / depth
        E = zero
        F = zero - R / depth
        G = zero
        H = zero - beta
        I = zero
        J = (-maskF / depth / rho0).where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return J, initS, (A, B, C, D, E, F, G, H, I)


def __coeffs_StommelArons(Q, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Stommel-Arons model."""
    epsilon = mParams['epsilon']
    f0      = mParams['f0']
    beta    = mParams['beta']
    Omega   = mParams['Omega']
    Rearth  = mParams['Rearth']
    
    maskF, initS, zero = __mask_FS(Q, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(Q[dims[0]])
        cosL = np.cos(lats)
        
        f = 2.0 * Omega * np.sin(lats)
        
        c1 = epsilon / (epsilon**2. + f**2.)
        c2 = f       / (epsilon**2. + f**2.)
        deg2m = Rearth / 180. * np.pi
        
        A = zero + c1
        B = zero
        C = zero + c1 / cosL**2.
        D = zero + (c1.differentiate(dims[0]) / deg2m + c1*np.tan(lats)/Rearth)
        E = zero - c2.differentiate(dims[0]) / deg2m / cosL
        F = zero
        G = maskF.where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is y, dims[1] is x
        ydef = Q[dims[0]]
        f    = f0 + beta * ydef
        
        c1 = epsilon / (epsilon**2. + f**2.)
        c2 = f       / (epsilon**2. + f**2.)
        
        A = zero + c1
        B = zero
        C = zero + c1
        D = zero + c1.differentiate(dims[0])
        E = zero - c2.differentiate(dims[0])
        F = zero
        G = maskF.where(maskF!=_undeftmp, _undeftmp)
        
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return G, initS, (A, B, C, D, E, F)


def __coeffs_geostrophic(lapPhi, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for geostrophic equation."""
    f0    = mParams['f0']
    beta  = mParams['beta']
    Omega = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(lapPhi, dims, iParams, icbc)

    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(maskF[dims[0]])
        
        sinG = np.sin(lats)
        sinH = np.sin((lats+lats.shift({dims[0]:1}))/2.)
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.)
        
        fH = 2. * Omega * sinH
        fG = 2. * Omega * sinG
        
        # regulation for near-zero f
        fH = xr.where(np.abs(fH)<2e-05, fH*1.5, fH)
        fG = xr.where(np.abs(fG)<2e-05, fG*1.5, fG)
        
        A = zero + fH * cosH
        B = zero
        C = zero + fG / cosG
        F =(lapPhi * cosG).where(lapPhi!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[0] is y, dims[1] is x
        ydef = maskF[dims[0]]
        fG = f0 + beta * ydef
        fH = f0 + beta * (ydef+ydef.shift({dims[0]:1}))/2.
        
        A = zero + fH
        B = zero
        C = zero + fG
        F = lapPhi.where(lapPhi!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return F, initS, (A, B, C)


def __coeffs_Bretherton(h, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Bretherton-Haiduogel equation."""
    f0    = mParams['f0']
    beta  = mParams['beta']
    depth = mParams['D']
    lamb  = mParams['lambda']
    Omega = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(h, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(maskF[dims[0]])
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.0) # shift half grid
        f    = 2. * Omega * np.sin(lats)
        
        A = zero + cosH
        B = zero
        C = zero
        D = zero + 1.0 / cosG
        E = zero - lamb * depth * cosG
        F = (-maskF*f/depth * cosG).where(maskF!=_undeftmp, _undeftmp)
        
    elif coords.lower() == 'cartesian':
        ydef = maskF[dims[0]]
        f = f0 + beta * ydef
        
        A = zero + 1.0
        B = zero
        C = zero
        D = zero + 1.0
        E = zero - lamb * depth
        F = (-maskF*f/depth).where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return F, initS, (A, B, C, D, E)


def __coeffs_Fofonoff(f, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for Bretherton-Haiduogel equation."""
    f0    = mParams['f0']
    beta  = mParams['beta']
    c0    = mParams['c0']
    c1    = mParams['c1']
    Omega = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(f, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[0] is lat, dims[1] is lon
        lats = np.deg2rad(maskF[dims[0]])
        cosG = np.cos(lats)
        cosH = np.cos((lats+lats.shift({dims[0]:1}))/2.0) # shift half grid
        f    = 2. * Omega * np.sin(lats)
        
        A = zero + cosH
        B = zero
        C = zero
        D = zero + 1.0 / cosG
        E = zero - c0 * cosG
        F = ((zero + c1 - f) * cosG).where(maskF!=_undeftmp, _undeftmp)
        
    elif coords.lower() == 'cartesian':
        ydef = maskF[dims[0]]
        f = f0 + beta * ydef
        
        A = zero + 1.0
        B = zero
        C = zero
        D = zero + 1.0
        E = zero - c0
        F = (zero + c1 - f).where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return F, initS, (A, B, C, D, E)


def __coeffs_omega(force, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for QG omega equation."""
    f0    = mParams['f0']
    beta  = mParams['beta']
    N2    = mParams['N2']
    Omega = mParams['Omega']
    
    maskF, initS, zero = __mask_FS(force, dims, iParams, icbc)

    if coords.lower() == 'lat-lon': # dims[1] is lat, dims[2] is lon
        lats = np.deg2rad(force[dims[1]])
        
        cosH = np.cos((lats+lats.shift({dims[1]:1}))/2.)
        cosG = np.cos(lats)
        
        f = 2. * Omega * np.sin(lats)
        
        A = zero + f**2 * cosG
        B = zero + N2*cosH
        C = zero + N2/cosG
        F = (maskF * cosG).where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[1] is y, dims[2] is x
        ydef = force[dims[1]]
        
        f = f0 + beta * ydef
        
        A = zero + f**2.
        B = zero + N2
        C = zero + N2
        F = maskF.where(maskF!=_undeftmp, _undeftmp)

    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return F, initS, (A, B, C)


def __coeffs_3DOcean(force, dims, coords, mParams, iParams, icbc):
    """Calculating coefficients for 3D ocean model."""
    f0      = mParams['f0']
    beta    = mParams['beta']
    epsilon = mParams['epsilon']
    N2      = mParams['N2']
    k       = mParams['k']
    Omega   = mParams['Omega']
    Rearth  = mParams['Rearth']
    
    maskF, initS, zero = __mask_FS(force, dims, iParams, icbc)
    
    if coords.lower() == 'lat-lon': # dims[1] is lat, dims[2] is lon
        lats = np.deg2rad(force[dims[1]])
        cosL = np.cos(lats)
        
        f = 2. * Omega * np.sin(lats)
        
        c1 = epsilon / (epsilon**2. + f**2.)
        c2 = f       / (epsilon**2. + f**2.)
        c3 = maskF[dims[0]] - maskF[dims[0]] + k / N2
        deg2m = Rearth / 180. * np.pi
        
        A = zero + c3
        B = zero + c1
        C = zero + c1 / cosL**2.
        D = zero + c3.differentiate(dims[0])
        E = zero +(c1.differentiate(dims[1]) / deg2m - c1*np.tan(lats)/Rearth)
        F = zero - c2.differentiate(dims[1]) / deg2m / cosL
        G = zero
        H = maskF.where(maskF!=_undeftmp, _undeftmp)
    
    elif coords.lower() == 'cartesian': # dims[1] is y, dims[2] is x
        ydef = force[dims[1]]
        f = f0 + beta * ydef
        
        c1 = epsilon / (epsilon**2. + f**2.)
        c2 = f       / (epsilon**2. + f**2.)
        c3 = maskF[dims[0]] - maskF[dims[0]] + k / N2
        const = 1 # check this
        
        A = zero + c3
        B = zero + c1
        C = zero + c1
        D = zero + c3.differentiate(dims[0])
        E = zero + c1.differentiate(dims[1]) / const
        F = zero - c2.differentiate(dims[1]) / const
        G = zero
        H = maskF.where(maskF!=_undeftmp, _undeftmp)
        
    else:
        raise Exception('unsupported coords ' + coords +
                        ', should be in [lat-lon, cartesian]')
    
    return H, initS, (A, B, C, D, E, F, G)


def __mask_FS(F, dims, iParams, icbc):
    r"""Properly mask forcing and output with _undeftmp.

    Parameters
    ----------
    F: xarray.DataArray
        Forcing function.
    dims: list of str
        Dimension combination for the inversion e.g., ['lat', 'lon'].
    iParams: dict
        Parameters.
    out: xarray.DataArray
        Output array.

    Returns
    -------
    maskF: xarray.DataArray
        Masked forcing function.
    initS: xarray.DataArray
        Initialized output.
    zero: xarray.DataArray
        Allocated array for later use.
    """
    ######  1. properly masking forcing with _undeftmp  ######
    if np.isnan(iParams['undef']):
        maskF = F.fillna(_undeftmp)
    else:
        maskF = F.where(F!=iParams['undef'], other=_undeftmp)
    
    zero = maskF - maskF
    
    ######  2. properly masking output with _undeftmp and BCs  ######
    if icbc is None:
        initS = zero.copy()
    else:
        dimVs = [maskF[dim] for dim in dims]
        conds = [dimV.isin([dimV[0], dimV[-1]]) for dimV in dimVs] 
        mask  = maskF == _undeftmp
        
        # applied fixed boundaries
        for cond, BC in zip(conds, iParams['BCs']):
            if BC != 'periodic':
                mask = np.logical_or(mask, cond)
        
        initS = xr.where(mask, icbc, 0)
    
    # loaded initS because dask cannot be modified
    return maskF, initS.load(), zero


def __cal_params3D(dim3_var, dim2_var, dim1_var, coords,
                   Rearth=default_mParams['Rearth'], debug=False):
    r"""Pre-calculate some parameters needed in SOR for the 3D cases.

    Parameters
    ----------
    dim3_var: xarray.DataArray
        Dimension variable of third dimension (e.g., lev).
    dim2_var: xarray.DataArray
        Dimension variable of second dimension (e.g., lat).
    dim1_var: xarray.DataArray
        Dimension variable of first  dimension (e.g., lon).
    debug: boolean
        Print result for debugging. The default is False.

    Returns
    -------
    re: dict
        Pre-calculated parameters.
    """
    gc3  = len(dim3_var)
    gc2  = len(dim2_var)
    gc1  = len(dim1_var)
    del3 = dim3_var.diff(dim3_var.name).values[0] # assumed uniform
    del2 = dim2_var.diff(dim2_var.name).values[0] # assumed uniform
    del1 = dim1_var.diff(dim1_var.name).values[0] # assumed uniform
    __uniform_interval(dim3_var, del3)
    __uniform_interval(dim2_var, del2)
    __uniform_interval(dim1_var, del1)
    
    if coords.lower() == 'lat-lon':
        del2 = np.deg2rad(del2) * Rearth # convert lat to m
        del1 = np.deg2rad(del1) * Rearth # convert lon to m
    elif coords.lower() == 'cartesian':
        pass
    else:
        raise Exception('unsupported coords for 3D case: ' + coords +
                        ', should be in [\'lat-lon\', \'cartesian\']')
    
    ratio1    = del1 / del2
    ratio2    = del1 / del3
    ratio1Sqr = ratio1 ** 2.0
    ratio2Sqr = ratio2 ** 2.0
    del1Sqr   = del1 ** 2.0
    epsilon   = (np.sin(np.pi/(2.0*gc1+2.0)) **2.0 +
                 np.sin(np.pi/(2.0*gc2+2.0)) **2.0 +
                 np.sin(np.pi/(2.0*gc3+3.0)) **2.0)
    optArg    = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    flags     = np.array([0.0, 1.0, 0.0])
    
    if debug:
        print('dim3_var: ', dim3_var)
        print('dim2_var: ', dim2_var)
        print('dim1_var: ', dim1_var)
        print('dim grids:', gc3, gc2, gc1)
        print('dim intervals: ', del3, del2, del1)
        print('ratio1Sqr, 2Sqr: ', ratio1Sqr, ratio2Sqr)
        print('del1Sqr: ', del1Sqr)
        print('optArg: ' , optArg)
    
    # store all and return
    re = {}
    
    re['gc3'      ] = gc3       # grid count in second dimension (e.g., lev)
    re['gc2'      ] = gc2       # grid count in second dimension (e.g., lat)
    re['gc1'      ] = gc1       # grid count in first  dimension (e.g., lon)
    re['del3'     ] = del3      # distance in third  dimension (unit: m or Pa)
    re['del2'     ] = del2      # distance in second dimension (unit: m)
    re['del1'     ] = del1      # distance in first  dimension (unit: m)
    re['ratio1'   ] = ratio1    # distance ratio: del1 / del2
    re['ratio2'   ] = ratio2    # ratio ** 4
    re['ratio1Sqr'] = ratio1Sqr # distance ratio: del1 / del2
    re['ratio2Sqr'] = ratio2Sqr # ratio ** 4
    re['del1Sqr'  ] = del1Sqr   # del1 ** 2
    re['optArg'   ] = optArg    # optimal argument for SOR
    re['flags'    ] = flags     # outputs of the SOR iteration:
                                #   [0] overflow or not
                                #   [1] tolerance
                                #   [2] loop count
    
    return re


def __cal_params2D(dim2_var, dim1_var, coords, Rearth=default_mParams['Rearth']):
    r"""Pre-calculate some parameters needed in SOR.

    Parameters
    ----------
    dim2_var: xarray.DataArray
        Dimension variable of second dimension (e.g., lat).
    dim1_var: xarray.DataArray
        Dimension variable of first  dimension (e.g., lon).
    debug: boolean
        Print result for debugging. The default is False.

    Returns
    -------
    re: dict
        Pre-calculated parameters.
    """
    gc2  = len(dim2_var)
    gc1  = len(dim1_var)
    del2 = dim2_var.diff(dim2_var.name).values[0] # assumed uniform
    del1 = dim1_var.diff(dim1_var.name).values[0] # assumed uniform
    __uniform_interval(dim2_var, del2)
    __uniform_interval(dim1_var, del1)
    
    if coords.lower() == 'lat-lon':
        del2 = np.deg2rad(del2) * Rearth # convert lat to m
        del1 = np.deg2rad(del1) * Rearth # convert lon to m
    elif coords.lower() == 'z-lat':
        del1 = np.deg2rad(del1) * Rearth # convert lat to m
    elif coords.lower() == 'z-lon':
        del1 = np.deg2rad(del1) * Rearth # convert lon to m
    elif coords.lower() == 'cartesian':
        pass
    else:
        raise Exception('unsupported coords for 2D case: ' + coords +
                        ', should be [lat-lon, cartesian]')
    
    ratio    = del1 / del2
    ratioSSr = ratio ** 4.0
    ratioSqr = ratio ** 2.0
    ratioQtr = ratio / 4.0
    del1Sqr  = del1 ** 2.0
    del1Tr   = del1 ** 3.0
    del1SSr  = del1 ** 4.0
    epsilon  = np.sin(np.pi/(2.0*gc1+2.0))**2 + np.sin(np.pi/(2.0*gc2+2.0))**2
    optArg   = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    flags    = np.array([0.0, 1.0, 0.0])
    
    # store all and return
    re = {}
    
    re['gc2'     ] = gc2       # grid count in second dimension (e.g., lat)
    re['gc1'     ] = gc1       # grid count in first  dimension (e.g., lon)
    re['del2'    ] = del2      # distance in second dimension (unit: m)
    re['del1'    ] = del1      # distance in first  dimension (unit: m)
    re['ratio'   ] = ratio     # distance ratio: del1 / del2
    re['ratioSSr'] = ratioSSr  # ratio ** 4
    re['ratioSqr'] = ratioSqr  # ratio ** 2
    re['ratioQtr'] = ratioQtr  # ratio ** 2 / 4
    re['del1Sqr' ] = del1Sqr   # del1 ** 2
    re['del1Tr'  ] = del1Tr    # del1 ** 3
    re['del1SSr' ] = del1SSr   # del1 ** 4
    re['optArg'  ] = optArg    # optimal argument for SOR
    re['flags'   ] = flags     # outputs of the SOR iteration:
                               #   [0] overflow or not
                               #   [1] tolerance
                               #   [2] loop count
    
    return re


def __cal_params1D(dim1_var, coords, Rearth=default_mParams['Rearth']):
    r"""Pre-calculate some parameters needed in SOR.

    Parameters
    ----------
    dim1_var: xarray.DataArray
        Dimension variable of first  dimension (e.g., lon).
    debug: boolean
        Print result for debugging. The default is False.

    Returns
    -------
    re: dict
        Pre-calculated parameters.
    """
    gc1  = len(dim1_var)
    del1 = dim1_var.diff(dim1_var.name).values[0] # assumed uniform
    __uniform_interval(dim1_var, del1)
    
    if coords.lower() == 'lat':
        del1 = np.deg2rad(del1) * Rearth # convert lon to m
    else:
        raise Exception('unsupported coords for 2D case: ' + coords +
                        ', should be [lat-lon, cartesian]')
    
    del1Sqr = del1 ** 2.0
    epsilon = np.sin(np.pi/(2.0*gc1+2.0))**2
    optArg  = 2.0 / (1.0 + np.sqrt((2.0 - epsilon) * epsilon))
    flags   = np.array([0.0, 1.0, 0.0])
    
    # store all and return
    re = {}
    
    re['gc1'     ] = gc1       # grid count in first  dimension (e.g., lon)
    re['del1'    ] = del1      # distance in first  dimension (unit: m)
    re['del1Sqr' ] = del1Sqr   # del1 ** 2
    re['optArg'  ] = optArg    # optimal argument for SOR
    re['flags'   ] = flags     # outputs of the SOR iteration:
                               #   [0] overflow or not
                               #   [1] tolerance
                               #   [2] loop count
    
    return re


def __update(default, users, valid=None):
    """Update default invert parameters with user-defined ones."""
    
    if valid is not None and users != default:
        for k, v in users.items():
            if k not in valid:
                raise Exception(f'mParams[\'{k}\'] is not used, valid are {valid}')
    
    default_cp = copy.deepcopy(default)
    
    for k, v in users.items():
        if v is not None:
            default_cp[k] = v
    
    return default_cp

def __uniform_interval(coord1D, value):
    if not np.isclose(coord1D.diff(coord1D.name), value).all():
        raise Exception(f'coordinate {coord1D.name} is non-uniform:\n{coord1D}')

def __print_params(params):
    """Print parameters for debugging."""
    if 'ratio' in params:
        print('dim grids  : ',
              params['gc2'], params['gc1'])
        print('dim intervs: ',
              params['del2'], params['del1'])
        print('ratio, Qtr : ',
              params['ratio'], params['ratioQtr'])
        print('optArg     : ',
              params['optArg'])
        print('max loops  : ',
              params['mxLoop'])
        print('tolerance  : ',
              params['tolerance'])
        print('printInfo  : ',
              params['printInfo'])
        print('debug      : ',
              params['debug'])
        print('undef      : ',
              params['undef'])
        print('boundaries : ',
              params['BCs'])
    elif 'ratio1Sqr' in params:
        print('dim grids  : ',
              params['gc3'], params['gc2'], params['gc1'])
        print('dim intervs: ',
              params['del3'], params['del2'], params['del1'])
        print('ratio1Sqr, ratio2Sqr : ',
              params['ratio1Sqr'], params['ratio2Sqr'])
        print('ratio1, ratio2 : ',
              params['ratio1'], params['ratio2'])
        print('optArg     : ',
              params['optArg'])
        print('max loops  : ',
              params['mxLoop'])
        print('tolerance  : ',
              params['tolerance'])
        print('printInfo  : ',
              params['printInfo'])
        print('debug      : ',
              params['debug'])
        print('undef      : ',
              params['undef'])
        print('boundaries : ',
              params['BCs'])
    else:
        print('dim grids  : ',
              params['gc1'])
        print('dim intervs: ',
              params['del1'])
        print('del1Sqr : ',
              params['del1Sqr'],)
        print('optArg     : ',
              params['optArg'])
        print('max loops  : ',
              params['mxLoop'])
        print('tolerance  : ',
              params['tolerance'])
        print('printInfo  : ',
              params['printInfo'])
        print('debug      : ',
              params['debug'])
        print('undef      : ',
              params['undef'])
        print('boundaries : ',
              params['BCs'])

