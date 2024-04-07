# -*- coding: utf-8 -*-
"""
Created on 2021.01.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import numpy as np
import xarray as xr


class FiniteDiff(object):
    """
    This class wrap some basic finite-difference operators supported for
    Cartesian coordinates (coords='cartesian') or latitude/longitude
    coordinates (coords='lat/lon'), using centered different scheme.
    
    This is designed particularly for Arakawa A grid (all the variables are
    defined on the same grid points).  For grids of other types (variables
    are staggered), please use `xgcm` to calculate the finite difference in
    finite volumn fashion.
    
    For derivative along a dimension, one may use xarray's `differentiate()`.
    The problem with xarray's `differentiate()` is that the boundary conditions
    are not flexible enough for our purpose.  So we implement each operator
    here using `xr.DataArray.pad()` method to account for different BCs.

    Attributes
    ----------
    dmap: dict
        Dimension mapping from those in `xarray.DataArray` to ['T', 'Z', 'Y', 'X'].
    BCs: dict
        Default boundary conditions e.g., BCs={'X': 'periodic'} for both end
        points along 'X' dimension; or BCs={'Z': ('fixed','reflect')} for fixed
        left BC and reflected right BC.  Left indicates lower indices along 'X'.
    fill: float or dict
        Fill value if BCs are fixed.  A typical example can be:
        {'Z':(1,2), 'Y':(0,0), 'X':(1,0)}
    coords: {'lat-lon', 'cartesian'}
        Types of coords.  Should be one of ['lat-lon', 'cartesian'].
    R: float
        Radius of Earth.
    
    Methods
    -------
    grad(scalar)
        3D gradient.
    divg(vector)
        3D divergence.
    vort(vector)
        3D vorticity.
    curl(vector)
        vertical vorticity of vector.
    laplacian(scalar)
        Laplacian.
    tension_strain(vector)
        Tension strain.
    shear_strain(vector)
        Shear strain.
    deformation_rate(vector)
        Deformation rate.
    Okubo_Weiss(vector)
        Okubo Weiss parameter.
    """
    def __init__(self, dim_mapping, BCs='extend', coords='lat-lon', fill=0, R=6371200.0):
        """Construct a FiniteDiff instance given dimension mapping
        
        Parameters
        ----------
        dim_mapping: dict
            Mapping 4D coordinates into ['T', 'Z', 'Y', 'X']. A typical case is:
                {'T':'time', 'Z':'lev', 'Y':'lat', 'X':'lon'}.
            Note that if coords is 'lat-lon', 'X' will be treated as longitude
            and 'Y' as latitude.  Finite difference along these coordinates will
            be properly scaled and weighted.
        BCs: dict
            Boundary conditions along each dimension, one can specify different
            BCs at different end points along a single dimension.  BCs includes:
            * 'fixed': pad with fixed value.
            * 'extend': pad with edge value.
            * 'reflect': pad with first inner value.  1st derivative is exactly zero.
            * 'periodic': pad with cyclic values.
            * 'extrapolate': pad with extrapolated value. (NOT implemented). 1st
              derivative equals the first inner point. 2nd derivative is exactly zero.
        coords: str
            Types of coords.  Should be one of ['lat-lon', 'cartesian'].
        fills: float or dict
            Fill value if BCs are fixed.  A typical example can be:
                {'Z':(1,2), 'Y':(0,0), 'X':(1,0)}
        R: float
            Radius of Earth.
        """
        # Align dims and BCs with default one ('extend').
        if BCs is None:
            BCs = {}
            for dim in dim_mapping:
                BCs[dim] = ('extend', 'extend')
        
        elif type(BCs) == str:
            BC  = BCs
            BCs = {}
            for dim in dim_mapping:
                BCs[dim] = (BC, BC)
        
        elif type(BCs) == dict:
            for dim in dim_mapping:
                if not dim in BCs:
                    BCs[dim] = ('extend', 'extend')
                elif type(BCs[dim]) == str:
                    BCs[dim] = (BCs[dim], BCs[dim])
                    
        # Align dims and fill with default one (0).
        if fill is None:
            fill = {}
            for dim in dim_mapping:
                fill[dim] = (0, 0)
        
        elif type(fill) in [float, int]:
            fil  = fill
            fill = {}
            for dim in dim_mapping:
                fill[dim] = (fil, fil)
        
        elif type(fill) == dict:
            for dim in dim_mapping:
                if not dim in fill:
                    fill[dim] = (0, 0)
        
        self.dmap   = dim_mapping
        self.BCs    = BCs
        self.fill   = fill
        self.coords = coords
        self.R      = R
        
        if coords not in ['lat-lon', 'cartesian']:
            raise Exception('unsupported coords: ' + coords +
                            ', should be one of [\'lat-lon\', \'cartesian\']')
    
    
    def __repr__(self):
        typ = '     Name,               BCs (l-r),     fills  => \'{:s}\' coords\n'\
              .format(self.coords)
        out = ['{:>1s}: {:>6s}  {:>24s}  {:>8s}\n'.format(
               str(dim), str(name), str(self.BCs[dim]), str(self.fill[dim]))
               for dim, name in self.dmap.items()]
        
        return typ + ''.join(out)

    
    def grad(self, v, dims=['X','Y'], BCs=None, fill=None):
        """Calculate spatial gradient components along each dimension given.
        
        Parameters
        ----------
        v: xarray.DataArray
            A scalar variable.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC.  If provided, overwrite the default one per call.
        
        Examples
        --------
        >>> vx = grad(v, 'X', coords='cartesian')
        >>> vx, vy = grad(v, ['lon', 'lat'])
        >>> vz, vy, vx = grad(v, ['lev', 'lat', 'lon'])
        
        Returns
        -------
        xarray.DataArray or list
            Gradient components.
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'lat-lon'
        re   = []
        
        for dim in dims:
            dimName = self.dmap[dim]
            
            if dim == 'Y' and llc:
                scale = np.pi * self.R / 180.0 # deg2m
            elif dim == 'X' and llc:
                if 'Y' in self.dmap and self.dmap['Y'] in v.dims:
                    cos = np.cos(np.deg2rad(v[self.dmap['Y']]))
                else:
                    cos = 1
                scale = np.pi * self.R / 180.0 * cos # deg2m
            else:
                scale = 1
            
            grd = deriv(v, dimName, BCs[dim], fill, scale)
            
            # if llc and dim == 'X' and 'Y' in self.dmap:
            #     grd = grd.where(np.abs(grd[self.dmap['Y']])!=90, other=0)
                
            re.append(grd)
        
        if len(re) == 1:
            return re[0]
        else:
            return re
    
    def divg(self, vector, dims, BCs=None, fill=None):
        """Calculate divergence as du/dx + dv/dy + dw/dz.
        
        Parameters
        ----------
        vector: xarray.DataArray or a list (tuple) of xarray.DataArray
            Component(s) of a vector.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Examples
        --------
        >>> divX   = divg(u, 'X')            # du/dx
        >>> divY   = divg(v, 'Y')            # dv/dy
        >>> divZ   = divg(w, 'Z')            # dw/dz
        >>> divXY  = divg((u,v), ['X','Y'])  # du/dx+dv/dy
        >>> divVW  = divg((v,w), ['Y','Z'])  # dv/dy+dw/dz
        >>> divXZ  = divg((u,w), ['X','Z'])  # du/dx+dw/dz
        >>> divXYZ = divg((u,v,w), ['X','Y','Z'])  # du/dx+dv/dy+dw/dz
        
        Returns
        -------
        xarray.DataArray
            Divergence.
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'lat-lon'
        
        if type(dims) is str:
            dims = [dims]
        
        if type(vector) is xr.DataArray:
            vector = [vector]
        
        if len(vector) != len(dims):
            raise Exception('lengths of vector and dims are not equal')
        
        re = []
        
        for comp, dim in zip(vector, dims):
            dimName = self.dmap[dim]
            
            if llc and dim == 'Y':
                # weight cos(lat)
                cos = np.cos(np.deg2rad(comp[self.dmap['Y']]))
                scale = np.pi * self.R / 180.0 * cos
                tmp = comp * cos
            elif llc and dim == 'X':
                # weight cos(lat)
                if 'Y' in self.dmap and self.dmap['Y'] in vector[0].dims:
                    cos = np.cos(np.deg2rad(comp[self.dmap['Y']]))
                else:
                    cos = 1
                scale = np.pi * self.R / 180.0 * cos
                tmp = comp
            else:
                scale = 1
                tmp = comp
            
            div = deriv(tmp, dimName, BCs[dim], fill, scale)
            
            # if llc and dim in ['X', 'Y'] and 'Y' in self.dmap:
            #     div = div.where(np.abs(div[self.dmap['Y']])!=90, other=0)
            
            re.append(div)
        
        return sum(re)
    
    def vort(self, u=None, v=None, w=None, components='k', BCs=None, fill=None):
        """
        Calculate vorticity component.  All the three components satisfy
        the right-hand rule so that we only need one function and input
        different components accordingly.
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        w: xarray.DataArray
            Z-component velocity.
        components: str or list of str
            Component(s) of the vorticity.  Order of component is the same as
            that of the outputs: vork, vorj, vori = vort(u,v,w, ['k','j','i'])
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Examples
        --------
        >>> vori = vort(v=v, w=w, 'i')   # x-component (i) is: dw/dy - dv/dz
        >>> vorj = vort(u=u, w=w, 'j')   # y-component (j) is: du/dz - dw/dx
        >>> vork = vort(u=u, v=v, 'k')   # z-component (k) is: dv/dx - du/dy

        >>> vori,vorj      = vort(u=u,v=v,w=w, ['i','j'])      # i,j components
        >>> vori,vorj,vork = vort(u=u,v=v,w=w, ['i','j','k'])  # all components
        
        Returns
        -------
        xarray.DataArray or list
            vorticity components.
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'lat-lon'
        dims = self.dmap
        
        if type(components) is str:
            components = [components]
        
        if llc:
            tmp  = None
            for a in [u, v, w]:
                if a is not None:
                    tmp = a
                    break
            
            if dims['Y'] in tmp.dims:
                cos = np.cos(np.deg2rad(tmp[dims['Y']]))
            else:
                cos = 1
                
            scale = np.deg2rad(1.0) * self.R * cos
        else:
            scale = 1.0
        
        vors = []
        for comp in components:
            if comp == 'i': # wy - vz
                t  = w * cos if llc else w # weighted by cos(lat)
                c1 = deriv(t, dims['Y'], BCs['Y'], fill['Y'], scale)
                c2 = deriv(v, dims['Z'], BCs['Z'], fill['Z'], 1.0)
                vor= c1 - c2
            elif comp == 'j': # uz - wx
                c1 = deriv(u, dims['Z'], BCs['Z'], fill['Z'], 1.0)
                c2 = deriv(w, dims['X'], BCs['X'], fill['X'], scale)
                vor= c1 - c2
            elif comp == 'k': # vx - uy
                t  = u * cos if llc else u # weighted by cos(lat)
                c1 = deriv(v, dims['X'], BCs['X'], fill['X'], scale)
                c2 = deriv(t, dims['Y'], BCs['Y'], fill['Y'], scale)
                vor= c1 - c2
            else:
                raise Exception('invalid component ' + str(comp) +
                                ', only in [i, j, k]')
            
            # if llc and comp in ['i', 'k']:
            #     vors.append(vor.where(np.abs(v[dims['Y']])!=90, other=0))
            # else:
            vors.append(vor)
        
        return vors if len(vors) != 1 else vors[0]
    
    def curl(self, u, v, BCs=None, fill=None):
        """Calculate vertical vorticity (k) component.
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        """
        return self.vort(u=u, v=v, components='k', BCs=BCs, fill=fill)
    
    def Laplacian(self, v, dims=['X', 'Y'], BCs=None, fill=None):
        """Calculate the Laplacian of a scalar.
        
        Parameters
        ----------
        v: xarray.DataArray
            A given scale variable.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Returns
        -------
        xarray.DataArray
            Laplacian of a scalar.
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'lat-lon'
        dmap = self.dmap
        
        re = []
        for dim in dims:
            if llc and dim in ['X', 'Y']:
                dimN  = dmap['Y']
                latR  = np.deg2rad(v[dimN])
                cosL  = np.cos(latR)
                
                if dim == 'Y':
                    scale  = np.pi * self.R / 180.0
                    metric = -deriv(v, dmap['Y'], BCs['Y'], fill['Y'],
                                    scale) * np.tan(latR) / self.R
                else:
                    scale = np.pi * self.R / 180.0 * cosL
                    metric = 0
            else:
                scale = 1.0
                metric = 0
            
            re.append(deriv2(v, dmap[dim], BCs[dim], fill[dim], scale) + metric)
        
        if llc and 'Y' in dims: # maskout poles with 0
            return sum(re).where(np.abs(v[dmap['Y']])!=90, other=0)
        else:
            return sum(re)
    
    def tension_strain(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """Calculate tension strain as du/dx - dv/dy.
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Returns
        -------
        xarray.DataArray
            tension strain.
        """
        # defined at tracer point
        return self.divg((u, -v), dims, BCs, fill)
    
    def shear_strain(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """Calculate tension strain as dv/dx + du/dy.
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Returns
        -------
        xarray.DataArray
            shear strain.
        """
        # defined at vorticity point
        return self.vort(u=v, v=-u, dims=dims, BCs=BCs, fill=fill)
    
    def deformation_rate(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """Calculate sqrt(tension^2+shear^2).
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Returns
        -------
        xarray.DataArray
            Deformation rate.
        """
        tension = self.tension_strain(u, v, dims, BCs, fill)
        shear   = self.shear_strain  (u, v, dims, BCs, fill)
        
        return np.hypot(tension + shear)
    
    def Okubo_Weiss(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """Calculate Okubo-Weiss parameter.
        
        Parameters
        ----------
        u: xarray.DataArray
            X-component velocity.
        v: xarray.DataArray
            Y-component velocity.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        
        Returns
        -------
        xarray.DataArray
            Okubo-Weiss parameter.
        """
        deform = self.deformation_rate(u, v, dims, BCs, fill)
        curlZ  = self.vort(u=v, v=u, components='j', dims=dims, BCs=BCs, fill=fill)
        
        return deform**2.0 - curlZ**2.0



def padBCs(v, dim, BCs, fill=(0,0)):
    """Pad array with boundary conditions.

    Pad (add two extra endpoints) original DataArray with BCs along a
    specific dimension.  Types of boundary conditions are:

        * 'fixed': pad with fixed value.
        * 'extend': pad with original edge value.
        * 'reflect: pad with first inner value. 1st-order derivative is exactly zero.
        * 'extrapolate': pad with extrapolated value. (NOT implemented). 1st-order
          derivative equals the first inner point. 2nd-order derivative is exactly zero.
        * 'periodic': pad with cyclic values.
    
    Parameters
    ----------
    v: xarray.DataArray
        A scalar variable.
    dim: list of str
        Dimension along which it is padded.
    BCs: tuple or list of str
        Boundary conditions for the two end points e.g., ('fixed','fixed').
    fill: tuple or list of floats
        Fill values as BCs if BC is fixed at two end points e.g., (0,0).
    
    Returns
    -------
    p: xarray.DataArray
        Padded array.
    """
    if type(BCs) == str:
        BCs = (BCs, BCs)
    
    p = v
    
    if 'periodic' in BCs: # pad with periodic BC
        if BCs[0] != BCs[1]:
            raise Exception('\'periodic\' cannot be mixed with other BCs')
        
        p = p.pad({dim:(1,1)}, mode='wrap')
        
    else: # pad with other mixed type of BCs
        for B, shp in zip(BCs, [(1,0), (0,1)]):
            if   B == 'fixed':
                p = p.pad({dim:shp}, mode='constant', constant_values=fill)
            elif B == 'extend':
                p = p.pad({dim:shp}, mode='edge')
            elif B == 'reflect':
                p = p.pad({dim:shp}, mode='reflect')
            else:
                raise Exception('unsupported BC: ' + str(BCs))
    
    # deal with extra (padded) coordinate values
    coord = p[dim].values
    coord[ 0] = coord[ 1] * 2 - coord[ 2]
    coord[-1] = coord[-2] * 2 - coord[-3]
    
    p[dim] = coord
    
    return p


def deriv(v, dim, BCs=('extend','extend'), fill=(0,0), scale=1,
          scheme='center'):
    """First-order derivative along a given dimension.
    
    The first-order derivative is calculated with proper boundary conditions
    (BCs) and finite difference scheme.
    
    Parameters
    ----------
    v: xarray.DataArray
        A scalar variable.
    dim: str
        Dimension along which difference is taken.
    BCs: tuple or list of str
        Boundary conditions for the two end points. Types of BCs are:

        * 'fixed': pad with fixed value.
        * 'extend': pad with original edge value.
        * 'reflect: pad with first inner value. 1st-order derivative is exactly zero.
        * 'extrapolate': pad with extrapolated value. (NOT implemented). 1st-order
          derivative equals the first inner point. 2nd-order derivative is exactly zero.
        * 'periodic': pad with cyclic values.

    fill: tuple of floats
        Fill values as BCs if BC is fixed at two end points.
    scale: float or xarray.DataArray
        Scale of the result, usually the metric of the dimension.
    scheme: str
        Finite difference scheme in ['center', 'forward', 'backward']
    
    Returns
    -------
    xarray.DataArray
         First-order derivative along the dimension
    """
    if   scheme == 'center':
        pad = padBCs(v, dim, BCs, fill).chunk({dim:len(v[dim])+2})
        grd = pad.differentiate(dim).isel({dim:slice(1,-1)})
        
    elif scheme == 'forward':
        grd = (v-v.shift({dim:-1})) / (v[dim]-v[dim].shift({dim:-1}))
        
    elif scheme == 'backward':
        grd = (v.shift({dim: 1})-v) / (v[dim].shift({dim: 1})-v[dim])
        
    else:
        raise Exception('unsupported scheme: ' + scheme +
                        ', should be in [\'center\', \'forward\', \'backward\']')
    
    # trimming the original range and scaling
    return grd / scale


def deriv2(v, dim, BCs=('extend','extend'), fill=(0,0), scale=1):
    """Second-order derivative along a given dimension
    
    The second-order derivative is calculated with proper boundary conditions
    (BCs) and centered finite-difference scheme.
    
    Parameters
    ----------
    v: xarray.DataArray
        A scalar variable.
    dim: str
        Dimension along which difference is taken.
    BCs: tuple or list of str
        Boundary conditions for the two end points. Types of BCs are:

        * 'fixed': pad with fixed value.
        * 'extend': pad with original edge value.
        * 'reflect: pad with first inner value. 1st-order derivative is exactly zero.
        * 'extrapolate': pad with extrapolated value. (NOT implemented). 1st-order
          derivative equals the first inner point. 2nd-order derivative is exactly zero.
        * 'periodic': pad with cyclic values.
        
    fill: tuple of floats
        Fill values as BCs if BC is fixed at two end points.
    scale: float or xarray.DataArray
        Scale of the result, usually the metric of the dimension.
    scheme: {'center', 'forward', 'backward'}
        Finite difference scheme in ['center', 'forward', 'backward'].
    
    Returns
    -------
    xarray.DataArray
         Second-order derivative along the dimension
    """
    pad = padBCs(v, dim, BCs, fill)
    lap = pad.diff(dim, 2, 'lower') / pad[dim].diff(dim) ** 2 / scale ** 2
    lap[dim] = v[dim]
    
    return lap



"""
helper methods.
"""
def _dimsBCs(dims, BCs):
    """
    Align dims and BCs with default one ('extend').
    """
    if type(dims) == str:
        dims = [dims]
        
    if BCs is None:
        BCs = {}
        for dim in dims:
            BCs[dim] = ('extend', 'extend')
    
    elif type(BCs) == str:
        BC  = BCs
        BCs = {}
        for dim in dims:
            BCs[dim] = (BC, BC)
    
    elif type(BCs) == dict:
        for dim in dims:
            if not dim in BCs:
                BCs[dim] = ('extend', 'extend')
    
    return dims, BCs

def _overwriteBCs(BCsNew, BCsOld):
    if BCsNew is None:
        return BCsOld
    
    BCs = BCsOld.copy()
    
    if type(BCsNew) == str:
        BC = BCsNew
        
        for B in BCsOld:
            BCs[B] = (BC, BC)
    
    elif type(BCsNew) == dict:
        for B in BCsNew:
            if B in BCsOld:
                if type(BCsNew[B]) == str:
                    BCs[B] = (BCsNew[B], BCsNew[B])
                else:
                    BCs[B] = BCsNew[B]
    
    return BCs

def _overwriteFills(fillsNew, fillsOld):
    if fillsNew is None:
        return fillsOld
    
    fills = fillsOld.copy()
    
    if type(fillsNew) in [float, int]:
        fill = fillsNew
        
        for f in fillsOld:
            fills[f] = (fill, fill)
    
    elif type(fillsNew) == dict:
        for f in fillsNew:
            if f in fillsOld:
                fills[f] = fillsNew[f]
    
    return fills

