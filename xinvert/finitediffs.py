# -*- coding: utf-8 -*-
"""
Created on 2021.01.03

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import numpy as np
import xarray as xr
from .utils import _deg2m, _R_earth


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
    here using `DataArray.pad()` method to account for different BCs.
    """
    def __init__(self, dim_mapping, BCs='extend', coords='latlon', fill=0):
        """
        Construct a FiniteDiff instance given dimension mapping
        
        Parameters
        ----------
        dim_mapping: dict
            Mapping 4D coordinates into ['T', 'Z', 'Y', 'X']. A typical case is:
                {'T':'time', 'Z':'lev', 'Y':'lat', 'X':'lon'}.
            Note that if coords is 'latlon', 'X' will be treated as longitude
            and 'Y' as latitude.  Finite difference along these coordinates will
            be properly scaled and weighted.
        BCs: dict
            Boundary conditions along each dimension, one can specify different
            BCs at different end points along a single dimension.  BCs includes:
              - 'fixed'       # pad with fixed value.
              - 'extend'      # pad with edge value.
              - 'reflect      # pad with first inner value.
                                1st derivative is exactly zero.
              - 'periodic'    # pad with cyclic values.
              - 'extrapolate' # pad with extrapolated value. (NOT implemented)
                                1st derivative equals the first inner point.
                                2nd derivative is exactly zero.
        coords: str
            Types of coords.  Should be one of ['latlon', 'cartesian'].
        fills: float or dict
            Fill value if BCs are fixed.  A typical example can be:
                {'Z':(1,2), 'Y':(0,0), 'X':(1,0)}
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
        
        self.dim_mapping = dim_mapping
        self.BCs = BCs
        self.fill = fill
        self.coords = coords
    
    
    def __repr__(self):
        typ = '     Name,               BCs (l-r),     fills  => \'{:s}\' coords\n'\
              .format(self.coords)
        out = ['{:>1s}: {:>6s}  {:>24s}  {:>8s}\n'.format(
               str(dim), str(name), str(self.BCs[dim]), str(self.fill[dim]))
               for dim, name in self.dim_mapping.items()]
        
        return typ + ''.join(out)

    
    def grad(self, v, dims=['X','Y'], BCs=None, fill=None):
        """
        Calculate spatial gradient components along each dimension given.
        
        For example:
            vx = grad(v, 'X', coords='cartesian')
            vx, vy = grad(v, ['lon','lat'])
        
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
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        re   = []
        llc  = self.coords == 'latlon'
        
        for dim in dims:
            dimName = self.dim_mapping[dim]
            
            if dim == 'Y' and llc:
                scale = _deg2m
                grd = self._diff1st(v, dimName, BCs[dim], fill, scale)
                
            elif dim == 'X' and llc:
                scale = _deg2m * np.cos(np.deg2rad(v[self.dim_mapping['Y']]))
                grd = self._diff1st(v, dimName, BCs[dim], fill, scale)
                
            else:
                scale = 1
                grd = self._diff1st(v, dimName, BCs[dim], fill, scale)
            
            re.append(grd)
        
        return re
    
    def divg(self, vector, dims, BCs=None, fill=None):
        """
        Calculate divergence as du/dx + dv/dy + dw/dz ...
        
        For example:
            du/dx            ->   divX   = divg(u, 'X')
            dv/dy            ->   divY   = divg(v, 'Y')
            dw/dz            ->   divZ   = divg(w, 'Z')
            du/dx+dv/dy      ->   divXY  = divg((u,v), ['X','Y'])
            dv/dy+dw/dz      ->   divVW  = divg((v,w), ['Y','Z'])
            du/dx+dw/dz      ->   divXZ  = divg((u,w), ['X','Z'])
            du/dx+dv/dy+dw/dz->   divXYZ = divg((u,v,w), ['X','Y','Z'])
        
        Parameters
        ----------
        vector: xarray.DataArray or list of xarray.DataArray
            Component(s) of a vector.
        dims: list of str
            Dimensions for gradient.  Order of dims is the same as
            that of the outputs: `vx, vy, vz = grad(v, ['X', 'Y', 'Z'])`.
            Here use ['X', 'Y'] in dim_mapping instead of true dimension names.
        BCs: dict
            Boundary condition.  If provided, overwrite the default one per call.
        fill: tuple of floats
            Fill values of fixed BC. If provided, overwrite the default one per call.
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'latlon'
        
        if type(dims) is str:
            dims = [dims]
        
        if type(vector) is xr.DataArray:
            vector = [vector]
        
        if len(vector) != len(dims):
            raise Exception('lengths of vector and dims are not equal')
        
        if llc:
            cos = np.cos(np.deg2rad(vector[0][self.dim_mapping['Y']]))
            scale = _deg2m * cos
        else:
            scale = 1
        
        re = []
        
        for comp, dim in zip(vector, dims):
            dimName = self.dim_mapping[dim]
            
            if dim == 'Y' and llc: # weighted by cos(lat)
                tmp = comp * cos
            else:
                tmp = comp
            
            div = self._diff1st(tmp, dimName, BCs[dim], fill, scale)
            
            re.append(div)
        
        return sum(re)
    
    def vort(self, u=None, v=None, w=None, components='k', BCs=None, fill=None):
        """
        Calculate vorticity component.  All the three components satisfy
        the right-hand rule so that we only need one function and input
        different components accordingly.
        
        For example:
            x-component (i) is: dw/dy - dv/dz  ->  vori = vort(v=v, w=w, 'i')
            y-component (j) is: du/dz - dw/dx  ->  vorj = vort(u=u, w=w, 'j')
            z-component (k) is: dv/dx - du/dy  ->  vork = vort(u=u, v=v, 'k')
            
            i,j components:  ->  vori,vorj      = vort(u=u,v=v,w=w, ['i','j'])
            all components:  ->  vori,vorj,vork = vort(u=u,v=v,w=w, ['i','j','k'])
        
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
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'latlon'
        dims = self.dim_mapping
        
        if type(components) is str:
            components = [components]
        
        if llc:
            tmp  = None
            for a in [u, v, w]:
                if a is not None:
                    tmp = a
                    break
            
            cos = np.cos(np.deg2rad(tmp[dims['Y']]))
            scale = _deg2m * cos
        else:
            scale = 1.0
        
        vors = []
        for comp in components:
            if comp == 'i': # wy - vz
                t  = w * cos if llc else w # weighted by cos(lat)
                c1 = self._diff1st(t, dims['Y'], BCs['Y'], fill['Y'], scale)
                c2 = self._diff1st(v, dims['Z'], BCs['Z'], fill['Z'], scale)
                vor= c1 - c2
            elif comp == 'j': # uz - wx
                c1 = self._diff1st(u, dims['Z'], BCs['Z'], fill['Z'], scale)
                c2 = self._diff1st(w, dims['X'], BCs['X'], fill['X'], scale)
                vor= c1 - c2
            elif comp == 'k': # vx - uy
                t  = u * cos if llc else u # weighted by cos(lat)
                c1 = self._diff1st(v, dims['X'], BCs['X'], fill['X'], scale)
                c2 = self._diff1st(u*cos, dims['Y'], BCs['Y'], fill['Y'], scale)
                vor= c1 - c2
            else:
                raise Exception('invalid component ' + str(comp) +
                                ', only in [i, j, k]')
            
            vors.append(vor)
        
        return vors if len(vors) != 1 else vors[0]
    
    def curl(self, u, v, BCs=None, fill=None):
        """
        Calculate vertical vorticity component.
        
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
        """
        Calculate $\nabla^2 v$.
        
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
        """
        BCs  = _overwriteBCs(BCs, self.BCs)
        fill = _overwriteFills(fill, self.fill)
        llc  = self.coords == 'latlon'
        dmap = self.dim_mapping
        
        re = []
        for dim in dims:
            if llc and dim in ['X', 'Y']:
                dimN = dmap['Y']
                latR = np.deg2rad(v[dimN])
                cos = np.cos(latR)
                scale = _deg2m * cos
                
                if dim == 'Y':
                    metric = self._diff1st(v, dmap['Y'], BCs['Y'], fill['Y'],
                                           scale) * np.tan(latR) / _R_earth
                else:
                    metric = 0
            else:
                scale = 1.0
                metric = 0
            
            re.append(self._diff2nd(v, dmap[dim], BCs[dim], fill[dim], scale) +
                      metric)
        
        return sum(re)
    
    def tension_strain(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """
        Calculate tension strain as du/dx - dv/dy.
        
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
        """
        # defined at tracer point
        return self.divg((u, -v), dims, BCs, fill)
    
    def shear_strain(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """
        Calculate tension strain as dv/dx + du/dy.
        
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
        """
        # defined at vorticity point
        return self.vort(u=v, v=-u, dims=dims, BCs=BCs, fill=fill)
    
    def deformation_rate(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """
        Calculate sqrt(tension^2+shear^2).
        
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
        """
        tension = self.tension_strain(u, v, dims, BCs, fill)
        shear   = self.shear_strain  (u, v, dims, BCs, fill)
        
        return np.hypot(tension + shear)
    
    def Okubo_Weiss(self, u, v, dims=['X', 'Y'], BCs=None, fill=None):
        """
        Calculate Okubo-Weiss parameter.
        
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
        """
        deform = self.deformation_rate(u, v, dims, BCs, fill)
        curlZ  = self.vort(u=v, v=u, components='j', dims=dims, BCs=BCs, fill=fill)
        
        return deform**2.0 - curlZ**2.0
    
    
    """
    helper methods.
    """
    def _padBCs(self, v, dim, BC, fill=(0,0)):
        """
        Pad original DataArray with BCs along a specific dimension.  Types
        of boundary conditions are:
            - 'fixed'       # pad with fixed value.
            - 'extend'      # pad with edge value.
            - 'reflect      # pad with first inner value.
                              1st-order derivative is exactly zero.
            - 'extrapolate' # pad with extrapolated value.
                              1st-order derivative equals the first inner point.
                              2nd-order derivative is exactly zero. (NOT implemented)
            - 'periodic'    # pad with cyclic values.
        
        Parameters
        ----------
        v: xarray.DataArray
            A scalar variable.
        dim: list of str
            Dimension along which it is padded.
        BC: tuple or list of str
            Boundary conditions for the two end points, e.g. ('fixed','fixed').
        fill: tuple of floats
            Fill values as BCs if BC is fixed at two end points.
        
        Returns
        ----------
        p: xarray.DataArray
            Padded array.
        """
        p = v
        
        if 'periodic' in BC: # pad with periodic BC
            if BC[0] != BC[1]:
                raise Exception('\'periodic\' BC cannot be mixed with other type of BC')
            
            p = p.pad({dim:(1,1)}, mode='wrap')
            
        else: # pad with other mixed type of BCs
            for B, shp in zip(BC, [(1,0), (0,1)]):
                if   B == 'fixed':
                    p = p.pad({dim:shp}, mode='constant', constant_values=fill)
                elif B == 'extend':
                    p = p.pad({dim:shp}, mode='edge')
                elif B == 'reflect':
                    p = p.pad({dim:shp}, mode='reflect')
                else:
                    raise Exception('unsupported BC: ' + str(BC))
        
        # deal with coordinate values
        coord = p[dim].values
        coord[ 0] = coord[ 1] * 2 - coord[ 2]
        coord[-1] = coord[-2] * 2 - coord[-3]
        
        p[dim] = coord
        
        return p
    
    def _diff1st(self, v, dim, BC, fill, scale):
        pad = self._padBCs(v, dim, BC, fill)
        grd = pad.differentiate(dim).isel({dim:slice(1,-2)}) / scale
        
        return grd
        
    def _diff2nd(self, v, dim, BC, fill, scale):
        pad = self._padBCs(v, dim, BC, fill)
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

