"""Conservative face-field projection on closed regular wet domains.

This application layer reuses invert_Poisson; it adds no iteration kernel.
Inputs are face components, not collocated velocities: x_face[y, x] is
the east face of cell (y, x), and y_face[y, x] is its north face. Increasing
coordinates define the positive directions. Face coordinates are stored
using the adjacent cell's coordinate labels, not the geometric midpoint.
"""
import numpy as np
import xarray as xr

from .apps import default_mParams, invert_Poisson

__all__ = ['project_vector']


def _spacing(field, dimension):
    """Return positive uniform spacing for a named one-dimensional coordinate.

    Parameters
    ----------
    field : xarray.DataArray
        Input cell/face-labelled array with explicit dimension coordinates.
    dimension : str
        Coordinate name; its units define the spatial units of the operator.

    Returns
    -------
    float
        Uniform spacing. Invalid, nonfinite or descending grids raise ValueError.
    """
    if dimension not in field.coords or field[dimension].dims != (dimension,):
        raise ValueError('projection requires explicit one-dimensional coordinates')
    values = np.asarray(field[dimension], dtype=float)
    delta = np.diff(values)
    if (len(values) < 3 or not np.isfinite(values).all() or
            not (delta > 0).all() or
            not np.allclose(delta, delta[0], rtol=1e-10, atol=0)):
        raise ValueError('projection requires increasing, uniform coordinates')
    return float(delta[0])


def _neighbour(field, dimension, offset, periodic=False, fill_value=np.nan):
    """Return values at neighbouring cells without changing coordinate labels.

    Parameters
    ----------
    field : xarray.DataArray
        Labelled array to shift.
    dimension : str
        Shifted dimension name.
    offset : int
        Positive for the next cell, negative for the previous cell.
    periodic : bool
        Wrap values across the seam instead of filling the outside.
    fill_value : scalar
        Fill value for the nonperiodic outside.

    Returns
    -------
    xarray.DataArray
        Neighbour values at the original coordinate labels.
    """
    if periodic:
        return field.roll({dimension: -offset}, roll_coords=False)
    return field.shift({dimension: -offset}, fill_value=fill_value)


def _divergence(east, north, wet, dims, dx, dy, periodic_x=False,
                cos_cell=1., cos_north=1.):
    """Compute conservative divergence of finite, zero-blocked face components.

    Parameters
    ----------
    east, north : xarray.DataArray
        East/north face values, with blocked faces represented by exact zeros.
    wet : xarray.DataArray
        Boolean cell mask, with inactive nonperiodic boundary rows/columns.
    dims : tuple of str
        Ordered (y, x) dimensions.
    dx, dy : float
        Cartesian spacings, or radius times angular spacings for lat-lon.
    periodic_x : bool
        Wrap eastern/western face values across the x seam.
    cos_cell, cos_north : float or xarray.DataArray
        Unity for Cartesian geometry; cos(latitude) at cells/north faces
        for lat-lon geometry.

    Returns
    -------
    xarray.DataArray
        Cell-centred divergence, NaN on inactive cells.
    """
    y, x = dims
    north_flux = north * cos_north
    return ((east - _neighbour(east, x, -1, periodic_x, 0))/dx +
            (north_flux - _neighbour(north_flux, y, -1, fill_value=0))/dy
            ).where(wet)/cos_cell


def project_vector(x_face, y_face, wet, dims, *, coords='cartesian',
                   periodic_x=False, mParams=None, iParams=None, icbc=None):
    """Split a face vector field into gradient and divergence-free parts.

    Solve Laplacian(phi)=div(F) with homogeneous coastal Neumann boundaries,
    then return grad(phi) and F-grad(phi). The finite-volume divergence and
    forward face gradient compose exactly to the regular masked five-point
    operator used by invert_Poisson. One weighted-zero-mean gauge is applied
    to each connected wet component by the existing solver.

    Parameters
    ----------
    x_face, y_face : xarray.DataArray
        Eager real numeric two-dimensional arrays on the same labelled grid.
        Values denote east and north faces, respectively, oriented towards
        increasing coordinates. These are velocities or transports themselves,
        not face-integrated fluxes. Nonzero wet-dry boundary flux is rejected.
        Inactive faces may contain zero or NaN; active faces must be finite.
    wet : xarray.DataArray
        Boolean cell-centred mask on the same grid. Nonperiodic outer rows and
        columns must be inactive. Lazy/batched inputs and nonuniform
        thickness-weighted projection are not implemented.
    dims : sequence of str
        Exactly two dimension names in (y, x) order. Coordinates must increase
        uniformly. Cartesian coordinates use consistent length units; lat-lon
        coordinates use degrees, with latitude first and longitude second.
    coords : {'cartesian', 'lat-lon'}
        Regular-grid geometry. Latitude must lie strictly between -90 and 90;
        original LLC multi-face connectivity and pole rows are unsupported.
    periodic_x : bool
        Connect the last east face to the first column. Lat-lon longitude must
        cover exactly 360 degrees without a duplicated endpoint when periodic.
        CPU-mc/GPU require an even periodic x size, as in invert_Poisson.
    mParams : dict or None
        Optional {'Rearth': positive_radius} for lat-lon geometry, using the
        same radius as invert_Poisson. The default is xinvert's Earth radius.
    iParams : dict or None
        Normal inversion controls, including architect, dtype, mxLoop,
        tolerance, optArg and diag_interval. Defaults here use float64.
        BCs=['fixed','periodic' if periodic_x else 'fixed'],
        internal_bcs='neumann' and undef=NaN are required. Fixed outer-array
        edges are inactive, not physical Dirichlet boundaries.
    icbc : xarray.DataArray or None
        Optional initial potential for warm starts, using the same dimensions
        and coordinates as the input grid. Finite values are required on wet
        cells; inactive values are ignored by internal Neumann. The existing
        inversion validates alignment. Only use a stored field as a restart
        when its geometry, mask and face-field definition remain appropriate.

    Returns
    -------
    xarray.Dataset or tuple of (xarray.Dataset, xarray.Dataset)
        Dataset with potential, divergence, x_gradient, y_gradient,
        x_solenoidal, y_solenoidal and remaining_divergence. Component fields
        retain the input face convention and are NaN on blocked faces.
        If diag_interval>0, also return the usual solver diagnostics separately.
        remaining_divergence is an independently evaluated equation defect;
        it is distinct from the solver's normalized SOR-correction error.

    Notes
    -----
    The remaining field can retain harmonic circulation around islands.
    Collocated fields must not be passed without an explicit, physically
    appropriate conversion to faces. This function does not silently remove
    incompatible forcing or nonzero coastal flux.
    """
    dims = tuple(dims)
    arrays = (x_face, y_face, wet)
    if len(dims) != 2 or len(set(dims)) != 2:
        raise ValueError('dims must contain two distinct names in (y, x) order')
    if any(not isinstance(a, xr.DataArray) for a in arrays):
        raise TypeError('face components and wet must be xarray.DataArray objects')
    if any(set(a.dims) != set(dims) or a.ndim != 2 for a in arrays):
        raise ValueError('projection currently requires a single two-dimensional slice')
    if any(a.chunks is not None for a in arrays):
        raise NotImplementedError('lazy projection inputs are not implemented yet')
    if wet.dtype != np.dtype(bool):
        raise ValueError('wet must be a boolean cell mask')
    if any(not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a)
           for a in arrays[:2]):
        raise ValueError('face components must be real numeric arrays')
    x_face, y_face, wet = xr.align(*arrays, join='exact', copy=False)
    x_face, y_face, wet = (a.transpose(*dims) for a in (x_face, y_face, wet))
    y, x = dims
    dy, dx = _spacing(wet, y), _spacing(wet, x)
    if not isinstance(periodic_x, (bool, np.bool_)):
        raise TypeError('periodic_x must be boolean')
    if not isinstance(coords, str) or coords.lower() not in ('cartesian', 'lat-lon'):
        raise NotImplementedError('projection supports Cartesian and nonpolar lat-lon grids')
    coords = coords.lower()
    model = dict(mParams or {})
    if set(model) - {'Rearth'}:
        raise ValueError('projection mParams only accepts Rearth')
    radius = model.get('Rearth', default_mParams['Rearth'])
    if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Rearth must be a positive finite radius')
    cos_cell, cos_north = 1., 1.
    if coords == 'lat-lon':
        if np.any(np.abs(wet[y].values) >= 90):
            raise ValueError('projection excludes pole/singular latitude rows')
        if periodic_x and not np.isclose(dx*wet.sizes[x], 360., rtol=1e-10, atol=1e-8):
            raise ValueError('periodic longitude must span 360 degrees without a duplicate endpoint')
        cos_cell = np.cos(np.deg2rad(wet[y]))
        cos_north = np.cos(np.deg2rad(wet[y] + dy/2))
        dy, dx = np.deg2rad(dy)*radius, np.deg2rad(dx)*radius
    mask = wet.values
    if (not mask.any() or mask[[0, -1], :].any() or
            (not periodic_x and mask[:, [0, -1]].any())):
        raise ValueError('wet must be nonempty with an inactive outer ring on nonperiodic axes')
    east_next = _neighbour(wet, x, 1, periodic_x, False)
    east = wet & east_next
    north = wet & wet.shift({y: -1}, fill_value=False)
    for field, allowed, adjacent in (
            (x_face, east, wet | east_next),
            (y_face, north, wet | wet.shift({y: -1}, fill_value=False))):
        data = field.values
        if not np.isfinite(data[allowed.values]).all():
            raise ValueError('wet-wet face values must be finite')
        coast = (adjacent & ~allowed).values
        if np.any((~np.isnan(data[coast])) & (data[coast] != 0)):
            raise ValueError('nonzero coastal normal flux requires nonhomogeneous Neumann support')
    params = dict(iParams or {})
    bcs = ['fixed', 'periodic' if periodic_x else 'fixed']
    if ('BCs' in params and list(params['BCs']) != bcs or
            str(params.get('internal_bcs', 'neumann')).strip().lower() != 'neumann' or
            not np.isnan(params.get('undef', np.nan))):
        raise ValueError('closed projection requires fixed outer BCs, internal Neumann and undef=NaN')
    params.update(BCs=bcs, internal_bcs='neumann', undef=np.nan)
    params.setdefault('dtype', 'float64')
    # Accumulate conservative face differences in float64 even when stored
    # components are float32. Otherwise cancellation errors in a float32 RHS
    # can violate the subsequent float64 component-compatibility check.
    fx, fy = x_face.astype('float64').where(east, 0), y_face.astype('float64').where(north, 0)
    forcing = _divergence(fx, fy, wet, dims, dx, dy, periodic_x, cos_cell, cos_north)
    solved = invert_Poisson(forcing, list(dims), coords=coords, mParams=model,
                           iParams=params, icbc=icbc)
    potential, diagnostic = solved if isinstance(solved, tuple) else (solved, None)
    gx = ((_neighbour(potential, x, 1, periodic_x) - potential)/(dx*cos_cell)).where(east, 0)
    gy = ((potential.shift({y: -1}) - potential)/dy).where(north, 0)
    sx, sy = fx - gx, fy - gy
    result = xr.Dataset({
        'potential': potential, 'divergence': forcing,
        'x_gradient': gx.where(east), 'y_gradient': gy.where(north),
        'x_solenoidal': sx.where(east), 'y_solenoidal': sy.where(north),
        'remaining_divergence': _divergence(sx, sy, wet, dims, dx, dy,
                                            periodic_x, cos_cell, cos_north)})
    result.attrs.update(geometry=coords, boundary='closed homogeneous Neumann',
                        periodic_x=int(periodic_x),
                        face_convention='east/north face stored at the adjacent cell label',
                        experimental='true')
    return (result, diagnostic) if diagnostic is not None else result
