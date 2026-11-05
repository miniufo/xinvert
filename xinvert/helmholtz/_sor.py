"""Single-potential SOR with standard scalar boundary controls.

The face operators reuse invert_Poisson and add no SOR iteration kernel.
Matching operators satisfy L_h = D_h G_h. Blocked normal fluxes are zero;
the remainder retains any harmonic island circulation. Internally, east
and north faces are indexed using the adjacent cell labels. The wrapper
returns true face coordinates or interpolates velocities back to centers.

Only the velocity potential is solved; the scalar solver selects its boundary kernel.
"""

import numpy as np
import xarray as xr

from ..apps import _solve_dtype, default_mParams, invert_Poisson


def decompose(u, v, dims, coords, mask, mp, ip, initial, diagnostics, *, grid="arakawa-c"):
    if mask is None and grid == "arakawa-c":
        raise ValueError(
            "sor with grid='arakawa-c' requires an explicit boolean cell-center mask"
        )
    input_u, input_v = u, v
    ip = ip.copy()
    if ip.pop("solver", "sor") != "sor":
        raise ValueError("sor currently supports solver=sor")
    from ..core import _normalize_internal_bcs, _normalize_scalar_bcs

    ip.setdefault("BCs", ["fixed", "fixed"])
    ip['BCs'] = _normalize_scalar_bcs(ip['BCs'], 2, valid=('fixed', 'extend', 'periodic'))
    ip['internal_bcs'] = _normalize_internal_bcs(ip.get('internal_bcs', 'fixed'))
    bcs = ip['BCs']
    if bcs[0] == 'periodic':
        raise NotImplementedError('SOR decomposition does not implement periodic y')
    ip.setdefault("printInfo", False)
    if diagnostics and ip.get("diag_interval", 0) <= 0:
        ip["diag_interval"] = 10
    if isinstance(initial, xr.Dataset):
        initial = initial["velocity_potential"]
    if initial is not None and not isinstance(initial, xr.DataArray):
        raise TypeError("SOR initial must be a velocity-potential DataArray or Dataset")
    periodic = bcs[1] == "periodic"
    if grid == "arakawa-a":
        valid = np.isfinite(u) & np.isfinite(v)
        mask = valid if mask is None else mask & valid
        east, north = _face_masks(mask, dims, periodic)
        next_u = u.roll({dims[1]: -1}, roll_coords=False) if periodic else u.shift({dims[1]: -1})
        u = (0.5 * (u.astype("float64") + next_u.astype("float64"))).where(east, 0)
        v = (0.5 * (v.astype("float64") + v.shift({dims[0]: -1}).astype("float64"))).where(north, 0)
    solved = _project_vector(
        u,
        v,
        mask,
        list(dims),
        coords=coords,
        periodic_x=bcs[1] == "periodic",
        mParams=mp,
        iParams=ip,
        icbc=initial,
        _standard_bcs=True,
    )
    original, detail = solved if isinstance(solved, tuple) else (solved, None)
    result = original.rename(
        {
            "potential": "velocity_potential",
            "x_gradient": "udiv",
            "y_gradient": "vdiv",
            "x_solenoidal": "urot",
            "y_solenoidal": "vrot",
        }
    )
    diagnostic = None
    if diagnostics:
        diagnostic = xr.Dataset() if detail is None else detail.copy()
        diagnostic["divergence_before_rms"] = (result.divergence**2).mean(dims) ** 0.5
        diagnostic["divergence_after_rms"] = (result.remaining_divergence**2).mean(dims) ** 0.5
        diagnostic.attrs["divergence_grid"] = "native C-grid faces and cell divergence"
    result = result.drop_vars(["divergence", "remaining_divergence"]).transpose(*input_u.dims)
    if grid == "arakawa-a":
        east, north = _face_masks(mask, dims, periodic)
        for name in ("udiv",):
            result[name] = _face_average(result[name], east, dims[1], periodic).where(mask)
        for name in ("vdiv",):
            result[name] = _face_average(result[name], north, dims[0], False).where(mask)
        # Subtract on the user's grid, after interpolating the gradient. Averaging
        # the face remainder would instead reconstruct a smoothed input field.
        # This preserves the center velocity sum, not exact center divergence.
        result["urot"] = (input_u - result.udiv).where(mask)
        result["vrot"] = (input_v - result.vdiv).where(mask)
        result.attrs.update(
            output_grid="cell_center",
            velocity_interpolation="mean of adjacent available wet faces",
            input_interpolation="neighbor center average to wet-wet faces; blocked flux zero",
        )
        for name in ("udiv", "vdiv", "urot", "vrot"):
            result[name].attrs.update(location="cell_center", units="m s-1")
        if diagnostics:
            from . import _errors
            errors = _errors(
                input_u.where(mask), input_v.where(mask),
                result.udiv + result.urot,
                result.vdiv + result.vrot, list(dims),
            )
            diagnostic.update(errors)
            diagnostic.attrs["velocity_error_grid"] = "original input centers"
    else:
        if diagnostics:
            east, north = _face_masks(mask, dims, periodic)
            a = (result.udiv + result.urot - input_u).where(east)
            b = (result.vdiv + result.vrot - input_v).where(north)
            error = (a**2).sum(list(dims)) + (b**2).sum(list(dims))
            energy = (input_u.astype("float64")**2).where(east).sum(list(dims)) + (input_v.astype("float64")**2).where(north).sum(list(dims))
            diagnostic["relative_velocity_error"] = xr.where(
                energy > 0, np.sqrt(error / xr.where(energy > 0, energy, 1)),
                xr.where(error == 0, 0.0, np.inf),
            )
        result = _physical_face_coordinates(result, dims)
    return result, diagnostic


def _face_masks(wet, dims, periodic):
    y, x = dims
    neighbour = wet.roll({x: -1}, roll_coords=False) if periodic else wet.shift({x: -1}, fill_value=False)
    return wet & neighbour, wet & wet.shift({y: -1}, fill_value=False)


def _face_average(value, valid, dim, periodic):
    value = value.where(valid, 0)
    previous = value.roll({dim: 1}, roll_coords=False) if periodic else value.shift({dim: 1}, fill_value=0)
    before = valid.roll({dim: 1}, roll_coords=False) if periodic else valid.shift({dim: 1}, fill_value=False)
    count = valid.astype("int8") + before.astype("int8")
    return (value + previous) / xr.where(count > 0, count, 1)


def _physical_face_coordinates(result, dims):
    """Expose true east/north positions without xarray aligning them to centers."""
    y, x = dims
    face_dims = {d: f"{d}_face" for d in dims}
    if any(d in result.coords or d in result.dims for d in face_dims.values()):
        raise ValueError("generated face coordinate name conflicts with an existing coordinate")
    output = xr.Dataset(attrs=result.attrs.copy())
    for name, field in result.data_vars.items():
        if name in ("udiv", "vdiv", "urot", "vrot"):
            dim = x if name.startswith("u") else y
            face = face_dims[dim]
            values = np.asarray(field[dim], dtype=float)
            step = (values[-1] - values[0]) / (values.size - 1)
            updated = field.rename({dim: face}).assign_coords({face: values + step / 2})
            updated[face].attrs = dict(field[dim].attrs)
            updated[face].attrs["location"] = "east_face" if dim == x else "north_face"
            updated.attrs.update(location=updated[face].attrs["location"], units="m s-1")
            output[name] = updated
        else:
            output[name] = field
    output.attrs.update(output_grid="arakawa-c", face_coordinates="physical east/north face positions")
    return output


def reconstruct(result, components):
    required = ("udiv", "vdiv", "urot", "vrot")
    if any(name not in result for name in required):
        raise ValueError(
            "sor reconstruction needs stored gradient and remainder fields"
        )
    velocity = xr.Dataset(
        {
            "u": result.udiv + result.urot,
            "v": result.vdiv + result.vrot,
        }
    )
    if components:
        velocity.update({key: result[key] for key in required})
    return velocity


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


def _project_vector(x_face, y_face, wet, dims, *, coords='cartesian',
                   periodic_x=False, mParams=None, iParams=None, icbc=None,
                   _standard_bcs=False):
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
        tolerance, optArg and diag_interval. None follows the face-component precision.
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
    if not _standard_bcs and (not mask.any() or mask[[0, -1], :].any() or
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
    from ..core import _normalize_internal_bcs

    params['internal_bcs'] = _normalize_internal_bcs(
        params.get('internal_bcs', 'fixed' if _standard_bcs else 'neumann'))
    if _standard_bcs:
        from ..core import _normalize_internal_bcs, _normalize_scalar_bcs

        params.setdefault('BCs', ['fixed', 'periodic' if periodic_x else 'fixed'])
        params['BCs'] = _normalize_scalar_bcs(params['BCs'], 2, valid=('fixed', 'extend', 'periodic'))
        params['internal_bcs'] = _normalize_internal_bcs(params.get('internal_bcs', 'fixed'))
        if not np.isnan(params.get('undef', np.nan)):
            raise ValueError('Helmholtz requires undef=NaN')
        params['undef'] = np.nan
    else:
        bcs = ['fixed', 'periodic' if periodic_x else 'fixed']
        if ('BCs' in params and list(params['BCs']) != bcs or
                str(params.get('internal_bcs', 'neumann')).strip().lower() != 'neumann' or
                not np.isnan(params.get('undef', np.nan))):
            raise ValueError('closed projection requires fixed outer BCs, internal Neumann and undef=NaN')
        params.update(BCs=bcs, internal_bcs='neumann', undef=np.nan)
    params['dtype'] = _solve_dtype(params, x_face.dtype, y_face.dtype)
    # Accumulate conservative face differences in float64 even when stored
    # components are float32. This avoids introducing avoidable flux imbalance
    # before casting the conservative RHS to the chosen solve precision.
    fx, fy = x_face.astype('float64').where(east, 0), y_face.astype('float64').where(north, 0)
    forcing = _divergence(fx, fy, wet, dims, dx, dy, periodic_x, cos_cell, cos_north)
    solved = invert_Poisson(forcing, list(dims), coords=coords, mParams=model,
                           iParams=params, icbc=icbc)
    potential, diagnostic = solved if isinstance(solved, tuple) else (solved, None)
    gx = ((_neighbour(potential, x, 1, periodic_x) - potential)/(dx*cos_cell)).where(east, 0)
    gy = ((potential.shift({y: -1}) - potential)/dy).where(north, 0)
    sx, sy = fx - gx, fy - gy
    diagnostic_wet = wet
    if _standard_bcs:
        diagnostic_wet = wet.copy(deep=True)
        if params['internal_bcs'] == 'fixed':
            for dimension, wraps in [(y, False), (x, periodic_x)]:
                for offset in (-1, 1):
                    diagnostic_wet = diagnostic_wet & _neighbour(
                        wet, dimension, offset, wraps, True)
        diagnostic_wet[{y: [0, -1]}] = False
        if not periodic_x:
            diagnostic_wet[{x: [0, -1]}] = False
    result = xr.Dataset({
        'potential': potential, 'divergence': forcing,
        'x_gradient': gx.where(east), 'y_gradient': gy.where(north),
        'x_solenoidal': sx.where(east), 'y_solenoidal': sy.where(north),
        'remaining_divergence': _divergence(sx, sy, wet, dims, dx, dy,
                                            periodic_x, cos_cell, cos_north).where(diagnostic_wet)})
    result.attrs.update(geometry=coords, boundary=('scalar potential BCs' if _standard_bcs
                                                     else 'closed homogeneous Neumann'),
                        internal_bcs=params['internal_bcs'],
                        periodic_x=int(periodic_x),
                        face_convention='east/north face stored at the adjacent cell label',
                        experimental='true')
    result = result.astype(params['dtype'])
    return (result, diagnostic) if diagnostic is not None else result
