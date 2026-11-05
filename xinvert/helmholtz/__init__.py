"""Helmholtz decomposition with method-specific, reproducible reconstruction.

decompose returns four velocity components; streamfunction_potential returns
the available scalar potentials. reconstruct sums velocity components or
differentiates Li2006 scalar potentials. Results are Datasets.
The closed projection remainder can include harmonic transport and is not
represented as an invented streamfunction.
Method and grid metadata travel with the result, including boundary rules and
Earth radius. This interface does not make different boundary-value problems
mathematically equivalent.
"""

from collections.abc import Mapping

import numpy as np
import xarray as xr

__all__ = ["decompose", "reconstruct", "streamfunction_potential"]


def _mapping(value, name):
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value or {})


def _inputs(u, v, dims, mask):
    if not isinstance(u, xr.DataArray) or not isinstance(v, xr.DataArray):
        raise TypeError("u and v must be DataArrays")
    if set(u.dims) != set(v.dims):
        raise ValueError("u and v must have the same dimensions")
    u, v = xr.align(u, v, join="exact", copy=False)
    v = v.transpose(*u.dims)
    if dims is None:
        candidates = [
            p
            for p in [("lat", "lon"), ("latitude", "longitude")]
            if set(p).issubset(u.dims)
        ]
        if len(candidates) != 1:
            raise ValueError("specify dims=[Y, X]")
        dims = candidates[0]
    if (
        not isinstance(dims, (tuple, list))
        or len(dims) != 2
        or len(set(dims)) != 2
        or not set(dims).issubset(u.dims)
    ):
        raise ValueError("dims must be two distinct input dimensions [Y, X]")
    if any(a.dtype.kind not in "fiu" for a in (u, v)):
        raise TypeError("real numeric velocities are required")
    if mask is not None:
        if not isinstance(mask, xr.DataArray) or mask.dtype.kind != "b":
            raise TypeError("mask must be a boolean DataArray")
        if not set(mask.dims).issubset(u.dims):
            raise ValueError("mask has dimensions absent from the velocity field")
        xr.align(u, mask, join="exact", copy=False)
    return u, v, tuple(dims), mask


def _metadata(result, method, grid, dims, coords, mp, bcs):
    result.attrs.update(
        method=method,
        grid_layout=grid,
        y_dim=dims[0],
        x_dim=dims[1],
        geometry=coords,
        Rearth=float(mp.get("Rearth", 6371200.0)),
        boundary_y=bcs[0],
        boundary_x=bcs[1],
    )
    return result


def _errors(u, v, ur, vr, dims):
    u, v, ur, vr = (a.astype("float64") for a in (u, v, ur, vr))
    active = np.isfinite(u) & np.isfinite(v) & np.isfinite(ur) & np.isfinite(vr)
    squared = ((ur - u) ** 2 + (vr - v) ** 2).where(active)
    total = squared.sum(dims, min_count=1)
    energy = (u**2 + v**2).where(active).sum(dims, min_count=1)
    relative = np.sqrt(total / xr.where(energy > 0, energy, 1))
    relative = xr.where(energy == 0, xr.where(total == 0, 0.0, np.inf), relative)
    return xr.Dataset(
        {
            "velocity_rmse": np.sqrt(total / (2 * active.sum(dims))),
            "relative_velocity_error": relative,
        }
    )


def _solve(
    u,
    v,
    dims=None,
    coords="lat-lon",
    *,
    method="li2006",
    grid="arakawa-a",
    mask=None,
    mParams=None,
    iParams=None,
    initial=None,
    return_diagnostics=False,
    _components=False,
):
    """Decompose two velocity DataArrays; return a Dataset.

    With return_diagnostics=True return (result, diagnostics). Methods:

    * li2006 (default): joint regularized inversion on an internal C grid. All algorithm
      details and parameters live in _li2006; float32/float64 CPU and optional
      solid-boundary GPU/LSMR are supported. BCs default to fixed/fixed;
      coasts are automatic solid boundaries, longitude may be periodic.
    * sor: one velocity-potential Poisson equation, solved by SOR with
      standard scalar BCs and matching C-grid operators.
      The rotational component is input minus the divergent component.
      Requires one eager 2-D slice.
      A inputs infer the wet mask; C inputs require an explicit center mask.
      A outputs are interpolated to centers before subtraction; C outputs
      use physical face coordinates. No streamfunction is computed.

    dims are [Y, X]; infer lat/lon or latitude/longitude. coords is lat-lon
    (degrees) or cartesian (length coordinates). initial is (psi0, chi0) or a
    named potential Dataset for Li; one DataArray or a Dataset with
    velocity_potential for SOR. Model/iteration dictionaries are copied.
    Results record reconstruction settings. Solver diagnostic histories retain
    their algorithm-specific definitions; velocity error is reported separately.

    Examples
    --------
    >>> result = decompose(u, v, dims=['lat', 'lon'], method='li2006')
    >>> velocity = reconstruct(result)
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.strip().lower()
    if method not in ("li2006", "sor"):
        raise ValueError("method must be li2006 or sor")
    if coords not in ("lat-lon", "cartesian"):
        raise ValueError("coords must be lat-lon or cartesian")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise TypeError("return_diagnostics must be boolean")
    allowed = ("arakawa-a", "arakawa-c") if method == "sor" else ("arakawa-a",)
    if not isinstance(grid, str):
        raise TypeError("grid must be arakawa-a or arakawa-c")
    grid = grid.lower()
    if grid not in allowed:
        raise ValueError(f"{method} requires grid in {allowed!r}")
    u, v, dims, mask = _inputs(u, v, dims, mask)
    mp, ip = _mapping(mParams, "mParams"), _mapping(iParams, "iParams")
    from ..apps import _solve_dtype

    ip['dtype'] = _solve_dtype(ip, u.dtype, v.dtype)
    u, v = u.astype(ip['dtype']), v.astype(ip['dtype'])
    if return_diagnostics and ip.get('diag_interval', 0) <= 0:
        ip['diag_interval'] = 10
    if method == "li2006":
        from ..core import _normalize_scalar_bcs
        from . import _li2006

        bcs = _normalize_scalar_bcs(ip.get('BCs', ['fixed', 'fixed']), 2,
                                    valid=('fixed', 'extend', 'periodic'))
        if 'extend' in bcs:
            raise NotImplementedError('Li2006 uses automatic solid boundaries; extend is unsupported')
        if bcs[0] != 'fixed':
            raise NotImplementedError('Li2006 requires a fixed nonperiodic y boundary')
        if 'internal_bcs' in ip:
            raise ValueError('Li2006 sets coastal boundaries automatically; omit internal_bcs')
        backend_ip = dict(ip, BCs=['closed', 'periodic' if bcs[1] == 'periodic' else 'closed'])
        solve = _li2006._helmholtz_impl if _components else _li2006.helmholtz
        extra = {"return_components": True} if _components else {}
        solved = solve(
            u,
            v,
            list(dims),
            coords,
            mask=mask,
            mParams=mp,
            iParams=backend_ip,
            initial=initial,
            return_diagnostics=return_diagnostics,
            **extra,
        )
        psi, chi = solved[:2]
        result = xr.Dataset({"streamfunction": psi, "velocity_potential": chi})
        result.attrs.update(psi.attrs)
        if _components:
            result.update(solved[2])
            result.attrs.update(solved[2].attrs)
        diagnostic = solved[3 if _components else 2] if return_diagnostics else None
    else:
        from ..core import _normalize_scalar_bcs
        from . import _sor

        ip['BCs'] = _normalize_scalar_bcs(ip.get('BCs', ['fixed', 'fixed']), 2,
                                        valid=('fixed', 'extend', 'periodic'))
        result, diagnostic = _sor.decompose(
            u, v, dims, coords, mask, mp, ip, initial, return_diagnostics, grid=grid
        )
        bcs = ip.get("BCs", ("fixed", "fixed"))
    result = result.astype(ip["dtype"])
    result = _metadata(result, method, grid, dims, coords, mp, bcs)
    return (result, diagnostic) if return_diagnostics else result


def _reconstruct_potentials(result, *, components=False):
    """Return a velocity Dataset using the recorded decomposition method.

    result must be a Dataset returned by decompose with its metadata intact.
    Default variables u/v are the total reconstructed velocity. With
    components=True, additionally include urot, vrot, udiv and vdiv. For Li, total is the sum of potential-derived
    velocities and can differ from the input. For SOR, total includes the
    stored remainder; this identity does not measure projection convergence.
    No discretization, radius or boundary overrides are accepted: recompute
    decompose to change the mathematical problem.
    """
    if not isinstance(result, xr.Dataset):
        raise TypeError("result must be a Dataset returned by decompose")
    if not isinstance(components, (bool, np.bool_)):
        raise TypeError("components must be boolean")
    required = (
        "method",
        "grid_layout",
        "y_dim",
        "x_dim",
        "geometry",
        "Rearth",
        "boundary_y",
        "boundary_x",
    )
    if any(key not in result.attrs for key in required):
        raise ValueError("decomposition metadata is missing")
    method = result.attrs["method"]
    if method == "sor":
        from ._sor import reconstruct as implementation
    elif method == "li2006":
        from . import _li2006

        dims = [result.attrs["y_dim"], result.attrs["x_dim"]]
        kwargs = {
            "dims": dims,
            "coords": result.attrs["geometry"],
            "mParams": {"Rearth": result.attrs["Rearth"]},
            "iParams": {
                "BCs": ['closed' if bc == 'fixed' else bc for bc in
                        [result.attrs["boundary_y"], result.attrs["boundary_x"]]]
            },
        }
        psi, chi = result["streamfunction"], result["velocity_potential"]
        u, v = _li2006.reconstruct_flow(psi, chi, **kwargs)
        velocity = xr.Dataset({"u": u, "v": v})
        if components:
            zero = xr.zeros_like(psi).where(np.isfinite(psi))
            ur, vr = _li2006.reconstruct_flow(psi, zero, **kwargs)
            ud, vd = _li2006.reconstruct_flow(zero, chi, **kwargs)
            velocity.update(
                {
                    "urot": ur,
                    "vrot": vr,
                    "udiv": ud,
                    "vdiv": vd,
                }
            )
        velocity.attrs.update(result.attrs)
        return velocity
    else:
        raise ValueError("unknown decomposition method in result metadata")
    velocity = implementation(result, components)
    velocity = velocity.astype(result.velocity_potential.dtype)
    velocity.attrs.update(result.attrs)
    return velocity


_COMPONENTS = ("urot", "vrot", "udiv", "vdiv")


def decompose(
    u,
    v,
    dims=None,
    coords="lat-lon",
    *,
    method="li2006",
    grid="arakawa-a",
    mask=None,
    mParams=None,
    iParams=None,
    initial=None,
    return_potentials=False,
    return_diagnostics=False,
):
    """Decompose velocity into rotational and divergent components.

    Parameters
    ----------
    u, v : xarray.DataArray
        Real velocities on the same dimensions and coordinates. Select one
        eager 2-D slice for SOR; Li2006 also accepts independent batch slices.
    dims : sequence of str or None
        Spatial dimensions in [Y, X] order. None infers lat/lon or
        latitude/longitude; specify custom dimension names explicitly.
    coords : {'lat-lon', 'cartesian'}
        Coordinates in degrees or consistent Cartesian length units.
        Exact polar rows are unsupported.
    method : {'li2006', 'sor'}
        Case-insensitive. Li2006 fits both potentials jointly; SOR solves only
        the velocity potential, then subtracts its gradient from the input.
        The rotational remainder can include harmonic circulation.
    grid : {'arakawa-a', 'arakawa-c'}
        Case-insensitive. A-grid input/output velocities share cell centers.
        C is supported by SOR only: both arrays use cell-indexed east/north
        face storage. Output velocities have physical half-step face
        coordinates; potentials retain cell-center coordinates.
    mask : xarray.DataArray of bool or None
        True for active cells. A-grid inputs infer finite(u) & finite(v)
        when omitted; C-grid inputs require an explicit center mask.
    mParams : mapping or None
        Rearth sets the radius for latitude/longitude geometry. Li2006 also
        accepts alpha, the coefficient of the potential-amplitude penalty,
        in inverse squared length units (m^-2 for metre-based coordinates).
        Larger alpha favors smaller potentials and usually a larger velocity
        error; smaller alpha favors the fit but weakens poorly constrained
        modes. None uses the grid-scaled default. To assess sensitivity,
        reduce alpha by factors of ten and compare converged solutions; use
        the same explicit alpha across grids. Zero is allowed but leaves
        constant offsets nonunique.
    iParams : mapping or None
        Solver controls; the mapping is copied. Boundary settings are:

        * BCs: one entry per spatial dimension, default ['fixed', 'fixed'].
          SOR accepts fixed, extend and periodic longitude. Fixed retains
          initial potential values; extend copies adjacent values for zero
          normal gradient. Latitude periodicity is unsupported.
        * internal_bcs: SOR defaults to fixed (hold wet coastal points).
          Extend gives zero coastal normal derivative; neumann is an alias.
          Nonzero normal derivatives are not implemented.
        * Li2006 automatically imposes solid coasts and outer walls: zero
          normal derivative of chi and one constant psi per connected coast.
          Different coastal constants are solved independently. Its BCs
          accept fixed and periodic longitude only; extend is unsupported.
          Omit internal_bcs for Li2006. Here fixed means a solid wall, not a
          prescribed scalar potential value.

        Iteration/execution controls include architect, threads, dtype,
        mxLoop, tolerance and diag_interval. The latter samples the solver
        error history; it defaults to 10 when diagnostics are requested.
        Li2006 checks convergence every iteration independently of sampling.
    initial : xarray.DataArray, xarray.Dataset, pair or None
        Li2006 accepts (psi0, chi0) or a Dataset containing streamfunction and
        velocity_potential, as initial guesses only. SOR accepts a potential
        DataArray or a Dataset containing velocity_potential; its fixed
        boundary values are retained. Fixed SOR requires the same wet mask
        as the forcing and finite initial values at all wet points.
    return_potentials : bool
        False by default. Add velocity_potential, and streamfunction for
        Li2006, increasing the number of stored fields and memory use.
    return_diagnostics : bool
        Return (result, diagnostics). Common fields are error, final_error,
        converged, iterations and stop_reason; vel_error/div_error are
        evaluated once after solving on the returned velocity grid. SOR also
        retains solution_min/solution_max histories. SOR's stopping error is
        its normalized correction; Li's is the smaller normalized
        least-squares stopping test. Equal tolerances do not imply equal
        physical accuracy. Definitions are retained in variable attributes.

    Returns
    -------
    xarray.Dataset or tuple of xarray.Dataset
        Fields urot, vrot, udiv and vdiv, optional scalar potentials, and
        optionally a separate diagnostics Dataset. SOR's velocity sum equals
        the input by subtraction, so near-zero vel_error is not an independent
        accuracy test; also examine solver error and div_error.

    Notes
    -----
    Input A-grid velocities are averaged to internal C-grid faces; derived
    velocities are interpolated to centers before returning. SOR subtracts
    on that returned grid. Center interpolation can introduce divergence
    even when the face operators are consistent. Reconstructing Li velocities
    from centered potentials is approximate; retain the velocity components
    when their original reconstruction is needed.
    """
    if not isinstance(return_potentials, (bool, np.bool_)):
        raise TypeError("return_potentials must be boolean")
    solved = _solve(
        u,
        v,
        dims,
        coords,
        method=method,
        grid=grid,
        mask=mask,
        mParams=mParams,
        iParams=iParams,
        initial=initial,
        return_diagnostics=return_diagnostics,
        _components=True,
    )
    potential_result, diagnostic = solved if return_diagnostics else (solved, None)
    if potential_result.attrs["method"] == "li2006":
        velocity = potential_result
    else:
        velocity = _reconstruct_potentials(potential_result, components=True)
    result = velocity[list(_COMPONENTS)]
    if return_potentials:
        for name in ("streamfunction", "velocity_potential"):
            if name in potential_result:
                result[name] = potential_result[name]
    result = result.astype(potential_result.velocity_potential.dtype)
    result.attrs.update(potential_result.attrs, representation="velocity_components")
    if return_diagnostics:
        diagnostic = _diagnostics(u, v, result, dims, coords, mask, diagnostic)
    return (result, diagnostic) if return_diagnostics else result


def streamfunction_potential(
    u,
    v,
    dims=None,
    coords="lat-lon",
    *,
    method="li2006",
    grid="arakawa-a",
    mask=None,
    mParams=None,
    iParams=None,
    initial=None,
    return_diagnostics=False,
):
    """Return available potentials in an xarray.Dataset.

    Li2006 returns streamfunction and velocity_potential; SOR returns only
    velocity_potential. Inputs/settings match decompose. A scalar potential
    alone cannot reconstruct SOR's rotational remainder: use decompose to
    retain both velocity components. Centered Li scalars reconstruct an
    approximate velocity because corner information has been interpolated.
    return_diagnostics=True returns (potentials, diagnostics).
    """
    solved = _solve(
        u,
        v,
        dims,
        coords,
        method=method,
        grid=grid,
        mask=mask,
        mParams=mParams,
        iParams=iParams,
        initial=initial,
        return_diagnostics=return_diagnostics,
    )
    result, diagnostic = solved if return_diagnostics else (solved, None)
    result.attrs["representation"] = "scalar_potentials"
    if return_diagnostics:
        velocity = reconstruct(result, components=True)
        diagnostic = _diagnostics(u, v, velocity, dims, coords, mask, diagnostic)
    if result.attrs["method"] == "sor":
        result = result[["velocity_potential"]]
    return (result, diagnostic) if return_diagnostics else result


def reconstruct(result, *, components=False):
    """Return total u/v from velocity components or Li2006 scalar potentials.

    Pass a Dataset from decompose or streamfunction_potential. Velocity fields
    are added directly; potentials use their recorded method, boundary rules
    and radius. components=True also returns the four named velocity fields.
    This is reconstruction of the represented solution, not a guarantee that
    it matches the original input. Preserve the Dataset metadata when saving.
    """
    if not isinstance(result, xr.Dataset):
        raise TypeError("result must be a Dataset")
    if not isinstance(components, (bool, np.bool_)):
        raise TypeError("components must be boolean")
    if all(name in result for name in _COMPONENTS):
        velocity = xr.Dataset(
            {
                "u": result.urot + result.udiv,
                "v": result.vrot + result.vdiv,
            },
            attrs=result.attrs.copy(),
        )
        if components:
            velocity.update({name: result[name] for name in _COMPONENTS})
        return velocity
    velocity = _reconstruct_potentials(result, components=components)
    return velocity


def _diagnostics(u, v, velocity, dims, coords, mask, detail):
    """Two common errors evaluated on the returned velocity grid."""
    from ..finitediffs import FiniteDiff

    dims = (velocity.attrs["y_dim"], velocity.attrs["x_dim"])
    if velocity.attrs["grid_layout"] == "arakawa-c":
        result = xr.Dataset({
            "vel_error": detail["relative_velocity_error"],
            "div_error": detail["divergence_after_rms"],
        })
    else:
        valid = np.isfinite(u) & np.isfinite(v)
        if mask is not None:
            valid = valid & mask
        ur, vr = velocity.urot, velocity.vrot
        ud, vd = velocity.udiv, velocity.vdiv
        result = xr.Dataset({
            "vel_error": _errors(
                u.where(valid), v.where(valid), ur + ud, vr + vd, list(dims)
            ).relative_velocity_error,
        })
        bcs = {
            axis: "periodic" if velocity.attrs["boundary_" + axis.lower()] == "periodic"
            else "extend"
            for axis in ("Y", "X")
        }
        fd = FiniteDiff(
            {"Y": dims[0], "X": dims[1]}, BCs=bcs.copy(), coords=coords,
            R=velocity.attrs["Rearth"],
        )
        stencil = valid & np.isfinite(ur) & np.isfinite(vr)
        for axis, dim in zip(("Y", "X"), dims):
            for offset in (-1, 1):
                neighbour = (
                    valid.roll({dim: offset}, roll_coords=False)
                    if bcs[axis] == "periodic"
                    else valid.shift({dim: offset}, fill_value=False)
                )
                stencil = stencil & neighbour
        divergence = fd.divg(
            (ur.where(valid, 0), vr.where(valid, 0)), ["X", "Y"]
        ).where(stencil)
        result["div_error"] = np.sqrt((divergence**2).mean(list(dims)))
    common = _solver_diagnostics(detail)
    result = xr.merge([common, result], join='exact')
    result["vel_error"].attrs = {
        "long_name": "relative L2 velocity reconstruction error", "units": "1",
        "definition": "norm(rotational + divergent - input) / norm(input)",
    }
    result["div_error"].attrs = {
        "long_name": "root mean square divergence of rotational velocity",
        "units": "s-1",
        "definition": "sqrt(mean(divergence(rotational_velocity)**2))",
    }
    result.attrs = {**common.attrs, "grid": velocity.attrs["grid_layout"],
                    "divergence_evaluation": "returned grid; centered stencils require wet neighbours"}
    return result


_SOLVER_FIELDS = ('error', 'final_error', 'converged', 'iterations', 'stop_reason',
                  'solution_min', 'solution_max')


def _solver_diagnostics(detail):
    """Retain the common stopping history of the selected solver."""
    return detail[[key for key in _SOLVER_FIELDS if key in detail]].copy()
