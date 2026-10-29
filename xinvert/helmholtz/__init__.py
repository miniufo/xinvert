"""Helmholtz decomposition with method-specific, reproducible reconstruction.

decompose returns four velocity components; streamfunction_potential returns
two scalar potentials for methods that compute both; reconstruct recovers
the total velocity from either representation. Results are Datasets.
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

    * li2006 (default): joint regularized, collocated inversion. All algorithm
      details and parameters live in _li2006; float64 CPU and optional
      closed-boundary GPU/LSMR are supported.
    * sor: divergence/vorticity from FiniteDiff, two existing
      invert_Poisson calls and cal_flow reconstruction. iParams are ordinary
      xinvert SOR controls, with solver='sor' optionally accepted. BCs default
      to fixed/fixed; missing velocities are zero-extended when differentiating
      and their nodes excluded from inversion. This is not the Li objective.
    * forget: existing project_vector closed Neumann projection, requiring an
      explicit wet mask and one eager 2-D slice with inactive nonperiodic edges.
      Collocated input is averaged to wet faces internally; components return
      to input centers. Explicit grid='arakawa-c' accepts the cell-indexed east/north
      storage convention of xinvert.projection and returns physical face axes.
      BCs are fixed/fixed or fixed/periodic. No psi is invented.

    dims are [Y, X]; infer lat/lon or latitude/longitude. coords is lat-lon
    (degrees) or cartesian (length coordinates). initial is (psi0, chi0) or a
    named potential Dataset for dual solves; one DataArray or a Dataset with
    velocity_potential for forget. Model/iteration dictionaries are copied.
    Results record reconstruction settings. Solver diagnostic histories retain
    their backend-specific definitions; velocity error is reported separately.

    Examples
    --------
    >>> result = decompose(u, v, dims=['lat', 'lon'], method='li2006')
    >>> velocity = reconstruct(result)
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.lower()
    if method not in ("li2006", "sor", "forget"):
        raise ValueError("method must be li2006, sor, or forget")
    if coords not in ("lat-lon", "cartesian"):
        raise ValueError("coords must be lat-lon or cartesian")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise TypeError("return_diagnostics must be boolean")
    allowed = ("arakawa-a", "arakawa-c") if method == "forget" else ("arakawa-a",)
    if not isinstance(grid, str):
        raise TypeError("grid must be arakawa-a or arakawa-c")
    grid = grid.lower()
    if grid not in allowed:
        raise ValueError(f"{method} requires grid in {allowed!r}")
    u, v, dims, mask = _inputs(u, v, dims, mask)
    mp, ip = _mapping(mParams, "mParams"), _mapping(iParams, "iParams")
    if method == "li2006":
        from . import _li2006

        solve = _li2006._helmholtz_impl if _components else _li2006.helmholtz
        extra = {"return_components": True} if _components else {}
        solved = solve(
            u,
            v,
            list(dims),
            coords,
            mask=mask,
            mParams=mp,
            iParams=ip,
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
        bcs = ip.get("BCs", ("open", "open"))
        diagnostic = solved[3 if _components else 2] if return_diagnostics else None
    elif method == "sor":
        from . import _sor

        result, diagnostic = _sor.decompose(
            u, v, dims, coords, mask, mp, ip, initial, return_diagnostics
        )
        bcs = ip.get("BCs", ("fixed", "fixed"))
    else:
        from . import _forget

        result, diagnostic = _forget.decompose(
            u, v, dims, coords, mask, mp, ip, initial, return_diagnostics, grid=grid
        )
        bcs = ip.get("BCs", ("fixed", "fixed"))
    result = _metadata(result, method, grid, dims, coords, mp, bcs)
    return (result, diagnostic) if return_diagnostics else result


def _reconstruct_potentials(result, *, components=False):
    """Return a velocity Dataset using the recorded decomposition method.

    result must be a Dataset returned by decompose with its metadata intact.
    Default variables u/v are the total reconstructed velocity. With
    components=True, additionally include udiv/vdiv and
    urot/vrot (dual potentials) or u_remainder/v_remainder
    (closed projection). For Li/Poisson, total is the sum of potential-derived
    velocities and can differ from the input. For forget, total includes the
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
    if method == "forget":
        from ._forget import reconstruct as implementation
    elif method == "sor":
        from ._sor import reconstruct as implementation
    elif method == "li2006":
        from . import _li2006

        dims = [result.attrs["y_dim"], result.attrs["x_dim"]]
        kwargs = {
            "dims": dims,
            "coords": result.attrs["geometry"],
            "mParams": {"Rearth": result.attrs["Rearth"]},
            "iParams": {
                "BCs": [result.attrs["boundary_y"], result.attrs["boundary_x"]]
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
    """Return urot, vrot, udiv and vdiv.

    Inputs u/v are xarray.DataArray objects; dims=[Y, X].
    method is 'sor', 'forget' or 'li2006' (default).
    grid is 'arakawa-a' (default) or 'arakawa-c' (Forget only),
    case-insensitive. Arakawa A inputs and outputs share center coordinates.
    Explicit C input follows xinvert.projection's cell-indexed east/north
    storage; output carries physical face coordinates.

    mask is a boolean DataArray (True for active cells).
    mParams['alpha'] controls Li's potential-amplitude penalty;
    increasing alpha favors smaller potentials at the expense of velocity fit.
    iParams carries boundary, iteration and execution controls.
    return_potentials adds the available potentials, increasing output memory.
    return_diagnostics returns (result, diagnostics), with vel_error,
    div_error, converged and iterations; see the diagnostics guide.

    reconstruct(result) adds retained velocity components. Reconstructing
    from Li's centered scalars instead is approximate. Forget's rotational
    component is the subtraction remainder and can include harmonic flows.
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
    nondiv = (
        ("u_remainder", "v_remainder")
        if potential_result.attrs["method"] == "forget"
        else ("urot", "vrot")
    )
    names = dict(zip((*nondiv, "udiv", "vdiv"), _COMPONENTS))
    result = velocity[list(names)].rename(names)
    if return_potentials:
        for name in ("streamfunction", "velocity_potential"):
            if name in potential_result:
                result[name] = potential_result[name]
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
    """Return streamfunction and velocity_potential in an xarray.Dataset.

    Inputs/settings match decompose. Only li2006 and sor compute
    both scalar potentials. For forget, use decompose(return_potentials=True)
    to retrieve chi with the two vector components; that projection does not
    uniquely determine a streamfunction. This entry point avoids allocating
    four output velocity arrays when only the scalar potentials are needed.
    return_diagnostics=True returns (potentials, diagnostics). Centered Li psi
    has lost vertex information: reconstruct of these two scalars is approximate
    and its diagnostic velocity error refers to that scalar reconstruction.
    Use decompose for velocities interpolated directly from the native fit.
    """
    if isinstance(method, str) and method.lower() == "forget":
        raise NotImplementedError(
            "forget determines one potential; use decompose(return_potentials=True)"
        )
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
    return (result, diagnostic) if return_diagnostics else result


def reconstruct(result, *, components=False):
    """Return total u/v from a decomposition or a pair of scalar potentials.

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
    if components:
        nondiv = (
            ("u_remainder", "v_remainder")
            if result.attrs["method"] == "forget"
            else ("urot", "vrot")
        )
        velocity = velocity.rename(
            dict(zip((*nondiv, "udiv", "vdiv"), _COMPONENTS))
        )
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
    if "converged" in detail:
        result["converged"] = detail["converged"]
        result["iterations"] = detail["iterations"]
    else:
        result["converged"] = detail["psi_converged"] & detail["chi_converged"]
        result["iterations"] = xr.apply_ufunc(
            np.maximum, detail["psi_iterations"], detail["chi_iterations"]
        )
    result["vel_error"].attrs = {
        "long_name": "relative L2 velocity reconstruction error", "units": "1",
        "definition": "norm(rotational + divergent - input) / norm(input)",
    }
    result["div_error"].attrs = {
        "long_name": "root mean square divergence of rotational velocity",
        "units": "s-1",
        "definition": "sqrt(mean(divergence(rotational_velocity)**2))",
    }
    result.attrs = {"grid": velocity.attrs["grid_layout"],
                    "divergence_evaluation": "returned grid; centered stencils require wet neighbours"}
    return result
