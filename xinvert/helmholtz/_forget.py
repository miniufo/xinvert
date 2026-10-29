"""Closed Neumann projection assembled from xinvert.projection, without kernels.

This exposes the previously discussed Forget/Ferreira-style divergent
projection convention, not a new reproduction of the full published workflow.
"""

import numpy as np
import xarray as xr

from ..projection import project_vector


def decompose(u, v, dims, coords, mask, mp, ip, initial, diagnostics, *, grid="arakawa-c"):
    if mask is None:
        raise ValueError("forget requires an explicit boolean wet mask")
    input_u, input_v = u, v
    ip = ip.copy()
    if ip.pop("solver", "sor") != "sor":
        raise ValueError("forget currently supports solver=sor")
    bcs = ip.get("BCs", ["fixed", "fixed"])
    if list(bcs) not in (["fixed", "fixed"], ["fixed", "periodic"]):
        raise ValueError("forget BCs must be fixed/fixed or fixed/periodic")
    ip.setdefault("printInfo", False)
    if diagnostics and ip.get("diag_interval", 0) <= 0:
        ip["diag_interval"] = 10
    if isinstance(initial, xr.Dataset):
        initial = initial["velocity_potential"]
    periodic = bcs[1] == "periodic"
    if grid == "arakawa-a":
        mask = mask & np.isfinite(u) & np.isfinite(v)
        east, north = _face_masks(mask, dims, periodic)
        next_u = u.roll({dims[1]: -1}, roll_coords=False) if periodic else u.shift({dims[1]: -1})
        u = (0.5 * (u.astype("float64") + next_u.astype("float64"))).where(east, 0)
        v = (0.5 * (v.astype("float64") + v.shift({dims[0]: -1}).astype("float64"))).where(north, 0)
    solved = project_vector(
        u,
        v,
        mask,
        list(dims),
        coords=coords,
        periodic_x=bcs[1] == "periodic",
        mParams=mp,
        iParams=ip,
        icbc=initial,
    )
    original, detail = solved if isinstance(solved, tuple) else (solved, None)
    result = original.rename(
        {
            "potential": "velocity_potential",
            "x_gradient": "udiv",
            "y_gradient": "vdiv",
            "x_solenoidal": "u_remainder",
            "y_solenoidal": "v_remainder",
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
        for name in ("udiv", "u_remainder"):
            result[name] = _face_average(result[name], east, dims[1], periodic).where(mask)
        for name in ("vdiv", "v_remainder"):
            result[name] = _face_average(result[name], north, dims[0], False).where(mask)
        result.attrs.update(
            output_grid="cell_center",
            velocity_interpolation="mean of adjacent available wet faces",
            input_interpolation="neighbor center average to wet-wet faces; blocked flux zero",
        )
        for name in ("udiv", "vdiv", "u_remainder", "v_remainder"):
            result[name].attrs.update(location="cell_center", units="m s-1")
        if diagnostics:
            from . import _errors
            errors = _errors(
                input_u.where(mask), input_v.where(mask),
                result.udiv + result.u_remainder,
                result.vdiv + result.v_remainder, list(dims),
            )
            diagnostic.update(errors)
            diagnostic.attrs["velocity_error_grid"] = "original input centers"
    else:
        if diagnostics:
            east, north = _face_masks(mask, dims, periodic)
            a = (result.udiv + result.u_remainder - input_u).where(east)
            b = (result.vdiv + result.v_remainder - input_v).where(north)
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
        if name in ("udiv", "vdiv", "u_remainder", "v_remainder"):
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
    required = ("udiv", "vdiv", "u_remainder", "v_remainder")
    if any(name not in result for name in required):
        raise ValueError(
            "forget reconstruction needs stored gradient and remainder fields"
        )
    velocity = xr.Dataset(
        {
            "u": result.udiv + result.u_remainder,
            "v": result.vdiv + result.v_remainder,
        }
    )
    if components:
        velocity.update({key: result[key] for key in required})
    return velocity
