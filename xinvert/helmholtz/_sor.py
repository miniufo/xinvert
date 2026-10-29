"""Classical two-Poisson assembly using existing xinvert solvers/operators."""

import numpy as np
import xarray as xr

from ..apps import invert_Poisson
from ..finitediffs import FiniteDiff
from ..utils import cal_flow


def _params(ip, diagnostics):
    ip = ip.copy()
    if ip.pop("solver", "sor") != "sor":
        raise ValueError("sor currently supports solver=sor")
    ip.setdefault("BCs", ["fixed", "fixed"])
    ip.setdefault("dtype", "float64")
    ip.setdefault("printInfo", False)
    if "undef" in ip and not np.isnan(ip["undef"]):
        raise ValueError("Helmholtz wrappers require undef=NaN")
    ip["undef"] = np.nan
    if diagnostics and ip.get("diag_interval", 0) <= 0:
        ip["diag_interval"] = 10
    return ip


def _split(solved):
    return solved if isinstance(solved, tuple) else (solved, None)


def _named(diagnostic, prefix, batch):
    renames = {key: f"{prefix}_{key}" for key in diagnostic.dims if key not in batch}
    renames.update({key: f"{prefix}_{key}" for key in diagnostic.data_vars})
    return diagnostic.rename(renames)


def decompose(u, v, dims, coords, mask, mp, ip, initial, diagnostics):
    from . import _errors, _metadata

    if set(mp) - {"Rearth"}:
        raise ValueError("sor mParams accepts Rearth only")
    ip = _params(ip, diagnostics)
    valid = np.isfinite(u) & np.isfinite(v)
    if mask is not None:
        valid = valid & mask
    finite_u, finite_v = u.where(valid, 0), v.where(valid, 0)
    fd = FiniteDiff(
        {"Y": dims[0], "X": dims[1]},
        BCs={"Y": ip["BCs"][0], "X": ip["BCs"][1]},
        coords=coords,
        R=mp.get("Rearth", 6371200.0),
    )
    divergence = fd.divg((finite_u, finite_v), ["X", "Y"]).where(valid)
    vorticity = fd.curl(finite_u, finite_v).where(valid)
    if isinstance(initial, xr.Dataset):
        initial = (initial["streamfunction"], initial["velocity_potential"])
    if initial is None:
        initial = (None, None)
    if not isinstance(initial, (tuple, list)) or len(initial) != 2:
        raise TypeError("initial must be (psi0, chi0) or a named potential Dataset")
    psi, pdiag = _split(
        invert_Poisson(
            vorticity, list(dims), coords, icbc=initial[0], mParams=mp, iParams=ip
        )
    )
    chi, cdiag = _split(
        invert_Poisson(
            divergence, list(dims), coords, icbc=initial[1], mParams=mp, iParams=ip
        )
    )
    result = xr.Dataset(
        {
            "streamfunction": psi.where(valid).transpose(*u.dims),
            "velocity_potential": chi.where(valid).transpose(*u.dims),
        }
    )
    result.attrs.update(
        discretization="xinvert FiniteDiff and scalar Poisson",
        internal_bcs=ip.get("internal_bcs", "fixed"),
    )
    for value in result.data_vars.values():
        value.attrs.update(units="m2 s-1")
    if not diagnostics:
        return result, None
    _metadata(result, "sor", "arakawa-a", dims, coords, mp, ip["BCs"])
    velocity = reconstruct(result, False)
    diagnostic = _errors(
        u.where(valid), v.where(valid), velocity.u, velocity.v, list(dims)
    )
    batch = set(u.dims) - set(dims)
    for prefix, detail in [("psi", pdiag), ("chi", cdiag)]:
        if detail is not None:
            diagnostic = xr.merge([diagnostic, _named(detail, prefix, batch)])
    return result, diagnostic


def reconstruct(result, components):
    dims = [result.attrs["y_dim"], result.attrs["x_dim"]]
    kwargs = {
        "dims": dims,
        "coords": result.attrs["geometry"],
        "BCs": [result.attrs["boundary_y"], result.attrs["boundary_x"]],
        "mParams": {"Rearth": result.attrs["Rearth"]},
    }
    # Reapply the same zero-land extension used when computing diagnostics.
    psi, chi = result.streamfunction, result.velocity_potential
    valid = np.isfinite(psi) & np.isfinite(chi)
    ur, vr = cal_flow(psi.where(valid, 0), vtype="streamfunction", **kwargs)
    ud, vd = cal_flow(chi.where(valid, 0), vtype="velocitypotential", **kwargs)
    velocity = xr.Dataset({"u": (ur + ud).where(valid), "v": (vr + vd).where(valid)})
    if components:
        velocity.update(
            {
                "urot": ur.where(valid),
                "vrot": vr.where(valid),
                "udiv": ud.where(valid),
                "vdiv": vd.where(valid),
            }
        )
    return velocity
