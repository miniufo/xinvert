"""Li2006 regularized inversion on an internal C grid.

chi is at cell centers, psi at vertices, and velocities on east/north faces.
Input center velocities are explicitly averaged onto wet-wet faces. Open
outside faces are omitted, rather than set to zero. The exact matrix-free
Euclidean adjoint includes masks and spherical metrics. Positive Tikhonov
regularization selects a unique discrete solution.

Outputs retain input coordinates: vertex psi is averaged to centers; chi
already lies at centers. This interpolation is lossy. Public reconstruct_flow
differentiates the returned centered potentials and does NOT reproduce the
native fit exactly. Diagnostics distinguish native_velocity_rmse from the
returned-field velocity_rmse. Four-component public output follows the latter.

Eager float64 is implemented. Explicit closed boundaries use compact compiled
operators, Jacobi column scaling, CPU threading and optional CUDA LSMR. Default open
boundaries retain their original unweighted objective and NumPy operator.
Independent batch slices may be loaded from Dask. No sparse matrix is stored.
Reference: doi:10.1175/MWR3249.1.
"""

import time
import warnings
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
import xarray as xr
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from .._threads import DEFAULT_THREADS, numba_threads, validate_threads

__all__ = ["helmholtz", "reconstruct_flow"]


def _options(values, defaults, label):
    if values is not None and not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = defaults.copy()
    if values:
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
        result.update(values)
    return result


def _positive(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _prepare(u, v, dims, coords, mask, mParams, iParams):
    if not isinstance(u, xr.DataArray) or not isinstance(v, xr.DataArray):
        raise TypeError("u and v must be xarray.DataArray objects")
    if set(u.dims) != set(v.dims):
        raise ValueError("u and v must have the same dimensions")
    u, v = xr.align(u, v, join="exact", copy=False)
    v = v.transpose(*u.dims)
    if any(a.dtype.kind not in "fiu" for a in (u, v)):
        raise TypeError("real numeric input is required")
    if dims is None:
        pairs = [("lat", "lon"), ("latitude", "longitude")]
        found = [p for p in pairs if all(d in u.dims for d in p)]
        if len(found) != 1:
            raise ValueError("specify dims=[Y, X] explicitly")
        dims = found[0]
    if (
        not isinstance(dims, (tuple, list))
        or len(dims) != 2
        or dims[0] == dims[1]
        or any(d not in u.dims for d in dims)
    ):
        raise ValueError("dims must contain two distinct input dimensions [Y, X]")
    dims = tuple(dims)
    if coords not in ("lat-lon", "cartesian"):
        raise ValueError("coords must be 'lat-lon' or 'cartesian'")
    mp = _options(
        mParams, {"alpha": None, "Rearth": 6371200.0}, "mParams"
    )
    ip = _options(
        iParams,
        {
            "solver": "lsmr",
            "architect": "cpu",
            "threads": DEFAULT_THREADS,
            "dtype": "float64",
            "BCs": ("open", "open"),
            "tolerance": 1e-8,
            "mxLoop": 5000,
            "conlim": 1e8,
            "printInfo": False,
            "on_nonconvergence": "warn",
            "diagnostics_fields": False,
        },
        "iParams",
    )
    if ip["architect"] not in ("cpu", "cpu-mc", "gpu"):
        raise NotImplementedError(
            "li2006 implements cpu and closed-boundary cpu-mc/gpu"
        )
    if ip["solver"] not in ("lsmr", "lsqr"):
        raise ValueError("solver must be lsmr or lsqr")
    dtype = np.dtype(ip["dtype"])
    if dtype != np.dtype("float64"):
        raise ValueError("this initial li2006 implementation requires dtype=float64")
    ip["dtype"] = dtype
    ip["tolerance"] = _positive(ip["tolerance"], "tolerance")
    ip["conlim"] = _positive(ip["conlim"], "conlim", allow_zero=True)
    if (
        isinstance(ip["mxLoop"], (bool, np.bool_))
        or not isinstance(ip["mxLoop"], (int, np.integer))
        or ip["mxLoop"] < 1
    ):
        raise ValueError("mxLoop must be a positive integer")
    if ip["on_nonconvergence"] not in ("warn", "raise", "ignore"):
        raise ValueError("on_nonconvergence must be warn, raise or ignore")
    for name in ("printInfo", "diagnostics_fields"):
        if not isinstance(ip[name], (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean")
    bcs = ip["BCs"]
    if (
        not isinstance(bcs, (list, tuple))
        or len(bcs) != 2
        or any(b not in ("open", "periodic", "closed") for b in bcs)
    ):
        raise ValueError("BCs must contain two entries from open, periodic, closed")
    closed = "closed" in bcs
    if closed and (bcs[0] != "closed" or bcs[1] not in ("closed", "periodic")):
        raise ValueError(
            "closed Li requires BCs=[closed, closed] or [closed, periodic]"
        )
    if not closed and ip["architect"] != "cpu":
        raise NotImplementedError("cpu-mc/gpu requires explicit closed Li boundaries")
    if ip["architect"] == "gpu" and ip["solver"] != "lsmr":
        raise NotImplementedError("Li GPU currently supports solver='lsmr' only")
    validate_threads(ip["threads"])
    ip["closed"] = closed
    if coords == "lat-lon" and bcs[0] == "periodic":
        raise ValueError("latitude cannot be periodic")
    steps, positions = [], []
    for dim in dims:
        if dim not in u.coords or u[dim].dims != (dim,) or u.sizes[dim] < 3:
            raise ValueError(f"{dim} needs a one-dimensional coordinate of length >= 3")
        pos = np.asarray(u[dim].values, dtype=float)
        step = np.diff(pos)
        nominal = (pos[-1] - pos[0]) / (pos.size - 1)
        precision = (
            np.finfo(u[dim].dtype).eps
            if u[dim].dtype.kind == "f"
            else np.finfo(float).eps
        )
        rounding = 4 * precision * max(float(np.max(np.abs(pos))), abs(nominal))
        if (
            not np.isfinite(pos).all()
            or nominal == 0
            or not np.all(step * nominal > 0)
            or not np.allclose(step, nominal, rtol=1e-6, atol=rounding)
        ):
            raise ValueError(f"{dim} must be finite, monotonic and uniformly spaced")
        steps.append(float(nominal))
        positions.append(pos)
    if coords == "lat-lon":
        radius = _positive(mp["Rearth"], "Rearth")
        if np.any(np.abs(positions[0]) >= 90):
            raise ValueError("latitude must lie strictly between -90 and 90 degrees")
        if abs(steps[1]) * (len(positions[1]) - 1) >= 360 - abs(steps[1]) * 1e-6:
            raise ValueError("longitude must not contain a duplicate cyclic endpoint")
        if bcs[1] == "periodic" and not np.isclose(
            abs(steps[1]) * len(positions[1]), 360, rtol=1e-6
        ):
            raise ValueError("periodic longitude must cover exactly 360 degrees")
        sy = 1 / (radius * np.deg2rad(steps[0]))
        sx = 1 / (radius * np.cos(np.deg2rad(positions[0])) * np.deg2rad(steps[1]))
    else:
        sy, sx = 1 / steps[0], np.full(len(positions[0]), 1 / steps[1])
    # Scalar column scaling improves numerical units without changing the objective.
    length = 1 / np.sqrt(sy**2 + np.mean(sx**2))
    # Keep the established default; explicit alpha is the physical coefficient
    # in the potential-amplitude term, as in Li et al. (2006).
    mp["alpha"] = (
        1e-8 / length**2 if mp["alpha"] is None
        else _positive(mp["alpha"], "alpha", allow_zero=closed)
    )
    if mask is None:
        mask = xr.DataArray(True)
    if not isinstance(mask, xr.DataArray) or mask.dtype.kind != "b":
        raise TypeError("mask must be a boolean DataArray (True means active)")
    if not set(mask.dims).issubset(u.dims):
        raise ValueError("mask dimensions must be a subset of input dimensions")
    xr.align(u, mask, join="exact", copy=False)
    return u, v, dims, mask.broadcast_like(u), sy, sx, length, mp, ip


def _diff(a, axis, periodic):
    return np.roll(a, -1, axis=axis) - a if periodic else np.diff(a, axis=axis)


def _diff_t(a, axis, periodic):
    if periodic:
        return np.roll(a, 1, axis=axis) - a
    shape = list(a.shape)
    shape[axis] += 1
    result = np.zeros(shape, dtype=a.dtype)
    first, second = [slice(None)] * 2, [slice(None)] * 2
    first[axis], second[axis] = slice(None, -1), slice(1, None)
    result[tuple(first)] -= a
    result[tuple(second)] += a
    return result


def _to_edges(a, valid, axis, periodic):
    """Linear centre-to-edge interpolation; extrapolate exposed array ends.

    Missing-neighbour edges use the available wet value, not a dry zero.
    Periodic vertex index i denotes the west/south face of centre i.
    """
    previous = np.roll(a, 1, axis=axis)
    pv = np.roll(valid, 1, axis=axis)
    total = valid.astype(float) + pv
    edge = np.divide(
        np.where(valid, a, 0) + np.where(pv, previous, 0),
        total,
        out=np.zeros_like(a),
        where=total > 0,
    )
    if periodic:
        return edge, total > 0
    shape = list(a.shape)
    shape[axis] += 1
    result = np.zeros(shape, dtype=a.dtype)
    rv = np.zeros(shape, dtype=bool)
    interior = [slice(None)] * 2
    interior[axis] = slice(1, -1)
    src = [slice(None)] * 2
    src[axis] = slice(1, None)
    result[tuple(interior)] = edge[tuple(src)]
    rv[tuple(interior)] = total[tuple(src)] > 0
    for side, near, next_ in ((0, 0, 1), (-1, -1, -2)):
        outsel, one, two = [slice(None)] * 2, [slice(None)] * 2, [slice(None)] * 2
        outsel[axis], one[axis], two[axis] = side, near, next_
        both = valid[tuple(one)] & valid[tuple(two)]
        result[tuple(outsel)] = np.where(
            both,
            1.5 * a[tuple(one)] - 0.5 * a[tuple(two)],
            np.where(valid[tuple(one)], a[tuple(one)], 0),
        )
        rv[tuple(outsel)] = valid[tuple(one)]
    return result, rv


class _Operator:
    """C-grid operator: chi at cells, psi at vertices, u/v at faces.

    Only wet-wet faces enter the objective; outside faces are not constrained.
    Open-axis vertices include both domain endpoints, periodic ones are unique.
    All transposes below are exact Euclidean transposes, including metrics/mask.
    """

    def __init__(self, active, sy, sx, bcs, dtype, sx_north=None):
        self.active, self.shape, self.dtype = active, active.shape, dtype
        ny, nx = active.shape
        self.periodic = tuple(b == "periodic" for b in bcs)
        py, px = self.periodic
        self.vertex_shape = (ny if py else ny + 1, nx if px else nx + 1)
        self.npsi = int(np.prod(self.vertex_shape))
        self.size = self.npsi + active.size
        self.sy, self.sx = sy, np.asarray(sx)[:, None]
        self.u_shape = (ny, nx if px else nx - 1)
        self.v_shape = (ny if py else ny - 1, nx)
        self.u_columns = (np.arange(self.u_shape[1]) + 1) % self.vertex_shape[1]
        self.v_rows = (np.arange(self.v_shape[0]) + 1) % self.vertex_shape[0]
        self.uwet = active & np.roll(active, -1, axis=1)
        self.vwet = active & np.roll(active, -1, axis=0)
        if not px:
            self.uwet = self.uwet[:, :-1]
        if not py:
            self.vwet = self.vwet[:-1, :]
        if sx_north is None:
            sx_north = (
                np.asarray(sx)
                if py
                else 0.5 * (np.asarray(sx)[:-1] + np.asarray(sx)[1:])
            )
        self.sx_north = np.asarray(sx_north)[:, None]
        self.nu = int(np.prod(self.u_shape))
        self.data_size = self.nu + int(np.prod(self.v_shape))
        self.face_count = int(self.uwet.sum() + self.vwet.sum())

    def split(self, x):
        x = np.asarray(x, dtype=self.dtype).ravel()
        return x[: self.npsi].reshape(self.vertex_shape), x[self.npsi :].reshape(
            self.shape
        )

    def forward(self, x):
        psi, chi = self.split(x)
        py, px = self.periodic
        u = (
            -_diff(psi[:, self.u_columns], 0, py) * self.sy
            + _diff(chi, 1, px) * self.sx
        ) * self.uwet
        v = (
            _diff(psi[self.v_rows, :], 1, px) * self.sx_north
            + _diff(chi, 0, py) * self.sy
        ) * self.vwet
        return np.concatenate((u.ravel(), v.ravel()))

    def adjoint(self, y):
        y = np.asarray(y, dtype=self.dtype).ravel()
        u = y[: self.nu].reshape(self.u_shape) * self.uwet
        v = y[self.nu :].reshape(self.v_shape) * self.vwet
        py, px = self.periodic
        psi = np.zeros(self.vertex_shape, dtype=self.dtype)
        psi[:, self.u_columns] -= _diff_t(u * self.sy, 0, py)
        psi[self.v_rows, :] += _diff_t(v * self.sx_north, 1, px)
        chi = _diff_t(u * self.sx, 1, px) + _diff_t(v * self.sy, 0, py)
        return np.concatenate((psi.ravel(), chi.ravel()))

    def sample_velocity(self, u, v):
        py, px = self.periodic
        uf = 0.5 * (u + np.roll(u, -1, axis=1))
        vf = 0.5 * (v + np.roll(v, -1, axis=0))
        if not px:
            uf = uf[:, :-1]
        if not py:
            vf = vf[:-1, :]
        return np.concatenate(
            (np.where(self.uwet, uf, 0).ravel(), np.where(self.vwet, vf, 0).ravel())
        )

    def from_centers(self, psi, chi):
        p, valid = _to_edges(psi, self.active, 1, self.periodic[1])
        p, _ = _to_edges(p, valid, 0, self.periodic[0])
        return np.concatenate((p.ravel(), np.where(self.active, chi, 0).ravel()))

    def potentials_to_centers(self, x):
        psi, chi = self.split(x)
        for axis, periodic in enumerate(self.periodic):
            if periodic:
                psi = 0.5 * (psi + np.roll(psi, -1, axis=axis))
            else:
                lo, hi = [slice(None)] * 2, [slice(None)] * 2
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                psi = 0.5 * (psi[tuple(lo)] + psi[tuple(hi)])
        return np.stack((psi, chi))


def _north_metric(field, dims, coords, mp, sx, bcs):
    if coords == "cartesian":
        if bcs[0] == "periodic":
            return 0.5 * (sx + np.roll(sx, -1))
        return 0.5 * (sx[:-1] + sx[1:])
    lat = np.asarray(field[dims[0]], dtype=float)
    lon = np.asarray(field[dims[1]], dtype=float)
    dx = np.deg2rad((lon[-1] - lon[0]) / (lon.size - 1)) * mp.get("Rearth", 6371200.0)
    return 1 / (dx * np.cos(np.deg2rad(0.5 * (lat[:-1] + lat[1:]))))


def _returned_velocity(op, potentials):
    """Reconstruction of returned centre potentials (not native C-grid fit).

    Center derivatives retain one-sided exposed boundaries. Interpolated psi
    cannot in general recover all native vertex modes. Report this separately.
    """
    if isinstance(op, _ClosedGrid):
        out = np.empty((2, *op.shape), dtype=np.float64)
        _closed_returned_velocity(
            op.active, potentials[0], potentials[1], op.sy, op.sx, op.periodic[1], out
        )
        return out
    result = []
    p, c = potentials
    for axis in (0, 1):
        active = op.active
        periodic = op.periodic[axis]
        edge = active & np.roll(active, -1, axis=axis)
        if not periodic:
            end = [slice(None)] * 2
            end[axis] = -1
            edge[tuple(end)] = False
        count = np.maximum(edge.astype(np.uint8) + np.roll(edge, 1, axis=axis), 1)
        scale = op.sy if axis == 0 else op.sx
        derivatives = []
        for a in (p, c):
            flux = (np.roll(a, -1, axis=axis) - a) * edge
            derivatives.append((flux + np.roll(flux, 1, axis=axis)) / count * scale)
        result.append(derivatives)
    return np.stack((-result[0][0] + result[1][1], result[1][0] + result[0][1]))


def _initial(initial, u):
    if initial is None:
        return None
    if isinstance(initial, xr.Dataset):
        initial = (initial["streamfunction"], initial["velocity_potential"])
    if not isinstance(initial, (list, tuple)) or len(initial) != 2:
        raise TypeError(
            "initial must be (psi0, chi0) or a Dataset with named potentials"
        )
    result = []
    for field in initial:
        if not isinstance(field, xr.DataArray) or set(field.dims) != set(u.dims):
            raise ValueError("initial potentials must have the same dimensions as u")
        xr.align(u, field, join="exact", copy=False)
        result.append(field.transpose(*u.dims))
    return result


def helmholtz(
    u,
    v,
    dims=None,
    coords="lat-lon",
    *,
    mask=None,
    mParams=None,
    iParams=None,
    initial=None,
    return_diagnostics=False,
):
    """Return same-grid ``(psi, chi)`` or ``(psi, chi, diagnostics)``.

    Parameters
    ----------
    u, v : xarray.DataArray
        Eastward and northward velocities on exactly matching coordinates.
        Additional dimensions (time, depth, etc.) are independent 2-D solves.
        Nonfinite values in either component remove that cell and incident faces.
    dims : sequence of str, optional
        Horizontal dimensions [Y, X]. Infer lat/lon or latitude/longitude only.
    coords : {'lat-lon', 'cartesian'}
        Uniform 1-D coordinates, in degrees or metres respectively. Poles and
        duplicate cyclic endpoints are excluded. Descending axes are allowed.
    mask : xarray.DataArray of bool, optional
        True for active nodes; may broadcast over batch dimensions. NaNs in
        velocities further restrict this mask. No connections cross dry nodes.
    mParams : dict, optional
        alpha: coefficient on both potential amplitudes (m^-2 with SI units).
        None (default) uses the established grid-scaled numerical default.
        Larger values favor smaller potentials; smaller values favor a closer
        velocity fit. Rearth defaults to 6371200 m.
    iParams : dict, optional
        ``solver``: lsmr (default) or lsqr (CPU); ``architect``: cpu, cpu-mc or gpu
        for explicit closed boundaries (gpu requires optional CuPy and LSMR);
        ``threads``: optional positive Numba
        thread count (cpu-mc defaults to None, inheriting the executing Numba
        thread setting; restored after each call);
        ``dtype``: float64 only in this initial implementation;
        ``BCs``: [open, open] or
        periodic per axis (longitude must span 360 degrees if periodic).
        Open means unconstrained outside faces, not homogeneous Neumann BC.
        ``BCs=[closed, periodic]`` or ``[closed, closed]`` explicitly select
        the compact closed-coast operator: zero coastal normal flux and an
        independent constant psi on each connected coast. Its objective uses
        cos(latitude) face weights (unit weights in Cartesian geometry) and
        native-vertex multiplicities in the regularization. Alpha may be zero
        for closed boundaries; potentials then need a gauge interpretation.
        ``tolerance``: 1e-8, supplied as both atol and btol; ``mxLoop``: 5000;
        ``conlim``: 1e8 (0 disables condition-limit stopping);
        ``printInfo``: False; ``on_nonconvergence``: warn/raise/ignore;
        ``diagnostics_fields``: False, optionally include u/v reconstruction
        and residual fields. Solver tolerance is not a velocity error target.
    initial : tuple of DataArray or Dataset, optional
        Starting potentials. This is a warm start, not a regularization prior.
    return_diagnostics : bool
        Return a Dataset with per-slice status, objective, RMS velocity error,
        relative L2 error, physical gradient norm, alpha, and active count.

    Notes
    -----
    A strictly positive alpha fixes constant and collocated null modes. No
    potential mean is subtracted after solving, and no component is filtered.
    All-dry slices return NaN potentials and undefined errors (zero work).
    Inputs are not modified. Output follows u's dimension order and coordinates.
    Native C-grid diagnostics refer to averaged input face velocities. Output
    psi is interpolated to centers, so reconstruct_flow measures a separate,
    generally larger returned-field error. It does not invert interpolation.

    Examples
    --------
    >>> from xinvert.li2006 import helmholtz
    >>> psi, chi = helmholtz(u, v, dims=['lat', 'lon'])
    """
    return _helmholtz_impl(
        u, v, dims, coords, mask=mask, mParams=mParams, iParams=iParams,
        initial=initial, return_diagnostics=return_diagnostics,
    )


def _helmholtz_impl(
    u, v, dims=None, coords="lat-lon", *, mask=None, mParams=None,
    iParams=None, initial=None, return_diagnostics=False, return_components=False,
):
    """Shared solve; optionally retain native-derived centered components."""
    u, v, dims, mask, sy, sx, length, mp, ip = _prepare(
        u, v, dims, coords, mask, mParams, iParams
    )
    guesses = _initial(initial, u)
    batch = tuple(d for d in u.dims if d not in dims)
    order = batch + dims
    sizes = tuple(u.sizes[d] for d in order)
    batch_shape = sizes[:-2]
    output = np.full((2,) + sizes, np.nan, dtype=ip["dtype"])
    component_output = (
        np.full((4,) + sizes, np.nan, dtype=ip["dtype"])
        if return_components else None
    )
    names = (
        "converged",
        "iterations",
        "stop_code",
        "active_count",
        "velocity_rmse",
        "relative_velocity_error",
        "gradient_norm",
        "native_velocity_rmse",
        "native_relative_velocity_error",
        "native_weighted_relative_velocity_error",
        "face_count",
        "objective_data",
        "objective_regularization",
    )
    stats = {n: np.full(batch_shape, np.nan) for n in names}
    fields = (
        np.full((4,) + sizes, np.nan, dtype=ip["dtype"])
        if return_diagnostics and ip["diagnostics_fields"]
        else None
    )
    with _closed_execution(ip):
        for index in np.ndindex(batch_shape):
            selection = dict(zip(batch, index))
            us = np.asarray(
                u.isel(selection).transpose(*dims).values,
                dtype=None
                if ip["closed"] and u.dtype == np.dtype("float32")
                else ip["dtype"],
            )
            vs = np.asarray(
                v.isel(selection).transpose(*dims).values,
                dtype=None
                if ip["closed"] and v.dtype == np.dtype("float32")
                else ip["dtype"],
            )
            active = (
                np.asarray(mask.isel(selection).transpose(*dims).values)
                & np.isfinite(us)
                & np.isfinite(vs)
            )
            count = int(active.sum())
            if count == 0:
                for key in ("iterations", "stop_code", "active_count", "face_count"):
                    stats[key][index] = 0
                stats["converged"][index] = True
                continue
            if ip["closed"]:
                lat = np.asarray(u[dims[0]], dtype=float)
                weight = (
                    np.cos(np.deg2rad(lat))
                    if coords == "lat-lon"
                    else np.ones(len(lat))
                )
                north_weight = (
                    np.cos(np.deg2rad(0.5 * (lat[:-1] + lat[1:])))
                    if coords == "lat-lon"
                    else np.ones(len(lat) - 1)
                )
                op = _ClosedGrid(
                    active,
                    sy,
                    sx,
                    _north_metric(u, dims, coords, mp, sx, ip["BCs"]),
                    ip["BCs"][1] == "periodic",
                    weight,
                    north_weight,
                )
                data = op.sample_velocity(us, vs)
                start = None
                if guesses is not None:
                    values = [
                        np.asarray(
                            g.isel(selection).transpose(*dims).values, dtype=ip["dtype"]
                        )
                        for g in guesses
                    ]
                    if any(not np.isfinite(a[active]).all() for a in values):
                        raise ValueError(
                            "initial potentials must be finite on active nodes"
                        )
                    start = op.from_centers(*values)
                solve = _solve_closed_grid
                if ip["architect"] == "gpu":
                    from ._li2006_gpu import solve_closed_gpu

                    solve = solve_closed_gpu
                x, status = solve(
                    op,
                    data,
                    length,
                    mp["alpha"] * length**2,
                    maxiter=ip["mxLoop"],
                    tolerance=ip["tolerance"],
                    conlim=ip["conlim"],
                    show=ip["printInfo"],
                    initial=start,
                    solver=ip["solver"],
                )
                code, iterations = status["stop_code"], status["iterations"]
                del start
            else:
                op = _Operator(
                    active,
                    sy,
                    sx,
                    ip["BCs"],
                    ip["dtype"],
                    _north_metric(u, dims, coords, mp, sx, ip["BCs"]),
                )
                data = op.sample_velocity(
                    np.where(active, us, 0), np.where(active, vs, 0)
                )
                n = op.size
                m = data.size
                damping = np.sqrt(mp["alpha"]) * length

                def matvec(z, op=op, damping=damping):
                    return np.concatenate((op.forward(z) * length, damping * z))

                def rmatvec(r, op=op, damping=damping, m=m):
                    return op.adjoint(r[:m]) * length + damping * r[m:]

                operator = LinearOperator(
                    (m + n, n), matvec=matvec, rmatvec=rmatvec, dtype=ip["dtype"]
                )
                rhs = np.concatenate((data, np.zeros(n, dtype=ip["dtype"])))
                x0 = None
                if guesses is not None:
                    values = [
                        np.asarray(
                            g.isel(selection).transpose(*dims).values, dtype=ip["dtype"]
                        )
                        for g in guesses
                    ]
                    if any(not np.isfinite(a[active]).all() for a in values):
                        raise ValueError(
                            "initial potentials must be finite on active nodes"
                        )
                    x0 = (
                        op.from_centers(*[np.where(active, a, 0) for a in values])
                        / length
                    )
                kwargs = {
                    "atol": ip["tolerance"],
                    "btol": ip["tolerance"],
                    "conlim": ip["conlim"],
                    "show": ip["printInfo"],
                    "x0": x0,
                }
                if ip["solver"] == "lsmr":
                    result = lsmr(operator, rhs, maxiter=ip["mxLoop"], **kwargs)
                else:
                    result = lsqr(operator, rhs, iter_lim=ip["mxLoop"], **kwargs)
                x = np.asarray(result[0] * length, dtype=ip["dtype"])
                code, iterations = result[1:3]
                del result, operator, rhs, kwargs, matvec, rmatvec, x0
            converged = code in (0, 1, 2, 4, 5) and np.isfinite(x).all()
            if not converged:
                message = (
                    f"li2006 slice {selection}: {ip['solver']} stopped with code {code}"
                )
                if ip["on_nonconvergence"] == "raise":
                    raise RuntimeError(message)
                if ip["on_nonconvergence"] == "warn":
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
            native_components = None
            if return_components:
                native_components = component_output[(slice(None),) + index]
                _native_velocity_to_centers(op, x, components=True, out=native_components)
                native_components[:, ~active] = np.nan
            potentials = op.potentials_to_centers(x)
            output[(slice(None),) + index] = np.where(active, potentials, np.nan)
            if not return_diagnostics:
                del op, data, x, potentials, us, vs
                continue
            predicted = op.forward(x)
            residual = predicted - data
            del predicted
            native_norm = np.linalg.norm(data)
            native_error = np.linalg.norm(residual)
            stats["native_velocity_rmse"][index] = (
                native_error / np.sqrt(op.face_count) if op.face_count else np.nan
            )
            stats["native_relative_velocity_error"][index] = (
                native_error / native_norm
                if native_norm
                else (0.0 if native_error == 0 else np.inf)
            )
            if ip["closed"]:
                energy = op.weighted_dot(data, data)
                error_energy = op.weighted_dot(residual, residual)
                stats["native_weighted_relative_velocity_error"][index] = (
                    np.sqrt(error_energy / energy)
                    if energy
                    else (0.0 if error_energy == 0 else np.inf)
                )
            else:
                stats["native_weighted_relative_velocity_error"][index] = stats[
                    "native_relative_velocity_error"
                ][index]
            stats["face_count"][index] = op.face_count
            returned = (
                np.stack((native_components[0] + native_components[2],
                          native_components[1] + native_components[3]))
                if return_components else _returned_velocity(op, potentials)
            )
            center_data = np.stack(
                (np.where(active, us, 0), np.where(active, vs, 0))
            ).astype(ip["dtype"], copy=False)
            center_residual = returned - center_data
            center_residual[:, ~active] = 0
            residual_norm = np.linalg.norm(center_residual)
            norm = np.linalg.norm(center_data)
            stats["converged"][index] = converged
            stats["iterations"][index] = iterations
            stats["stop_code"][index] = code
            stats["active_count"][index] = count
            stats["velocity_rmse"][index] = residual_norm / np.sqrt(2 * count)
            stats["relative_velocity_error"][index] = (
                residual_norm / norm
                if norm
                else (0.0 if residual_norm == 0 else np.inf)
            )
            if fields is not None:
                for j in range(2):
                    fields[(j,) + index] = np.where(active, returned[j], np.nan)
                    fields[(j + 2,) + index] = np.where(
                        active, center_residual[j], np.nan
                    )
            del returned, center_data, center_residual, us, vs
            stats["gradient_norm"][index] = np.linalg.norm(
                op.add_regularization_gradient(
                    op.adjoint(op.weight_faces(residual)), x, mp["alpha"]
                )
                if ip["closed"]
                else op.adjoint(residual) + mp["alpha"] * x
            )
            stats["objective_data"][index] = 0.5 * (
                op.weighted_dot(residual, residual)
                if ip["closed"]
                else float(np.dot(residual, residual))
            )
            stats["objective_regularization"][index] = (
                0.5
                * mp["alpha"]
                * (op.regularization_norm(x) if ip["closed"] else float(np.dot(x, x)))
            )
            del op, data, x, residual, potentials
    template = u.transpose(*order)
    attrs = {
        "method": "Li et al. (2006), internal C grid",
        "internal_grid": "chi at cells; psi at vertices; velocity at faces",
        "output_interpolation": "four-vertex average for psi; chi unchanged",
        "reconstruction": (
            "native C-grid velocity averaged to input centers"
            if return_components else "center differences of interpolated potentials"
        ),
        "velocity_location": "cell_center",
        "velocity_interpolation": (
            "mean of adjacent available wet faces; one face copied at exposed cells"
            if return_components else "center derivatives of returned potentials"
        ),
        "input_interpolation": "neighbor center average to wet-wet faces",
        "alpha": mp["alpha"],
        "solver": ip["solver"],
        "grid_layout": "arakawa-a",
        "boundary_y": ip["BCs"][0],
        "boundary_x": ip["BCs"][1],
        "norm": "cos-latitude face weights; native-node penalty"
        if ip["closed"] and coords == "lat-lon"
        else "unweighted Euclidean node sum",
        "operator_backend": "cuda matrix-free (CuPy LSMR)"
        if ip["architect"] == "gpu"
        else "compact compiled matrix-free"
        if ip["closed"]
        else "numpy matrix-free",
        "coords": coords,
    }
    potentials = []
    for j, name in enumerate(("streamfunction", "velocity_potential")):
        field = template.copy(data=output[j]).transpose(*u.dims).rename(name)
        field.attrs = dict(attrs, units="m2 s-1")
        potentials.append(field)
    if return_components:
        components = xr.Dataset()
        for j, name in enumerate(("urot", "vrot", "udiv", "vdiv")):
            field = template.copy(data=component_output[j]).transpose(*u.dims).rename(name)
            field.attrs = {"units": "m s-1", "location": "cell_center",
                           "reconstruction": attrs["reconstruction"]}
            components[name] = field
        components.attrs.update(attrs)
        potentials.append(components)
    if not return_diagnostics:
        return tuple(potentials)
    diagnostic = xr.Dataset(attrs=attrs)
    batch_coords = {d: u[d] for d in batch if d in u.coords}
    for key, value in stats.items():
        if key == "converged":
            value = value.astype(bool)
        elif key in ("iterations", "stop_code", "active_count", "face_count"):
            value = value.astype(np.int64)
        diagnostic[key] = xr.DataArray(value, dims=batch, coords=batch_coords)
    if fields is not None:
        for j, name in enumerate(
            ("u_reconstructed", "v_reconstructed", "u_residual", "v_residual")
        ):
            diagnostic[name] = template.copy(data=fields[j]).transpose(*u.dims)
    return (*potentials, diagnostic)


def reconstruct_flow(
    psi, chi, dims=None, coords="lat-lon", *, mask=None, mParams=None, iParams=None
):
    """Reconstruct u/v by differentiating the returned centered potentials.

    This intentionally evaluates the interpolated user-visible fields, not
    the unrecoverable native vertex solution. Native-fit error is reported
    separately by helmholtz. Pass the same coordinates, mask and boundary rules.
    """
    psi, chi, dims, mask, sy, sx, _, _, ip = _prepare(
        psi, chi, dims, coords, mask, mParams, iParams
    )
    batch = tuple(d for d in psi.dims if d not in dims)
    order = batch + dims
    shape = tuple(psi.sizes[d] for d in order)
    result = np.full((2,) + shape, np.nan, dtype=ip["dtype"])
    for index in np.ndindex(shape[:-2]):
        selection = dict(zip(batch, index))
        p, c = [
            np.asarray(a.isel(selection).transpose(*dims).values, dtype=ip["dtype"])
            for a in (psi, chi)
        ]
        active = (
            np.asarray(mask.isel(selection).transpose(*dims).values)
            & np.isfinite(p)
            & np.isfinite(c)
        )
        op = _Operator(active, sy, sx, ip["BCs"], ip["dtype"])
        values = np.stack((np.where(active, p, 0), np.where(active, c, 0)))
        velocity = _returned_velocity(op, values)
        result[(slice(None),) + index] = np.where(active, velocity, np.nan)
    template = psi.transpose(*order)
    outputs = []
    for i, name in enumerate(("u", "v")):
        field = template.copy(data=result[i]).transpose(*psi.dims).rename(name)
        field.attrs = {
            "units": "m s-1",
            "method": "li2006 reconstruction after centering",
        }
        outputs.append(field)
    return tuple(outputs)


def _native_velocity_to_centers(op, x, *, components=False, out=None):
    """Differentiate native potentials first, then average available wet faces.

    Output order with components=True is rotational u/v, divergent u/v.
    Missing/blocked faces are excluded from the average, as for the existing
    one-sided center gradient. No dry value is interpolated into a wet cell.
    """
    count = 4 if components else 2
    if out is None:
        out = np.empty((count, *op.shape), dtype=np.float64)
    if isinstance(op, _ClosedGrid):
        _closed_native_velocity(
            x, op.vm, op.cm, op.ui, op.vi, op.npsi,
            op.sy, op.sx, op.sx_north, op.periodic[1], out,
        )
        return out
    psi, chi = op.split(x)
    py, px = op.periodic
    ru = -_diff(psi[:, op.u_columns], 0, py) * op.sy
    rv = _diff(psi[op.v_rows, :], 1, px) * op.sx_north
    du = _diff(chi, 1, px) * op.sx
    dv = _diff(chi, 0, py) * op.sy

    def average(face, valid, axis, periodic):
        value = np.where(valid, face, 0)
        if periodic:
            total = value + np.roll(value, 1, axis=axis)
            number = valid.astype(np.int8) + np.roll(valid, 1, axis=axis)
        else:
            total = np.zeros(op.shape, dtype=np.float64)
            number = np.zeros(op.shape, dtype=np.int8)
            low, high = [slice(None), slice(None)], [slice(None), slice(None)]
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            total[tuple(low)] += value
            total[tuple(high)] += value
            number[tuple(low)] += valid
            number[tuple(high)] += valid
        return total / np.maximum(number, 1)

    rotational_u = average(ru, op.uwet, 1, px)
    rotational_v = average(rv, op.vwet, 0, py)
    divergent_u = average(du, op.uwet, 1, px)
    divergent_v = average(dv, op.vwet, 0, py)
    if components:
        out[:] = (rotational_u, rotational_v, divergent_u, divergent_v)
    else:
        out[0] = rotational_u + divergent_u
        out[1] = rotational_v + divergent_v
    return out


@njit(cache=True, parallel=True)
def _closed_native_velocity(z, vm, cm, ui, vi, npot, sy, sx, sxn, px, out):
    """Native C-grid derivatives and face averaging, without face work arrays."""
    ny, nx = cm.shape
    nvx = vm.shape[1]
    for j in prange(ny):
        for i in range(nx):
            ur = vr = ud = vd = 0.0
            cx = cy = 0
            k = cm[j, i]
            if k >= 0:
                east = (i + 1) % nx
                west = (i - 1) % nx
                corner = (i + 1) % nvx
                center = z[npot + k]
                if ui[j, i] >= 0:
                    ur -= (z[vm[j + 1, corner]] - z[vm[j, corner]]) * sy
                    ud += (z[npot + cm[j, east]] - center) * sx[j, 0]
                    cx += 1
                if (px or i > 0) and ui[j, west] >= 0:
                    ur -= (z[vm[j + 1, i]] - z[vm[j, i]]) * sy
                    ud += (center - z[npot + cm[j, west]]) * sx[j, 0]
                    cx += 1
                if vi[j, i] >= 0:
                    vr += (z[vm[j + 1, corner]] - z[vm[j + 1, i]]) * sxn[j]
                    vd += (z[npot + cm[j + 1, i]] - center) * sy
                    cy += 1
                if j > 0 and vi[j - 1, i] >= 0:
                    vr += (z[vm[j, corner]] - z[vm[j, i]]) * sxn[j - 1]
                    vd += (center - z[npot + cm[j - 1, i]]) * sy
                    cy += 1
            ur /= max(cx, 1)
            ud /= max(cx, 1)
            vr /= max(cy, 1)
            vd /= max(cy, 1)
            if out.shape[0] == 4:
                out[0, j, i] = ur
                out[1, j, i] = vr
                out[2, j, i] = ud
                out[3, j, i] = vd
            else:
                out[0, j, i] = ur + ud
                out[1, j, i] = vr + vd


@contextmanager
def _closed_execution(ip):
    if not ip["closed"]:
        yield
        return
    threads = ip["threads"] if ip["architect"] == "cpu-mc" else 1
    with numba_threads("cpu-mc", threads):
        yield


@njit(cache=True)
def _closed_root(parent, k):
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k


@njit(cache=True)
def _closed_union(parent, a, b):
    a = _closed_root(parent, a)
    b = _closed_root(parent, b)
    if a != b:
        parent[max(a, b)] = min(a, b)


@njit(cache=True)
def _closed_maps(wet, px, parent, vm, cm, ui, vi):
    ny, nx = wet.shape
    nvx = vm.shape[1]
    # Wet/dry vertical edges, including the nonperiodic outside.
    for j in range(ny):
        for i in range(nvx):
            right = i < nx and wet[j, i]
            left = wet[j, (i - 1) % nx] if px or i > 0 else False
            if right != left:
                _closed_union(parent, j * nvx + i, (j + 1) * nvx + i)
    for j in range(ny + 1):
        for i in range(nx):
            south = j > 0 and wet[j - 1, i]
            north = j < ny and wet[j, i]
            if south != north:
                _closed_union(parent, j * nvx + i, j * nvx + (i + 1) % nvx)
    # Identify vertices adjacent to a wet cell, then number roots in index order.
    active = np.zeros(vm.shape, np.bool_)
    for j in range(ny):
        for i in range(nx):
            if wet[j, i]:
                east = (i + 1) % nvx
                active[j, i] = True
                active[j, east] = True
                active[j + 1, i] = True
                active[j + 1, east] = True
    npot = 0
    for j in range(ny + 1):
        for i in range(nvx):
            if active[j, i]:
                k = j * nvx + i
                parent[k] = _closed_root(parent, k)
                if parent[k] == k:
                    vm[j, i] = npot
                    npot += 1
    for j in range(ny + 1):
        for i in range(nvx):
            if active[j, i]:
                r = parent[j * nvx + i]
                vm[j, i] = vm[r // nvx, r % nvx]
    nc = 0
    for j in range(ny):
        for i in range(nx):
            if wet[j, i]:
                cm[j, i] = nc
                nc += 1
    m = 0
    for j in range(ny):
        for i in range(nx if px else nx - 1):
            if wet[j, i] and wet[j, (i + 1) % nx]:
                ui[j, i] = m
                m += 1
    nu = m
    for j in range(ny - 1):
        for i in range(nx):
            if wet[j, i] and wet[j + 1, i]:
                vi[j, i] = m
                m += 1
    counts = np.zeros(npot, np.uint32)
    for j in range(ny + 1):
        for i in range(nvx):
            if vm[j, i] >= 0:
                counts[vm[j, i]] += 1
    return npot, nc, nu, m, counts


@njit(cache=True, parallel=True)
def _closed_pack(u, v, cm, ui, vi, out, centers):
    ny, nx = cm.shape
    for j in prange(ny):
        for i in range(nx):
            k = ui[j, i]
            if k >= 0:
                out[k] = (
                    0.5 * (np.float64(u[j, i]) + np.float64(u[j, (i + 1) % nx]))
                    if centers
                    else u[j, i]
                )
            k = vi[j, i]
            if k >= 0:
                out[k] = (
                    0.5 * (np.float64(v[j, i]) + np.float64(v[j + 1, i]))
                    if centers
                    else v[j, i]
                )


@njit(cache=True, inline="always")
def _closed_vertex(j, i, vm, ui, vi, up, vp, r, sc, diag):
    ny, nx = ui.shape
    nvx = vm.shape[1]
    px = nvx == nx
    total = 0.0
    left = i - 1 if i > 0 else nx - 1
    if px or i > 0:
        if j > 0:
            k = ui[j - 1, left]
            if k >= 0 and vm[j - 1, i] != vm[j, i]:
                q = -up[j - 1]
                total += q * q if diag else (q * sc) * r[k]
        if j < ny:
            k = ui[j, left]
            if k >= 0 and vm[j, i] != vm[j + 1, i]:
                q = up[j]
                total += q * q if diag else (q * sc) * r[k]
    if j > 0:
        kl = vi[j - 1, left] if px or i > 0 else -1
        kr = vi[j - 1, i] if i < nx else -1
        if kl >= 0 and vm[j, left] == vm[j, i]:
            kl = -1
        if kr >= 0 and vm[j, i] == vm[j, i + 1 if i + 1 < nvx else 0]:
            kr = -1
        q = vp[j - 1]
        if kl >= 0:
            if not diag and kr >= 0 and kr < kl:
                total += (-q * sc) * r[kr]
                kr = -1
            total += q * q if diag else (q * sc) * r[kl]
        if kr >= 0:
            total += q * q if diag else (-q * sc) * r[kr]
    return total


@njit(cache=True, inline="always")
def _closed_reg(r, p, k, m):
    return p[k] * r[m + k] if len(p) else 0.0


@njit(cache=True, parallel=True)
def _closed_transpose(
    r,
    out,
    vm,
    cm,
    ui,
    vi,
    counts,
    coast_ids,
    offsets,
    vertices,
    up,
    uc,
    vp,
    vc,
    scale,
    penalty,
    diag,
    alpha,
):
    ny, nx = cm.shape
    npot = len(counts)
    nvx = vm.shape[1]
    m = (len(r) - len(out) if len(penalty) else len(r)) if not diag else 0
    for j in prange(ny):
        for i in range(nx):
            ci = cm[j, i]
            if ci < 0:
                continue
            c = npot + ci
            s = 1.0 if diag else _closed_col(scale, c)
            total = 0.0
            west = ui[j, i - 1 if i > 0 else nx - 1]
            east = ui[j, i]
            # A nonperiodic west endpoint has no west face.
            if nvx != nx and i == 0:
                west = -1
            if west >= 0:
                if not diag and east >= 0 and east < west:
                    total += (-uc[j] * s) * r[east]
                    east = -1
                total += uc[j] ** 2 if diag else (uc[j] * s) * r[west]
            if east >= 0:
                total += uc[j] ** 2 if diag else (-uc[j] * s) * r[east]
            south = vi[j - 1, i] if j > 0 else -1
            north = vi[j, i]
            if south >= 0:
                total += vc[j - 1] ** 2 if diag else (vc[j - 1] * s) * r[south]
            if north >= 0:
                total += vc[j] ** 2 if diag else (-vc[j] * s) * r[north]
            out[c] = total + (alpha if diag else _closed_reg(r, penalty, c, m))
    for j in prange(vm.shape[0]):
        for i in range(nvx):
            c = vm[j, i]
            if c >= 0 and counts[c] == 1:
                s = 1.0 if diag else _closed_col(scale, c)
                val = _closed_vertex(j, i, vm, ui, vi, up, vp, r, s, diag)
                out[c] = val + (alpha if diag else _closed_reg(r, penalty, c, m))
    for b in prange(len(coast_ids)):
        c = coast_ids[b]
        s = 1.0 if diag else _closed_col(scale, c)
        total = 0.0
        for k in range(offsets[b], offsets[b + 1]):
            j = vertices[k] // nvx
            i = vertices[k] % nvx
            total += _closed_vertex(j, i, vm, ui, vi, up, vp, r, s, diag)
        out[c] = total + (alpha * counts[c] if diag else _closed_reg(r, penalty, c, m))


@njit(cache=True, inline="always")
def _closed_col(scale, k):
    return scale[k] if len(scale) else 1.0


@njit(cache=True, parallel=True)
def _closed_forward(z, out, vm, cm, ui, vi, up, uc, vp, vc, scale, penalty, m, npot):
    ny, nx = cm.shape
    # Descending columns match the sparse reference's final CSR row order.
    for j in prange(ny):
        for i in range(nx):
            k = ui[j, i]
            if k >= 0:
                east = i + 1 if i + 1 < nx else 0
                a = npot + cm[j, i]
                b = npot + cm[j, east]
                if a > b:
                    val = (-uc[j] * _closed_col(scale, a)) * z[a] + (
                        uc[j] * _closed_col(scale, b)
                    ) * z[b]
                else:
                    val = (uc[j] * _closed_col(scale, b)) * z[b] + (
                        -uc[j] * _closed_col(scale, a)
                    ) * z[a]
                a = vm[j, (i + 1) % vm.shape[1]]
                b = vm[j + 1, (i + 1) % vm.shape[1]]
                if a != b:
                    if a > b:
                        val += (up[j] * _closed_col(scale, a)) * z[a]
                        val += (-up[j] * _closed_col(scale, b)) * z[b]
                    else:
                        val += (-up[j] * _closed_col(scale, b)) * z[b]
                        val += (up[j] * _closed_col(scale, a)) * z[a]
                out[k] = val
            k = vi[j, i]
            if k >= 0:
                a = npot + cm[j, i]
                b = npot + cm[j + 1, i]
                val = (vc[j] * _closed_col(scale, b)) * z[b] + (
                    -vc[j] * _closed_col(scale, a)
                ) * z[a]
                a = vm[j + 1, i]
                b = vm[j + 1, (i + 1) % vm.shape[1]]
                if a != b:
                    if a > b:
                        val += (-vp[j] * _closed_col(scale, a)) * z[a]
                        val += (vp[j] * _closed_col(scale, b)) * z[b]
                    else:
                        val += (vp[j] * _closed_col(scale, b)) * z[b]
                        val += (-vp[j] * _closed_col(scale, a)) * z[a]
                out[k] = val
    if len(out) > m:
        for c in prange(len(z)):
            out[m + c] = penalty[c] * z[c]


@njit(cache=True, parallel=True)
def _closed_prepare_diagonal(diagonal, counts, penalty):
    for k in prange(len(diagonal)):
        val = diagonal[k]
        diagonal[k] = 1 / np.sqrt(val) if val > 0 else 1.0


@njit(cache=True, parallel=True)
def _closed_penalty(scale, counts, alpha, out):
    for k in prange(len(scale)):
        out[k] = np.sqrt(alpha * (counts[k] if k < len(counts) else 1)) * scale[k]


@njit(cache=True, parallel=True)
def _closed_rhs(data, out, ui, vi, wu, wv):
    for j in prange(ui.shape[0]):
        for i in range(ui.shape[1]):
            k = ui[j, i]
            if k >= 0:
                out[k] = wu[j] * data[k]
            k = vi[j, i]
            if k >= 0:
                out[k] = wv[j] * data[k]
    for k in prange(len(data), len(out)):
        out[k] = 0.0


@njit(cache=True, parallel=True)
def _closed_recover(z, scale, length):
    for k in prange(len(z)):
        z[k] = (length * scale[k]) * z[k]


@njit(cache=True, parallel=True)
def _closed_centers(z, vm, cm, npot, out):
    nvx = vm.shape[1]
    for j in prange(cm.shape[0]):
        for i in range(cm.shape[1]):
            c = cm[j, i]
            if c >= 0:
                east = (i + 1) % nvx
                out[0, j, i] = 0.25 * (
                    z[vm[j, i]] + z[vm[j + 1, i]] + z[vm[j, east]] + z[vm[j + 1, east]]
                )
                out[1, j, i] = z[npot + c]
            else:
                out[0, j, i] = 0.0
                out[1, j, i] = 0.0


@njit(cache=True, parallel=True)
def _closed_weighted_dot(a, b, ui, vi, wu, wv):
    totals = np.zeros(ui.shape[0])
    for j in prange(ui.shape[0]):
        total = 0.0
        for i in range(ui.shape[1]):
            k = ui[j, i]
            if k >= 0:
                total += (wu[j] * a[k]) * (wu[j] * b[k])
            k = vi[j, i]
            if k >= 0:
                total += (wv[j] * a[k]) * (wv[j] * b[k])
        totals[j] = total
    return totals


@njit(cache=True, parallel=True)
def _closed_weight(r, ui, vi, wu, wv):
    for j in prange(ui.shape[0]):
        for i in range(ui.shape[1]):
            k = ui[j, i]
            if k >= 0:
                r[k] *= wu[j] ** 2
            k = vi[j, i]
            if k >= 0:
                r[k] *= wv[j] ** 2


@njit(cache=True, parallel=True)
def _closed_reg_norm(z, counts):
    total = 0.0
    for k in prange(len(z)):
        total += (counts[k] if k < len(counts) else 1) * z[k] * z[k]
    return total


@njit(cache=True, parallel=True)
def _closed_reg_gradient(out, z, alpha, counts):
    for k in prange(len(z)):
        out[k] += alpha * (counts[k] if k < len(counts) else 1) * z[k]


@njit(cache=True)
def _closed_initial(psi, chi, vm, cm, counts, out):
    ny, nx = cm.shape
    nvx = vm.shape[1]
    px = nvx == nx
    for j in range(ny + 1):
        for i in range(nvx):
            k = vm[j, i]
            if k < 0:
                continue
            total = 0.0
            count = 0
            for row in range(max(0, j - 1), min(ny, j + 1)):
                if i < nx and cm[row, i] >= 0:
                    total += psi[row, i]
                    count += 1
                if (px or i > 0) and cm[row, (i - 1) % nx] >= 0:
                    total += psi[row, (i - 1) % nx]
                    count += 1
            if count:
                out[k] += total / count
    for k in range(len(counts)):
        out[k] /= counts[k]
    for j in range(ny):
        for i in range(nx):
            k = cm[j, i]
            if k >= 0:
                out[len(counts) + k] = chi[j, i]


@njit(cache=True)
def _closed_logical_nnz(vm, ui, vi):
    total = 0
    for j in range(ui.shape[0]):
        for i in range(ui.shape[1]):
            east = (i + 1) % vm.shape[1]
            if ui[j, i] >= 0:
                total += 2 + (2 if vm[j, east] != vm[j + 1, east] else 0)
            if vi[j, i] >= 0:
                total += 2 + (2 if vm[j + 1, i] != vm[j + 1, east] else 0)
    return total


@njit(cache=True, parallel=True)
def _closed_returned_velocity(active, psi, chi, sy, sx, periodic_x, out):
    """Fuse the original masked centered differences without roll temporaries."""
    ny, nx = active.shape
    for j in prange(ny):
        north = (j + 1) % ny
        south = (j - 1) % ny
        for i in range(nx):
            east = (i + 1) % nx
            west = (i - 1) % nx
            en = active[j, i] and active[north, i] and j < ny - 1
            es = active[south, i] and active[j, i] and j > 0
            ee = active[j, i] and active[j, east] and (periodic_x or i < nx - 1)
            ew = active[j, west] and active[j, i] and (periodic_x or i > 0)
            cy = max(int(en) + int(es), 1)
            cx = max(int(ee) + int(ew), 1)
            # Multiplication by the mask intentionally retains NaN propagation.
            pn = (psi[north, i] - psi[j, i]) * en
            ps = (psi[j, i] - psi[south, i]) * es
            pe = (psi[j, east] - psi[j, i]) * ee
            pw = (psi[j, i] - psi[j, west]) * ew
            cn = (chi[north, i] - chi[j, i]) * en
            cs = (chi[j, i] - chi[south, i]) * es
            ce = (chi[j, east] - chi[j, i]) * ee
            cw = (chi[j, i] - chi[j, west]) * ew
            pdy = (pn + ps) / cy * sy
            pdx = (pe + pw) / cx * sx[j, 0]
            cdy = (cn + cs) / cy * sy
            cdx = (ce + cw) / cx * sx[j, 0]
            out[0, j, i] = -pdy + cdx
            out[1, j, i] = cdy + pdx


class _ClosedGrid:
    """Compact closed y, closed/periodic x grid, with exact wet-coast constraints."""

    def __init__(
        self,
        active,
        sy,
        sx,
        sx_north,
        periodic_x=True,
        cell_weights=None,
        north_weights=None,
    ):
        self.active = np.asarray(active, bool)
        self.shape = active.shape
        ny, nx = self.shape
        self.periodic = (False, periodic_x)
        self.sy = sy
        self.sx = np.asarray(sx)[:, None]
        self.sx_north = np.asarray(sx_north)
        nvx = nx if periodic_x else nx + 1
        dtype = (
            np.int32
            if (ny + 1) * nvx + 2 * int(active.sum()) < np.iinfo(np.int32).max
            else np.int64
        )
        parent = np.arange((ny + 1) * nvx, dtype=dtype)
        self.vm = np.full((ny + 1, nvx), -1, dtype=dtype)
        self.cm = np.full(self.shape, -1, dtype=dtype)
        self.ui = np.full(self.shape, -1, dtype=dtype)
        self.vi = np.full(self.shape, -1, dtype=dtype)
        self.npsi, self.nchi, self.nu, self.data_size, self.counts = _closed_maps(
            self.active, periodic_x, parent, self.vm, self.cm, self.ui, self.vi
        )
        self.size = self.npsi + self.nchi
        self.face_count = self.data_size
        del parent
        # Only shared variables require a reduction over multiple vertices.
        self.coast_ids = np.flatnonzero(self.counts > 1).astype(dtype)
        shared = np.zeros(self.vm.shape, bool)
        for j in range(ny + 1):
            v = self.vm[j]
            valid = v >= 0
            shared[j, valid] = self.counts[v[valid]] > 1
        vertices = np.flatnonzero(shared.ravel()).astype(dtype)
        del shared
        owners = self.vm.ravel()[vertices]
        order = np.argsort(owners, kind="stable")
        self.vertices = vertices[order]
        _, starts = np.unique(owners[order], return_index=True)
        self.offsets = np.r_[starts, len(vertices)]
        self.wu = np.sqrt(np.ones(ny) if cell_weights is None else cell_weights)
        self.wv = np.sqrt(np.ones(ny - 1) if north_weights is None else north_weights)
        self.scale = None
        self.penalty = None

    def _coefficients(self, length=1.0, weighted=False):
        wu = self.wu if weighted else np.ones(self.shape[0])
        wv = self.wv if weighted else np.ones(self.shape[0] - 1)
        return (
            (wu * self.sy) * length,
            (wu * self.sx[:, 0]) * length,
            (wv * self.sx_north) * length,
            (wv * self.sy) * length,
        )

    def pack_faces(self, u, v):
        out = np.empty(self.data_size)
        _closed_pack(u, v, self.cm, self.ui, self.vi, out, False)
        return out

    def sample_velocity(self, u, v):
        out = np.empty(self.data_size)
        _closed_pack(u, v, self.cm, self.ui, self.vi, out, True)
        return out

    def prepare(self, length, alpha):
        coeff = self._coefficients(length, True)
        scale = np.empty(self.size)
        dummy = np.empty(0)
        _closed_transpose(
            dummy,
            scale,
            self.vm,
            self.cm,
            self.ui,
            self.vi,
            self.counts,
            self.coast_ids,
            self.offsets,
            self.vertices,
            *coeff,
            dummy,
            dummy,
            True,
            alpha,
        )
        _closed_prepare_diagonal(scale, self.counts, alpha)
        penalty = np.empty(self.size)
        _closed_penalty(scale, self.counts, alpha, penalty)
        self.scale, self.penalty, self.coeff = scale, penalty, coeff

    def matvec(self, z, out=None):
        if out is None:
            out = np.empty(self.data_size + self.size)
        _closed_forward(
            z,
            out,
            self.vm,
            self.cm,
            self.ui,
            self.vi,
            *self.coeff,
            self.scale,
            self.penalty,
            self.data_size,
            self.npsi,
        )
        return out

    def rmatvec(self, r, out=None):
        if out is None:
            out = np.empty(self.size)
        _closed_transpose(
            r,
            out,
            self.vm,
            self.cm,
            self.ui,
            self.vi,
            self.counts,
            self.coast_ids,
            self.offsets,
            self.vertices,
            *self.coeff,
            self.scale,
            self.penalty,
            False,
            0.0,
        )
        return out

    def adjoint(self, r):
        out = np.empty(self.size)
        dummy = np.empty(0)
        _closed_transpose(
            r,
            out,
            self.vm,
            self.cm,
            self.ui,
            self.vi,
            self.counts,
            self.coast_ids,
            self.offsets,
            self.vertices,
            *self._coefficients(),
            dummy,
            dummy,
            False,
            0.0,
        )
        return out

    def weighted_dot(self, a, b):
        return float(
            _closed_weighted_dot(a, b, self.ui, self.vi, self.wu, self.wv).sum()
        )

    def weight_faces(self, r):
        out = r.copy()
        _closed_weight(out, self.ui, self.vi, self.wu, self.wv)
        return out

    def regularization_norm(self, z):
        return float(_closed_reg_norm(z, self.counts))

    def add_regularization_gradient(self, out, z, alpha):
        _closed_reg_gradient(out, z, alpha, self.counts)
        return out

    def from_centers(self, psi, chi):
        out = np.zeros(self.size)
        _closed_initial(psi, chi, self.vm, self.cm, self.counts, out)
        return out

    def potentials_to_centers(self, z):
        out = np.empty((2,) + self.shape)
        _closed_centers(z, self.vm, self.cm, self.npsi, out)
        return out

    def forward(self, z):
        # Native physical operator; avoids persistent dense row weights.
        dummy = np.empty(0)
        out = np.empty(self.data_size)
        _closed_forward(
            z,
            out,
            self.vm,
            self.cm,
            self.ui,
            self.vi,
            *self._coefficients(),
            dummy,
            dummy,
            self.data_size,
            self.npsi,
        )
        return out


def _solve_closed_grid(
    op,
    data,
    length,
    alpha,
    *,
    maxiter=5000,
    tolerance=1e-9,
    conlim=1e12,
    show=False,
    initial=None,
    solver="lsmr",
    check_adjoint=False,
):
    """Same augmented Jacobi-scaled objective as the CSR reference."""
    op.prepare(length, alpha)
    operator = LinearOperator(
        (op.data_size + op.size, op.size),
        matvec=op.matvec,
        rmatvec=op.rmatvec,
        dtype=float,
    )
    adjerror = None
    if check_adjoint:
        rng = np.random.default_rng(1440)
        x = rng.normal(size=op.size)
        y = rng.normal(size=op.data_size + op.size)
        a, b = np.dot(op.matvec(x), y), np.dot(x, op.rmatvec(y))
        adjerror = float(abs(a - b) / max(abs(a), abs(b), 1.0))
        if adjerror > 1e-12:
            raise RuntimeError(f"Adjoint error: {adjerror}")
        del x, y
    rhs = np.empty(op.data_size + op.size)
    _closed_rhs(data, rhs, op.ui, op.vi, op.wu, op.wv)
    x0 = None if initial is None else initial / (length * op.scale)
    start = time.perf_counter()
    # Neither Krylov method parallelizes its iteration sequence. Both reuse
    # the same parallel matrix-free forward/adjoint kernels. Our closed CPU
    # LSMR additionally fuses and parallelizes vector updates and reductions;
    # SciPy LSQR below does not have those xinvert-specific optimizations.
    # This is an implementation difference, not an inherent inability of LSQR
    # to run in parallel. There is no automatic solver selection or speed
    # guarantee; comparisons must use the same objective and achieved accuracy.
    if solver == "lsmr":
        result = _lsmr_closed_cpu(
            op,
            rhs,
            overwrite_b=True,
            atol=tolerance,
            btol=tolerance,
            conlim=conlim,
            maxiter=maxiter,
            show=show,
            x0=x0,
        )
    else:
        result = lsqr(
            operator,
            rhs,
            atol=tolerance,
            btol=tolerance,
            conlim=conlim,
            iter_lim=maxiter,
            show=show,
            x0=x0,
        )
    del rhs, operator, x0
    z = result[0]
    _closed_recover(z, op.scale, length)
    op.scale = None
    op.penalty = None
    status = {
        "stop_code": int(result[1]),
        "iterations": int(result[2]),
        "converged": result[1] in (0, 1, 2, 4, 5),
        "elapsed_seconds": time.perf_counter() - start,
        "regularization": alpha,
        "atol_btol": tolerance,
        "adjoint_relative_error": adjerror,
    }
    return z, status


# Deterministic block partitions keep reductions independent of thread count.
_CPU_BLOCK = 8192


@njit(cache=True, inline="always")
def _cpu_scaled_squares(x, start, end, scale):
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return scale
    s0 = s1 = s2 = s3 = 0.0
    k = start
    if scale >= np.finfo(np.float64).tiny:
        inv = 1.0 / scale
        while k + 3 < end:
            a = x[k] * inv
            b = x[k + 1] * inv
            c = x[k + 2] * inv
            d = x[k + 3] * inv
            s0 += a * a
            s1 += b * b
            s2 += c * c
            s3 += d * d
            k += 4
        while k < end:
            a = x[k] * inv
            s0 += a * a
            k += 1
    else:
        while k < end:
            a = x[k] / scale
            s0 += a * a
            k += 1
    return (s0 + s1) + (s2 + s3)


@njit(cache=True)
def _cpu_finish_norm(parts, blocks):
    scale = 0.0
    for k in range(blocks):
        value = parts[k, 0]
        if np.isnan(value):
            return np.nan
        scale = max(scale, value)
    if scale == 0.0 or np.isinf(scale):
        return scale
    total = 0.0
    for k in range(blocks):
        ratio = parts[k, 0] / scale
        total += parts[k, 1] * (ratio * ratio)
    return scale * np.sqrt(total)


@njit(cache=True, inline="always")
def _cpu_raw_sumsq(x, start, end):
    s0 = s1 = s2 = s3 = 0.0
    k = start
    while k + 3 < end:
        a, b, c, d = x[k], x[k + 1], x[k + 2], x[k + 3]
        s0 += a * a
        s1 += b * b
        s2 += c * c
        s3 += d * d
        k += 4
    while k < end:
        a = x[k]
        s0 += a * a
        k += 1
    return (s0 + s1) + (s2 + s3)


@njit(cache=True, inline="always")
def _cpu_save_norm_part(x, start, end, raw, parts, block):
    # Ordinary data need no maximum scan or second pass. This conservative
    # lower bound makes lost underflow terms negligible even for a full block.
    if np.isfinite(raw) and raw >= 1e-200:
        parts[block, 0] = np.sqrt(raw)
        parts[block, 1] = 1.0
        return
    scale = 0.0
    for k in range(start, end):
        value = abs(x[k])
        if np.isnan(value):
            scale = np.nan
            break
        scale = max(scale, value)
    parts[block, 0] = scale
    parts[block, 1] = _cpu_scaled_squares(x, start, end, scale)


@njit(cache=True, parallel=True)
def _cpu_norm_parts(x, parts):
    blocks = (len(x) + _CPU_BLOCK - 1) // _CPU_BLOCK
    for block in prange(blocks):
        start = block * _CPU_BLOCK
        end = min(start + _CPU_BLOCK, len(x))
        raw = _cpu_raw_sumsq(x, start, end)
        _cpu_save_norm_part(x, start, end, raw, parts, block)


@njit(cache=True, parallel=True)
def _cpu_axpy_parts(dest, source, a, b, parts):
    blocks = (len(dest) + _CPU_BLOCK - 1) // _CPU_BLOCK
    for block in prange(blocks):
        start = block * _CPU_BLOCK
        end = min(start + _CPU_BLOCK, len(dest))
        s0 = s1 = s2 = s3 = 0.0
        k = start
        while k + 3 < end:
            q0 = dest[k] * a + source[k] * b
            q1 = dest[k + 1] * a + source[k + 1] * b
            q2 = dest[k + 2] * a + source[k + 2] * b
            q3 = dest[k + 3] * a + source[k + 3] * b
            dest[k], dest[k + 1], dest[k + 2], dest[k + 3] = q0, q1, q2, q3
            s0 += q0 * q0
            s1 += q1 * q1
            s2 += q2 * q2
            s3 += q3 * q3
            k += 4
        while k < end:
            q0 = dest[k] * a + source[k] * b
            dest[k] = q0
            s0 += q0 * q0
            k += 1
        raw = (s0 + s1) + (s2 + s3)
        _cpu_save_norm_part(dest, start, end, raw, parts, block)


@njit(cache=True, inline="always")
def _cpu_update_one(k, x, h, hbar, v, a, b, c):
    old_h = h[k]
    value = hbar[k] * a
    value += old_h
    hbar[k] = value
    new_x = x[k] + b * value
    x[k] = new_x
    h[k] = old_h * c + v[k]
    return new_x


@njit(cache=True, parallel=True)
def _cpu_update_parts(x, h, hbar, v, a, b, c, parts):
    blocks = (len(x) + _CPU_BLOCK - 1) // _CPU_BLOCK
    for block in prange(blocks):
        start = block * _CPU_BLOCK
        end = min(start + _CPU_BLOCK, len(x))
        s0 = s1 = s2 = s3 = 0.0
        k = start
        while k + 3 < end:
            q0 = _cpu_update_one(k, x, h, hbar, v, a, b, c)
            q1 = _cpu_update_one(k + 1, x, h, hbar, v, a, b, c)
            q2 = _cpu_update_one(k + 2, x, h, hbar, v, a, b, c)
            q3 = _cpu_update_one(k + 3, x, h, hbar, v, a, b, c)
            s0 += q0 * q0
            s1 += q1 * q1
            s2 += q2 * q2
            s3 += q3 * q3
            k += 4
        while k < end:
            q0 = _cpu_update_one(k, x, h, hbar, v, a, b, c)
            s0 += q0 * q0
            k += 1
        raw = (s0 + s1) + (s2 + s3)
        _cpu_save_norm_part(x, start, end, raw, parts, block)


def _cpu_norm(x, parts):
    _cpu_norm_parts(x, parts)
    return _cpu_finish_norm(parts, (len(x) + _CPU_BLOCK - 1) // _CPU_BLOCK)


@njit(cache=True, parallel=True)
def _cpu_scale(x, value):
    for k in prange(len(x)):
        x[k] *= value


def _cpu_axpy_norm(dest, source, a, b, parts):
    _cpu_axpy_parts(dest, source, a, b, parts)
    return _cpu_finish_norm(parts, (len(dest) + _CPU_BLOCK - 1) // _CPU_BLOCK)


def _cpu_update_norm(x, h, hbar, v, a, b, c, parts):
    _cpu_update_parts(x, h, hbar, v, a, b, c, parts)
    return _cpu_finish_norm(parts, (len(x) + _CPU_BLOCK - 1) // _CPU_BLOCK)


# Scalar recurrence adapted from SciPy 1.16.1 scipy/sparse/linalg/_isolve/lsmr.py.
# Copyright (C) 2010 David Fong and Michael Saunders
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Givens helper adapted from SciPy LSQR; retain its upstream notice.
# Copyright (c) 2006, Systems Optimization Laboratory
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# 
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
# 
#     * Neither the name of Stanford University nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 


def _cpu_sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    """
    if b == 0:
        return (np.sign(a), 0, abs(a))
    elif a == 0:
        return (0, np.sign(b), abs(b))
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / np.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / np.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return (c, s, r)


def _lsmr_closed_cpu(
    op,
    b,
    damp=0.0,
    atol=1e-06,
    btol=1e-06,
    conlim=100000000.0,
    maxiter=None,
    show=False,
    x0=None,
    *,
    overwrite_b=False,
):
    """Float64 closed-grid LSMR with reusable buffers and fused CPU kernels."""
    A = op
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()
    msg = (
        "The exact solution is x = 0, or x = x0, if x0 was given  ",
        "Ax - b is small enough, given atol, btol                  ",
        "The least-squares solution is good enough, given atol     ",
        "The estimate of cond(Abar) has exceeded conlim            ",
        "Ax - b is small enough for this machine                   ",
        "The least-squares solution is good enough for this machine",
        "Cond(Abar) seems to be too large for this machine         ",
        "The iteration limit has been reached                      ",
    )
    hdg1 = "   itn      x(1)       norm r    norm Ar"
    hdg2 = " compatible   LS      norm A   cond A"
    pfreq = 20
    pcount = 0
    m, n = (op.data_size + op.size, op.size)
    minDim = min([m, n])
    if maxiter is None:
        maxiter = minDim
    dtype = np.float64
    if show:
        print(" ")
        print("LSMR            Least-squares solution of  Ax = b\n")
        print(f"The matrix A has {m} rows and {n} columns")
        print(f"damp = {damp:20.14e}\n")
        print(f"atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n")
        print(f"btol = {btol:8.2e}             maxiter = {maxiter:8g}\n")
    scratch = np.empty(m)
    parts = np.empty(((m + _CPU_BLOCK - 1) // _CPU_BLOCK, 2))
    u = b if overwrite_b else b.copy()
    normb = _cpu_norm(b, parts)
    if x0 is None:
        x = np.zeros(n, dtype)
        beta = normb
    else:
        x = np.atleast_1d(x0.copy()).ravel()
        A.matvec(x, out=scratch)
        beta = _cpu_axpy_norm(u, scratch, 1.0, -1.0, parts)
    if beta > 0:
        _cpu_scale(u, 1 / beta)
        v = np.empty(n)
        A.rmatvec(u, out=v)
        alpha = _cpu_norm(v, parts)
    else:
        v = np.zeros(n, dtype)
        alpha = 0
    if alpha > 0:
        _cpu_scale(v, 1 / alpha)
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0
    h = v.copy()
    hbar = np.zeros(n, dtype)
    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0
    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e100
    normA = np.sqrt(normA2)
    condA = 1
    normx = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return (x, istop, itn, normr, normar, normA, condA, normx)
    if normb == 0:
        x[()] = 0
        return (x, istop, itn, normr, normar, normA, condA, normx)
    if show:
        print(" ")
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f"{itn:6g} {x[0]:12.5e}"
        str2 = f" {normr:10.3e} {normar:10.3e}"
        str3 = f"  {test1:8.1e} {test2:8.1e}"
        print(f"{str1}{str2}{str3}")
    while itn < maxiter:
        itn = itn + 1
        A.matvec(v, out=scratch)
        beta = _cpu_axpy_norm(u, scratch, -alpha, 1.0, parts)
        if beta > 0:
            _cpu_scale(u, 1 / beta)
            A.rmatvec(u, out=scratch[:n])
            alpha = _cpu_axpy_norm(v, scratch[:n], -beta, 1.0, parts)
            if alpha > 0:
                _cpu_scale(v, 1 / alpha)
        chat, shat, alphahat = _cpu_sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _cpu_sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _cpu_sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        normx = _cpu_update_norm(
            x,
            h,
            hbar,
            v,
            -(thetabar * rho / (rhoold * rhobarold)),
            zeta / (rho * rhobar),
            -(thetanew / rho),
            parts,
        )
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _cpu_sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = np.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 = normA2 + beta * beta
        normA = np.sqrt(normA2)
        normA2 = normA2 + alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        test1 = normr / normb
        if normA * normr != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = np.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if show and (
            n <= 40
            or itn <= 10
            or itn >= maxiter - 10
            or (itn % 10 == 0)
            or (test3 <= 1.1 * ctol)
            or (test2 <= 1.1 * atol)
            or (test1 <= 1.1 * rtol)
            or (istop != 0)
        ):
            if pcount >= pfreq:
                pcount = 0
                print(" ")
                print(hdg1, hdg2)
            pcount = pcount + 1
            str1 = f"{itn:6g} {x[0]:12.5e}"
            str2 = f" {normr:10.3e} {normar:10.3e}"
            str3 = f"  {test1:8.1e} {test2:8.1e}"
            str4 = f" {normA:8.1e} {condA:8.1e}"
            print(f"{str1}{str2}{str3}{str4}")
        if istop > 0:
            break
    if show:
        print(" ")
        print("LSMR finished")
        print(msg[istop])
        print(f"istop ={istop:8g}    normr ={normr:8.1e}")
        print(f"    normA ={normA:8.1e}    normAr ={normar:8.1e}")
        print(f"itn   ={itn:8g}    condA ={condA:8.1e}")
        print(f"    normx ={normx:8.1e}")
        print(str1, str2)
        print(str3, str4)
    return (x, istop, itn, normr, normar, normA, condA, normx)
