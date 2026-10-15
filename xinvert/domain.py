"""Topology helpers for masked two-dimensional xarray domains.

This module is intentionally independent of xinvert's PDE solvers.  It turns
a static two-dimensional wet/active-cell mask into connected-component labels
and directional wet-to-wet face masks.  Internal boundary conditions are
derived from the resulting topology in :mod:`xinvert.internal_bcs`.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import xarray as xr
from numba import njit
from scipy.ndimage import label as label_connected

_DIRECTIONS = ("east", "west", "north", "south")


@dataclass(frozen=True)
class DomainTopology:
    """Discrete topology of a static two-dimensional active domain.

    All masks are cell-centred.  A directional ``open_*`` value is true at an
    active cell when the neighbouring cell in that direction is also active.
    Array-edge faces are not internal faces; periodic edges are joined before
    connected components and face masks are constructed.
    """

    active: xr.DataArray
    component_labels: xr.DataArray
    n_components: int
    open_east: xr.DataArray
    open_west: xr.DataArray
    open_north: xr.DataArray
    open_south: xr.DataArray
    dims: tuple[str, str]
    periodic: tuple[bool, bool]

    @property
    def open_faces(self) -> xr.Dataset:
        """Return the four directional wet-to-wet face masks."""
        return xr.Dataset({
            direction: getattr(self, f"open_{direction}")
            for direction in _DIRECTIONS
        })

    @property
    def component_sizes(self) -> dict[int, int]:
        """Return active-cell counts keyed by one-based component label."""
        labels = np.asarray(self.component_labels.values)
        counts = np.zeros(self.n_components + 1, dtype=np.int64)
        for start in range(0, labels.shape[0], 128):
            counts += np.bincount(labels[start:start + 128].ravel(),
                                  minlength=counts.size)
        return {label: int(counts[label])
                for label in range(1, self.n_components + 1)}


def analyze_domain(data, dims=None, periodic=None, connectivity=4, valid=None):
    """Identify active cells, components, and open faces in a 2-D domain.

    Parameters
    ----------
    data : xarray.DataArray
        A two-dimensional boolean mask or data field.  Boolean values are used
        directly.  For other dtypes, finite/non-null cells are active unless
        ``valid`` is supplied.
    dims : sequence of str, optional
        Ordered ``(y, x)`` dimensions.  Defaults to ``data.dims``.
    periodic : bool or sequence of bool/str, optional
        Periodic dimensions.  A scalar bool applies to both dimensions.  A
        sequence of booleans follows ``dims``; a sequence of names marks those
        dimensions periodic.  The common global-grid choice is ``(False,
        True)`` for ``(lat, lon)``.
    connectivity : int, default 4
        Cell connectivity.  Only four-neighbour connectivity is currently
        supported because it matches a five-point elliptic stencil.
    valid : xarray.DataArray, optional
        Explicit boolean active-cell mask with exactly the same dimensions,
        shape, and coordinates as ``data``.

    Returns
    -------
    DomainTopology
        One-based component labels use zero for inactive cells.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if data.ndim != 2:
        raise ValueError("domain analysis currently requires a 2-D DataArray")

    dims = _normalize_dims(data, dims)
    periodic = _normalize_periodic(dims, periodic)
    if isinstance(connectivity, bool) or not isinstance(connectivity, Integral):
        raise TypeError("connectivity must be the integer 4")
    if connectivity != 4:
        raise ValueError("only 4-cell connectivity is currently supported")

    active = _active_mask(data, valid).transpose(*dims)
    values = np.asarray(active.values, dtype=bool)
    labels, n_components = _label_components(values, periodic)

    label_da = xr.DataArray(
        labels,
        dims=dims,
        coords={dim: active.coords[dim] for dim in dims if dim in active.coords},
        name="component_labels",
        attrs={
            "long_name": "one-based connected-component labels",
            "inactive_label": 0,
            "connectivity": connectivity,
        },
    )
    active = active.astype(bool).rename("active")
    active.attrs = {
        "long_name": "active domain cells",
        "source": "boolean values" if data.dtype.kind == "b" and valid is None
        else "non-null data values" if valid is None else "explicit valid mask",
    }

    open_faces = {}
    for direction in _DIRECTIONS:
        neighbour = _neighbour_values(values, direction, periodic)
        open_faces[direction] = xr.DataArray(
            values & neighbour,
            dims=dims,
            coords={dim: active.coords[dim] for dim in dims if dim in active.coords},
            name=f"open_{direction}",
            attrs={"long_name": f"active face toward {direction}"},
        )

    return DomainTopology(
        active=active,
        component_labels=label_da,
        n_components=n_components,
        open_east=open_faces["east"],
        open_west=open_faces["west"],
        open_north=open_faces["north"],
        open_south=open_faces["south"],
        dims=dims,
        periodic=periodic,
    )


def _normalize_dims(data, dims):
    if dims is None:
        return tuple(data.dims)
    if isinstance(dims, str):
        raise TypeError("dims must contain the ordered (y, x) dimension names")
    dims = tuple(dims)
    if len(dims) != 2:
        raise ValueError("dims must contain exactly two dimension names")
    if any(not isinstance(dim, str) for dim in dims):
        raise TypeError("each domain dimension must be a string")
    if len(set(dims)) != 2:
        raise ValueError("domain dimensions must not contain duplicates")
    if set(dims) != set(data.dims):
        raise ValueError("dims must name exactly the two DataArray dimensions")
    return dims


def _normalize_periodic(dims, periodic):
    if periodic is None:
        return False, False
    if isinstance(periodic, (bool, np.bool_)):
        value = bool(periodic)
        return value, value
    if isinstance(periodic, str):
        periodic = (periodic,)
    else:
        periodic = tuple(periodic)
    if all(isinstance(item, str) for item in periodic):
        unknown = set(periodic) - set(dims)
        if unknown:
            raise ValueError(f"periodic contains unknown dimensions: {sorted(unknown)}")
        return tuple(dim in periodic for dim in dims)
    if len(periodic) != 2 or not all(
            isinstance(item, (bool, np.bool_)) for item in periodic):
        raise ValueError(
            "periodic must be a bool, dimension name(s), or two booleans")
    return tuple(bool(item) for item in periodic)


def _active_mask(data, valid):
    if valid is None:
        return data if data.dtype.kind == "b" else data.notnull()
    if not isinstance(valid, xr.DataArray):
        raise TypeError("valid must be an xarray.DataArray")
    if valid.dtype.kind != "b":
        raise TypeError("valid must have boolean dtype")
    if valid.dims != data.dims or valid.shape != data.shape:
        raise ValueError("valid must have the same dimensions and shape as data")
    for dim in data.dims:
        if dim in data.coords or dim in valid.coords:
            if dim not in data.coords or dim not in valid.coords:
                raise ValueError("valid coordinates must exactly match data")
            if not data.indexes[dim].equals(valid.indexes[dim]):
                raise ValueError("valid coordinates must exactly match data")
    return valid


def _label_components(active, periodic):
    """Label a boolean 2-D mask; preserve row-major numbering after wrapping.

    Parameters are a NumPy mask and ordered (y, x) periodic flags. Returns
    int32 labels (zero inactive) and the number of components. SciPy labels
    the non-periodic grid; Numba joins seam components and relabels in place.
    """
    labels, count = label_connected(active, output=np.int32)
    if count and any(periodic) and all(active.shape):
        count = _join_periodic_labels(labels, count, periodic)
    return labels, int(count)


@njit(cache=True)
def _join_periodic_labels(labels, count, periodic):
    """Merge int32 labels across periodic seams; return the dense count.

    Mutates labels in place. Union roots are the smallest original label,
    retaining the first-cell ordering of a row-major flood fill.
    """
    parent = np.arange(count + 1, dtype=np.int32)
    ny, nx = labels.shape
    for axis in range(2):
        if not periodic[axis]:
            continue
        for index in range(nx if axis == 0 else ny):
            a = labels[0, index] if axis == 0 else labels[index, 0]
            b = labels[-1, index] if axis == 0 else labels[index, -1]
            if a == 0 or b == 0:
                continue
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            parent[max(a, b)] = min(a, b)
    lookup = np.zeros(count + 1, dtype=np.int32)
    total = 0
    for label in range(1, count + 1):
        root = label
        while parent[root] != root:
            root = parent[root]
        if root == label:
            total += 1
        lookup[label] = total if root == label else lookup[root]
    for y in range(ny):
        for x in range(nx):
            labels[y, x] = lookup[labels[y, x]]
    return total


def _neighbour_indices(y, x, ny, nx, periodic):
    for dy, dx in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        yy, xx = y + dy, x + dx
        if periodic[0]:
            yy %= ny
        if periodic[1]:
            xx %= nx
        if 0 <= yy < ny and 0 <= xx < nx:
            yield yy, xx


def _neighbour_values(active, direction, periodic):
    axis = 1 if direction in ("east", "west") else 0
    step = -1 if direction in ("east", "south") else 1
    result = np.roll(active, step, axis=axis)
    if not periodic[axis] and active.shape[axis]:
        edge = [slice(None), slice(None)]
        edge[axis] = -1 if direction in ("east", "south") else 0
        result[tuple(edge)] = False
    return result
