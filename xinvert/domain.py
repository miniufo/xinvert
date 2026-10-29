"""Memory-efficient topology helpers for masked two-dimensional domains."""
from dataclasses import dataclass
from numbers import Integral, Real

import numba as nb
import numpy as np
import xarray as xr

OPEN_EAST = np.uint8(1)
OPEN_WEST = np.uint8(2)
OPEN_NORTH = np.uint8(4)
OPEN_SOUTH = np.uint8(8)
ACTIVE = np.uint8(16)

_DIRECTION_BITS = {
    "east": OPEN_EAST, "west": OPEN_WEST,
    "north": OPEN_NORTH, "south": OPEN_SOUTH,
}
_DIRECTIONS = tuple(_DIRECTION_BITS)
_EARTH_RADIUS = 6_371_200.0


@dataclass(frozen=True)
class DomainTopology:
    """Compact topology of a static two-dimensional active domain."""

    face_code: xr.DataArray
    component_labels: xr.DataArray
    component_stats: xr.Dataset
    n_components: int
    dims: tuple[str, str]
    periodic: tuple[bool, bool]
    cell_area: xr.DataArray | None = None

    @property
    def active(self):
        """Boolean active-cell mask decoded from ``face_code``."""
        result = ((self.face_code & ACTIVE) != 0).rename("active")
        result.attrs = {"long_name": "active domain cells"}
        return result

    def open_face(self, direction):
        """Decode one directional wet-to-wet face mask."""
        direction = _normalize_direction(direction)
        result = (
            (self.face_code & _DIRECTION_BITS[direction]) != 0
        ).rename(f"open_{direction}")
        result.attrs = {"long_name": f"active face toward {direction}"}
        return result

    @property
    def open_east(self):
        return self.open_face("east")

    @property
    def open_west(self):
        return self.open_face("west")

    @property
    def open_north(self):
        return self.open_face("north")

    @property
    def open_south(self):
        return self.open_face("south")

    @property
    def open_faces(self):
        """Materialize all four directional wet-to-wet masks."""
        return xr.Dataset({name: self.open_face(name) for name in _DIRECTIONS})

    @property
    def component_sizes(self):
        """Return active-cell counts keyed by one-based component label."""
        labels = self.component_stats.component.values
        counts = self.component_stats.cell_count.values
        return dict(zip(labels.tolist(), counts.tolist()))

    def labels(self, domain="active"):
        """Return labels for active cells or their complement."""
        domain = _normalize_plot_domain(domain)
        if domain == "active":
            return self.component_labels
        return _complement_components(self)[0]

    def statistics(self, domain="active"):
        """Return component counts and optional areas for either domain."""
        domain = _normalize_plot_domain(domain)
        if domain == "active":
            return self.component_stats
        return _complement_components(self)[1]

    def plot(self, domain="active", *, ax=None, cmap="tab20",
             add_colorbar=False, **kwargs):
        """Plot component labels using xarray's two-dimensional plot method."""
        domain = _normalize_plot_domain(domain)
        labels = self.labels(domain)
        shown = labels.where(labels != 0)
        shown.name = f"{domain}_component_labels"
        defaults = {"cmap": cmap, "add_colorbar": add_colorbar}
        defaults.update(kwargs)
        return shown.plot(ax=ax, **defaults)

    def plot_histogram(
            self, domain="land", *, metric=None, bins=30, log=True,
            ax=None, **kwargs):
        """Plot a histogram of component area or cell count."""
        import matplotlib.pyplot as plt

        domain = _normalize_plot_domain(domain)
        stats = self.statistics(domain)
        if metric is None:
            metric = "area_km2" if "area_km2" in stats else "cell_count"
        if metric not in stats:
            raise ValueError(
                f"metric {metric!r} is unavailable; choose one of "
                f"{tuple(stats.data_vars)}")
        values = np.asarray(stats[metric].values)
        values = values[np.isfinite(values) & (values > 0)]
        if ax is None:
            _, ax = plt.subplots()
        hist_bins = bins
        if log and values.size and isinstance(bins, Integral):
            lower, upper = values.min(), values.max()
            if lower < upper:
                hist_bins = np.geomspace(lower, upper, bins + 1)
                ax.set_xscale("log")
        defaults = {"edgecolor": "black", "linewidth": 0.5}
        defaults.update(kwargs)
        result = ax.hist(values, bins=hist_bins, **defaults)
        label = stats[metric].attrs.get("long_name", metric.replace("_", " "))
        units = stats[metric].attrs.get("units")
        ax.set_xlabel(f"{label} ({units})" if units else label)
        ax.set_ylabel("number of components")
        ax.set_title(f"{domain.capitalize()} component distribution")
        return result

    def filter_components(
            self, *, min_cells=None, min_area_km2=None, keep="all",
            keep_labels=None):
        """Return a new topology with selected components removed.

        Nothing is removed by default.  Selection rules are combined, and the
        original object is never modified.  Face codes and labels are rebuilt
        because a removed water body becomes inactive domain.
        """
        keep = str(keep).strip().lower()
        if keep not in ("all", "largest"):
            raise ValueError("keep must be 'all' or 'largest'")
        if min_cells is not None:
            if isinstance(min_cells, bool) or not isinstance(min_cells, Integral):
                raise TypeError("min_cells must be a positive integer")
            if min_cells < 1:
                raise ValueError("min_cells must be a positive integer")
        if min_area_km2 is not None:
            if isinstance(min_area_km2, bool) or not isinstance(min_area_km2, Real):
                raise TypeError("min_area_km2 must be a non-negative real scalar")
            if not np.isfinite(min_area_km2) or min_area_km2 < 0:
                raise ValueError("min_area_km2 must be a non-negative real scalar")
            if "area_km2" not in self.component_stats:
                raise ValueError(
                    "min_area_km2 requires analyze_domain(..., cell_area=...)")

        selected = np.ones(self.n_components + 1, dtype=np.bool_)
        selected[0] = False
        stats = self.component_stats
        if min_cells is not None:
            selected[1:] &= stats.cell_count.values >= min_cells
        if min_area_km2 is not None:
            selected[1:] &= stats.area_km2.values >= min_area_km2
        if keep == "largest" and np.any(selected):
            candidates = np.flatnonzero(selected)
            index = np.argmax(stats.cell_count.values[candidates - 1])
            selected[:] = False
            selected[candidates[index]] = True
        if keep_labels is not None:
            requested = np.asarray(tuple(keep_labels), dtype=np.int64)
            if requested.ndim != 1:
                raise ValueError("keep_labels must be a one-dimensional iterable")
            if np.any((requested < 1) | (requested > self.n_components)):
                raise ValueError("keep_labels contains an unknown component label")
            explicit = np.zeros_like(selected)
            explicit[requested] = True
            selected &= explicit

        if np.all(selected[1:]):
            return self
        active = _selected_active(
            np.asarray(self.component_labels.values), selected)
        return _build_topology(
            active, self.face_code, self.dims, self.periodic, self.cell_area)


def analyze_domain(
        data, dims=None, periodic=None, connectivity=4, valid=None,
        cell_area=None, radius=_EARTH_RADIUS):
    """Identify active cells, components, open faces, and component sizes.

    ``cell_area="spherical"`` derives memory-efficient latitude-band areas
    from uniform one-dimensional latitude and longitude coordinates.
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
    mask = _active_mask(data, valid).transpose(*dims)
    active = np.ascontiguousarray(mask.values, dtype=np.bool_)
    if active.size > np.iinfo(np.int32).max:
        raise ValueError(
            "domain exceeds the current int32 connected-component index limit")
    template = data.transpose(*dims)
    area = _normalize_cell_area(cell_area, template, dims, radius)
    return _build_topology(active, template, dims, periodic, area)


def spherical_cell_area(data, dims=None, radius=_EARTH_RADIUS):
    """Return one-dimensional latitude-band cell areas for a regular grid."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    dims = _normalize_dims(data, dims)
    return _spherical_cell_area(data.transpose(*dims), dims, radius)


def _build_topology(active, template, dims, periodic, cell_area):
    labels32, n_components = _label_components_numba(
        active, periodic[0], periodic[1])
    labels = labels32.astype(_smallest_label_dtype(n_components))
    del labels32
    codes = _encode_face_codes(active, periodic[0], periodic[1])
    coords = {dim: template.coords[dim] for dim in dims if dim in template.coords}
    face_code = xr.DataArray(
        codes, dims=dims, coords=coords, name="face_code",
        attrs={
            "long_name": "active-cell and open-face bit mask",
            "active_bit": int(ACTIVE), "east_bit": int(OPEN_EAST),
            "west_bit": int(OPEN_WEST), "north_bit": int(OPEN_NORTH),
            "south_bit": int(OPEN_SOUTH),
        },
    )
    label_da = xr.DataArray(
        labels, dims=dims, coords=coords, name="component_labels",
        attrs={"inactive_label": 0, "connectivity": 4},
    )
    stats = _component_statistics(labels, n_components, cell_area)
    return DomainTopology(
        face_code, label_da, stats, n_components, dims, periodic, cell_area)


def _component_statistics(labels, n_components, cell_area):
    counts = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
    components = np.arange(1, n_components + 1, dtype=np.int64)
    total = counts.sum()
    variables = {
        "cell_count": ("component", counts.astype(np.int64)),
        "cell_fraction": (
            "component", counts / total if total else counts.astype(float)),
    }
    if cell_area is not None:
        areas = _sum_component_area(labels, np.asarray(cell_area.values))
        area_total = areas.sum()
        variables.update({
            "area": ("component", areas),
            "area_km2": ("component", areas / 1.0e6),
            "area_fraction": (
                "component", areas / area_total if area_total else areas),
        })
    result = xr.Dataset(variables, coords={"component": components})
    result.cell_count.attrs = {"long_name": "active cells in component"}
    if "area" in result:
        result.area.attrs = {"long_name": "component area", "units": "m2"}
        result.area_km2.attrs = {"long_name": "component area", "units": "km2"}
    return result


def _complement_components(topology):
    active = np.ascontiguousarray(~np.asarray(topology.active.values))
    labels32, n_components = _label_components_numba(
        active, topology.periodic[0], topology.periodic[1])
    labels = labels32.astype(_smallest_label_dtype(n_components))
    coords = {
        dim: topology.face_code.coords[dim]
        for dim in topology.dims if dim in topology.face_code.coords
    }
    label_da = xr.DataArray(
        labels, dims=topology.dims, coords=coords,
        name="inactive_component_labels",
        attrs={"inactive_label": 0, "connectivity": 4},
    )
    return label_da, _component_statistics(
        labels, n_components, topology.cell_area)


def _normalize_plot_domain(domain):
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    domain = domain.strip().lower()
    aliases = {
        "active": "active", "water": "active", "ocean": "active",
        "inactive": "inactive", "land": "inactive",
    }
    if domain not in aliases:
        raise ValueError(
            "domain must be 'active'/'water'/'ocean' or 'inactive'/'land'")
    return aliases[domain]


def _normalize_cell_area(cell_area, template, dims, radius):
    if cell_area is None:
        return None
    if isinstance(cell_area, str):
        if cell_area.strip().lower() != "spherical":
            raise ValueError("cell_area string must be 'spherical'")
        return _spherical_cell_area(template, dims, radius)
    if isinstance(cell_area, Real) and not isinstance(cell_area, bool):
        if not np.isfinite(cell_area) or cell_area <= 0:
            raise ValueError("cell_area scalar must be positive and finite")
        return xr.DataArray(float(cell_area), name="cell_area")
    if not isinstance(cell_area, xr.DataArray):
        raise TypeError(
            "cell_area must be None, 'spherical', a scalar, or DataArray")
    if cell_area.ndim > 2 or not set(cell_area.dims).issubset(dims):
        raise ValueError("cell_area dimensions must be a subset of domain dims")
    for dim in cell_area.dims:
        if (dim in cell_area.coords and dim in template.coords
                and not cell_area.indexes[dim].equals(template.indexes[dim])):
            raise ValueError("cell_area coordinates must exactly match data")
    try:
        xr.broadcast(cell_area, template)
    except ValueError as error:
        raise ValueError("cell_area is not broadcast-compatible with data") from error
    values = np.asarray(cell_area.values)
    if np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError("cell_area values must be positive and finite")
    return cell_area


def _spherical_cell_area(template, dims, radius):
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a positive finite real scalar")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive finite real scalar")
    ydim, xdim = dims
    if ydim not in template.coords or xdim not in template.coords:
        raise ValueError("spherical cell areas require dimension coordinates")
    lat = np.asarray(template.coords[ydim].values, dtype=np.float64)
    lon = np.asarray(template.coords[xdim].values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("spherical cell areas require 1-D coordinates")
    dlat, dlon = np.diff(lat), np.diff(lon)
    dlat0, dlon0 = np.median(dlat), np.median(dlon)
    if not np.allclose(dlat, dlat0, rtol=2e-3, atol=1e-8):
        raise ValueError("latitude coordinates must be uniformly spaced")
    if not np.allclose(dlon, dlon0, rtol=2e-3, atol=1e-8):
        raise ValueError("longitude coordinates must be uniformly spaced")
    edges = np.empty(lat.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    edges[0] = max(-90.0, lat[0] - 0.5 * dlat0)
    edges[-1] = min(90.0, lat[-1] + 0.5 * dlat0)
    band = (
        float(radius) ** 2 * abs(np.deg2rad(dlon0))
        * (np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1])))
    )
    return xr.DataArray(
        band, dims=(ydim,), coords={ydim: template.coords[ydim]},
        name="cell_area", attrs={"units": "m2"})


def _normalize_direction(direction):
    if not isinstance(direction, str):
        raise TypeError("direction must be a string")
    direction = direction.strip().lower()
    if direction not in _DIRECTION_BITS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    return direction


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
        return bool(periodic), bool(periodic)
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


def _smallest_label_dtype(n_components):
    if n_components <= np.iinfo(np.uint8).max:
        return np.uint8
    if n_components <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _label_components(active, periodic):
    """Compatibility wrapper used by the masked solver.

    The solver requires signed ``int32`` labels for its compiled kernels,
    whereas :class:`DomainTopology` stores the smallest unsigned dtype.
    """
    active = np.asarray(active, dtype=np.bool_)
    if active.size > np.iinfo(np.int32).max:
        raise ValueError("domain contains too many cells for int32 indexing")
    labels, count = _label_components_numba(
        active, bool(periodic[0]), bool(periodic[1]))
    return labels, int(count)


@nb.njit(parallel=True, cache=True)
def _encode_face_codes(active, periodic_y, periodic_x):
    ny, nx = active.shape
    result = np.zeros((ny, nx), dtype=np.uint8)
    for y in nb.prange(ny):
        for x in range(nx):
            if not active[y, x]:
                continue
            code = ACTIVE
            east = x + 1 if x + 1 < nx else (0 if periodic_x else -1)
            west = x - 1 if x > 0 else (nx - 1 if periodic_x else -1)
            north = y - 1 if y > 0 else (ny - 1 if periodic_y else -1)
            south = y + 1 if y + 1 < ny else (0 if periodic_y else -1)
            if east >= 0 and active[y, east]:
                code |= OPEN_EAST
            if west >= 0 and active[y, west]:
                code |= OPEN_WEST
            if north >= 0 and active[north, x]:
                code |= OPEN_NORTH
            if south >= 0 and active[south, x]:
                code |= OPEN_SOUTH
            result[y, x] = code
    return result


@nb.njit(cache=True)
def _label_components_numba(active, periodic_y, periodic_x):
    ny, nx = active.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    queue = np.empty(ny * nx, dtype=np.int32)
    label = 0
    for y0 in range(ny):
        for x0 in range(nx):
            if not active[y0, x0] or labels[y0, x0] != 0:
                continue
            label += 1
            head, tail = 0, 1
            queue[0] = y0 * nx + x0
            labels[y0, x0] = label
            while head < tail:
                index = queue[head]
                head += 1
                y, x = index // nx, index % nx
                east = x + 1 if x + 1 < nx else (0 if periodic_x else -1)
                west = x - 1 if x > 0 else (nx - 1 if periodic_x else -1)
                north = y - 1 if y > 0 else (ny - 1 if periodic_y else -1)
                south = y + 1 if y + 1 < ny else (0 if periodic_y else -1)
                if east >= 0 and active[y, east] and labels[y, east] == 0:
                    labels[y, east] = label
                    queue[tail], tail = y * nx + east, tail + 1
                if west >= 0 and active[y, west] and labels[y, west] == 0:
                    labels[y, west] = label
                    queue[tail], tail = y * nx + west, tail + 1
                if north >= 0 and active[north, x] and labels[north, x] == 0:
                    labels[north, x] = label
                    queue[tail], tail = north * nx + x, tail + 1
                if south >= 0 and active[south, x] and labels[south, x] == 0:
                    labels[south, x] = label
                    queue[tail], tail = south * nx + x, tail + 1
    return labels, label


@nb.njit(cache=True)
def _selected_active(labels, selected):
    result = np.empty(labels.shape, dtype=np.bool_)
    flat_labels = labels.ravel()
    flat_result = result.ravel()
    for index in range(flat_labels.size):
        flat_result[index] = selected[flat_labels[index]]
    return result


def _sum_component_area(labels, area):
    n_components = int(labels.max())
    if area.ndim == 0:
        counts = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
        return counts * float(area)
    if area.ndim == 1 and area.size == labels.shape[0]:
        return _sum_row_area(labels, area, n_components)
    if area.ndim == 1 and area.size == labels.shape[1]:
        return _sum_column_area(labels, area, n_components)
    if area.shape == labels.shape:
        return np.bincount(
            labels.ravel(), weights=area.ravel(),
            minlength=n_components + 1)[1:]
    raise ValueError("cell_area shape is not compatible with component labels")


@nb.njit(cache=True)
def _sum_row_area(labels, row_area, n_components):
    result = np.zeros(n_components + 1, dtype=np.float64)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label:
                result[label] += row_area[y]
    return result[1:]


@nb.njit(cache=True)
def _sum_column_area(labels, column_area, n_components):
    result = np.zeros(n_components + 1, dtype=np.float64)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label:
                result[label] += column_area[x]
    return result[1:]
