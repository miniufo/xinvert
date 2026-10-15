"""Identify internal boundary faces from :mod:`xinvert.domain` topology."""
from dataclasses import dataclass
from numbers import Real

import numpy as np
import xarray as xr
from numba import njit

from .domain import DomainTopology

_DIRECTIONS = ("east", "west", "north", "south")


@dataclass(frozen=True)
class InternalBoundaryCondition:
    """A fixed or homogeneous Neumann condition on directional internal faces.

    The directional masks are true at active cells whose neighbour in that
    direction is inactive.  Faces at non-periodic array edges are deliberately
    excluded because they belong to xinvert's existing external-BC mechanism.
    ``face_code`` is the packed uint8 array; ``topology`` retains the original
    domain metadata by reference. Directional properties allocate boolean
    views on demand. ``value`` records the prescribed boundary value as
    metadata; the public inversion API instead uses its iParams setting.
    """

    kind: str
    value: float
    face_code: xr.DataArray
    topology: DomainTopology

    def _face(self, direction: str) -> xr.DataArray:
        """Decode one direction into a newly allocated boolean DataArray."""
        bit = 1 << _DIRECTIONS.index(direction)
        face = ((self.face_code & bit) != 0).rename(direction)
        face.attrs = {"long_name": f"internal boundary face toward {direction}",
                      "boundary_condition": self.kind}
        return face

    @property
    def east(self) -> xr.DataArray:
        """Boolean east-facing internal boundary mask (decoded on demand)."""
        return self._face("east")

    @property
    def west(self) -> xr.DataArray:
        """Boolean west-facing internal boundary mask (decoded on demand)."""
        return self._face("west")

    @property
    def north(self) -> xr.DataArray:
        """Boolean north-facing internal boundary mask (decoded on demand)."""
        return self._face("north")

    @property
    def south(self) -> xr.DataArray:
        """Boolean south-facing internal boundary mask (decoded on demand)."""
        return self._face("south")

    @property
    def faces(self) -> xr.Dataset:
        """Return directional internal-boundary masks as a Dataset."""
        return xr.Dataset({
            direction: getattr(self, direction) for direction in _DIRECTIONS
        })

    @property
    def n_directed_faces(self) -> int:
        """Number of active-cell sides adjacent to inactive cells."""
        return int(_count_faces(self.face_code.values))


def identify_internal_bcs(topology, kind="neumann", value=0.0):
    """Build internal boundary masks for a previously analysed domain.

    Parameters
    ----------
    topology : DomainTopology
        Static domain; retained by reference for coordinate/periodic validation.
    kind : {'neumann', 'fixed'}, default 'neumann'
        Homogeneous zero-flux faces, or fixed inactive-neighbour values.
    value : float, default 0.0
        Neumann must be zero. Fixed records a finite prescribed scalar value.

    Returns
    -------
    InternalBoundaryCondition
        Packed directional faces and original topology for inspection or
        preprocessing. Public Poisson inversion infers its own topology from
        forcing via iParams['internal_bcs'], rather than accepting this object.
    """
    if not isinstance(topology, DomainTopology):
        raise TypeError("topology must be a DomainTopology")
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    kind = kind.strip().lower()
    if kind not in ("neumann", "fixed"):
        raise NotImplementedError(
            "only fixed and homogeneous Neumann internal boundaries are supported")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("value must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("value must be a finite real scalar")
    if kind == "neumann" and value != 0.0:
        raise NotImplementedError(
            "only homogeneous Neumann internal boundaries are supported")

    active = topology.active
    codes = _encode_internal_faces(np.asarray(active.values, dtype=bool),
                                   topology.periodic)
    face_code = xr.DataArray(codes, dims=active.dims, coords=active.coords,
                            name="internal_face_code", attrs={
                                "long_name": "packed internal boundary faces",
                                "flag_masks": [1, 2, 4, 8, 16],
                                "flag_meanings": "east west north south active",
                            })

    return InternalBoundaryCondition(
        kind=kind,
        value=value,
        face_code=face_code,
        topology=topology,
    )


@njit(cache=True)
def _encode_internal_faces(active, periodic):
    """Encode a boolean (y, x) mask using uint8 internal-face/active bits.

    periodic is a (y, x) boolean pair. Non-periodic exterior faces are
    excluded; inactive cells are zero. Returns one byte per grid cell.
    """
    ny, nx = active.shape
    result = np.zeros(active.shape, dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            if not active[y, x]:
                continue
            code = 16
            if (x + 1 < nx or periodic[1]) and not active[y, (x + 1) % nx]:
                code |= 1
            if (x > 0 or periodic[1]) and not active[y, (x - 1) % nx]:
                code |= 2
            if (y > 0 or periodic[0]) and not active[(y - 1) % ny, x]:
                code |= 4
            if (y + 1 < ny or periodic[0]) and not active[(y + 1) % ny, x]:
                code |= 8
            result[y, x] = code
    return result


@njit(cache=True)
def _count_faces(codes):
    """Return the number of low-four-bit faces without allocating masks."""
    total = 0
    for y in range(codes.shape[0]):
        for x in range(codes.shape[1]):
            code = codes[y, x]
            for bit in (1, 2, 4, 8):
                total += (code & bit) != 0
    return total


def _has_neighbour_mask(topology, direction):
    shape = topology.active.shape
    values = np.ones(shape, dtype=bool)
    axis = 1 if direction in ("east", "west") else 0
    if not topology.periodic[axis]:
        edge = [slice(None), slice(None)]
        edge[axis] = -1 if direction in ("east", "south") else 0
        values[tuple(edge)] = False
    return xr.DataArray(
        values,
        dims=topology.dims,
        coords={
            dim: topology.active.coords[dim]
            for dim in topology.dims if dim in topology.active.coords
        },
    )
