"""Identify internal boundary faces from compact domain topology."""
from dataclasses import dataclass
from numbers import Real

import numba as nb
import numpy as np
import xarray as xr

from .domain import (
    ACTIVE,
    OPEN_EAST,
    OPEN_NORTH,
    OPEN_SOUTH,
    OPEN_WEST,
    DomainTopology,
)

_DIRECTIONS = ("east", "west", "north", "south")
_OPEN_BITS = {
    "east": OPEN_EAST, "west": OPEN_WEST,
    "north": OPEN_NORTH, "south": OPEN_SOUTH,
}


@dataclass(frozen=True)
class InternalBoundaryCondition:
    """A homogeneous Neumann condition derived from compact face codes."""

    topology: DomainTopology
    kind: str
    value: float
    n_directed_faces: int

    @property
    def face_code(self):
        """Packed internal-face mask, materialized only when requested."""
        topology = self.topology
        codes = _encode_internal_face_codes(
            np.asarray(topology.face_code.values),
            topology.periodic[0], topology.periodic[1])
        return xr.DataArray(
            codes, dims=topology.dims, coords=topology.face_code.coords,
            name="internal_face_code", attrs={
                "long_name": "packed internal boundary faces",
                "flag_masks": [1, 2, 4, 8, 16],
                "flag_meanings": "east west north south active",
            })

    def boundary_face(self, direction):
        """Decode one active-to-inactive internal boundary mask."""
        if not isinstance(direction, str):
            raise TypeError("direction must be a string")
        direction = direction.strip().lower()
        if direction not in _OPEN_BITS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")
        codes = self.topology.face_code
        result = ((codes & ACTIVE) != 0) & (
            (codes & _OPEN_BITS[direction]) == 0)
        result &= _has_neighbour_mask(self.topology, direction)
        result = result.rename(direction)
        result.attrs = {
            "long_name": f"internal boundary face toward {direction}",
            "boundary_condition": self.kind,
        }
        return result

    @property
    def east(self):
        return self.boundary_face("east")

    @property
    def west(self):
        return self.boundary_face("west")

    @property
    def north(self):
        return self.boundary_face("north")

    @property
    def south(self):
        return self.boundary_face("south")

    @property
    def faces(self):
        """Materialize all four directional internal-boundary masks."""
        return xr.Dataset({name: self.boundary_face(name) for name in _DIRECTIONS})


def identify_internal_bcs(topology, kind="neumann", value=0.0):
    """Describe homogeneous-Neumann faces for an analysed domain."""
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
    count = _count_internal_faces(
        np.asarray(topology.face_code.values),
        topology.periodic[0], topology.periodic[1])
    return InternalBoundaryCondition(topology, kind, value, int(count))


def _has_neighbour_mask(topology, direction):
    values = np.ones(topology.face_code.shape, dtype=bool)
    axis = 1 if direction in ("east", "west") else 0
    if not topology.periodic[axis]:
        edge = [slice(None), slice(None)]
        edge[axis] = -1 if direction in ("east", "south") else 0
        if values.shape[axis]:
            values[tuple(edge)] = False
    return xr.DataArray(
        values, dims=topology.dims,
        coords={dim: topology.face_code.coords[dim] for dim in topology.dims})


@nb.njit(cache=True)
def _count_internal_faces(codes, periodic_y, periodic_x):
    ny, nx = codes.shape
    count = 0
    for y in range(ny):
        for x in range(nx):
            code = codes[y, x]
            if not code & ACTIVE:
                continue
            if (x + 1 < nx or periodic_x) and not code & OPEN_EAST:
                count += 1
            if (x > 0 or periodic_x) and not code & OPEN_WEST:
                count += 1
            if (y > 0 or periodic_y) and not code & OPEN_NORTH:
                count += 1
            if (y + 1 < ny or periodic_y) and not code & OPEN_SOUTH:
                count += 1
    return count


@nb.njit(cache=True)
def _encode_internal_face_codes(codes, periodic_y, periodic_x):
    """Encode blocked internal faces while excluding exterior array edges."""
    ny, nx = codes.shape
    result = np.zeros((ny, nx), dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            code = codes[y, x]
            if not code & ACTIVE:
                continue
            blocked = ACTIVE
            if (x + 1 < nx or periodic_x) and not code & OPEN_EAST:
                blocked |= OPEN_EAST
            if (x > 0 or periodic_x) and not code & OPEN_WEST:
                blocked |= OPEN_WEST
            if (y > 0 or periodic_y) and not code & OPEN_NORTH:
                blocked |= OPEN_NORTH
            if (y + 1 < ny or periodic_y) and not code & OPEN_SOUTH:
                blocked |= OPEN_SOUTH
            result[y, x] = blocked
    return result


@nb.njit(cache=True)
def _encode_internal_faces(active, periodic):
    """Compatibility helper used by the masked solver."""
    ny, nx = active.shape
    result = np.zeros(active.shape, dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            if not active[y, x]:
                continue
            code = ACTIVE
            if (x + 1 < nx or periodic[1]) and not active[y, (x + 1) % nx]:
                code |= OPEN_EAST
            if (x > 0 or periodic[1]) and not active[y, (x - 1) % nx]:
                code |= OPEN_WEST
            if (y > 0 or periodic[0]) and not active[(y - 1) % ny, x]:
                code |= OPEN_NORTH
            if (y + 1 < ny or periodic[0]) and not active[(y + 1) % ny, x]:
                code |= OPEN_SOUTH
            result[y, x] = code
    return result
