"""Independent tests for internal-boundary identification."""
import numpy as np
import pytest
import xarray as xr

from xinvert.domain import analyze_domain
from xinvert.internal_bcs import identify_internal_bcs


def _topology(values, periodic=None):
    mask = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.arange(len(values)), "x": np.arange(len(values[0]))},
    )
    return analyze_domain(mask, periodic=periodic)


def test_single_land_cell_produces_four_directional_faces():
    topology = _topology([
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ])

    bc = identify_internal_bcs(topology)

    assert bc.kind == "neumann"
    assert bc.value == 0.0
    assert bc.n_directed_faces == 4
    assert bool(bc.east[1, 0])
    assert bool(bc.west[1, 2])
    assert bool(bc.south[0, 1])
    assert bool(bc.north[2, 1])
    assert not bool(bc.east[1, 2])


def test_nonperiodic_array_edges_are_not_internal_boundaries():
    bc = identify_internal_bcs(_topology([
        [True, True],
        [True, True],
    ]))

    assert bc.n_directed_faces == 0


def test_periodic_seam_can_contain_internal_boundary():
    bc = identify_internal_bcs(_topology([
        [True, True, True, False],
        [True, True, True, True],
    ], periodic="x"))

    assert bool(bc.west[0, 0])
    assert bool(bc.east[0, 2])


def test_face_dataset_preserves_coordinates_and_boolean_dtype():
    topology = _topology([[True, False], [True, True]])
    faces = identify_internal_bcs(topology).faces

    assert set(faces.data_vars) == {"east", "west", "north", "south"}
    assert all(faces[name].dtype == bool for name in faces.data_vars)
    np.testing.assert_array_equal(faces.x, topology.active.x)
    np.testing.assert_array_equal(faces.y, topology.active.y)


@pytest.mark.parametrize("kind", ["dirichlet", "robin"])
def test_unimplemented_internal_boundary_kinds_are_explicit(kind):
    with pytest.raises(NotImplementedError, match="homogeneous Neumann"):
        identify_internal_bcs(_topology([[True]]), kind=kind)


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_nonzero_neumann_values_are_not_silently_accepted(value):
    with pytest.raises(NotImplementedError, match="homogeneous Neumann"):
        identify_internal_bcs(_topology([[True]]), value=value)


def test_invalid_topology_and_value_fail_clearly():
    with pytest.raises(TypeError, match="DomainTopology"):
        identify_internal_bcs(xr.DataArray([[True]]))
    with pytest.raises(TypeError, match="finite real scalar"):
        identify_internal_bcs(_topology([[True]]), value=True)
    with pytest.raises(ValueError, match="finite real scalar"):
        identify_internal_bcs(_topology([[True]]), value=np.nan)
