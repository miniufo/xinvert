"""Independent tests for masked-domain topology analysis."""
import numpy as np
import pytest
import xarray as xr

from xinvert.domain import analyze_domain


def _array(values):
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": np.arange(len(values)), "lon": np.arange(len(values[0]))},
    )


def test_boolean_mask_components_and_sizes():
    mask = _array([
        [True, True, False, False],
        [True, False, False, True],
        [False, False, True, True],
    ])

    topology = analyze_domain(mask)

    assert topology.n_components == 2
    assert topology.component_sizes == {1: 3, 2: 3}
    np.testing.assert_array_equal(topology.component_labels.where(mask, 0), [
        [1, 1, 0, 0],
        [1, 0, 0, 2],
        [0, 0, 2, 2],
    ])
    assert topology.component_labels.dtype == np.int32
    assert topology.active.dims == ("lat", "lon")
    np.testing.assert_array_equal(topology.active.lon, mask.lon)


def test_periodic_longitude_connects_across_seam():
    mask = _array([
        [True, False, False, True],
        [False, False, False, False],
    ])

    closed = analyze_domain(mask)
    periodic = analyze_domain(mask, periodic="lon")

    assert closed.n_components == 2
    assert periodic.n_components == 1
    assert bool(periodic.open_west[0, 0])
    assert bool(periodic.open_east[0, -1])


def test_nonboolean_data_use_nonnull_cells_by_default():
    data = _array([[1.0, np.nan], [0.0, -2.0]])
    topology = analyze_domain(data)

    np.testing.assert_array_equal(
        topology.active, [[True, False], [True, True]])
    assert topology.n_components == 1


def test_explicit_valid_mask_overrides_data_values():
    data = _array([[np.nan, np.nan], [np.nan, np.nan]])
    valid = _array([[True, False], [False, True]])

    topology = analyze_domain(data, valid=valid)

    assert topology.n_components == 2
    np.testing.assert_array_equal(topology.active, valid)


def test_open_faces_are_reciprocal():
    mask = _array([
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ])
    topology = analyze_domain(mask)

    np.testing.assert_array_equal(
        topology.open_east[:, :-1], topology.open_west[:, 1:])
    np.testing.assert_array_equal(
        topology.open_south[:-1, :], topology.open_north[1:, :])


@pytest.mark.parametrize(
    "kwargs,error,message",
    [
        ({"dims": "lat"}, TypeError, "ordered"),
        ({"dims": ("lat",)}, ValueError, "exactly two"),
        ({"periodic": "time"}, ValueError, "unknown"),
        ({"periodic": (True,)}, ValueError, "two booleans"),
        ({"connectivity": 8}, ValueError, "only 4-cell"),
    ],
)
def test_invalid_options_fail_clearly(kwargs, error, message):
    with pytest.raises(error, match=message):
        analyze_domain(_array([[True, True], [True, True]]), **kwargs)


def test_rejects_non_dataarray_and_non_2d_data():
    with pytest.raises(TypeError, match="xarray.DataArray"):
        analyze_domain(np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="2-D"):
        analyze_domain(xr.DataArray(np.ones((2, 2, 2), dtype=bool)))
