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
    assert topology.component_labels.dtype == np.uint8
    assert topology.face_code.dtype == np.uint8
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


def test_component_statistics_include_count_and_spherical_area():
    mask = xr.DataArray(
        [[True, True, False], [False, False, True]],
        dims=("lat", "lon"),
        coords={"lat": [-0.5, 0.5], "lon": [0.5, 1.5, 2.5]},
    )
    topology = analyze_domain(mask, cell_area="spherical")

    assert topology.n_components == 2
    np.testing.assert_array_equal(topology.component_stats.cell_count, [2, 1])
    assert "area_km2" in topology.component_stats
    assert float(topology.component_stats.area.sum()) > 0.0
    np.testing.assert_allclose(topology.component_stats.area_fraction.sum(), 1.0)


def test_filter_components_is_explicit_and_rebuilds_faces():
    mask = _array([
        [True, True, False, False],
        [True, False, False, True],
        [False, False, True, True],
    ])
    topology = analyze_domain(mask, cell_area=2.0)

    unchanged = topology.filter_components()
    filtered = topology.filter_components(min_cells=3, keep_labels=[2])

    assert unchanged is topology
    assert filtered.n_components == 1
    assert filtered.component_sizes == {1: 3}
    assert float(filtered.component_stats.area_km2[0]) == pytest.approx(6e-6)
    np.testing.assert_array_equal(filtered.active, [
        [False, False, False, False],
        [False, False, False, True],
        [False, False, True, True],
    ])


def test_filter_largest_and_area_threshold():
    mask = _array([
        [True, True, False, True],
        [True, True, False, False],
    ])
    topology = analyze_domain(mask, cell_area=1.0e6)

    assert topology.filter_components(keep="largest").component_sizes == {1: 4}
    assert topology.filter_components(min_area_km2=2).component_sizes == {1: 4}


def test_invalid_filter_options_fail_clearly():
    topology = analyze_domain(_array([[True, False], [False, True]]))
    with pytest.raises(ValueError, match="requires"):
        topology.filter_components(min_area_km2=1)
    with pytest.raises(ValueError, match="unknown"):
        topology.filter_components(keep_labels=[3])


def test_land_alias_labels_and_statistics_are_computed_on_demand():
    mask = _array([
        [True, False, True, False],
        [True, False, True, False],
    ])
    topology = analyze_domain(mask, periodic="lon", cell_area=1.0e6)

    labels = topology.labels("land")
    stats = topology.statistics("inactive")

    np.testing.assert_array_equal(labels, [
        [0, 1, 0, 2],
        [0, 1, 0, 2],
    ])
    np.testing.assert_array_equal(stats.cell_count, [2, 2])
    np.testing.assert_allclose(stats.area_km2, [2, 2])
    assert topology.labels("water") is topology.component_labels
    assert topology.statistics("ocean") is topology.component_stats


def test_plot_methods_return_xarray_artist_and_histogram():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topology = analyze_domain(
        _array([[True, False, False], [True, True, False]]),
        cell_area=1.0e6,
    )
    figure, axes = plt.subplots(1, 2)
    artist = topology.plot(domain="land", ax=axes[0])
    counts, edges, patches = topology.plot_histogram(
        domain="land", ax=axes[1], bins=4)

    assert artist.axes is axes[0]
    assert counts.sum() == 1
    assert len(edges) == 5
    assert len(patches) == 4
    assert axes[1].get_xscale() == "linear"
    plt.close(figure)


@pytest.mark.parametrize("domain", [None, "atmosphere"])
def test_invalid_plot_domain_fails_clearly(domain):
    topology = analyze_domain(_array([[True]]))
    error = TypeError if domain is None else ValueError
    with pytest.raises(error, match="domain"):
        topology.labels(domain)
