"""Public naming, parameter, and diagnostic contracts."""
import numpy as np
import pytest
import xarray as xr

from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose, reconstruct, streamfunction_potential

from .test_helmholtz_output_grids import faces, inputs

ERRORS = {"vel_error", "div_error", "converged", "iterations",
          "error", "final_error", "stop_reason"}
FIELDS = {"urot", "vrot", "udiv", "vdiv"}


@pytest.mark.parametrize("method", ["sor", "li2006"])
@pytest.mark.parametrize("grid", ["arakawa-a", "Arakawa-A", "ARAKAWA-A"])
def test_common_output_and_errors(method, grid):
    u, v, wet = inputs(False)
    opts = {"mxLoop": 6000, "tolerance": 1e-10, "printInfo": False,
            "BCs": ["fixed", "fixed"] if method == "li2006" else ["fixed", "fixed"]}
    result, d = decompose(u, v, ["y", "x"], coords="cartesian", grid=grid,
                          method=method, mask=wet, iParams=opts, return_diagnostics=True)
    assert set(result) == FIELDS
    assert ERRORS.issubset(d)
    assert result.attrs["method"] == method
    assert result.attrs["grid_layout"] == "arakawa-a"
    assert bool(d.converged)
    assert d.converged.dtype == bool
    vel = reconstruct(result, components=True)
    assert set(vel) == FIELDS | {"u", "v"}
    delta = ((vel.u-u)**2+(vel.v-v)**2).where(wet)
    reference = (u*u+v*v).where(wet)
    np.testing.assert_allclose(d.vel_error, np.sqrt(delta.sum()/reference.sum()))
    # Independent uniform Cartesian central differences on complete wet stencils.
    a, b, mask = result.urot.values, result.vrot.values, wet.values
    samples = []
    for j in range(1, a.shape[0]-1):
        for i in range(1, a.shape[1]-1):
            if all(mask[y, x] for y, x in [(j,i),(j-1,i),(j+1,i),(j,i-1),(j,i+1)]):
                samples.append((a[j,i+1]-a[j,i-1])/2+(b[j+1,i]-b[j-1,i])/2)
    np.testing.assert_allclose(d.div_error, np.sqrt(np.mean(np.square(samples))))
    assert d.vel_error.attrs["units"] == "1"
    assert d.div_error.attrs["units"] == "s-1"


@pytest.mark.parametrize("grid", ["Arakawa-C", "ARAKAWA-C", "arakawa-c"])
def test_c_grid_labels_and_diagnostics(grid):
    u, v, wet = inputs(True)
    u, v, _, _ = faces(u, v, wet, True)
    result, d = decompose(u, v, ["y","x"], coords="cartesian", method="sor",
                          grid=grid, mask=wet, iParams={"BCs":["fixed","periodic"],
                          "printInfo":False, "mxLoop":6000, "tolerance":1e-10},
                          return_diagnostics=True)
    assert set(result) == FIELDS
    assert ERRORS.issubset(d)
    assert result.urot.dims == ("y","x_face")
    assert result.vrot.dims == ("y_face","x")
    assert float(d.vel_error) < 1e-14
    assert float(d.div_error) < 1e-8


@pytest.mark.parametrize("kw", [
    {"method": "poisson"}, {"method": "forget"}, {"grid": "collocated"}, {"grid": "face"},
    {"mParams":{"regularization":1e-8}}, {"mParams":{"rho":1e-8}},
    {"grid":"arakawa-c"},
])
def test_removed_options_are_not_silent_aliases(kw):
    u, v, wet = inputs(False)
    with pytest.raises(ValueError):
        decompose(u, v, ["y","x"], coords="cartesian", mask=wet, **kw)


def test_alpha_preserves_default_objective_and_has_expected_effect():
    u, v, wet = inputs(False)
    opts = {"BCs":["fixed","fixed"],"mxLoop":6000,"tolerance":1e-12,
            "on_nonconvergence":"raise"}
    prepared = li._prepare(u,v,["y","x"],"cartesian",wet,None,opts)
    length, mp = prepared[6:8]
    assert mp["alpha"] == 1e-8/length**2
    low = li.helmholtz(u,v,["y","x"],"cartesian",mask=wet,
                      mParams={"alpha":1e-5},iParams=opts,return_diagnostics=True)
    high = li.helmholtz(u,v,["y","x"],"cartesian",mask=wet,
                       mParams={"alpha":1.0},iParams=opts,return_diagnostics=True)
    assert float(high[-1].objective_data) > float(low[-1].objective_data)
    lo_norm = sum(float((p*p).sum()) for p in low[:2])
    hi_norm = sum(float((p*p).sum()) for p in high[:2])
    assert hi_norm < lo_norm
    default = streamfunction_potential(u,v,["y","x"],"cartesian",mask=wet,iParams=opts)
    explicit = streamfunction_potential(u,v,["y","x"],"cartesian",mask=wet,iParams=opts,
                                       mParams={"alpha":mp["alpha"]})
    xr.testing.assert_identical(default, explicit)
    assert "alpha" in default.attrs and "rho" not in default.attrs


@pytest.mark.parametrize("method", ["sor", "li2006"])
def test_zero_and_batch_errors(method):
    u,v,wet = inputs(False)
    opts = {"BCs":["fixed","fixed"] if method=="li2006" else ["fixed","fixed"],
            "printInfo":False,"mxLoop":6000,"tolerance":1e-10}
    _,d = decompose(u*0,v*0,["y","x"],"cartesian",method=method,
                         mask=wet,iParams=opts,return_diagnostics=True)
    assert float(d.vel_error) == float(d.div_error) == 0
    if method=="sor":
        return
    uu=xr.concat([u,u*.5],dim="time")
    vv=xr.concat([v,v*.5],dim="time")
    _,d=decompose(uu,vv,["y","x"],"cartesian",method=method,
                  mask=wet,iParams=opts,return_diagnostics=True)
    assert all(d[key].dims == ("time",) for key in
               ("vel_error", "div_error", "converged", "iterations", "stop_reason"))
    np.testing.assert_allclose(d.vel_error[0],d.vel_error[1],rtol=1e-10)
    np.testing.assert_allclose(d.div_error[0],2*d.div_error[1],rtol=1e-8)


@pytest.mark.parametrize("periodic", [False, True])
def test_sor_a_grid_infers_joint_velocity_mask(architect, periodic):
    u, v, wet = inputs(periodic)
    u, v = u.where(wet), v.where(wet)
    # A missing value in either component excludes the cell.
    u[2, 3] = np.nan
    v[6, 5] = np.inf
    joint = np.isfinite(u) & np.isfinite(v)
    saved_u, saved_v = u.copy(deep=True), v.copy(deep=True)
    opts = {"architect": architect, "BCs": ["fixed", "periodic" if periodic else "fixed"],
            "mxLoop": 6000, "tolerance": 1e-10, "printInfo": False}
    kwargs = {"dims": ["y", "x"], "coords": "cartesian", "method": "sor",
              "grid": "Arakawa-A", "iParams": opts,
              "return_potentials": True, "return_diagnostics": True}
    inferred, diagnostics = decompose(u, v, **kwargs)
    explicit, reference = decompose(u, v, mask=joint, **kwargs)
    xr.testing.assert_allclose(inferred, explicit, rtol=1e-11, atol=1e-11)
    xr.testing.assert_allclose(diagnostics, reference, rtol=1e-11, atol=1e-11)
    for name in FIELDS:
        np.testing.assert_array_equal(np.isfinite(inferred[name]), joint)
    xr.testing.assert_identical(u, saved_u)
    xr.testing.assert_identical(v, saved_v)


def test_sor_c_grid_requires_cell_center_mask():
    u, v, wet = inputs(False)
    u, v, _, _ = faces(u, v, wet, False)
    with pytest.raises(ValueError, match="requires an explicit boolean cell-center mask"):
        decompose(u, v, ["y", "x"], coords="cartesian", method="sor", grid="Arakawa-C")


def test_sor_scalar_output_and_warm_start(architect):
    u, v, wet = inputs(False)
    opts = {"architect": architect, "mxLoop": 6000, "tolerance": 1e-10,
            "printInfo": False, "diag_interval": 13}
    components, diag = decompose(
        u, v, ["y", "x"], "cartesian", method="SoR", mask=wet,
        iParams=opts, return_potentials=True, return_diagnostics=True)
    assert set(components) == FIELDS | {"velocity_potential"}
    assert "potential" not in diag.dims
    scalar, scalar_diag = streamfunction_potential(
        u, v, ["y", "x"], "cartesian", method="sor", mask=wet,
        iParams=opts, return_diagnostics=True)
    assert set(scalar) == {"velocity_potential"}
    xr.testing.assert_allclose(scalar.velocity_potential, components.velocity_potential)
    xr.testing.assert_allclose(scalar_diag, diag)
    with pytest.raises(ValueError, match="stored gradient and remainder"):
        reconstruct(scalar)
    for initial in [scalar, scalar.velocity_potential]:
        warm = decompose(u, v, ["y", "x"], "cartesian", method="sor",
                         mask=wet, iParams=opts, initial=initial,
                         return_potentials=True)
        for name in components:
            xr.testing.assert_allclose(warm[name], components[name], rtol=2e-7, atol=2e-8)
    xr.testing.assert_allclose(reconstruct(components).u, u.where(wet))
    xr.testing.assert_allclose(reconstruct(components).v, v.where(wet))


def test_sor_updates_full_array_with_standard_fixed_boundary():
    u, v, _ = inputs(False)
    result = decompose(u, v, ["y", "x"], "cartesian", method="sor",
                       return_potentials=True, iParams={"printInfo": False})
    assert np.isfinite(result.velocity_potential).all()
    np.testing.assert_array_equal(result.velocity_potential.isel(y=[0, -1]), 0.)
