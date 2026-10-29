"""Public naming, parameter, and diagnostic contracts."""
import numpy as np
import pytest
import xarray as xr

from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose, reconstruct, streamfunction_potential

from .test_helmholtz_output_grids import faces, inputs

ERRORS = {"vel_error", "div_error", "converged", "iterations"}
FIELDS = {"urot", "vrot", "udiv", "vdiv"}


@pytest.mark.parametrize("method", ["sor", "forget", "li2006"])
@pytest.mark.parametrize("grid", ["arakawa-a", "Arakawa-A", "ARAKAWA-A"])
def test_common_output_and_errors(method, grid):
    u, v, wet = inputs(False)
    opts = {"mxLoop": 6000, "tolerance": 1e-10, "printInfo": False,
            "BCs": ["closed", "closed"] if method == "li2006" else ["fixed", "fixed"]}
    result, d = decompose(u, v, ["y", "x"], coords="cartesian", grid=grid,
                          method=method, mask=wet, iParams=opts, return_diagnostics=True)
    assert set(result) == FIELDS
    assert set(d) == ERRORS
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
    result, d = decompose(u, v, ["y","x"], coords="cartesian", method="forget",
                          grid=grid, mask=wet, iParams={"BCs":["fixed","periodic"],
                          "printInfo":False, "mxLoop":6000, "tolerance":1e-10},
                          return_diagnostics=True)
    assert set(result) == FIELDS
    assert set(d) == ERRORS
    assert result.urot.dims == ("y","x_face")
    assert result.vrot.dims == ("y_face","x")
    assert float(d.vel_error) < 1e-14
    assert float(d.div_error) < 1e-8


@pytest.mark.parametrize("kw", [
    {"method": "poisson"}, {"grid": "collocated"}, {"grid": "face"},
    {"mParams":{"regularization":1e-8}}, {"mParams":{"rho":1e-8}},
    {"grid":"arakawa-c"},
])
def test_removed_options_are_not_silent_aliases(kw):
    u, v, wet = inputs(False)
    with pytest.raises(ValueError):
        decompose(u, v, ["y","x"], coords="cartesian", mask=wet, **kw)


def test_alpha_preserves_default_objective_and_has_expected_effect():
    u, v, wet = inputs(False)
    opts = {"BCs":["closed","closed"],"mxLoop":6000,"tolerance":1e-12,
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


@pytest.mark.parametrize("method", ["sor", "forget", "li2006"])
def test_zero_and_batch_errors(method):
    u,v,wet = inputs(False)
    opts = {"BCs":["closed","closed"] if method=="li2006" else ["fixed","fixed"],
            "printInfo":False,"mxLoop":6000,"tolerance":1e-10}
    _,d = decompose(u*0,v*0,["y","x"],"cartesian",method=method,
                         mask=wet,iParams=opts,return_diagnostics=True)
    assert float(d.vel_error) == float(d.div_error) == 0
    if method=="forget":
        return
    uu=xr.concat([u,u*.5],dim="time")
    vv=xr.concat([v,v*.5],dim="time")
    _,d=decompose(uu,vv,["y","x"],"cartesian",method=method,
                  mask=wet,iParams=opts,return_diagnostics=True)
    assert all(a.dims == ("time",) for a in d.values())
    np.testing.assert_allclose(d.vel_error[0],d.vel_error[1],rtol=1e-10)
    np.testing.assert_allclose(d.div_error[0],2*d.div_error[1],rtol=1e-8)
