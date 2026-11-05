"""Common diagnostic schema and independent Krylov stopping-error oracles."""
import numpy as np
import pytest
import xarray as xr
from scipy.sparse.linalg import lsmr, lsqr

from xinvert.helmholtz import _li2006 as li
from xinvert.helmholtz import decompose, streamfunction_potential

from .test_helmholtz_output_grids import inputs
from .test_li_cpu_lsmr import DenseClosed

COMMON = {"error", "final_error", "converged", "iterations", "stop_reason",
          "vel_error", "div_error"}


@pytest.mark.parametrize("method", ["sor", "li2006"])
def test_common_history_and_last_error(architect, method):
    if architect == "gpu" and method == "li2006":
        pytest.importorskip("cupy")
    u, v, wet = inputs(False)
    result, diag = decompose(
        u, v, ["y", "x"], "cartesian", method=method, mask=wet,
        return_diagnostics=True,
        iParams={"architect": architect, "diag_interval": 13, "mxLoop": 3000,
                 "BCs": ["fixed", "fixed"] if method == "li2006" else ["fixed", "fixed"],
                 "tolerance": 1e-8, "printInfo": False})
    assert COMMON <= set(diag)
    assert bool(diag.converged)
    assert diag.stop_reason.item() == "converged"
    assert bool((diag.final_error <= 1e-8).all())
    assert not any("iteration" in diag[key].dims for key in ("vel_error", "div_error"))
    assert "potential" not in diag.dims
    history = diag.error.dropna("iteration")
    assert history[-1].item() == diag.final_error.item()
    assert int(history.iteration[-1]) == diag.iterations.item()
    assert diag.error.attrs["units"] == "1"
    assert "definition" in diag.error.attrs or method == "sor"
    assert set(result) == {"urot", "vrot", "udiv", "vdiv"}


@pytest.mark.parametrize("closed", [True])
@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
def test_li_sampling_does_not_change_fixed_work(closed, solver):
    u, v, wet = inputs(False)
    opts = {"BCs": ["fixed", "fixed"] if closed else ["open", "open"],
            "mxLoop": 7, "tolerance": 1e-14, "on_nonconvergence": "ignore",
            "solver": solver}
    args = {"dims": ["y", "x"], "coords": "cartesian", "method": "li2006", "mask": wet}
    original = decompose(u, v, iParams=opts, **args)
    outputs = []
    for interval in (2, 3):
        result, diag = decompose(u, v, iParams={**opts, "diag_interval": interval},
                                 return_diagnostics=True, **args)
        xr.testing.assert_allclose(result, original, rtol=1e-11, atol=1e-11)
        assert not bool(diag.converged)
        assert diag.stop_reason.item() == "max_iterations"
        assert diag.final_error.item() > opts["tolerance"]
        assert diag.iteration.values.tolist() == list(range(interval, 7, interval))+[7]
        outputs.append(diag.final_error.item())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
@pytest.mark.parametrize("warm", [False, True])
def test_scalar_error_matches_independent_scipy_stopping_estimates(solver, warm):
    rng = np.random.default_rng(61)
    a = rng.normal(size=(33, 15))
    b = rng.normal(size=33)
    x0 = rng.normal(size=15) if warm else None
    monitor = li._KrylovDiagnostics(3)
    kwargs = {"atol": 1e-12, "btol": 1e-12, "x0": x0}
    if solver == "lsmr":
        actual = li._lsmr_closed_cpu(DenseClosed(a), b, maxiter=7,
                                     diagnostics=monitor, **kwargs)
        reference = lsmr(a, b, maxiter=7, **kwargs)
        r, ar, anorm, xnorm = reference[3], reference[4], reference[5], reference[7]
    else:
        actual = li._lsqr_diagnostics(a, b, iter_lim=7, diagnostics=monitor, **kwargs)
        reference = lsqr(a, b, iter_lim=7, **kwargs)
        r, ar, anorm, xnorm = reference[4], reference[7], reference[5], reference[8]
    expected = min(r/(np.linalg.norm(b)+anorm*xnorm), ar/(anorm*r))
    np.testing.assert_allclose(monitor.final_error, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual[0], reference[0], rtol=1e-12, atol=1e-13)
    assert [step for step, _ in monitor.records] == [3, 6, 7]


def test_li_exact_zero_and_batched_early_stops():
    u, v, wet = inputs(False)
    uu, vv = xr.concat([u*0, u], dim="time"), xr.concat([v*0, v], dim="time")
    _, diag = streamfunction_potential(
        uu, vv, ["y", "x"], "cartesian", mask=wet, return_diagnostics=True,
        iParams={"BCs": ["fixed", "fixed"], "diag_interval": 7, "mxLoop": 2000,
                 "tolerance": 1e-8})
    assert bool(diag.converged.all())
    assert diag.final_error[0].item() == 0.
    assert diag.iterations[0].item() == 0
    assert diag.error.isel(time=0).dropna("iteration").iteration.item() == 0
    last = diag.error.isel(time=1).dropna("iteration")
    assert int(last.iteration[-1]) == diag.iterations[1].item()
    assert np.isnan(diag.error.isel(time=0).sel(iteration=last.iteration[-1]))


@pytest.mark.parametrize("solver", ["lsmr", "lsqr"])
def test_li_float32_diagnostics_preserve_precision(solver):
    u, v, wet = inputs(False)
    result, diag = decompose(
        u.astype("float32"), v.astype("float32"), ["y", "x"], "cartesian",
        mask=wet, return_diagnostics=True,
        iParams={"BCs": ["fixed", "fixed"], "solver": solver, "diag_interval": 7,
                 "mxLoop": 2000, "tolerance": 1e-5})
    assert all(field.dtype == np.dtype("float32") for field in result.values())
    assert bool(diag.converged)
    assert diag.final_error.item() <= 1e-5
    assert diag.error.dropna("iteration")[-1].item() == diag.final_error.item()


def test_condition_limit_is_distinguished_from_convergence():
    a = np.diag(np.geomspace(1., 1e-6, 12))
    monitor = li._KrylovDiagnostics(5)
    result = li._lsmr_closed_cpu(DenseClosed(a), np.ones(12),
                                 atol=1e-14, btol=1e-14, conlim=1.01,
                                 maxiter=30, diagnostics=monitor)
    assert result[1] == 3
    assert monitor.stop_reason == "condition_limit"
    assert monitor.final_error > 1e-14
    assert monitor.records[-1][0] == result[2]


@pytest.mark.parametrize("backend", ["CPU", "cPu", "CPU-mc", "GPU"])
def test_li_backend_is_case_insensitive_and_does_not_mutate_options(backend):
    if backend.lower() == "gpu":
        pytest.importorskip("cupy")
        from .gpu_utils import gpu_available
        if not gpu_available():
            pytest.skip("CUDA is unavailable")
    u, v, wet = inputs(False)
    opts = {"architect": backend, "BCs": ["fixed", "fixed"], "threads": 2,
            "mxLoop": 2000, "tolerance": 1e-8}
    result, diag = decompose(u, v, ["y", "x"], "cartesian", mask=wet,
                             iParams=opts, return_diagnostics=True)
    assert bool(diag.converged)
    assert opts["architect"] == backend
    assert set(result) == {"urot", "vrot", "udiv", "vdiv"}
