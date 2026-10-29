"""GPU contracts remain explicit on CPU-only installations."""

import sys

import pytest

from xinvert.helmholtz import streamfunction_potential
from xinvert.helmholtz._li2006_gpu import require_cupy

from .test_helmholtz_closed import fields


def test_optional_dependency_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match="optional CuPy"):
        require_cupy()


def test_gpu_rejects_unsupported_solver_and_open_boundaries():
    u, v, _ = fields()
    with pytest.raises(NotImplementedError, match="explicit closed"):
        streamfunction_potential(u, v, iParams={"architect": "gpu"})
    with pytest.raises(NotImplementedError, match="solver='lsmr'"):
        streamfunction_potential(
            u,
            v,
            iParams={
                "architect": "gpu",
                "solver": "lsqr",
                "BCs": ["closed", "periodic"],
            },
        )
