# -*- coding: utf-8 -*-
"""Backend dispatch should fail loudly and explain optional dependencies."""
import pytest

import xinvert.core as core
from xinvert.cpus import invert_standard_2D


def _params(architect):
    return {'architect': architect, 'convergence': 'norm'}


def test_missing_gpu_dependency_has_actionable_error(monkeypatch):
    missing = ImportError('numba-cuda is missing')
    monkeypatch.setattr(core, '_gpu_kernel_map', {})
    monkeypatch.setattr(core, '_gpu_import_error', missing)

    with pytest.warns(UserWarning, match='GPU backend is experimental'):
        with pytest.raises(
                ImportError, match=r'pip install xinvert\[gpu\]') as exc:
            core._make_kernel(invert_standard_2D, [], _params('gpu'))

    assert exc.value.__cause__ is missing


def test_unimplemented_gpu_kernel_is_distinct_from_missing_dependency(
        monkeypatch):
    def unsupported_kernel():
        pass

    monkeypatch.setattr(core, '_gpu_import_error', None)
    with pytest.warns(UserWarning, match='GPU backend is experimental'):
        with pytest.raises(NotImplementedError, match='unsupported_kernel'):
            core._make_kernel(unsupported_kernel, [], _params('gpu'))


def test_invalid_architecture_is_rejected():
    with pytest.raises(ValueError, match="should be 'cpu' or 'gpu'"):
        core._make_kernel(invert_standard_2D, [], _params('tpu'))
