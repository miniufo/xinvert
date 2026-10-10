"""Shared backend selection; unavailable CUDA is an explicit skip."""
import numba
import pytest

from .gpu_utils import gpu_available

# Large scientific-data regressions remain in the default full suite.
# Use "-m not slow" for a quick developer/CI check.
DATA_REGRESSION_FILES = {
    'test_AnimateConverge.py', 'test_Bretherton.py', 'test_Eliassen.py', 'test_FDs.py',
    'test_Fofonoff.py', 'test_GeoAdjustment.py', 'test_Geopotential.py',
    'test_GillMatsuno.py', 'test_Ishida.py', 'test_MunkWBC.py',
    'test_OmegaEq.py', 'test_Poisson.py', 'test_QGPV.py',
    'test_RefStateSWM.py', 'test_StommelArons.py', 'test_StommelWBC.py',
}


def pytest_collection_modifyitems(items):
    """Separate data-dependent regressions without dropping them from pytest."""
    for item in items:
        if item.path.name in DATA_REGRESSION_FILES:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(params=['cpu', 'cpu-mc', 'gpu'])
def architect(request):
    backend = request.param
    if backend == 'gpu' and not gpu_available():
        pytest.skip('CUDA kernel probe failed')
    original = numba.get_num_threads()
    if backend == 'cpu-mc':
        numba.set_num_threads(min(4, original))
    try:
        yield backend
    finally:
        numba.set_num_threads(original)
