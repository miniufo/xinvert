"""
xinvert: invert geophysical fluid dynamics problems using SOR iteration.

Built on xarray and numba, this package solves classical elliptic PDEs
(Poisson, Gill-Matsuno, Stommel-Munk, QG-omega, Eliassen, PV inversion,
reference-state, etc.) via successive over-relaxation with spatially-varying
coefficients and dask-enabled parallel computation.

Notes on numerical precision
----------------------------
The whole inversion (coefficients, iteration, output) is carried out in
the dtype selected by ``iParams['dtype']`` (``float32`` by default).
``float32`` roughly halves memory traffic and storage compared with
``float64``, at the cost of reduced residual accuracy.  The forcing and an
optional ``icbc`` are converted to the selected compute dtype before the
coefficient arrays are built.
"""
from .apps import (
    animate_iteration,
    cal_flow,
    invert_3DOcean,
    invert_BrethertonHaidvogel,
    invert_Eliassen,
    invert_Fofonoff,
    invert_GeoAdjustment,
    invert_geostrophic,
    invert_GillMatsuno,
    invert_GillMatsunoFlux,
    invert_omega,
    invert_Poisson,
    invert_PV2D,
    invert_RefState,
    invert_RefStateSWM,
    invert_Stommel,
    invert_StommelArons,
    invert_StommelFlux,
    invert_StommelMunk,
)
from .core import (
    inv_general2D,
    inv_general2D_bih,
    inv_general3D,
    inv_standard2D,
    inv_standard2D_full,
    inv_standard3D,
)
from .finitediffs import FiniteDiff, deriv, deriv2, padBCs
from .utils import loop_noncore

__version__ = "0.3.1"
