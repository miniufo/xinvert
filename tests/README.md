# Test suite

Tests are organized by purpose; performance measurements belong in
`benchmarks/`, and runnable demonstrations belong in `examples/`.

* `test_backend_math.py`: the same discrete manufactured solutions for all
  seven templates on CPU, CPU-mc and GPU, in float32 and float64. Assertions
  check the solution and independently evaluated equation residual.
* `test_CpuMc.py`, `test_cpu_mc_templates.py`, `test_Gpu*.py` and
  `test_CpuGpuConsistency.py`: specialized boundaries, masks, coloring,
  thread reproducibility and backend comparisons. Agreement alone is not
  proof of mathematical correctness.
* `test_backend_apps.py`: shared small public-application smoke tests and a
  constant-height GeoAdjustment oracle. Nonlinear RefState/RefStateSWM
  integrations remain in their dedicated scientific-data tests.
* `test_Convergence.py` and validation/parameter tests: public API,
  diagnostics, stopping controls, batching and Dask behavior.
* Named physical-application tests: integration and historical regression
  checks, often requiring NetCDF inputs in `Data/`. These are not a complete
  three-backend matrix.

The shared `architect` fixture selects CPU, CPU-mc and GPU. It limits CPU-mc
to four threads and restores the previous setting afterward. A failed CUDA
kernel probe explicitly skips GPU cases; CPU-only CI is not GPU validation.

## Reference values and convergence

Do not regenerate historical reference values just because stopping controls
changed. Data files may contain independent reference solutions; hard-coded
extrema or energies are historical snapshots, not independent oracles.

Before updating a snapshot:

1. State the compute dtype and explicit comparison `rtol`/`atol`.
2. Enable diagnostics and inspect convergence, stop reason and iteration count.
3. Compare with a tighter float64 solve and, where possible, an independently
   evaluated discrete equation residual or analytical solution.
4. Record why the old reference is obsolete and how the new one was validated.

The current reported residual is a normalized maximum SOR correction, not
an independently evaluated `F - L(S)`. It is not an upper bound on solution
error. Extremely small tolerances in float32 do not guarantee accuracy;
rounding can produce zero updates or the solver may hit its iteration limit.

## Running

Run from the repository root:

```sh
NUMBA_NUM_THREADS=4 python -m pytest -q tests/test_backend_math.py tests/test_Convergence.py
python -m pytest -q
python -m pytest -q -m "not slow"
python -m pytest -q -p tests.reference_audit tests/test_MunkWBC.py
```

The optional GPU CI job runs all GPU-specific files plus the shared
mathematical and diagnostics tests. Long real-data and Jupyter/distributed
tests should be run separately when their runtime is unsuitable for a quick
development check. Keep MultiGrid deferred.

The optional reference audit plugin reports actual CPU slice flags without
changing solver controls or expected values. A historical snapshot can pass
even when tolerance was not met; such a result must not be presented as an
accurate converged reference. This audit does not measure equation residual
or certify that float32 rounding has not caused stagnation.

The audited Bretherton, Gill--Matsuno, Munk, Stommel, Stommel--Arons and
RefStateSWM snapshots previously exhausted their iteration limits without
meeting their positive tolerances. They now explicitly retain float32 and
the same iteration limits with tolerance=0 (fixed work). Their numerical
reference values and comparison tolerances are unchanged. These tests
detect historical output changes, not convergence or mathematical accuracy.
