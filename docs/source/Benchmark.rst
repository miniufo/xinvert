.. xinvert GPU benchmark documentation

GPU Benchmark
=============

This page documents the GPU performance of xinvert's Red-Black SOR solver,
the optimisations applied, and how to reproduce the measurements.

Hardware
--------

* CPU: x86_64
* GPU: NVIDIA GeForce RTX 3090 (82 SMs, compute capability 8.6)
* Software: Python 3.13, numba / numba-cuda

Benchmark method
----------------

The benchmark solves a synthetic Poisson equation with a known analytic
solution (:math:`\psi = \sin(\pi x)\sin(\pi y)`) on square grids from
128×128 up to 4096×4096.

Both CPU and GPU run a **fixed 1000 iterations** (``tolerance=0``) so that
they do exactly the same amount of work.  The speedup therefore reflects
pure per-iteration throughput, not "who converged first".  Timings include
host↔device data transfer for the GPU.

Reproduce::

    python tests/benchmark_cpu_gpu.py 128 256 512 1024 2048 4096

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 15 15 12 15 15

   * - Grid
     - CPU (s)
     - GPU (s)
     - Speedup
     - CPU error
     - GPU error
   * - 128×128
     - 0.103
     - 0.071
     - 1.45×
     - 5.10e-05
     - 5.10e-05
   * - 256×256
     - 0.423
     - 0.075
     - 5.68×
     - 1.26e-05
     - 1.26e-05
   * - 512×512
     - 1.684
     - 0.126
     - 13.41×
     - 1.40e-04
     - 4.71e-05
   * - 1024×1024
     - 6.762
     - 0.431
     - 15.68×
     - 4.01e-02
     - 1.51e-02
   * - 2048×2048
     - 28.008
     - 1.707
     - 16.41×
     - 2.79e-01
     - 1.89e-01
   * - 4096×4096
     - 116.814
     - 6.825
     - 17.12×
     - —
     - —

The speedup grows with grid size and saturates near 17×, because larger
grids amortise the fixed per-iteration overhead (kernel launch, host
synchronisation, data transfer).

.. note::

    At large grids (512+) with ``optArg ≈ 2.0`` SOR over-relaxation has not
    fully converged in 1000 iterations, so the error grows with grid size
    for **both** back-ends.  This is expected and does not indicate an
    implementation bug; the speedup comparison remains valid (equal work).

    CPU uses standard SOR (Gauss-Seidel); GPU uses Red-Black SOR (required
    for parallelism).  Both converge to the same fixed point; small
    per-iteration differences exist but the final result agrees.

Correctness verification
------------------------

``tests/test_CpuGpuConsistency.py`` verifies CPU↔GPU consistency on a
51×41 Poisson problem (``mxLoop=20000``, ``tolerance=1e-10``)::

    CPU  max error vs analytic: 4.216179e-04
    GPU  max error vs analytic: 4.206634e-04
    CPU-GPU max diff:           1.106681e-06
    RESULT: PASS

Optimisations applied
---------------------

The GPU wrapper went through several optimisation rounds.  All changes are
in ``xinvert/gpus.py``.

Round 1 — sparse convergence checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On non-check iterations only the Red + Black update kernels run; the norm
reduction kernel and host sync are skipped.  This cut the host-device
synchronisation cost dramatically.

Round 2 — adaptive check interval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_compute_check_interval(mxLoop, tolerance)`` scales the check interval with
the total iteration budget so the number of host-device synchronisations
stays roughly constant regardless of problem size:

- ``tolerance > 0`` (real convergence): ``clamp(mxLoop/50, 10, 100)``
- ``tolerance ≤ 0`` (fixed-iter / benchmark): ``clamp(mxLoop/20, 50, 500)``

This benefits large iteration counts the most — e.g. 10000 iterations need
only ~20 synchronisations (same as 1000), so the speedup grows with
iteration count:

.. list-table::
   :header-rows: 1

   * - mxLoop (512², tol=0)
     - CPU (s)
     - GPU (s)
     - Speedup
   * - 1000
     - 1.69
     - 0.14
     - 11.9×
   * - 10000
     - 16.96
     - 0.99
     - 17.1×

Round 2 also extracted the convergence-evaluation logic into
``_evaluate_gpu_norm(...)`` and the interval logic into
``_compute_check_interval(...)``, both dimension-agnostic, so future 3-D /
general / biharmonic GPU wrappers can reuse the same loop skeleton.

Round 3 — boundary-kernel consolidation & block configurability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Folded the four-corner handling into ``_extend_x_boundary``, removing a
  separate single-block kernel launch (which emitted a ``Grid size 1``
  under-utilisation warning) and saving one kernel launch per iteration.
* Made the 2-D thread-block shape configurable via the
  ``XINVERT_GPU_BLOCK2D`` env var, read at call time (no module reload).

Thread-block shape sweep
------------------------

A sweep over six block shapes (``tests/benchmark_blocks.py``) tested whether
warp-coalesced thin blocks (``32×8``) beat square blocks (``16×16``).

.. list-table::
   :header-rows: 1
   :widths: 15 14 14 14 14 14 14

   * - Grid
     - 16×16
     - 32×8
     - 32×16
     - 32×32
     - 64×4
     - 16×32
   * - 512×512
     - 0.1251
     - 0.1244
     - 0.1331
     - 0.1498
     - 0.1258
     - 0.1356
   * - 1024×1024
     - 0.4392
     - 0.4403
     - 0.4526
     - 0.4837
     - 0.4403
     - 0.4536
   * - 2048×2048
     - 1.7139
     - 1.7616
     - 1.7638
     - 1.8657
     - 1.9462
     - 1.8159
   * - 4096×4096
     - 6.9775
     - 7.1847
     - 7.2191
     - 7.5225
     - 8.2796
     - 7.3437

**Finding**: the square ``(16, 16)`` block is ~3 % faster than the
warp-coalesced ``(32, 8)`` at 4096².  Coalescing theory predicts ``(32,8)``
wins for row-stride access, but the SOR stencil accesses neighbours in
**both** x and y, so square blocks give better 2-D cache locality.  Large
blocks (``32×32`` = 1024 threads) reduce occupancy and are consistently
slowest.

The default remains ``(16, 16)``; tune via::

    XINVERT_GPU_BLOCK2D=32,8 python tests/benchmark_cpu_gpu.py

Reproduce the sweep::

    python tests/benchmark_blocks.py 128 256 512 1024 2048 4096

Numba performance warnings
--------------------------

numba-cuda emits ``NumbaPerformanceWarning: Grid size N will likely result
in GPU under-utilization`` when the launched grid has fewer blocks than the
GPU has SMs (82 on the RTX 3090).  This is **expected and correct** for
small grids and for the single-block boundary kernels — it does not affect
results.

The warning category lives in a different module path depending on the
numba / numba-cuda version, so a category-based ``filterwarnings`` is
unreliable.  The benchmark scripts filter by **message text** instead::

    warnings.filterwarnings('ignore', message='.*under-utilization.*')

Future optimisation directions
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Direction
     - Gain
     - Notes
   * - Block-level norm reduction
     - medium
     - reduces atomic contention on large grids; benefits 3-D even more
   * - Shared-memory tiling for SOR
     - high
     - reduces global-memory traffic; Red-Black halves utilisation, needs
       benchmarking
   * - Block-size auto-tuning
     - low–medium
     - pick block shape from grid size / CC at call time
   * - Pinned-memory host buffers
     - low
     - faster ``copy_to_host``
   * - Coefficient caching across solves
     - high
     - reuse A/B/C/F device buffers for animation / frame-by-frame solves
       (needs API change)
   * - max|update| lightweight convergence signal
     - medium
     - replaces the full-array norm; especially useful for 3-D
