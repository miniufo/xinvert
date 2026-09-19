.. xinvert architecture documentation

Architecture
============

xinvert uses a three-layer architecture for solving elliptic partial
differential equations (Poisson, omega, Gill-Matsuno, Stommel-Munk, etc.)
encountered in geophysical fluid dynamics.  The equations are solved at the
bottom layer by SOR (Successive Over-Relaxation) iteration, with two compute
back-ends: CPU (numba) and GPU (CUDA).

Three-layer design
------------------

::

    ┌──────────────────────────────────────────────────────────────┐
    │  apps.py  — top layer: physical-model layer                  │
    │  invert_Poisson / invert_omega / invert_GillMatsuno / ...    │
    │    responsibility: compute coefficients A/B/C/F for a given  │
    │    physical equation, then delegate to the core layer        │
    ├──────────────────────────────────────────────────────────────┤
    │  core.py  — middle layer: xarray bridge layer                │
    │  inv_standard2D / inv_standard3D / inv_general2D / ...       │
    │    responsibility: schedule non-core dims via xr.apply_ufunc │
    │    and call a low-level kernel; _make_kernel() is the ONLY   │
    │    architecture dispatch point                               │
    ├──────────────────────────────────────────────────────────────┤
    │  cpus.py  — bottom layer: CPU kernels (numba @njit)          │
    │  gpus.py  — bottom layer: GPU kernels (numba @cuda.jit)      │
    │    responsibility: pure numpy arrays + scalar parameters,    │
    │    execute the SOR iteration loop                            │
    └──────────────────────────────────────────────────────────────┘

Data flow
---------

Using ``invert_Poisson`` as an example::

    user call
      │  invert_Poisson(F, dims=['lat','lon'], iParams={'architect':'gpu'})
      ▼
    apps.py: __template(__coeffs_Poisson, inv_standard2D, 2, F, dims, ...)
      │  1. __coeffs_Poisson → compute coefficients (A, B, C) and forcing F
      │  2. __cal_params2D   → compute grid params (gc2, gc1, del1Sqr, ratioSqr, ...)
      │  3. call inv_standard2D(A, B, C, F, S, dims, iParams)
      ▼
    core.py: inv_standard2D(A, B, C, F, S, dims, iParams)
      │  1. assemble grid_args = [gc2, gc1, BCy, BCx, del1Sqr, ratioQtr, ratioSqr]
      │  2. _make_kernel(kernel_func, grid_args, iParams)
      │     ├─ iParams['architect'] == 'cpu' → use cpus.invert_standard_2D
      │     └─ iParams['architect'] == 'gpu' → use gpus.invert_standard_2D_gpu
      │  3. xr.apply_ufunc(_kernel_, S, A, B, C, F, info, ...)
      ▼
    cpus.py / gpus.py: kernel(S, A, B, C, F, info, *grid_args, optArg, undef, flags, mxLoop, tolerance)
      │  CPU: kernel runs the full while(True) iteration loop internally
      │  GPU: Python wrapper runs a while loop, launching Red/Black CUDA kernels each iteration
      └─ modifies S (in-place), sets flags[overflow, error, loops]

Architecture dispatch (architect)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 25 15 30

   * - Parameter
     - Location
     - Allowed values
     - Default
     - Description
   * - ``architect``
     - ``iParams``
     - ``'cpu'``, ``'gpu'``
     - ``'cpu'``
     - Compute back-end selection

Any other value raises: ``ValueError: unsupported architect 'xxx', should be 'cpu' or 'gpu'``

Dispatch location
~~~~~~~~~~~~~~~~~

Dispatch happens inside ``_make_kernel()`` in ``core.py`` — this is the
**only coupling point between the core layer and the bottom layer**:

.. code-block:: python

    def _make_kernel(kernel_func, grid_args, iParams):
        architect = iParams.get('architect', 'cpu')

        if architect == 'cpu':
            func = kernel_func                          # numba @njit kernel
        elif architect == 'gpu':
            func = _gpu_kernel_map[kernel_func]         # look up GPU equivalent
        else:
            raise ValueError(...)

        def _kernel_(s, *args):
            # ... call func; CPU/GPU interfaces are identical ...
        return _kernel_

GPU kernel registry
~~~~~~~~~~~~~~~~~~~

``core.py`` maintains a registry ``_gpu_kernel_map`` that is populated at
module-load time:

.. code-block:: python

    # top of core.py
    _gpu_kernel_map = {}
    try:
        from .gpus import invert_standard_2D_gpu
        _gpu_kernel_map[invert_standard_2D] = invert_standard_2D_gpu
    except Exception:
        pass  # silently fall back to CPU-only when CUDA is unavailable

When the user requests ``architect='gpu'`` but no GPU kernel is implemented
for the requested equation, it raises:
``NotImplementedError: GPU kernel not implemented for invert_standard_2D``

CPU vs GPU implementation comparison
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - CPU (cpus.py)
     - GPU (gpus.py)
   * - Decorator
     - ``@nb.njit``
     - ``@cuda.jit`` (kernels) + Python wrapper
   * - Iteration control
     - ``while True`` inside the kernel
     - ``while True`` in the Python wrapper
   * - Per call
     - runs all iterations
     - launches a Red + a Black kernel each iteration
   * - Parallelism
     - serial (point-by-point update)
     - massively parallel (Red-Black parallel update)
   * - Convergence check
     - ``absNorm2D`` inside the kernel
     - GPU reduction kernel → host comparison
   * - Data location
     - numpy host arrays
     - CUDA device arrays
   * - Function signature
     - identical
     - identical

Red-Black SOR principle
-----------------------

Classical SOR is inherently serial (each point update depends on the latest
neighbour values), so it cannot be parallelised directly.  Red-Black SOR
partitions the grid into two colours in a checkerboard pattern::

    R B R B R
    B R B R B
    R B R B R

1. **Red phase**: all red points update in parallel (their neighbours are all
   black, read at the old value).
2. **Black phase**: all black points update in parallel (their neighbours are
   all red, already updated).

Together the two phases are equivalent to one SOR iteration.

.. note::

    When B≠0 (cross-derivative term) the diagonal neighbours share the same
    colour, so the cross terms are effectively Jacobi-updated.  For the
    Poisson equation (B=0) Red-Black SOR is strictly correct.

GPU convergence-check strategy
------------------------------

The GPU wrapper cannot check convergence every iteration cheaply, because each
check requires a host-device synchronisation.  Two optimisations are applied:

Adaptive check interval
~~~~~~~~~~~~~~~~~~~~~~~

``_compute_check_interval(mxLoop, tolerance)`` scales the check interval with
the total iteration budget so that the number of host-device synchronisations
stays roughly constant regardless of problem size:

- **tolerance > 0** (real convergence run): ``clamp(mxLoop/50, 10, 100)``
  e.g. mxLoop = 1000 → 20, 5000 → 100, 10000 → 100, 100000 → 100
- **tolerance ≤ 0** (fixed-iteration / benchmark, no early exit): only an
  overflow guard is needed, so a larger interval is safe:
  ``clamp(mxLoop/20, 50, 500)``
  e.g. mxLoop = 1000 → 50, 5000 → 250, 10000 → 500, 100000 → 500

On non-check iterations only the Red + Black update kernels run; the norm
reduction kernel and host sync are skipped entirely.

Shared norm-evaluation helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_evaluate_gpu_norm(...)`` reduces the ``[sum|S|, count]`` atomic accumulators
into a convergence signal that mirrors the CPU ``absNorm*`` metric:

.. math::

    \text{norm} = \frac{1}{N}\sum |S|,\qquad
    \text{error} = \frac{|\text{norm} - \text{norm}_{prev}|}{\text{norm}_{prev}}

These two helpers are dimension-agnostic, so future 3-D / general / biharmonic
GPU wrappers can reuse the same loop skeleton and only need their own update
kernel.

Configurable thread-block shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU 2-D thread-block shape defaults to ``(16, 16)`` and can be overridden
**per call** via ``iParams['gpu_block2d']`` (no module reload needed):

.. code-block:: python

    iParams['gpu_block2d'] = (32, 8)   # warp-coalesced thin block

A block-shape sweep (see :doc:`Benchmark`) showed that square blocks are ~3 %
faster than warp-coalesced thin blocks for the 2-D stencil, because stencil
access is two-dimensional and square blocks give better cache locality.

The 1-D boundary/norm kernels have **no user-facing block parameter**: their
block size is chosen adaptively by ``_auto_bsize_1d(n)`` from the extent each
kernel actually sweeps (32 for *n* < 64, 128 for *n* < 512, else 256).  Since
these kernels are off the hot loop, the block size has negligible effect on
total runtime, so the parameter was removed to keep the API surface small.
``gpu_block2d`` in ``iParams`` is the only remaining GPU tuning knob.

GPU with dask datasets
~~~~~~~~~~~~~~~~~~~~~~

When the input is dask-backed and ``architect='gpu'``, each time step
becomes a dask task and the result stays **lazy** — compute is deferred
until the user requests it (``.compute()``, ``.to_netcdf()``, plotting,
...), so results can be streamed to disk chunk by chunk without ever
holding the whole dataset in memory.

GPU solves are bounded to one at a time per process
(``gpus._GPU_MAX_CONCURRENT``): concurrent solves gain nothing (kernels
serialise on the default stream, and the blocking convergence-check syncs
create a convoy effect — measured ~0.4x on small grids), while each
in-flight solve holds ~5 device buffers in VRAM.  With the semaphore in
place, dask's worker threads simply pipeline disk I/O against the GPU
solves.  Note that numba-cuda manages CUDA contexts thread-locally, so
``gpus.ensure_context()`` is called once in the main thread at dispatch
time; without it, dask worker threads would fail with
``CUDA_ERROR_NOT_INITIALIZED`` on their first launch.

Live ``printInfo`` output under parallelism
-------------------------------------------

With ``dask='parallelized'`` the per-time-step inversions run concurrently in
dask worker threads (the numba kernels release the GIL via ``nogil=True``), and
each call prints its diagnostic line as soon as it finishes.  Plain ``print``
does not survive this setting, for two reasons:

**1. Line splicing.**  ``print(msg)`` issues *two* ``write`` calls — ``msg``,
then ``'\n'``.  ipykernel's ``OutStream`` flushes on newline, so a concurrent
thread can interleave between the two writes and splice two messages into one
corrupted line.

**2. Output misattribution.**  ipykernel binds each *new* thread to the parent
header (the cell) that spawned it.  dask reuses its worker pool across cells,
so on a re-run those threads still carry the *previous* cell's parent header
and their output is attributed to a finished cell — the current cell appears to
produce nothing.

``core._print_live`` handles both:

.. code-block:: python

    def _print_live(msg):
        out   = sys.stdout
        ident = threading.get_ident()
        # drop this thread's stale parent-header registration so the header
        # falls back to ipykernel's global = the *currently executing* cell
        for attr in ('_thread_to_parent', '_thread_to_parent_header'):
            reg = getattr(out, attr, None)
            if reg is not None:
                try:
                    reg.pop(ident, None)
                except Exception:
                    pass
        out.write(msg + '\n')   # single write => one atomic line
        out.flush()

Notes:

* It is a **module-level function**, not a closure.  The ``_kernel_`` closure
  only ever *references* it, and cloudpickle transfers referenced module-level
  functions **by reference** — so the dask graph stays picklable.  Capturing a
  lock or buffer in the closure would break ``dask.distributed``.
* It only touches ipykernel internals **when they exist**; under a plain Python
  interpreter both ``getattr`` calls return ``None`` and the function degrades
  to a normal atomic ``print``.
* Under ``dask.distributed`` the worker processes print to their own stdout,
  visible in the worker logs rather than in the client notebook.

Regression checks live in ``tests/test_jupyter_printinfo.py`` (executes a real
2-cell notebook through ``nbclient`` and asserts every cell sees all 12 lines)
and ``tests/test_distributed.py`` (asserts the graph serialises and solves
correctly under a real ``distributed.Client``).

How to add a new GPU kernel
---------------------------

Using ``invert_general_2D_gpu`` as an example:

1. Implement it in ``gpus.py``:

.. code-block:: python

    def invert_general_2D_gpu(S, A, B, C, D, E, F, G, info,
                               yc, xc, delx, BCy, BCx,
                               delxSqr, ratio, ratioQtr, ratioSqr,
                               optArg, undef, flags, mxLoop, tolerance):
        # 1. transfer data to GPU
        # 2. while loop: Red/Black kernel + norm kernel (reuse the shared
        #    _compute_check_interval and _evaluate_gpu_norm helpers)
        # 3. copy result back, set flags

2. Register it in ``core.py``:

.. code-block:: python

    from .gpus import invert_general_2D_gpu
    _gpu_kernel_map[invert_general_2D] = invert_general_2D_gpu

3. No changes needed in ``apps.py``:

The ``invert_*`` functions in ``apps.py`` are completely unchanged; users
enable GPU via ``iParams={'architect': 'gpu'}``.

User usage
----------

.. code-block:: python

    import xinvert

    # CPU (default)
    S = xinvert.invert_Poisson(F, dims=['lat', 'lon'])

    # GPU
    S = xinvert.invert_Poisson(F, dims=['lat', 'lon'],
                                iParams={'architect': 'gpu'})

    # GPU with custom parameters
    S = xinvert.invert_Poisson(F, dims=['lat', 'lon'],
                                iParams={'architect': 'gpu',
                                         'mxLoop': 10000,
                                         'tolerance': 1e-10})

File structure
--------------

::

    xinvert/
    ├── __init__.py      # package exports
    ├── apps.py          # top layer: physical models (Poisson, omega, ...)
    ├── core.py          # middle layer: xarray bridge + _make_kernel dispatcher
    ├── cpus.py          # bottom layer: CPU kernels (numba @njit)
    ├── gpus.py          # bottom layer: GPU kernels (numba @cuda.jit)
    ├── utils.py         # utility functions
    └── finitediffs.py   # finite-difference utilities

Current GPU implementation status
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 30 10

   * - Equation type
     - CPU kernel
     - GPU kernel
     - Status
   * - standard 2D (Poisson etc.)
     - ``invert_standard_2D``
     - ``invert_standard_2D_gpu``
     - ✅ implemented
   * - standard 3D (omega etc.)
     - ``invert_standard_3D``
     - —
     - TODO
   * - general 2D (GillMatsuno etc.)
     - ``invert_general_2D``
     - —
     - TODO
   * - general 3D (3DOcean etc.)
     - ``invert_general_3D``
     - —
     - TODO
   * - general 2D bih (StommelMunk etc.)
     - ``invert_general_bih_2D``
     - —
     - TODO
   * - standard 2D test
     - ``invert_standard_2D_test``
     - —
     - TODO
   * - standard 1D
     - ``invert_standard_1D``
     - —
     - TODO
