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

    ┌─────────────────────────────────────────────────────────────┐
    │  apps.py  — top layer: physical-model layer                 │
    │  invert_Poisson / invert_omega / invert_GillMatsuno / ...   │
    │    responsibility: compute coefficients A/B/C/F for a given  │
    │    physical equation, then delegate to the core layer        │
    ├─────────────────────────────────────────────────────────────┤
    │  core.py  — middle layer: xarray bridge layer                │
    │  inv_standard2D / inv_standard3D / inv_general2D / ...      │
    │    responsibility: schedule non-core dims via xr.apply_ufunc│
    │    and call a low-level kernel; _make_kernel() is the ONLY  │
    │    architecture dispatch point                               │
    ├─────────────────────────────────────────────────────────────┤
    │  cpus.py  — bottom layer: CPU kernels (numba @njit)          │
    │  gpus.py  — bottom layer: GPU kernels (numba @cuda.jit)     │
    │    responsibility: pure numpy arrays + scalar parameters,   │
    │    execute the SOR iteration loop                            │
    └─────────────────────────────────────────────────────────────┘

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
      │  └─ iParams['architect'] == 'gpu' → use gpus.invert_standard_2D_gpu
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

The GPU thread-block shape is read from the ``XINVERT_GPU_BLOCK2D`` env var at
**call time** (no module reload needed), defaulting to ``(16, 16)``.
A block-shape sweep (see :doc:`Benchmark`) showed that square blocks are ~3 %
faster than warp-coalesced thin blocks for the 2-D stencil, because stencil
access is two-dimensional and square blocks give better cache locality.

.. code-block:: bash

    # tune block shape without recompiling kernels
    XINVERT_GPU_BLOCK2D=32,8 python tests/benchmark_cpu_gpu.py

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
