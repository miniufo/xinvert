.. xinvert architecture documentation

Architecture
============

xinvert solves elliptic equations with successive over-relaxation (SOR).  The
same high-level applications can run on a numba CPU backend or an experimental
multi-color CPU or CUDA GPU backend, while retaining xarray dimensions,
coordinates and batching.

Three-layer design
------------------

::

    ┌──────────────────────────────────────────────────────────────┐
    │ apps.py  — physical equations                                │
    │ coefficients, masks, grid parameters and user-facing calls   │
    ├──────────────────────────────────────────────────────────────┤
    │ core.py  — xarray bridge                                     │
    │ validation, vectorized batch scheduling and diagnostics      │
    ├──────────────────────────────────────────────────────────────┤
    │ cpus.py / cpu_mc.py / gpus.py — numerical kernels            │
    │ SOR sweeps, boundary updates and residual checks             │
    └──────────────────────────────────────────────────────────────┘

For example, ``invert_Poisson`` first constructs the equation coefficients in
``apps.py``.  ``core.py`` then applies the selected low-level solver over the
requested spatial dimensions with ``xarray.apply_ufunc``.  Dimensions not
listed in ``dims`` (for example ``time`` or ``member``) are independent batch
dimensions and are preserved in the result.

Selecting a backend
-------------------

The backend is selected only through ``iParams['architect']``. CPU is the
default; use ``'cpu-mc'`` for the experimental multi-core, multi-color CPU
solver or ``'gpu'`` for CUDA. Names are case-insensitive and surrounding
whitespace is ignored. All three backends implement the same seven elliptic
solver templates used by the ordinary inversion applications. There are no
equation families implemented only on one or two backends. CPU-mc and GPU
remain experimental; the Poisson benchmark does not imply the same speedup
for other templates. Existing boundary-mode restrictions still apply,
including fixed z boundaries for 3D inversion. MultiGrid is outside this
support claim.

For Poisson, ``iParams['internal_bcs'] = 'neumann'`` automatically identifies
the active mask per spatial slice on all three backends. See
:doc:`BoundaryConditions` for the external/internal boundary diagram and
:doc:`DomainTopology` for compatibility and gauge details. Internal Neumann
also supports PV2D, diagonal Eliassen, geostrophic, Bretherton--Haidvogel and
Fofonoff and RefState on all three backends. Nonzero mixed terms, general
first-derivative, 1-D/3-D and fourth-order operators, and animation still
raise explicit unsupported errors. Reaction and ellipticity restrictions
are documented in :doc:`DomainTopology`.

.. code-block:: python

    # CPU (default)
    solution = invert_Poisson(forcing, dims=['lat', 'lon'])

    # Multi-core, multi-color CPU (experimental)
    solution = invert_Poisson(
        forcing, dims=['lat', 'lon'], iParams={'architect': 'cpu-mc'},
    )

    # GPU
    solution = invert_Poisson(
        forcing, dims=['lat', 'lon'], iParams={'architect': 'gpu'},
    )


CPU thread counts
-----------------

SOR (including internal Neumann solves) and closed-boundary Li2006 share
``iParams['threads']`` for ``architect='cpu-mc'``. Its default is ``None``:
omitting it inherits the executing thread's current Numba mask. Importing xinvert
does not change the global Numba setting. An integer overrides the current
Numba setting temporarily; the previous setting is restored after the solve,
including on exceptions. Explicit counts must be positive integers no greater
than ``numba.config.NUMBA_NUM_THREADS``; booleans are rejected.

.. code-block:: python

    # Default: inherit the executing thread's Numba setting.
    solution = invert_Poisson(
        forcing, dims=['lat', 'lon'], iParams={'architect': 'cpu-mc'},
    )

    # Override only this solve.
    solution = invert_Poisson(
        forcing, dims=['lat', 'lon'],
        iParams={'architect': 'cpu-mc', 'threads': 8},
    )

    # External Numba control (threads=None can also be omitted).
    import numba
    numba.set_num_threads(8)
    solution = invert_Poisson(
        forcing, dims=['lat', 'lon'],
        iParams={'architect': 'cpu-mc', 'threads': None},
    )

The same dictionaries apply to Li2006's closed-boundary CPU-mc implementation.
``threads=None`` inherits the setting of the **executing thread**. Numba masks
are thread-local: setting a mask in the main thread does not configure Dask
worker threads. For lazy/Dask solves prefer an explicit integer, which is
applied inside each executing worker. Concurrent slices each receive that
many threads; limit Dask concurrency too when conserving total resources.
The setting does not control BLAS, SciPy vector operations, or GPU blocks.
Ordinary SOR ``architect='cpu'`` remains serial and ignores the thread count.

To cap Numba's maximum in every worker process, set ``NUMBA_NUM_THREADS``
before importing Numba (typically before starting Python). For example,
``NUMBA_NUM_THREADS=2`` limits each process to at most two Numba threads.
``threads=None`` is an inheritance policy, not a resource cap: an unconfigured
worker may inherit Numba's full available thread count. Direct calls to
low-level compiled kernels likewise follow Numba's current setting.


CPU and GPU implementation
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Aspect
     - CPU
     - GPU
   * - Implementation
     - numba ``@njit``
     - numba ``@cuda.jit`` kernels with a Python loop
   * - SOR ordering
     - lexicographic, point by point
     - dependency-safe multi-color sweeps
   * - Parallelism
     - one spatial solve per CPU kernel call
     - grid points of one color update in parallel
   * - Residual
     - maximum accumulated inside the kernel
     - two atomic maxima copied to the host at each check
   * - Data location
     - NumPy host arrays
     - CUDA device arrays during the solve
   * - Batch dimensions
     - vectorized by xarray/Dask
     - vectorized by xarray/Dask; GPU solves are serialized per process

The CPU and GPU algorithms use different update orderings, so their
intermediate iteration paths are not bitwise identical.  They solve the same
discrete equation and use the same residual stopping definition.  Numerical
agreement is covered by the corresponding ``tests/test_Gpu*.py`` tests.  See
:doc:`Benchmark` for measured performance and accuracy.

Multi-color SOR on CPU and GPU
------------------------------

Classical SOR uses newly updated neighbouring values and is therefore serial.
The ``cpu-mc`` and GPU backends partition the stencil dependency graph into
colors. Every point of one color can update concurrently, followed by a
synchronization before the next color:

::

    R B R B R
    B R B R B
    R B R B R

xinvert selects the coloring from the active stencil:

* two colors for five-point 2-D and seven-point 3-D operators;
* four colors when a 2-D mixed derivative couples diagonal neighbours;
* five colors on GPU and nine colors on CPU-mc for the biharmonic stencil,
  which couples points up to two cells away in each direction.

.. important::

   With ``BCs`` specifying ``'periodic'`` in x, the number of x grid points
   must preserve the backend's coloring:

   * Second-order equations: GPU and CPU-mc require an even x length.
   * Biharmonic equations (e.g. Stommel--Munk): GPU requires a multiple of
     five; CPU-mc requires a multiple of three.

   An incompatible length raises ``ValueError`` before SOR iteration begins;
   xinvert does not silently switch backends or change the grid. Choose a
   compatible grid or set ``iParams['architect'] = 'cpu'``, which has no
   coloring restriction. These constraints apply only to periodic x, not
   fixed or extend boundaries. Count the actual x grid points, without
   duplicating the periodic endpoint.

Automatic SOR relaxation factor
-------------------------------

When ``iParams['optArg']`` is ``None``, xinvert estimates `Young's optimal SOR
factor <https://doi.org/10.1016/C2013-0-11733-3>`_ from the spectral radius of
an anisotropic, constant-coefficient Jacobi model:

.. math::

    \rho_J = \frac{\sum_d h_d^{-2}\cos\theta_d}
                   {\sum_d h_d^{-2}}, \qquad
    \omega = \frac{2}{1+\sqrt{1-\rho_J^2}}.

Here :math:`h_d` is the grid spacing in direction :math:`d`, and :math:`N_d`
is the total number of grid points including boundaries.

.. list-table::
   :header-rows: 1

   * - Boundary
     - Mode in a mixed-boundary problem
     - First non-constant mode when all axes are free
   * - ``fixed``
     - :math:`\theta_d=\pi/(N_d-1)`
     - not applicable
   * - ``periodic``
     - :math:`\theta_d=0`
     - :math:`\theta_d=2\pi/N_d`
   * - ``extend``
     - :math:`\theta_d=0`
     - :math:`\theta_d=\pi/(N_d-1)`

When every direction is periodic or extend, the all-constant null mode is
excluded.  Xinvert evaluates the first non-constant mode along each axis and
uses the candidate with the largest :math:`\rho_J`.

For a two-dimensional fixed-boundary grid,

.. math::

    \rho_J =
    \frac{h_y^{-2}\cos(\pi/(N_y-1))+
          h_x^{-2}\cos(\pi/(N_x-1))}
         {h_y^{-2}+h_x^{-2}}.

The finer direction therefore carries more spectral weight.  The expression
is exact for the matching separable constant-coefficient model, but is an
estimate for variable coefficients, masks, mixed derivatives and biharmonic
operators.  A user-supplied ``optArg`` always overrides it.

The automatic value depends on grid geometry and boundary conditions, not on
the forcing.  It is shared across ``time`` or other batch dimensions when the
left-hand-side operator and effective domain do not change.  If coefficients
or masks vary between slices, their true optimal values may also vary.

For spherical latitude-longitude problems, longitude spacing in the actual
operator contains :math:`\cos\phi`.  No single rectangular-grid closed form is
exact at every latitude, so xinvert's value is a nominal global estimate.  A
representative empirical scan may improve repeated production solves.

Residual checks and diagnostics
-------------------------------

CPU and GPU stop on the same normalized, diagonally preconditioned residual.
Checks use an automatic interval unless ``iParams['diag_interval']`` is a
positive integer.  A positive value also returns the sampled history as an
``xarray.Dataset``.  The residual definition, interval behavior, Dataset
layout and plotting examples are documented in :doc:`Diagnostics`.

GPU execution notes
-------------------

The two-dimensional CUDA block defaults to ``(16, 16)`` and can be overridden
per call with ``iParams['gpu_block2d']``.  Compute precision is controlled by
``iParams['dtype']``; ``float32`` is the default and ``float64`` is available
for accuracy-sensitive work.

With Dask-backed input, non-spatial slices remain lazy until ``compute()``,
writing, or plotting is requested.  GPU solves are limited to one at a time
per process because concurrent default-stream solves increase VRAM use without
improving throughput.  Dask can still overlap surrounding I/O and scheduling.

File structure
--------------

::

    xinvert/
    ├── __init__.py      # package exports
    ├── apps.py          # physical equations and coefficient construction
    ├── animate.py       # streaming iteration frames and NetCDF output
    ├── core.py          # validation, xarray bridge and diagnostics
    ├── cpus.py          # numba CPU kernels
    ├── cpu_mc.py        # experimental multi-core multi-color CPU kernels
    ├── gpus.py          # CUDA kernels and GPU iteration loops
    ├── finitediffs.py   # finite-difference utilities
    └── utils.py         # flow diagnostics, experimental MultiGrid and utilities
