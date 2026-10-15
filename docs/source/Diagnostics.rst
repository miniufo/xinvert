Diagnostic analysis
===================

Xinvert can return the convergence history of every independent inversion as
an ``xarray.Dataset``.  The single ``diag_interval`` parameter both enables this
output and specifies its sampling interval in SOR sweeps.

Enabling diagnostics
--------------------

The default ``diag_interval=0`` returns only the solution and leaves convergence
checking on xinvert's automatic schedule.  Any non-positive integer disables
diagnostic output; a positive integer records the
residual at that interval and changes the return value to
``(solution, diagnostics)``::

    solution, diagnostics = invert_Poisson(
        forcing,
        dims=['lat', 'lon'],
        iParams={
            'tolerance': 1e-6,
            'diag_interval': 20,
        },
    )

Here convergence is checked and recorded after iterations 20, 40, 60, and so
on.  The last iteration is always checked, even when ``mxLoop`` is not an exact
multiple of ``diag_interval``.  A smaller interval gives a denser curve and may
detect convergence earlier, but requires more reductions and, on GPU, more
host-device synchronisations.

Residual and check interval
---------------------------

Xinvert uses the normalized maximum SOR correction as its stopping residual:

.. math::

    r_D = \frac{\max_i |\Delta S_i|}
                {\max_i\max(|S_i^{old}|, |S_i^{new}|)}.

Because an SOR correction satisfies
:math:`\Delta S_i=\omega D_{ii}^{-1}(L(S)-F)_i`, this is a diagonally
preconditioned equation residual.  If the solution scale is zero, xinvert uses
the absolute maximum correction instead.

When ``diag_interval <= 0``, residual checks follow an automatic schedule of
approximately ``mxLoop / 50``, limited to 10--100 sweeps.

When ``diag_interval > 0``, that value is both the residual-check interval and
the diagnostic sampling interval.  The final sweep is always checked.  CPU and
GPU use the same residual definition; on GPU, more frequent checks require
more device-to-host synchronisations.

Diagnostics layout
------------------

For the two-time atmospheric Poisson example used below, ``print(diagnostics)``
produces output of this form::

    <xarray.Dataset> Size: 14kB
    Dimensions:      (time: 2, iteration: 250)
    Coordinates:
      * time         (time) datetime64[ns] 16B 2008-01-01 2008-01-02
      * iteration    (iteration) int64 2kB 20 40 60 ... 4960 4980 5000
    Data variables:
        converged    (time) bool 2B True True
        iterations   (time) int64 16B 420 440
        error        (time, iteration) float64 4kB 0.06304 ... nan nan
        solution_min (time, iteration) float64 4kB ... nan nan
        solution_max (time, iteration) float64 4kB ... nan nan
        final_error  (time) float64 16B 8.66e-07 9.245e-07
        stop_reason  (time) object 16B 'converged' 'converged'
    Attributes:
        tolerance:       1e-06
        max_iterations:  5000
        diag_interval:   20

The ``iteration`` coordinate gives the SOR sweep at which each residual was
sampled.  ``error`` holds those samples; entries after an individual slice has
stopped are ``NaN``.  ``iterations`` and ``final_error`` provide the last valid
state directly, while ``converged`` and ``stop_reason`` summarize why each
solve ended.

``solution_min`` and ``solution_max`` record the signed minimum and maximum
of the current solution at the same checkpoints as ``error``. They include
valid fixed-boundary cells and exclude forcing-mask cells, undefined values
and non-finite values. An empty valid domain yields ``NaN``. Like ``error``,
entries after a slice stops remain ``NaN``; all batch dimensions are retained.
These variables are diagnostic only and do not change the stopping criterion.
No percentiles or outlier removal are applied.

CPU and CPU-mc scan the solution only at diagnostic checkpoints. GPU reduces
the extrema on device and transfers only two scalars, not the full field.
This adds work when diagnostics are enabled; ``diag_interval=0`` does not
compute or store extrema.

All non-inversion dimensions are retained.  An input
``forcing(member, time, lat, lon)`` therefore produces
``error(member, time, iteration)``.  A single ``(lat, lon)`` input has only the
``iteration`` dimension; xinvert does not invent a ``time`` dimension.

Plotting and derived convergence rates
--------------------------------------

The raw residual history is usually the most useful quantity to plot::

    diagnostics.error.plot.line(
        x='iteration', hue='time', yscale='log',
    )

.. figure:: _static/diagnostic_residual.png
   :width: 720px
   :align: center
   :alt: Residual histories for two atmospheric Poisson inversions

   Residual histories from ``Data/Helmholtz_atmos.nc`` using the CPU backend,
   ``diag_interval=20`` and ``tolerance=1e-6``.  Each time slice stops
   independently when its sampled residual falls below the dashed tolerance.

A ratio between successive samples is one possible convergence-rate measure::

    rate = diagnostics.error / diagnostics.error.shift(iteration=1)

Xinvert does not store a separate convergence-rate variable because several
definitions are useful in practice and each can be derived from ``error``.

Streaming iteration frames
--------------------------

``animate_iteration`` writes full solution snapshots synchronously, rather
than returning an in-memory or Dask-backed frame array. Specify a new output
file; an existing path raises ``FileExistsError``::

    from xinvert import animate_iteration
    import xarray as xr

    animate_iteration(
        'Poisson', forcing, dims=['lat', 'lon'],
        iParams={'architect': 'gpu'},
        loop_per_frame=20, max_frames=100, output='iteration.nc',
    )
    frames = xr.open_dataset('iteration.nc', chunks={'iter': 1})

Use ``complevels=1`` through ``9`` to losslessly compress the ``inverted``
variable with zlib; ``complevels=0`` (the default) disables compression.
For example, add ``complevels=4`` to the call above. The public parameter
maps to the standard NetCDF ``complevel`` setting; with xarray's
``to_netcdf`` the equivalent configuration is
``encoding={'inverted': {'zlib': True, 'complevel': 4}}``.
Coordinates are not compressed by this option.

The call executes immediately. By default it returns ``None``; explicitly
set ``iParams['diag_interval']`` to a positive integer to return scalar
diagnostics instead of a frame array::

    diagnostics = animate_iteration(
        'Poisson', forcing, dims=['lat', 'lon'],
        iParams={'architect': 'gpu', 'diag_interval': 10},
        loop_per_frame=20, max_frames=100, output='diagnosed_iteration.nc',
    )
    diagnostics.error.plot.line(x='iteration', yscale='log')
    print(diagnostics.attrs['completed_frames'])

The diagnostic interval is independent of the frame interval. Error and
signed solution extrema are sampled at that interval and at the final sweep;
GPU diagnostics do not download a complete frame. The returned Dataset uses
the same layout as inversion diagnostics and retains singleton batch
coordinates. Because animation disables convergence stopping, its effective
tolerance is zero, ``converged`` is false and ``stop_reason`` is
``'max_iterations'``; these describe fixed work, not a proof of inaccuracy.
Diagnostics are returned in memory, not automatically saved to the frame file.
Their scalar history grows with the number of checkpoints, so a very small
interval over an extremely long run still has a memory cost.

``completed_frames`` and ``status`` are always attributes of the output file;
they are also attributes of the optional returned diagnostics. For example::

    with xr.open_dataset('iteration.nc') as frames:
        count = frames.attrs['completed_frames']
        if count > 0:
            last_complete = frames.inverted.isel(iter=count - 1)

Each frame represents
exactly ``loop_per_frame`` additional sweeps; early convergence is disabled.
The file variable ``inverted`` has an ``iter`` dimension whose coordinate
counts cumulative SOR sweeps. Other dimensions and coordinates are preserved.
CPU, CPU-mc and GPU use the same sequential output workflow. GPU uploads its
solution and coefficients once and reuses a resident CUDA workspace until
the call ends. Only one snapshot is downloaded at a time.

The numerical workspace and live frame buffers do not grow with frame count.
Each frame is written and flushed before the next is computed; no temporary
files are created. A complete spatial problem and its coefficient/work arrays
must still fit in RAM and, for GPU, VRAM. Disk capacity and memory used by
other processes remain external constraints.

If a solve or write fails, the file is closed without removing earlier
frames. Inspect ``status`` and ``completed_frames`` attributes, and use only
``inverted.isel(iter=slice(0, completed_frames))``: an interrupted write may
leave an incomplete trailing frame. This is a progress record, not an
automatic restart/checkpoint format. Calling ``load()`` on the saved complete
history can still exhaust RAM; read selected frames or operate lazily.

``diag_interval`` versus ``printInfo``
--------------------------------------

Diagnostics are structured, retained after the solve, and suitable for plots,
filtering, serialization, and comparisons across time or ensemble members.
They become available when the inversion result is computed.

``printInfo=True`` instead emits one short text line as each independent solve
finishes.  It does not preserve the iteration history, but can provide an
immediate completion indication during a long batched or Dask calculation.
It is enabled by default; set ``printInfo=False`` when quiet execution or only
structured diagnostics are desired.  The two mechanisms are independent and
may be enabled together when both behaviours are useful.
