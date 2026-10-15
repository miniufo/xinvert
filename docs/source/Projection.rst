Vector-field projection (experimental)
======================================

``xinvert.projection`` is an application module, not a new SOR solver or a
replacement for the user's main program. It constructs a conservative
divergence, calls ``invert_Poisson``, and reconstructs face gradients:

.. math::

   L_h\phi = D_h\mathbf F,\qquad
   \mathbf F_{\rm grad}=G_h\phi,\qquad
   \mathbf F_{\rm sol}=\mathbf F-G_h\phi.

Here the operators satisfy ``L_h = D_h G_h``. The remaining field is
divergence-free to the equation-solving accuracy; harmonic circulation around
islands can remain. This is not a separate extraction of a harmonic component.

Supported grids
---------------

This version accepts one eager 2-D slice on an increasing, uniformly spaced
Cartesian or nonpolar lat-lon grid, with a boolean wet mask and inactive
outer cells along nonperiodic axes. Cartesian coordinates use consistent
length units; lat-lon coordinates use degrees. Latitude rows at the poles
are explicitly rejected. The inactive ring places the
physical closed boundary inside the computational array; it is not a
prescribed potential boundary. Multiple disconnected basins are permitted.

Inputs are **face components**, with the same array shape as the wet mask:
``x_face[j,i]`` is the east face of cell ``[j,i]`` and ``y_face[j,i]`` its
north face. Their coordinate labels are the adjacent cell's labels, not
face-midpoint positions. Values are vector components, not already multiplied
by face length. Wet-wet faces must be finite. Wet-dry faces must contain zero
or NaN; nonzero coastal flux raises an error instead of being silently removed.

Collocated vectors require an explicit conversion to faces before this call.
No universal interpolation is supplied. The x axis may be periodic; on a
lat-lon grid, longitude must cover exactly 360 degrees without a duplicated
endpoint. An even x size is required for CPU-mc and GPU periodic solves.
Open boundaries, original LLC multi-face connectivity, nonuniform grids,
lazy inputs and time batching are not implemented.
For global ocean data this is therefore an initial framework, not yet a
drop-in global projection implementation.

Spherical operators
-------------------

For latitude ``theta``, longitude ``lambda`` and radius ``R``, the face
gradient uses

.. math::

   (G_h\phi)^x_{j,i+1/2}
       = \frac{\phi_{j,i+1}-\phi_{j,i}}{R\cos\theta_j\,\Delta\lambda},
   \qquad
   (G_h\phi)^y_{j+1/2,i}
       = \frac{\phi_{j+1,i}-\phi_{j,i}}{R\,\Delta\theta}.

The conservative divergence uses the matching north/south face metrics:

.. math::

   D_h\mathbf F =
       \frac{F^x_{j,i+1/2}-F^x_{j,i-1/2}}
            {R\cos\theta_j\,\Delta\lambda}
       +\frac{\cos\theta_{j+1/2} F^y_{j+1/2,i}
                   -\cos\theta_{j-1/2} F^y_{j-1/2,i}}
            {R\cos\theta_j\,\Delta\theta}.

Angles in these equations are radians. Blocked faces contribute exactly zero;
periodic faces wrap without changing coordinate labels. These operators
compose to the existing spherical Poisson stencil, including its transformed
RHS ``cos(theta) * divergence``. Compatibility is checked separately for
each wet component, and its potential uses a cosine-weighted zero-mean gauge.
The radius is shared by all three operators via ``mParams={'Rearth': R}``.
This is a discrete-consistency statement, not a claim of exact continuum
derivatives at finite resolution.

Calling it from a main program
------------------------------

.. code-block:: python

   from xinvert.projection import project_vector

   result, diagnostics = project_vector(
       x_face, y_face, wet, dims=['y', 'x'],
       iParams={
           'architect': 'gpu',  # also 'cpu' and 'cpu-mc'
           'dtype': 'float64',
           'mxLoop': 30000,
           'tolerance': 1e-12,
           'diag_interval': 100,
           'printInfo': False,
       },
   )
   phi = result.potential
   fx_remaining = result.x_solenoidal
   fy_remaining = result.y_solenoidal

The module sets internal Neumann and the required outer-array controls.
Conflicting boundary settings are rejected. Without a positive
``diag_interval``, only ``result`` is returned.

For a regular latitude-longitude band with a periodic longitude axis:

.. code-block:: python

   result = project_vector(
       east_face, north_face, wet, ['lat', 'lon'],
       coords='lat-lon', periodic_x=True,
       iParams={'architect': 'gpu', 'dtype': 'float64',
                'tolerance': 1e-12, 'mxLoop': 30000},
   )

The first and last latitude rows must be inactive. For a closed regional
lat-lon crop, keep ``periodic_x=False`` and make all four outer edges inactive.
Closing the crop changes its physical boundary condition: do not do this
silently to real ocean transport crossing the crop perimeter.

The Dataset includes the potential, input divergence, gradient components,
remaining components and ``remaining_divergence``. Component arrays contain
NaN on blocked faces; use zero there when evaluating conservative fluxes.
``remaining_divergence`` evaluates the actual equation defect independently
of the solver's normalized SOR-correction ``error``. Check it rather than
assuming that reaching a stopping tolerance guarantees physical accuracy.
Face differences and spherical metrics are evaluated in float64, including
when input components are stored as float32. The selected ``dtype`` still
controls the Poisson inversion; float64 is the projection default.

A complete manufactured-data main program is available as
:download:`projection_closed_basin.py <../../examples/projection_closed_basin.py>`.
From the repository root, run:

.. code-block:: console

   PYTHONPATH=. python examples/projection_closed_basin.py

Its data preparation belongs to the main program; the decomposition belongs
to ``project_vector``. It performs no file writing unless the caller explicitly
adds it. The function returns data even when ``mxLoop`` is reached, following
normal xinvert behavior; inspect diagnostics and the equation defect.

LLC coastal-mask verification
-----------------------------

:download:`projection_llc_mask_check.py <../../examples/projection_llc_mask_check.py>`
reads a 128-by-128 nonpolar coastal crop from the interpolated LLC4320 SSH
file. SSH supplies only the wet mask; the vector field is manufactured from
a known potential. The crop perimeter is explicitly closed for this test.
It checks potential recovery modulo each component's gauge, and evaluates
the equation defect using an independent spherical stencil.

The source's float32 coordinate labels have rounding noise. This example
explicitly reconstructs nominal uniform float64 labels from the full-grid
endpoints after checking that deviations are within coordinate precision,
and reports the adjustment. It does not modify the source file.
``project_vector`` itself does not silently regularize coordinates: genuinely
nonuniform grids must not be relabelled to force acceptance.

.. code-block:: console

   PYTHONPATH=. python examples/projection_llc_mask_check.py --architect cpu
   PYTHONPATH=. python examples/projection_llc_mask_check.py --architect cpu-mc
   PYTHONPATH=. python examples/projection_llc_mask_check.py --architect gpu

These are correctness checks, not performance benchmarks: reported wall time
includes compilation, and RSS measures the whole process rather than the
solver alone. No real ocean velocity/transport decomposition is claimed.

API
---

.. automodule:: xinvert.projection
   :members: project_vector
