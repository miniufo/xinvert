Finite-difference utilities
===========================

The inversion requires users to provide the forcing and coefficients of
the equation in advance.  Generally, it does not care about how you
calculate all these.  But most of these adopt the finite differencing
of some given fields.  For convenience, we provide the utility
``FiniteDiff``, which computes derivatives for the forcing, equation
coefficients or derived fields. This module provides gradients, divergence,
vorticity and a Laplacian on both ``Cartesian`` and ``lat-lon`` coordinates.

.. code-block:: python

   from xinvert.finitediffs import FiniteDiff

   fd = FiniteDiff(
       {'Y': 'lat', 'X': 'lon'},
       BCs={'Y': 'extend', 'X': 'periodic'},
       coords='lat-lon',
   )
   divergence = fd.divg((u, v), dims=['X', 'Y'])
   vorticity  = fd.curl(u, v)
   dfdy, dfdx = fd.grad(field, dims=['Y', 'X'])
   nablafield = fd.Laplacian(field, dims=['Y', 'X'])

Use axis names ``X`` and ``Y`` in derivative calls; the mapping supplies
their xarray dimension names. Latitude and longitude are in degrees;
spherical derivatives include the radius and latitude metrics.

.. important::
    Sequence of ``dims=['X', 'Y']`` is important as it determines the outputs.
    In Cartesian coordinates, ``fd.divg((u, v), dims=['Y', 'X'])`` computes
    du/dy + dv/dx; on a lat-lon grid, each axis also applies its spherical metric.
    ``fd.grad(field, dims=['X', 'Y'])`` returns dfdx, dfdy.

Derivative boundaries
---------------------

``BCs`` controls ghost values for differentiation, independently of inversion
boundary settings:

* ``'fixed'``: pad with ``fill`` (zero by default).
* ``'extend'``: repeat the endpoint value; this does not force the centered
  derivative at that endpoint to zero.
* ``'reflect'``: mirror the adjacent interior value, ensuring zero derivative at the endpoint.
* ``'periodic'``: wrap values from the opposite end.

One string applies to both ends. A pair specifies the lower-index then
upper-index endpoint:

.. code-block:: python

   fd = FiniteDiff(
       {'Y': 'lat', 'X': 'lon'},
       BCs={'Y': 'extend', 'X': ('extend', 'fixed')},
       fill=0, coords='lat-lon',
   )

For increasing longitude this extends the west endpoint and pads the east
endpoint with zero.

``xinvert.cal_flow`` uses ``FiniteDiff`` to calculate velocity from a
streamfunction or velocity potential. It accepts one boundary string per
dimension (including ``'reflect'`` and periodic y); use ``FiniteDiff``
directly when different endpoint rules are needed.


.. note::
    These are differentiation rules, not applicable for an inversion.
    The underlying implementation is based on ``xarray``'s API. For centered first
    derivatives, ``FiniteDiff`` adds one ghost point at each
    endpoint with `DataArray.pad
    <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.pad.html>`_,
    calls `DataArray.differentiate
    <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.differentiate.html>`_,
    then removes the ghost points. It adds boundary handling and spherical
    metrics to xarray's coordinate derivative.
