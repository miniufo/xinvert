Boundary conditions
===================

xinvert has two related, but deliberately different, boundary-condition
interfaces:

* inversion routines use one scalar string per inversion dimension;
* :class:`xinvert.finitediffs.FiniteDiff` may specify the two endpoints of a
  dimension separately.

Inversion interface
-------------------

Pass ``iParams={'BCs': [...]}`` in the same order as ``dims``.  For example::

    psi = invert_Poisson(
        forcing,
        dims=['lat', 'lon'],
        iParams={'BCs': ['extend', 'periodic']},
    )

Each entry must be one of ``'fixed'``, ``'extend'``, or ``'periodic'``.
Capitalization and surrounding whitespace are normalized.  A two-dimensional
inversion therefore requires exactly two strings, and a three-dimensional
inversion requires exactly three strings.

``fixed``
    Keep the prescribed boundary value from ``icbc`` (or the initialized
    value when ``icbc`` is omitted).

``extend``
    Copy the adjacent interior value onto both boundaries of that dimension.
    The x and y dimensions are updated independently.  A corner is extended
    diagonally only when both adjoining dimensions use ``extend``; if either
    dimension is ``fixed``, that fixed corner is preserved.

``periodic``
    Wrap the stencil across the two ends of the dimension.  Periodic inversion
    is currently implemented only for x, the last inversion dimension.

Current support
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 35 30

   * - Dimension
     - Implemented
     - Accepted but not implemented
     - Behaviour when not implemented
   * - x (last)
     - ``fixed``, ``extend``, ``periodic``
     - none
     - --
   * - y
     - ``fixed``, ``extend``
     - ``periodic``
     - emits ``RuntimeWarning``; the requested periodic update is not applied
   * - z (3-D)
     - ``fixed``
     - ``extend``, ``periodic``
     - emits ``RuntimeWarning``; the requested z update is not applied

The not-yet-implemented values remain accepted so that the public interface
does not need to change when their numerical kernels are added.  Do not rely on
their numerical result until the warning is removed in a future release.

Initial guesses and missing values
----------------------------------

When supplied, ``icbc`` is used as the initial guess on every valid grid point,
not only on fixed boundaries.  It may omit non-inversion dimensions: for
example, a ``(lat, lon)`` initial guess is broadcast over the ``time`` dimension
of a ``(time, lat, lon)`` forcing array.  All shared coordinates must match the
forcing exactly.

Grid points where the forcing is undefined ignore the corresponding ``icbc``
value and remain zero in the internal solution.  The forcing sentinel prevents
SOR updates there, and the final output is restored to the user-selected
``undef`` value.

Asymmetric endpoints belong to FiniteDiff
-----------------------------------------

The SOR inversion interface does **not** accept an endpoint dictionary or a
two-element endpoint pair inside ``iParams['BCs']``.  Such inputs raise a
``ValueError`` instead of being interpreted ambiguously.

For differentiation, :class:`xinvert.finitediffs.FiniteDiff` supports separate
lower-index and upper-index endpoints::

    fd = FiniteDiff(
        {'Y': 'lat', 'X': 'lon'},
        BCs={'Y': 'fixed', 'X': ('extend', 'fixed')},
        coords='lat-lon',
    )

For ``X``, the pair is ordered lower-index then upper-index, conventionally
west then east.  Thus the example extends the west boundary and fixes the east
boundary.  The same lower/upper ordering applies to other dimensions.

``cal_flow`` follows the simple interface
-----------------------------------------

:func:`xinvert.apps.cal_flow` also accepts exactly one boundary-condition
string per dimension and applies it to both endpoints.  It supports the four
conditions implemented by ``FiniteDiff``: ``'fixed'``, ``'extend'``,
``'reflect'``, and ``'periodic'``.  Unlike the SOR inversion routines, a
periodic y boundary is fully implemented here and does not emit the pending
kernel warning.

To use different endpoint conditions when calculating a derivative, construct
``FiniteDiff`` directly as in the example above.  Passing an endpoint pair or
dictionary to ``cal_flow`` raises ``ValueError`` rather than silently changing
its meaning.
