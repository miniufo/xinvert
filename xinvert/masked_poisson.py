"""Masked diagonal 2-D divergence operators with optional reaction terms.

Conductances at closed Neumann faces are removed from both flux and
diagonal. The default fixed mode continues through the old solver. Its
unmasked application path is deliberately unchanged.
"""
import numpy as np
import xarray as xr
from numba import njit, prange

from ._threads import DEFAULT_THREADS, numba_threads
from .core import (
    _apply_solver,
    _get_info,
    _print_live,
    _undeftmp,
    _validate_inversion_dims,
)
from .cpus import _check_interval
from .domain import _label_components
from .internal_bcs import _encode_internal_faces


def invert_masked_poisson(F, dims, coords, icbc, mParams, iParams):
    """Infer a mask per slice and solve via the normal application template.

    Parameters follow invert_Poisson. internal_bcs='neumann' is selected
    through iParams; users do not supply a topology object.
    Returns the normal DataArray or (DataArray, diagnostics Dataset), with
    dask batch laziness preserved. Different slices may have different masks.
    """
    from . import apps

    if coords.lower() not in ('cartesian', 'lat-lon'):
        raise NotImplementedError("internal Poisson supports Cartesian and nonpolar lat-lon grids")
    return invert_masked_standard(
        F, dims, coords, icbc, mParams, iParams,
        apps.__coeffs_Poisson, ['g', 'Omega', 'Rearth'], False)


def invert_masked_standard(F, dims, coords, icbc, mParams, iParams,
                           builder, valid_params, full=False):
    """Solve a diagonal divergence-form 2-D operator with optional reaction.

    F/dims/coords/icbc/mParams/iParams follow the public application. builder
    is its coefficient factory; valid_params is the model-parameter whitelist.
    full selects (A,B,C,D,E) rather than (A,B,C). Nonzero mixed coefficients
    are rejected. Reaction must be nonpositive after a uniform diffusion-sign
    normalization per component; indefinite/degenerate graphs are rejected.
    Returns solution or (solution, diagnostics), preserving Dask batching.
    """
    from . import apps

    architect = str((iParams or {}).get('architect', 'cpu')).strip().lower()
    if architect not in ('cpu', 'cpu-mc', 'gpu'):
        raise ValueError("architect must be 'cpu', 'cpu-mc', or 'gpu'")
    if coords.lower() not in ('cartesian', 'lat-lon', 'z-lat', 'z-lon'):
        raise NotImplementedError("unsupported coordinates for masked 2-D divergence operator")
    dims = tuple(_validate_inversion_dims(F, dims, 2))

    def coefficients(force, dims, coords, mp, ip, initial):
        """Build native Poisson coefficients and topology-controlled forcing mask."""
        bcs = ip['BCs']
        if bcs[0] == 'periodic':
            raise NotImplementedError("periodic y is not implemented for masked Poisson")
        if any(force.sizes[d] < 3 for d in dims):
            raise ValueError("masked Poisson requires at least three grid points per axis")
        if coords.lower() == 'lat-lon' and np.any(np.abs(force[dims[0]].values) >= 90):
            raise ValueError("masked lat-lon Poisson excludes pole/singular latitude rows")
        if np.isfinite(ip['undef']):
            force = force.where(force != ip['undef'])
        force = force.where(np.isfinite(force))
        # NaN is the temporary topology mask regardless of the public sentinel.
        tmp = dict(ip, undef=np.nan)
        rhs, solution, coeffs = builder(force, dims, coords, mp, tmp, initial)
        rhs = rhs.where(np.isfinite(force), _undeftmp)
        coeffs = tuple(coef if coef.dtype == solution.dtype else coef.astype(solution.dtype)
                       for coef in coeffs)
        return rhs, solution, coeffs

    def solve(*arguments):
        """Prepare static unknown graph and wrap one-slice compiled SOR."""
        *coefs, rhs, solution, dims, ip = arguments
        if full:
            A, mixed_y, mixed_x, C, reaction = coefs
        else:
            A, mixed_y, C = coefs
            mixed_x = mixed_y
            reaction = xr.DataArray(np.array(0., dtype=solution.dtype))
        if architect in ('cpu-mc', 'gpu') and ip['BCs'][1] == 'periodic' and rhs.sizes[dims[1]] % 2:
            raise ValueError("red-black masked Poisson requires an even periodic x grid size")
        if architect == 'gpu':
            from . import masked_poisson_gpu
            masked_poisson_gpu.ensure_context()
        weight = (np.cos(np.deg2rad(rhs[dims[0]].values))
                  if coords.lower() == 'lat-lon' else np.ones(rhs.sizes[dims[0]]))
        info, _ = _get_info(rhs, dims)
        source_active = np.isfinite(F) & (F != ip['undef'])
        interval = max(int(ip['diag_interval']), 0)
        histories = (ip['mxLoop'] + interval - 1) // interval if interval else 0

        def kernel(s, a, c, e, by, bx, f, active, label_info):
            """Validate a batch slice, check compatibility and return SOR output."""
            if not active.any():
                raise ValueError(f"{label_info}: internal_bcs domain must contain active cells")
            unknown = active.copy()
            unknown[[0, -1], :] = False
            if ip['BCs'][1] != 'periodic':
                unknown[:, [0, -1]] = False
            labels, count = _label_components(unknown, (False, ip['BCs'][1] == 'periodic'))
            codes = _encode_internal_faces(active, (False, ip['BCs'][1] == 'periodic'))
            links, anchored = _prepare_graph(
                active, codes, unknown, labels, count, ip['BCs'][0] == 'fixed',
                ip['BCs'][1] == 'fixed', ip['BCs'][1] == 'periodic', False)
            del unknown, codes
            if np.any(~np.isfinite(f[active])) or np.any(f[active] == _undeftmp):
                raise ValueError(f"{label_info}: forcing must be finite on every active cell")
            if np.any(~np.isfinite(s[active])) or _invalid_neighbours(s, labels, links):
                raise ValueError(f"{label_info}: icbc/fixed values must be finite")
            b = np.zeros_like(s)
            np.multiply(f, ip['del1Sqr'], out=b, where=active, casting='unsafe')
            if not np.isfinite(b[active]).all():
                raise ValueError("scaled equation RHS exceeds the selected compute dtype")
            q = np.zeros_like(s) if full else np.zeros((1, 1), dtype=s.dtype)
            if full:
                np.multiply(e, ip['del1Sqr'], out=q, where=active, casting='unsafe')
            if (np.any(~np.isfinite(by[active])) or np.any(~np.isfinite(bx[active])) or
                    np.any(by[active] != 0) or np.any(bx[active] != 0)):
                raise NotImplementedError(
                    "internal Neumann for nonzero mixed-derivative coefficients "
                    "requires a normal-flux boundary derivation; coefficients are not ignored")
            status = _classify_operator(a, c, q, labels, links, anchored, ip['ratioSqr'])
            if status == 1:
                raise ValueError(
                    "active face coefficients must be finite and nonzero; reaction coefficients must be finite")
            if status == 2:
                raise NotImplementedError(
                    "diffusion must have one consistent nonzero sign per component")
            if status == 3:
                raise NotImplementedError(
                    "reaction must be nonpositive after diffusion-sign normalization; "
                    "indefinite Helmholtz operators need separate solvability handling")
            sums, absolute = _component_rhs(b, labels, count)
            threshold = 64 * np.finfo(s.dtype).eps
            for component in range(1, count + 1):
                if not anchored[component] and abs(sums[component]) > threshold * absolute[component]:
                    mean = sums[component] / max(1, np.count_nonzero(labels == component))
                    raise ValueError(
                        f"{label_info}: component {component} has incompatible forcing: "
                        f"scaled RHS sum={sums[component]:.6g}, mean={mean:.6g}. "
                        "A closed zero-flux diffusion component requires zero sum of its assembled RHS; "
                        "check the application's transformed RHS/model parameters, "
                        "correct the equation forcing explicitly or provide a fixed boundary.")
            flags = np.full(3 + 3 * histories, np.nan)
            out = np.array(s, copy=True, order='C')
            arguments = (out, a, c, b, q, active, labels, links, anchored, weight,
                         ip['ratioSqr'], ip['optArg'], ip['mxLoop'], ip['tolerance'],
                         interval, flags, ip['BCs'][0] == 'extend', ip['BCs'][1] == 'extend',
                         ip['BCs'][1] == 'periodic')
            if architect == 'gpu':
                from .masked_poisson_gpu import sor_gpu
                sor_gpu(*arguments, block=ip.get('gpu_block2d'))
            else:
                with numba_threads(architect, ip.get('threads', DEFAULT_THREADS)):
                    _sor(*arguments, multicore=architect == 'cpu-mc')
            if ip['printInfo']:
                _print_live(f"{label_info} loops {flags[2]:4.0f}, residual is {flags[1]:e}")
            return (out, flags) if interval else out

        return _apply_solver(kernel, [solution, A, C, reaction, mixed_y, mixed_x,
                                     rhs, source_active, info],
                             [dims, dims, dims, dims if full else [], dims, dims,
                              dims, dims, []], dims, solution, ip)

    coefficients._supports_internal_bcs = True
    result = apps.__template(coefficients, solve, 2, F, list(dims), coords,
                             icbc, valid_params, mParams, iParams)
    if isinstance(result, tuple):
        result[1].attrs.update(internal_boundary='neumann',
                               gauge='component weighted zero mean for unanchored components')
    return result


@njit(cache=True)
def _prepare_graph(active, codes, unknown, labels, count, fixed_y, fixed_x, periodic_x, fixed_land):
    """Return uint8 allowed-face links and component anchor flags.

    Input boolean grids and int32 labels have identical (y,x) shape. Fixed
    boundary cells are not unknowns; extend edge cells are eliminated ghosts.
    """
    ny, nx = active.shape
    links = np.zeros(active.shape, dtype=np.uint8)
    anchored = np.zeros(count + 1, dtype=np.bool_)
    for y in range(ny):
        for x in range(nx):
            label = labels[y, x]
            if not label:
                continue
            for direction in range(4):
                if not fixed_land and codes[y, x] & (1 << direction):
                    continue
                yy = y + (1 if direction == 3 else -1 if direction == 2 else 0)
                xx = x + (1 if direction == 0 else -1 if direction == 1 else 0)
                if periodic_x:
                    xx %= nx
                if not (0 <= yy < ny and 0 <= xx < nx):
                    continue
                # Array-edge ghost/fixed classification precedes land handling.
                edge_y = yy == 0 or yy == ny - 1
                edge_x = not periodic_x and (xx == 0 or xx == nx - 1)
                if (edge_y and not fixed_y) or (edge_x and not fixed_x):
                    continue
                permitted = unknown[yy, xx] or (
                    active[yy, xx] and (edge_y or edge_x)) or fixed_land
                if permitted:
                    links[y, x] |= 1 << direction
                    if not unknown[yy, xx]:
                        anchored[label] = True
    return links, anchored


@njit(cache=True)
def _component_rhs(rhs, labels, count):
    """Accumulate float64 signed/absolute scaled RHS sums by unknown component."""
    sums = np.zeros(count + 1, dtype=np.float64)
    absolute = np.zeros(count + 1, dtype=np.float64)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label:
                sums[label] += rhs[y, x]
                absolute[label] += abs(rhs[y, x])
    return sums, absolute


@njit(cache=True)
def _classify_operator(a, c, q, labels, links, anchored, ratio):
    """Validate signed ellipticity/coercivity; mark reactive components anchored.

    a/c are y/x face coefficients, q is the scaled zero-order term. Each
    allowed face must be finite/nonzero, with a uniform sign per component.
    Isolated components use the reaction sign. Return 0 on success, 1 for
    invalid coefficients, 2 for sign changes, 3 for noncoercive reaction.
    anchored is modified only when a reaction removes the constant nullspace.
    """
    signs = np.zeros(anchored.size, dtype=np.int8)
    ny, nx = labels.shape
    for y in range(ny):
        for x in range(nx):
            label = labels[y, x]
            if not label:
                continue
            reaction = q[0, 0] if q.size == 1 else q[y, x]
            if not np.isfinite(reaction):
                return 1
            for direction in range(4):
                if not (links[y, x] & (1 << direction)):
                    continue
                yy = y + (1 if direction == 3 else -1 if direction == 2 else 0)
                xx = (x + (1 if direction == 0 else -1 if direction == 1 else 0)) % nx
                g = (c[y, xx] if direction == 0 else c[y, x] if direction == 1
                     else a[y, x]*ratio if direction == 2 else a[yy, x]*ratio)
                if not np.isfinite(g) or g == 0:
                    return 1
                sign = 1 if g > 0 else -1
                if signs[label] and signs[label] != sign:
                    return 2
                signs[label] = sign
    for y in range(ny):
        for x in range(nx):
            label = labels[y, x]
            if not label:
                continue
            reaction = q[0, 0] if q.size == 1 else q[y, x]
            if reaction != 0:
                sign = signs[label] if signs[label] else (-1 if reaction > 0 else 1)
                if reaction*sign > 0:
                    return 3
                anchored[label] = True
    return 0


@njit(cache=True)
def _invalid_neighbours(s, labels, links):
    """Whether an allowed stencil read uses a nonfinite fixed/initial value."""
    ny, nx = s.shape
    for y in range(ny):
        for x in range(nx):
            if not labels[y, x]:
                continue
            for direction in range(4):
                if links[y, x] & (1 << direction):
                    yy = y + (1 if direction == 3 else -1 if direction == 2 else 0)
                    xx = (x + (1 if direction == 0 else -1 if direction == 1 else 0)) % nx
                    if not np.isfinite(s[yy, xx]):
                        return True
    return False


@njit(cache=True)
def _offsets(s, labels, anchored, weights):
    """Return per-component weighted solution means; anchored offsets are zero."""
    offsets = np.zeros(anchored.size)
    totals = np.zeros(anchored.size)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label and not anchored[label]:
                offsets[label] += weights[y] * s[y, x]
                totals[label] += weights[y]
    for label in range(1, anchored.size):
        if totals[label]:
            offsets[label] /= totals[label]
    return offsets


@njit(cache=True)
def _extend(s, active, extend_y, extend_x, periodic_x):
    """Fill eliminated exterior ghost rows/columns from adjacent active cells."""
    ny, nx = s.shape
    if extend_y:
        for x in range(0 if periodic_x else 1, nx if periodic_x else nx - 1):
            if active[0, x] and active[1, x]:
                s[0, x] = s[1, x]
            if active[-1, x] and active[-2, x]:
                s[-1, x] = s[-2, x]
    if extend_x:
        for y in range(1, ny - 1):
            if active[y, 0] and active[y, 1]:
                s[y, 0] = s[y, 1]
            if active[y, -1] and active[y, -2]:
                s[y, -1] = s[y, -2]
    if extend_y and extend_x:
        for y in (0, ny - 1):
            for x in (0, nx - 1):
                yy = 1 if y == 0 else ny - 2
                xx = 1 if x == 0 else nx - 2
                if active[y, x] and active[yy, xx]:
                    s[y, x] = s[yy, xx]


@njit(nogil=True)
def _sor(s, a, c, rhs, q, active, labels, links, anchored, weights,
         ratio, omega, loops, tolerance, requested, flags, extend_y, extend_x, periodic_x,
         multicore=False):
    """Mutate a slice using masked five-point SOR and existing diagnostic layout.

    This dispatcher is compiled per process: persisted code that references
    the parallel sweep dispatcher can crash when reloaded in another process.
    Its numerical helpers retain their ordinary compilation/cache policies.

    a/c are native y/x face coefficients, rhs and reaction q include dx^2.
    The SOR denominator is sum(conductances)-q; uniformly negative diffusion
    is handled algebraically without allocating sign-normalized copies. Static uint8
    links avoid reading closed faces. All unknown points retain their equations.
    Unanchored components use weighted means for checkpoint metrics and final
    output, without pinning a point. weights do not define RHS compatibility. flags stores
    overflow/error/iterations and optional error/min/max checkpoint histories.
    """
    ny, nx = s.shape
    interval = _check_interval(loops, tolerance, requested)
    row_updates = np.empty(ny, dtype=np.float64)
    history = (loops + requested - 1) // requested if requested else 0
    index = 0
    error = 0.0
    overflow = False
    for iteration in range(1, loops + 1):
        check = iteration % interval == 0 or iteration == loops
        max_update = 0.0
        if multicore:
            for color in range(2):
                max_update = max(max_update, _sweep_mc(
                    s, a, c, rhs, q, labels, links, ratio, omega, color, check,
                    row_updates))
        else:
            for y in range(ny):
                for x in range(nx):
                    update = _update_point(s, a, c, rhs, q, labels, links,
                                           ratio, omega, y, x)
                    if check:
                        max_update = max(max_update, abs(update))
        if not check:
            continue
        offsets = _offsets(s, labels, anchored, weights)
        minimum, maximum = np.inf, -np.inf
        scale = 0.0
        for y in range(ny):
            for x in range(nx):
                if active[y, x]:
                    yy, xx = y, x
                    if extend_y and y in (0, ny-1) and (periodic_x or 0 < x < nx-1 or extend_x):
                        yy = 1 if y == 0 else ny-2
                    if extend_x and x in (0, nx-1) and (0 < y < ny-1 or extend_y):
                        xx = 1 if x == 0 else nx-2
                    if not active[yy, xx]:
                        yy, xx = y, x
                    value = s[yy, xx] - offsets[labels[yy, xx]]
                    if not np.isfinite(value):
                        overflow = True
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    scale = max(scale, abs(value))
        error = max_update / scale if scale > 0 else max_update
        overflow = overflow or not np.isfinite(error) or error > 1e100
        if overflow:
            error = np.inf
        if requested:
            flags[3 + index] = error
            flags[3 + history + index] = minimum
            flags[3 + 2 * history + index] = maximum
            index += 1
        if overflow or (tolerance > 0 and error < tolerance):
            break
    offsets = _offsets(s, labels, anchored, weights)
    for y in range(ny):
        for x in range(nx):
            if labels[y, x]:
                s[y, x] -= offsets[labels[y, x]]
    _extend(s, active, extend_y, extend_x, periodic_x)
    flags[0], flags[1], flags[2] = overflow, error, iteration


@njit(inline='always')
def _update_point(s, a, c, rhs, q, labels, links, ratio, omega, y, x):
    """Update one eligible point in place; return signed SOR correction."""
    nx = s.shape[1]
    label = labels[y, x]
    if not label:
        return 0.0
    diagonal, flux = 0.0, 0.0
    for direction in range(4):
        if not (links[y, x] & (1 << direction)):
            continue
        yy = y + (1 if direction == 3 else -1 if direction == 2 else 0)
        xx = (x + (1 if direction == 0 else -1 if direction == 1 else 0)) % nx
        conductance = (c[y, xx] if direction == 0 else c[y, x]
                       if direction == 1 else a[y, x] * ratio
                       if direction == 2 else a[yy, x] * ratio)
        diagonal += conductance
        flux += conductance * (s[yy, xx] - s[y, x])
    reaction = q[0, 0] if q.size == 1 else q[y, x]
    diagonal -= reaction
    if diagonal == 0:
        return 0.0
    update = omega * (flux + reaction*s[y, x] - rhs[y, x]) / diagonal
    s[y, x] += update
    return update


@njit(cache=True, parallel=True, nogil=True)
def _sweep_mc(s, a, c, rhs, q, labels, links, ratio, omega, color, check,
              row_updates):
    """Parallel half-grid scan for one color; return maximum correction.

    Same-color axial neighbours are absent, so rows are dependency-safe.
    Completing this call is the red/black barrier. Periodic x must be even.
    """
    for row in prange(s.shape[0]):
        y = np.int64(row)
        row_maximum = 0.0
        for x in range((color - y) % 2, s.shape[1], 2):
            update = _update_point(s, a, c, rhs, q, labels, links,
                                   ratio, omega, y, x)
            if check:
                row_maximum = max(row_maximum, abs(update))
        row_updates[y] = row_maximum
    return np.max(row_updates)
