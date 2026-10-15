"""Resident-device red-black SOR for masked diffusion/reaction equations.

Static graph/gauge compatibility is prepared on the host. The solution,
coefficients and reduction workspaces remain on the device for all sweeps;
only four scalars per checkpoint and the final solution are downloaded.
"""
import math
from numbers import Integral

import numpy as np
from numba import cuda, njit

from .cpus import _check_interval
from .gpus import _get_gpu_sem, _launch_config_2d, ensure_context


@cuda.jit
def _sweep(s, a, c, rhs, q, labels, links, ratio, omega, color, check, metric):
    """Update one red/black half-grid point; optionally reduce corrections."""
    y, half_x = cuda.grid(2)
    x = 2 * half_x + ((color - y) % 2)
    ny, nx = s.shape
    if y >= ny or x >= nx:
        return
    label = labels[y, x]
    if not label:
        return
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
    if diagonal != 0:
        delta = omega * (flux + reaction*s[y, x] - rhs[y, x]) / diagonal
        s[y, x] += delta
        if check:
            if math.isfinite(delta):
                cuda.atomic.max(metric, 0, abs(delta))
            else:
                cuda.atomic.max(metric, 3, 1.)


@cuda.jit
def _reset(metric, offsets):
    """Clear reusable metric and component mean workspaces on the device."""
    index = cuda.grid(1)
    if index < offsets.size:
        offsets[index] = 0.
    if index == 0:
        metric[0], metric[1], metric[2], metric[3] = 0., math.inf, -math.inf, 0.


@cuda.jit
def _means(s, labels, anchored, weights, offsets):
    """Accumulate weighted component sums without downloading a full field."""
    y, x = cuda.grid(2)
    if y < s.shape[0] and x < s.shape[1]:
        label = labels[y, x]
        if label and not anchored[label]:
            cuda.atomic.add(offsets, label, weights[y] * s[y, x])


@cuda.jit
def _divide_means(offsets, totals):
    """Convert component sums into means using static host-prepared weights."""
    index = cuda.grid(1)
    if index < offsets.size and totals[index] > 0:
        offsets[index] /= totals[index]


@cuda.jit(device=True)
def _source(y, x, ny, nx, extend_y, extend_x, periodic_x):
    """Return the solved-cell source for an eliminated exterior ghost cell."""
    yy, xx = y, x
    if extend_y and (y == 0 or y == ny-1) and (periodic_x or 0 < x < nx-1 or extend_x):
        yy = 1 if y == 0 else ny-2
    if extend_x and (x == 0 or x == nx-1) and (0 < y < ny-1 or extend_y):
        xx = 1 if x == 0 else nx-2
    return yy, xx


@cuda.jit
def _extrema(s, active, labels, offsets, metric, extend_y, extend_x, periodic_x):
    """Reduce active output extrema in the final component gauge."""
    y, x = cuda.grid(2)
    ny, nx = s.shape
    if y >= ny or x >= nx or not active[y, x]:
        return
    yy, xx = _source(y, x, ny, nx, extend_y, extend_x, periodic_x)
    if not active[yy, xx]:
        yy, xx = y, x
    value = s[yy, xx] - offsets[labels[yy, xx]]
    if math.isfinite(value):
        cuda.atomic.min(metric, 1, value)
        cuda.atomic.max(metric, 2, value)
    else:
        cuda.atomic.max(metric, 3, 1.)


@cuda.jit
def _shift(s, labels, offsets):
    """Fix the additive constant only once, after all SOR sweeps."""
    y, x = cuda.grid(2)
    if y < s.shape[0] and x < s.shape[1] and labels[y, x]:
        s[y, x] -= offsets[labels[y, x]]


@cuda.jit
def _fill_edges(s, active, extend_y, extend_x, periodic_x):
    """Write eliminated edges after shifting; interior sources are immutable."""
    y, x = cuda.grid(2)
    ny, nx = s.shape
    if y >= ny or x >= nx or not active[y, x]:
        return
    yy, xx = _source(y, x, ny, nx, extend_y, extend_x, periodic_x)
    if (yy != y or xx != x) and active[yy, xx]:
        s[y, x] = s[yy, xx]


@njit(cache=True)
def _totals(labels, anchored, weights):
    """Return static float64 gauge-weight totals, one per unknown component."""
    totals = np.zeros(anchored.size)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label and not anchored[label]:
                totals[label] += weights[y]
    return totals


def sor_gpu(s, a, c, rhs, q, active, labels, links, anchored, weights,
            ratio, omega, loops, tolerance, requested, flags, extend_y, extend_x,
            periodic_x, *, block=None):
    """Mutate a host slice using resident CUDA state and fill diagnostic flags.

    Array/scalar arguments match masked_poisson._sor. block follows the existing
    gpu_block2d setting. Allocations/uploads occur once; checkpoints transfer
    only update/min/max/overflow scalars. A process-local shared semaphore
    limits concurrent Dask GPU solves. Periodic x must be even.
    """
    ny, nx = s.shape
    if periodic_x and nx % 2:
        raise ValueError("red-black masked Poisson requires an even periodic x grid size")
    if block is not None and (
            not isinstance(block, (tuple, list)) or len(block) != 2 or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                or v <= 0 or v > 1024 for v in block) or block[0]*block[1] > 1024):
        raise ValueError("gpu_block2d must contain two positive integers with product <= 1024")
    full_blocks, threads = _launch_config_2d(ny, nx, block)[:2]
    half_blocks, _ = _launch_config_2d(ny, (nx + 1)//2, block)[:2]
    interval = _check_interval(loops, tolerance, requested)
    history = (loops + requested - 1)//requested if requested else 0
    vector_blocks = (anchored.size + 127)//128
    with _get_gpu_sem():
        ensure_context()
        arrays = [cuda.to_device(np.ascontiguousarray(array)) for array in
                  (s, a, c, rhs, q, active, labels, links, anchored, weights)]
        ds, da, dc, df, dq, dactive, dlabels, dlinks, danchored, dweights = arrays
        offsets = cuda.device_array(anchored.size, dtype=np.float64)
        totals = cuda.to_device(_totals(labels, anchored, weights))
        metric = cuda.device_array(4, dtype=np.float64)
        index = 0
        for iteration in range(1, loops + 1):
            check = iteration % interval == 0 or iteration == loops
            if check:
                _reset[vector_blocks, 128](metric, offsets)
            for color in range(2):
                _sweep[half_blocks, threads](ds, da, dc, df, dq, dlabels, dlinks,
                                            ratio, omega, color, check, metric)
            if not check:
                continue
            _means[full_blocks, threads](ds, dlabels, danchored, dweights, offsets)
            _divide_means[vector_blocks, 128](offsets, totals)
            _extrema[full_blocks, threads](ds, dactive, dlabels, offsets, metric,
                                           extend_y, extend_x, periodic_x)
            update, minimum, maximum, bad = metric.copy_to_host()
            scale = max(abs(minimum), abs(maximum))
            error = update/scale if scale > 0 else update
            overflow = bool(bad) or not np.isfinite(error) or error > 1e100
            if overflow:
                error = np.inf
            if requested:
                flags[3 + index] = error
                flags[3 + history + index] = minimum
                flags[3 + 2*history + index] = maximum
                index += 1
            if overflow or (tolerance > 0 and error < tolerance):
                break
        _shift[full_blocks, threads](ds, dlabels, offsets)
        _fill_edges[full_blocks, threads](ds, dactive, extend_y, extend_x, periodic_x)
        ds.copy_to_host(s)
    flags[0], flags[1], flags[2] = overflow, error, iteration
