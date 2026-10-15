"""Compiled topology against independent small-grid flood-fill oracles."""
import numpy as np
import pytest
import xarray as xr

from xinvert.domain import analyze_domain
from xinvert.internal_bcs import identify_internal_bcs


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (8, 11), (0, 3)])
@pytest.mark.parametrize("periodic", [(False, False), (False, True),
                                      (True, False), (True, True)])
def test_compiled_against_flood_fill(shape, periodic):
    rng = np.random.default_rng(123)
    for probability in (0, 0.25, 0.65, 1):
        active = rng.random(shape) < probability
        expected = np.zeros(shape, dtype=np.int32)
        faces = np.zeros(shape, dtype=np.uint8)
        count = 0
        for y, x in np.ndindex(shape):
            if not active[y, x] or expected[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            expected[y, x] = count
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                    yy, xx = cy + dy, cx + dx
                    if periodic[0]:
                        yy %= shape[0]
                    if periodic[1]:
                        xx %= shape[1]
                    if not (0 <= yy < shape[0] and 0 <= xx < shape[1]):
                        continue
                    if active[yy, xx] and not expected[yy, xx]:
                        expected[yy, xx] = count
                        stack.append((yy, xx))
        for y, x in np.ndindex(shape):
            if not active[y, x]:
                continue
            faces[y, x] = 16
            for bit, (dy, dx) in enumerate(((0, 1), (0, -1), (-1, 0), (1, 0))):
                yy, xx = y + dy, x + dx
                if periodic[0]:
                    yy %= shape[0]
                if periodic[1]:
                    xx %= shape[1]
                if (0 <= yy < shape[0] and 0 <= xx < shape[1]
                        and not active[yy, xx]):
                    faces[y, x] |= 1 << bit
        topology = analyze_domain(xr.DataArray(active, dims=("y", "x")),
                                  periodic=periodic)
        np.testing.assert_array_equal(topology.component_labels, expected)
        assert topology.n_components == count
        assert topology.component_sizes == {
            i: int((expected == i).sum()) for i in range(1, count + 1)}
        bc = identify_internal_bcs(topology)
        assert bc.face_code.dtype == np.uint8
        assert bc.face_code.nbytes == active.size
        np.testing.assert_array_equal(bc.face_code, faces)
        assert bc.n_directed_faces == sum(
            int(((faces & (1 << bit)) != 0).sum()) for bit in range(4))
        for bit, direction in enumerate(("east", "west", "north", "south")):
            np.testing.assert_array_equal(getattr(bc, direction),
                                          (faces & (1 << bit)) != 0)
