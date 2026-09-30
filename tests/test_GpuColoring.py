"""Pure-Python checks for the dependency graphs used by GPU coloring."""

import itertools

import numpy as np
import pytest

from xinvert.gpus import (
    _has_active_coefficient,
    _validate_periodic_x_coloring,
)


def test_four_color_mapping_separates_nine_point_stencil():
    color = lambda j, i: (j % 2) * 2 + (i % 2)
    offsets = [
        (dj, di)
        for dj, di in itertools.product((-1, 0, 1), repeat=2)
        if (dj, di) != (0, 0)
    ]
    for j, i in itertools.product(range(4), repeat=2):
        for dj, di in offsets:
            assert color(j, i) != color(j + dj, i + di)


def test_five_color_mapping_separates_biharmonic_stencil():
    color = lambda j, i: (j + 2 * i) % 5
    offsets = set()
    for distance in (1, 2):
        offsets.update({
            (distance, 0), (-distance, 0),
            (0, distance), (0, -distance),
        })
    offsets.update(itertools.product((-1, 1), repeat=2))
    offsets.update(itertools.product((-2, 2), repeat=2))

    for j, i in itertools.product(range(5), repeat=2):
        for dj, di in offsets:
            assert color(j, i) != color(j + dj, i + di)


@pytest.mark.parametrize(
    'xc,n_color',
    [(9, 2), (9, 4), (34, 5)],
)
def test_incompatible_periodic_extent_is_rejected(xc, n_color):
    with pytest.raises(ValueError, match='periodic x'):
        _validate_periodic_x_coloring(xc, 'periodic', n_color)


@pytest.mark.parametrize(
    'xc,n_color',
    [(10, 2), (10, 4), (35, 5)],
)
def test_compatible_periodic_extent_is_accepted(xc, n_color):
    _validate_periodic_x_coloring(xc, 'periodic', n_color)


def test_mixed_coefficient_detection():
    zeros = np.zeros((3, 4))
    mixed = zeros.copy()
    mixed[1, 2] = 0.1
    assert not _has_active_coefficient(zeros)
    assert _has_active_coefficient(zeros, mixed)
