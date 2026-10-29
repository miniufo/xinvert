"""Per-call Numba thread masks shared by CPU-mc solvers."""

import numbers
from contextlib import contextmanager

from numba import config, get_num_threads, set_num_threads

# Inherit the executing thread's mask; do not override worker configuration.
DEFAULT_THREADS = None


def validate_threads(threads):
    """Accept a thread count or None (inherit the executing thread's mask)."""
    if threads is not None and (
        isinstance(threads, bool)
        or not isinstance(threads, numbers.Integral)
        or not 1 <= threads <= config.NUMBA_NUM_THREADS
    ):
        raise ValueError(
            "threads must be None or a positive integer within Numba's "
            f"thread limit ({config.NUMBA_NUM_THREADS})"
        )


@contextmanager
def numba_threads(architect, threads=DEFAULT_THREADS):
    """Temporarily select CPU-mc threads inside the executing worker thread."""
    if architect != 'cpu-mc':
        yield
        return
    validate_threads(threads)
    previous = get_num_threads()
    try:
        if threads is not None:
            set_num_threads(int(threads))
        yield
    finally:
        set_num_threads(previous)
