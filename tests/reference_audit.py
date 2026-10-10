"""Opt-in CPU baseline observation: pytest -p tests.reference_audit.

Records actual flags without changing dtype, controls or reference values.
"""
from functools import wraps

from xinvert import core

_original_factory = None
_records = []
_test_id = ''


def pytest_runtest_setup(item):
    global _test_id
    _test_id = item.nodeid


def pytest_configure():
    global _original_factory
    _original_factory = core._make_kernel
    _records.clear()

    def factory(kernel, grid_args, params):
        if str(params.get('architect', 'cpu')).strip().lower() != 'cpu':
            return _original_factory(kernel, grid_args, params)

        @wraps(kernel)
        def observed(*args, **kwargs):
            result = kernel(*args, **kwargs)
            flags, limit, tolerance = args[-3:]
            _records.append((
                _test_id, str(args[0].dtype), float(tolerance),
                int(flags[2]), int(limit), float(flags[1]), bool(flags[0]),
            ))
            return result

        return _original_factory(observed, grid_args, params)

    core._make_kernel = factory


def pytest_unconfigure():
    if _original_factory is not None:
        core._make_kernel = _original_factory


def pytest_terminal_summary(terminalreporter):
    terminalreporter.section('historical CPU convergence audit')
    for nodeid in sorted({row[0] for row in _records}):
        rows = [row for row in _records if row[0] == nodeid]
        failed = sum(row[6] or (row[2] > 0 and not row[5] < row[2])
                     for row in rows)
        limited = sum(row[3] >= row[4] for row in rows)
        terminalreporter.write_line(
            f'{nodeid}: slices={len(rows)}, '
            f'dtypes={sorted({row[1] for row in rows})}, '
            f'fixed_work={sum(row[2] == 0 for row in rows)}, '
            f'tolerance_not_met={failed}, at_limit={limited}, '
            f'max_reported_error={max(row[5] for row in rows):.3e}')
