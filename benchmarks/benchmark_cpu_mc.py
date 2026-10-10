"""Benchmark the experimental multi-color CPU backend.

The fixed-work experiment measures thread scaling for exactly 1000 SOR
iterations.  A second experiment measures time to the same residual tolerance
as the CPU/GPU benchmark using the selected production thread count.
"""
import argparse
import json
import os
import platform

import numba
from benchmark_cpu_gpu import (
    benchmark_solve,
    benchmark_trace,
    make_problem,
    repeat_count,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, 'results', 'cpu_mc.json')


def main():
    """Run fixed-work scaling and time-to-tolerance measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--grids', nargs='+', type=int,
                        default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument('--threads', nargs='+', type=int,
                        default=[8, 16, 32, 56])
    parser.add_argument(
        '--converged-threads', nargs='+', type=int,
        help='threads measured to tolerance (default: all --threads values)')
    parser.add_argument('--trace-threads', type=int, default=32)
    parser.add_argument('--trace-grid', type=int, default=512)
    parser.add_argument('--checkpoints', nargs='+', type=int,
                        default=[1, 25, 50, 100, 200, 400, 800, 1600, 3200])
    parser.add_argument('--dtype', choices=['float32', 'float64'],
                        default='float64')
    parser.add_argument('--fixed-iterations', type=int, default=1000)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--max-iterations', type=int, default=50000)
    parser.add_argument('--diag-interval', type=int, default=50)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--large-repeat', type=int, default=2)
    parser.add_argument('--large-grid', type=int, default=2048)
    parser.add_argument(
        '--skip-fixed', action='store_true',
        help='reuse fixed_scaling from the existing output file')
    parser.add_argument(
        '--skip-converged', action='store_true',
        help='run only the fixed-work thread-scaling experiment')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    available = numba.config.NUMBA_NUM_THREADS
    if any(not 1 <= value <= available for value in args.threads):
        parser.error(f'--threads values must be in [1, {available}]')
    converged_threads = args.converged_threads or args.threads
    if any(not 1 <= value <= available for value in converged_threads):
        parser.error(f'--converged-threads values must be in [1, {available}]')
    if not 1 <= args.trace_threads <= available:
        parser.error(f'--trace-threads must be in [1, {available}]')
    if args.tolerance <= 0:
        parser.error('--tolerance must be positive')
    if args.repeat < 1 or args.large_repeat < 1:
        parser.error('--repeat values must be positive')

    if args.skip_fixed:
        if not os.path.exists(args.output):
            parser.error('--skip-fixed requires an existing --output file')
        with open(args.output, encoding='utf-8') as stream:
            fixed = json.load(stream)['fixed_scaling']
    else:
        fixed = []
        for n in args.grids:
            forcing = make_problem(n, args.dtype)
            repeats = repeat_count(
                n, args.repeat, args.large_repeat, args.large_grid)
            by_thread = {}
            for threads in args.threads:
                record = benchmark_solve(
                    forcing, 'cpu-mc', args.dtype, 0.0,
                    args.fixed_iterations, args.fixed_iterations, repeats,
                    threads)
                record.update({
                    'grid': n, 'threads': threads, 'repeat': repeats})
                fixed.append(record)
                by_thread[threads] = record
            reference = by_thread[args.threads[0]]['seconds_median']
            summary = [f'{n:4d}x{n:<4d}']
            for threads in args.threads:
                seconds = by_thread[threads]['seconds_median']
                summary.append(
                    f'{threads}t={seconds:.4f}s/{reference / seconds:.2f}x')
            print('  '.join(summary), flush=True)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    converged = []

    def write_results(residual_trace):
        """Checkpoint completed measurements to the output JSON file."""
        payload = {
            'metadata': {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'cpu': platform.processor() or platform.machine(),
                'dtype': args.dtype,
                'threads': args.threads,
                'converged_threads': converged_threads,
                'trace_threads': args.trace_threads,
                'trace_grid': args.trace_grid,
                'fixed_iterations': args.fixed_iterations,
                'tolerance': args.tolerance,
                'max_iterations': args.max_iterations,
                'diag_interval': args.diag_interval,
                'repeat_small': args.repeat,
                'repeat_large': args.large_repeat,
                'large_grid': args.large_grid,
            },
            'fixed_scaling': fixed,
            'converged_solves': converged,
            'residual_trace': residual_trace,
        }
        temporary = f'{output}.tmp'
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary, output)

    if args.skip_converged:
        write_results([])
        print(f'saved {output}')
        return

    for n in args.grids:
        forcing = make_problem(n, args.dtype)
        repeats = repeat_count(
            n, args.repeat, args.large_repeat, args.large_grid)
        for threads in converged_threads:
            record = benchmark_solve(
                forcing, 'cpu-mc', args.dtype, args.tolerance,
                args.max_iterations, args.diag_interval, repeats, threads)
            record.update({
                'grid': n,
                'threads': threads,
                'repeat': repeats,
            })
            converged.append(record)
            write_results([])
            print(
                f'{n:4d}x{n:<4d}  converged {threads}t='
                f'{record["seconds_median"]:.4f}s  '
                f'iterations={record["iterations"]}',
                flush=True)

    trace_forcing = make_problem(args.trace_grid, args.dtype)
    residual_trace = benchmark_trace(
        trace_forcing, 'cpu-mc', args.dtype,
        sorted(set(args.checkpoints)), args.repeat, args.trace_threads)

    write_results(residual_trace)
    print(f'saved {output}')


if __name__ == '__main__':
    main()
