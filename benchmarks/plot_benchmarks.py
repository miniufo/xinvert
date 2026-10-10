"""Plot the public xinvert performance benchmarks."""
import json
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg', force=True)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_RESULT = os.path.join(HERE, 'results', 'fixed_work.json')
CONVERGED_RESULT = os.path.join(HERE, 'results', 'time_to_tolerance.json')
CPU_MC_RESULT = os.path.join(HERE, 'results', 'cpu_mc.json')
STATIC = os.path.join(HERE, '..', 'docs', 'source', '_static')


def save(fig, name):
    """Save a documentation figure and close it."""
    fig.savefig(os.path.join(STATIC, name), dpi=200, bbox_inches='tight')
    plt.close(fig)


def load(path):
    """Load one benchmark result, failing with a reproduction hint."""
    if not os.path.exists(path):
        raise SystemExit(
            f'missing {path}; run the corresponding benchmark first')
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def by_architecture(records):
    """Index solve records by architecture and grid size."""
    indexed = {}
    for record in records:
        indexed.setdefault(record['architect'], {})[record['grid']] = record
    return indexed


def by_thread(records):
    """Index CPU-mc records by thread count and grid size."""
    indexed = {}
    for record in records:
        indexed.setdefault(record['threads'], {})[record['grid']] = record
    return indexed


def format_grid_axis(ax, grids):
    """Use powers-of-two grid labels on a logarithmic x axis."""
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids)
    ax.set_xticklabels([rf'${n}^2$' for n in grids])
    ax.set_xlabel('grid size')
    ax.grid(alpha=0.3)


def plot_acceleration(reference, cpu_mc_records, thread_counts, title, name):
    """Plot GPU and multi-thread CPU speedups with common axes."""
    standard = by_architecture(reference)
    cpu = standard['cpu']
    gpu = standard['gpu']
    grids = sorted(cpu)
    multi = {
        threads: records
        for threads, records in by_thread(cpu_mc_records).items()
        if threads in thread_counts
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    axes[0].plot(
        grids,
        [cpu[n]['seconds_median'] / gpu[n]['seconds_median'] for n in grids],
        'o-', label='GPU')
    axes[0].set_title('GPU')

    for threads in sorted(multi):
        axes[1].plot(
            grids,
            [cpu[n]['seconds_median'] /
             multi[threads][n]['seconds_median'] for n in grids],
            'o-', label=f'{threads} threads')
    axes[1].set_title('CPU-mc')
    axes[1].legend(ncol=2, fontsize='x-small')

    for ax in axes:
        format_grid_axis(ax, grids)
    axes[0].set_ylabel('speedup over one CPU core')
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, name)


def plot_residual_convergence(converged, cpu_mc):
    """Compare residual convergence of CPU, GPU, and CPU-mc."""
    traces = dict(converged['residual_traces'])
    traces['cpu-mc'] = cpu_mc['residual_trace']
    labels = {
        'cpu': 'CPU, one core',
        'gpu': 'GPU',
        'cpu-mc': f'CPU-mc, {cpu_mc["metadata"]["trace_threads"]} threads',
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for architect in ('cpu', 'gpu', 'cpu-mc'):
        trace = traces[architect]
        residual = [row['residual'] for row in trace]
        axes[0].plot(
            [row['iterations'] for row in trace], residual, 'o-',
            label=labels[architect])
        axes[1].plot(
            [row['seconds_median'] for row in trace], residual, 'o-',
            label=labels[architect])
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.legend(fontsize='small')
    axes[0].set_ylabel('maximum relative residual')
    axes[0].set_xlabel('SOR iterations')
    axes[0].set_title('Convergence per iteration')
    axes[1].set_xlabel('wall time (s)')
    axes[1].set_title('Convergence per second')
    fig.tight_layout()
    save(fig, 'residual_convergence.png')


def main():
    """Generate all public benchmark figures from raw JSON results."""
    fixed = load(FIXED_RESULT)
    converged = load(CONVERGED_RESULT)
    cpu_mc = load(CPU_MC_RESULT)
    os.makedirs(STATIC, exist_ok=True)

    plot_acceleration(
        fixed['solves'], cpu_mc['fixed_scaling'],
        cpu_mc['metadata']['converged_threads'],
        'Fixed work: 1000 SOR iterations', 'fixed_work_acceleration.png')
    plot_acceleration(
        converged['solves'], cpu_mc['converged_solves'],
        cpu_mc['metadata']['converged_threads'],
        r'Time to residual $\leq 10^{-9}$',
        'time_to_residual_acceleration.png')
    plot_residual_convergence(converged, cpu_mc)
    print('figures written to docs/source/_static/')


if __name__ == '__main__':
    main()
