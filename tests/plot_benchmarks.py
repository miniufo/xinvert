# -*- coding: utf-8 -*-
"""Generate documentation figures for the GPU benchmark experiments.

Reads tests/results/*.json (from benchmark_convergence.py and
benchmark_gpu_overheads.py); figures are skipped if their JSON is absent.
All figures saved at DPI=200 with bbox_inches='tight'.

Run:  python tests/plot_benchmarks.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DPI = 200
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
STATIC = os.path.join(HERE, '..', 'docs', 'source', '_static')


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(STATIC, name), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def fig_speedup():
    grids = [128, 256, 512, 1024, 2048, 4096]
    speedup = [1.45, 5.68, 13.41, 15.68, 16.41, 17.12]  # 2026-07 session
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(grids, speedup, 'o-')
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids)
    ax.set_xticklabels([f'{g}^2' for g in grids])
    ax.set_xlabel('grid size')
    ax.set_ylabel('GPU speedup vs CPU (x)')
    ax.set_title('CPU vs GPU (1000 iterations, tol=0)')
    ax.grid(alpha=0.3)
    save(fig, 'gpu_cpu_speedup.png')


def fig_block_sweep():
    grids = [512, 1024, 2048, 4096]
    times = {'16,16': [0.1251, 0.4392, 1.7139, 6.9775],
             '32,8':  [0.1244, 0.4403, 1.7616, 7.1847],
             '32,16': [0.1331, 0.4526, 1.7638, 7.2191],
             '32,32': [0.1498, 0.4837, 1.8657, 7.5225],
             '64,4':  [0.1258, 0.4403, 1.9462, 8.2796],
             '16,32': [0.1356, 0.4536, 1.8159, 7.3437]}  # 2026-07 session
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for c, t in times.items():
        ax.plot(grids, t, 'o-', label=c)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(grids)
    ax.set_xticklabels([f'{g}^2' for g in grids])
    ax.set_xlabel('grid size')
    ax.set_ylabel('time (s)')
    ax.set_title('thread-block shape sweep (GPU, 1000 iterations)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, 'gpu_block_sweep.png')


def fig_convergence(data):
    if data is None:
        return
    iters = data['iters']
    grids = sorted(int(g) for g in data['grids'])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(grids)))
    for ax, arch, title in ((axes[0], 'cpu', 'CPU (lexicographic SOR)'),
                            (axes[1], 'gpu', 'GPU (red-black SOR)')):
        for c, g in zip(cmap, grids):
            errs = data['errors'][arch][str(g)]
            xs = [i for i, e in zip(iters, errs) if e is not None]
            ys = [e for e in errs if e is not None]
            ax.plot(xs, ys, 'o-', color=c, label=f'{g}^2')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('max |result - analytic|')
    axes[1].legend(fontsize=7)
    fig.suptitle('true-error convergence (tolerance=0, auto omega)')
    save(fig, 'convergence_curves.png')


def fig_gpu_overheads(data):
    if data is None:
        return
    micro = data['micro_bench_us']
    sub = data['dask_submission']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    keys = ['to_device_alloc_us', 'copy_to_device_us', 'copy_to_host_us',
            'kernel_launch_us']
    labels = ['to_device\n(alloc+copy)', 'copy to\ndevice buffer',
              'copy to\nhost', 'kernel launch\n(noop)']
    ax1.bar(labels, [micro[k] for k in keys], color='tab:orange')
    ax1.set_ylabel('cost (us)')
    ax1.set_title('fixed GPU-side costs (512^2, 2 MB)')
    ax1.tick_params(axis='x', labelsize=7)
    ax1.grid(alpha=0.3, axis='y')
    cases = [f"{r['nt']}x{r['grid']}^2" for r in sub]
    xx = range(len(cases))
    ax2.bar([x - 0.18 for x in xx],
            [r['schedulers']['single-threaded'] for r in sub],
            width=0.36, label='single-threaded')
    ax2.bar([x + 0.18 for x in xx],
            [r['schedulers']['threads'] for r in sub],
            width=0.36, label='threads')
    ax2.set_xticks(list(xx))
    ax2.set_xticklabels(cases, rotation=30, fontsize=7)
    ax2.set_ylabel('wall time (s)')
    ax2.set_title('dask submission strategies (GPU)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis='y')
    save(fig, 'gpu_overheads_dask.png')


def main():
    os.makedirs(STATIC, exist_ok=True)
    fig_speedup()
    fig_block_sweep()
    fig_convergence(load('convergence.json'))
    fig_gpu_overheads(load('gpu_overheads.json'))
    print('figures written to docs/source/_static/')


if __name__ == '__main__':
    main()
