# xinvert benchmarks

These are standalone performance experiments, not correctness tests. Run them
from the repository root on an otherwise idle machine.

## Experiments

- `benchmark_cpu_gpu.py` compares the GPU backend with the single-core CPU
  reference. Its `fixed` mode measures equal iteration counts, while
  `converged` measures time to a common residual tolerance.
- `benchmark_cpu_mc.py` measures CPU-mc thread scaling and time to a common
  residual tolerance.
- `plot_benchmarks.py` regenerates all public documentation figures.

The documented experiment uses float64, grids from 128 square through 4096
square, 1000 iterations for fixed work, and a residual tolerance of `1e-9`.
Small grids use five samples and grids of at least 2048 square use two.

```bash
taskset -c 0-55 python benchmarks/benchmark_cpu_gpu.py \
    --mode fixed --architectures cpu gpu \
    --grids 128 256 512 1024 2048 4096 --fixed-iterations 1000

taskset -c 0-55 python benchmarks/benchmark_cpu_gpu.py \
    --mode converged --architectures cpu gpu \
    --grids 128 256 512 1024 2048 4096 --tolerance 1e-9

taskset -c 0-55 python benchmarks/benchmark_cpu_mc.py
python benchmarks/plot_benchmarks.py
```

Raw measurements are stored in `results/fixed_work.json`,
`results/time_to_tolerance.json`, and `results/cpu_mc.json`. Numerical
correctness and agreement between backends belong to the pytest suite.
