# -*- coding: utf-8 -*-
"""Regression: printInfo diagnostic output survives Jupyter reruns.

Two failure modes are guarded against (see docs/source/Benchmark.rst,
"Live printInfo output under parallelism" and core._print_live):

1. Output misattribution: ipykernel binds new threads to the parent
   header of the cell that spawned them; dask reuses its worker pool
   across cells, which used to attribute the second run's output to the
   first cell (lines "disappearing").
2. Line splicing: concurrent prints from dask worker threads used to be
   spliced by ipykernel's buffer; each line is now a single atomic write.

The test executes a two-cell notebook in a real Jupyter kernel via
nbclient and requires all 12 diagnostic lines in BOTH cells.

Marked slow: spawns a real kernel and JIT-compiles numba kernels (~1 min).
Run:  pytest tests/test_jupyter_printinfo.py -v
"""
import os
import pytest

nbformat = pytest.importorskip('nbformat')
NotebookClient = pytest.importorskip('nbclient').NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CELL = '''\
import numpy as np, xarray as xr
from xinvert import invert_Poisson
print('MARK-START')
nt = 12
x = np.linspace(0,1,101); y = np.linspace(0,1,81); t = np.arange(nt)
X, Y = np.meshgrid(x, y)
vor3d = np.stack([-2.0*np.pi**2*np.sin(np.pi*X)*np.sin(np.pi*Y)]*nt)
da = xr.DataArray(vor3d, dims=['time','y','x'],
                  coords={'time':t,'y':y,'x':x}).chunk({'time':1})
ip = {'BCs':['fixed','fixed'],'mxLoop':50,'tolerance':0.0,
      'printInfo':True,'architect':'cpu'}
sf = invert_Poisson(da, dims=['y','x'], coords='cartesian', iParams=ip)
sf.compute()
print('MARK-END')
'''


def _diag_lines(cell):
    return sum(out.get('text', '').count('loops')
               for out in cell.get('outputs', [])
               if out.get('output_type') == 'stream')


@pytest.fixture(scope='module')
def executed_nb():
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(CELL) for _ in range(2)]
    client = NotebookClient(nb, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': ROOT}})
    client.execute()
    return nb


@pytest.mark.slow
def test_printinfo_complete_on_first_run(executed_nb):
    assert _diag_lines(executed_nb.cells[0]) == 12


@pytest.mark.slow
def test_printinfo_complete_on_rerun(executed_nb):
    # the second run used to lose most diagnostic lines
    assert _diag_lines(executed_nb.cells[1]) == 12
