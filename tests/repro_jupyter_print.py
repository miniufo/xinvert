# -*- coding: utf-8 -*-
"""Regression: batched printInfo in real Jupyter kernel (2 cells)."""
import nbformat
from nbclient import NotebookClient

CELL = '''import numpy as np, xarray as xr
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

nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(CELL) for _ in range(2)]
client = NotebookClient(nb, timeout=300, kernel_name='python3',
                        resources={'metadata': {'path': '.'}})
client.execute()

ok = True
for i, cell in enumerate(nb.cells, 1):
    n = sum(out.get('text', '').count('loops')
            for out in cell.get('outputs', [])
            if out.get('output_type') == 'stream')
    print(f'cell {i}: {n} / 12')
    ok = ok and n == 12
print('RESULT:', 'PASS' if ok else 'FAIL')
