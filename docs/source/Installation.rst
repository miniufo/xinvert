.. xinvert documentation master file, created by
   sphinx-quickstart on Wed April 19 21:26:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

Requirements
^^^^^^^^^^^^

xinvert supports Python 3.9 and newer.  Its core dependencies are xarray_,
dask_, numpy_, and numba_.  GPU acceleration is optional and additionally
requires an NVIDIA GPU, a compatible driver/runtime, and ``numba-cuda``.

Installation from conda forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xinvert can be installed via conda forge::

    conda install -c conda-forge xinvert

Installation from pip
^^^^^^^^^^^^^^^^^^^^^

One can do this by using pip::

    pip install xinvert

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

GPU support
^^^^^^^^^^^

.. warning::

   The GPU backend is experimental and is currently intended for testing and
   evaluation.  Its numerical kernels, configuration options, and performance
   characteristics may change before the backend is declared stable.  Use the
   CPU backend for production workflows that require the stable path.

For a machine with an existing CUDA runtime, install the GPU extra::

    pip install "xinvert[gpu]"

If the CUDA runtime should be installed into the Python environment, NVIDIA's
``numba-cuda`` package provides versioned extras, for example::

    pip install xinvert "numba-cuda[cu12]"

or with conda::

    conda install -c conda-forge xinvert numba-cuda "cuda-version=12"

CUDA 13 users can replace ``cu12`` / ``cuda-version=12`` with the
corresponding CUDA 13 option.  An up-to-date NVIDIA driver is required in all
cases.  CPU-only installations do not need ``numba-cuda``.

Installation from github
^^^^^^^^^^^^^^^^^^^^^^^^

xinvert is still under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/miniufo/xinvert>`_
and install it::

    git clone https://github.com/miniufo/xinvert.git
    cd xinvert
    python -m pip install .

or simply::

    pip install git+https://github.com/miniufo/xinvert.git


How to run the notebooks
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run the example notebooks in this documentation, you will need a
few extra dependencies that you can install via:::

    conda env create -f environment.yml
    conda activate xinvert



.. _dask: http://dask.pydata.org/
.. _numpy: https://numpy.org/
.. _xarray: http://xarray.pydata.org/
.. _numba: https://numba.pydata.org/
