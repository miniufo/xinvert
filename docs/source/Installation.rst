.. xinvert documentation master file, created by
   sphinx-quickstart on Wed April 19 21:26:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

Requirements
^^^^^^^^^^^^

xinvert is compatible with python 3 (>= version 3.6). It requires xarray_ dask_ 
numpy_ and numba_.

Installation from conda forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GCM-Filters can be installed via conda forge::

    conda install -c conda-forge xinvert

Installation from pip
^^^^^^^^^^^^^^^^^^^^^

One can do this by using pip::

    pip install xinvert

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from github
^^^^^^^^^^^^^^^^^^^^^^^^

xinvert is still under active development. To obtain the latest development version,
you may clone the `source repository <https://github.com/miniufo/xinvert>`_
and install it::

    git clone https://github.com/miniufo/xinvert.git
    cd xinvert
    python setup.py install

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
