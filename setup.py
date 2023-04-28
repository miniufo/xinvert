from setuptools import setup, find_packages
import codecs
from os import path
import io
import re

with io.open("xinvert/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='xinvert',

    version=version,

    description='Invert geofluid problems based on xarray, using SOR iteration',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/miniufo/xinvert',

    author='miniufo',
    author_email='miniufo@163.com',

    license='MIT',

    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],

    keywords='invert inversion atmosphere ocean SOR successive-overrelaxation-iteration',

    packages=find_packages(exclude=['docs', 'tests', "notebooks", "pics"]),

    install_requires=[
        "numpy",
        "xarray",
        "dask",
        "numba",
    ],
)
