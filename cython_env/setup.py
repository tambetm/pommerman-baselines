from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name='cpommerman',
    version='0.0.1',
    description='Cython version of Pommerman environment',
    author='Tambet Matiisen',
    author_email='tambet.matiisen@gmail.com',
    packages=find_packages(),
    ext_modules=cythonize(['cpommerman/*.pyx'], annotate=True),
    include_dirs=[np.get_include()],
    install_requires=['cython', 'numpy', 'gym']
)
