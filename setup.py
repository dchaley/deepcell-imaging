from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("src/grayscale_reconstruction/fast_hybrid_impl.pyx"))
