from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("src/deepcell_imaging/image_processing/fast_hybrid_impl.pyx")
)
