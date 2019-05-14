try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
	include_dirs = [numpy.get_include()],
    ext_modules = cythonize("_optics_inner.pyx")
)

