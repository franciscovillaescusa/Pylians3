from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


ext_modules = cythonize([

    Extension("integration_library", 
		["integration_library.pyx",'integration.c'],
		include_dirs=[numpy.get_include()],
		extra_compile_args=["-O3","-ffast-math","-march=native"]
		)])


setup(ext_modules = ext_modules)
