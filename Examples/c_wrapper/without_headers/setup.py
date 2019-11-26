from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


setup(name = 'c_wrapper',
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize([Extension("cython_code", 
        ["cython_code.pyx",'c_code.c'],
        compiler_directives={'language_level' : "3"},
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3","-ffast-math","-march=native"])]))
