from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

# details on installing python packages can be found here
# https://docs.python.org/3.7/install/

ext_modules = [
    Extension("MAS_library.MAS_library", ["MAS_library/MAS_library.pyx",
                                          "MAS_library/MAS_c.c"],
        extra_compile_args=['-O3','-ffast-math','-march=native','-fopenmp'],
              extra_link_args=['-fopenmp'], libraries=['m']),

    Extension("Pk_library.Pk_library", ["Pk_library/Pk_library.pyx"],
        extra_compile_args = ['-O3','-ffast-math','-march=native','-fopenmp']),

    Extension("Pk_library.bispectrum_library",
        ["Pk_library/bispectrum_library.pyx"]),

    Extension("MAS_library.field_properties",
        ["MAS_library/field_properties.pyx"]),

    Extension("redshift_space_library.redshift_space_library", 
              ["redshift_space_library/redshift_space_library.pyx"]),

    Extension("smoothing_library.smoothing_library",
              ["smoothing_library/smoothing_library.pyx"],
        extra_compile_args = ['-O3','-ffast-math','-march=native','-fopenmp'],
        extra_link_args=['-fopenmp'], libraries=['m']),

    Extension("void_library.void_library", 
              ["void_library/void_library.pyx",
               "void_library/void_openmp_library.c"],
        extra_compile_args = ['-O3','-ffast-math','-march=native','-fopenmp'],
        extra_link_args=['-fopenmp'], libraries=['m']),

    Extension("integration_library.integration_library",
              ["integration_library/integration_library.pyx",
               "integration_library/integration.c",
               "integration_library/runge_kutta.c"],
              extra_compile_args=["-O3","-ffast-math","-march=native"]),

    Extension("density_field_library.density_field_library", 
              ["density_field_library/density_field_library.pyx"]),

    Extension("sorting_library.sorting_library", 
              ["sorting_library/sorting_library.pyx"],
              extra_compile_args=['-O3','-ffast-math','-march=native']),

    Extension("HI_library.HI_library",
              ["HI_library/HI_library.pyx"]),

    Extension("HI_clusters_library.HI_clusters_library", 
              ["HI_clusters_library/HI_clusters_library.pyx"]),


]


setup(
    name    = 'Pylians3',
    version = "3.0", 
    author  = 'Francisco Villaescusa-Navarro',
    author_email = 'villaescusa.francisco@gmail.com',
    ext_modules = cythonize(ext_modules, 
                            compiler_directives={'language_level' : "3"},
                            include_path=['MAS_library/','void_library/',
                                          'integration_library/']),
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    py_modules=['bias_library', 'CAMB_library', 'correlation_function_library',
                'cosmology_library', 'HI_image_library/HI_image_library',
                'HOD_library', 'IM_library', 'mass_function_library',
                'plotting_library', 'readfof', 'readgadget', 'readsnapHDF5',
                'readsnap', 'readsnap2', 'readsnap_mpi', 
                'readsubf', 'routines', 'units_library']
)




