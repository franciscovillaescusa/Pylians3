from platform import machine, system

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


is_mac = system() == "Darwin"
is_arm = machine() == "arm64"
is_m1 = is_mac and is_arm

arch_flag = "-mcpu=apple-m1" if is_m1 else "-march=native"
omp_flag = "-Xpreprocessor -fopenmp" if is_m1 else "-fopenmp"
extra_compile_args = ["-O3", "-ffast-math", arch_flag]
extra_compile_args_omp = extra_compile_args.copy()
extra_compile_args_omp.append(omp_flag)
extra_link_args = [omp_flag]

ext_modules = [
    Extension(
        "MAS_library.MAS_library",
        ["library/MAS_library/MAS_library.pyx", "library/MAS_library/MAS_c.c"],
        extra_compile_args=extra_compile_args_omp,
        extra_link_args=extra_link_args,
        libraries=["m"],
    ),
    Extension(
        "Pk_library.Pk_library",
        ["library/Pk_library/Pk_library.pyx"],
        extra_compile_args=extra_compile_args_omp,
    ),
    Extension(
        "Pk_library.bispectrum_library", ["library/Pk_library/bispectrum_library.pyx"]
    ),
    Extension(
        "MAS_library.field_properties", ["library/MAS_library/field_properties.pyx"]
    ),
    Extension(
        "redshift_space_library.redshift_space_library",
        ["library/redshift_space_library/redshift_space_library.pyx"],
    ),
    Extension(
        "smoothing_library.smoothing_library",
        ["library/smoothing_library/smoothing_library.pyx"],
        extra_compile_args=extra_compile_args_omp,
        extra_link_args=extra_link_args,
        libraries=["m"],
    ),
    Extension(
        "void_library.void_library",
        [
            "library/void_library/void_library.pyx",
            "library/void_library/void_openmp_library.c",
        ],
        extra_compile_args=extra_compile_args_omp,
        extra_link_args=extra_link_args,
        libraries=["m"],
    ),
    Extension(
        "integration_library.integration_library",
        [
            "library/integration_library/integration_library.pyx",
            "library/integration_library/integration.c",
            "library/integration_library/runge_kutta.c",
        ],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "density_field_library.density_field_library",
        ["library/density_field_library/density_field_library.pyx"],
    ),
    Extension(
        "sorting_library.sorting_library",
        ["library/sorting_library/sorting_library.pyx"],
        extra_compile_args=extra_compile_args,
    ),
    Extension("HI_library.HI_library", ["library/HI_library/HI_library.pyx"]),
    Extension(
        "HI_clusters_library.HI_clusters_library",
        ["library/HI_clusters_library/HI_clusters_library.pyx"],
    ),
]


with open("README.md", "r") as f:
    documentation = f.read()

setup(
    name="Pylians",
    version="0.10",
    author="Francisco Villaescusa-Navarro",
    author_email="villaescusa.francisco@gmail.com",
    description="Python libraries for the analysis of numerical simulations",
    url="https://github.com/franciscovillaescusa/Pylians3",
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=documentation,
    packages=find_packages(where="library/"),
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
        include_path=[
            "library/MAS_library/",
            "library/void_library/",
            "library/integration_library/",
        ],
    ),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "h5py",
        "pyfftw; platform_system!='Darwin' and platform_machine!='arm64'",
        "scipy",
        "hdf5plugin",
        "Cython<3.0.0",
    ],
    package_dir={"": "library/"},
    py_modules=[
        "bias_library",
        "CAMB_library",
        "correlation_function_library",
        "cosmology_library",
        "HI_image_library/HI_image_library",
        "HOD_library",
        "IM_library",
        "mass_function_library",
        "plotting_library",
        "readfof",
        "readgadget",
        "readsnapHDF5",
        "readsnap",
        "readsnap2",
        "readsnap_mpi",
        "readsubf",
        "routines",
        "units_library",
    ],
)
