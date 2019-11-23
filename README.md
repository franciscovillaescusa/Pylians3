# Pylians3

Pylians stands for **Py**thon **li**braries for the **a**nalysis of **n**umerical **s**imulations. They are a set of python libraries, written in python, cython and C, whose purposes is to facilitate the analysis of numerical simulations (both N-body and hydro). Pylians3 evolved from Pylians to support python3. Among other things, they can be used to:

- Compute density fields
- Compute power spectra
- Compute bispectra
- Compute correlation functions
- Identify voids
- Populate halos with galaxies using an HOD
- Apply HI+H2 corrections to the output of hydrodynamic simulations
- Make 21cm maps
- Compute DLAs column density distribution functions
- Plot density fields and make movies

[Pylians](https://en.wikipedia.org/wiki/Nestor_(mythology)) were the native or inhabitant of the Homeric town of Pylos. 

## Requisites

- numpy
- scipy
- h5py
- pyfftw
- mpi4py
- cython
- openmp
 
We recommend installing the first packages with [anaconda](https://www.anaconda.com/download/?lang=en-us). 

## Installation
Pylians3 can be installed in two different ways:

1.)
```python
cd library
python setup.py build
```
the compiled libraries and scripts will be located in build/lib.XXX, where XXX depends on your machine. E.g. build/lib.linux-x86_64-3.7 or build/lib.macosx-10.7-x86_64-3.7

Add that folder to your PYTHONPATH in ~/.bashrc
```sh
export PYTHONPATH=$PYTHONPATH:$HOME/Pylians/library/build/lib.linux-x86_64-3.7
```
2.) 
```python
cd library
python setup.py install
```

We recommend using the first method since you will know exactly where the libraries are. If you want to uninstall Pylians3 and have used the first option, just delete build folder.

## Usage

The most updated documentation with examples can be found [here](https://github.com/franciscovillaescusa/Pylians/blob/master/documentation/Documentation.md).


## Contact

For comments, problems, bugs... etc you can reach me at [villaescusa.francisco@gmail.com](mailto:villaescusa.francisco@gmail.com).
