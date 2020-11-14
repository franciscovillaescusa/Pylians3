# Instructions to install Pylians on a MacOS

- Download the Pylians3 code from the repository

- In a new conda environment (conda create --name Pylians_env python=3.x.x), install the dependency packages laid out in the documentation.md file (numpy, scipy, h5py, pyfftw, mpi4py, cython, openmp) in this new environment.

- ```conda activate Pylians_env```

- Using homebrew, install llvm and libomp by running:

```sh
brew install llvm
brew install libomp
```

- navigate to the Pylians3-master/library directory. In there, open setup.py in a text editor. Delete every instance of '-fopenmp'

- in a terminal, while in Pylians3-master/library, run python setup.py install

- to test the installation, navigate to Pylians3-master directory, and run ```python Tests/import_libraries.py```

These instructions should work on High Sierra.

Many thanks to Alexander Gough for this!