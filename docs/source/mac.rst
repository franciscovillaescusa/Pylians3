***
Mac
***

Pylians3 make use of these packages:

- numpy
- scipy
- h5py
- pyfftw
- mpi4py
- cython
- openmp
 
We recommend installing all packages, with the exception of openmp, with `anaconda <https://www.anaconda.com/download/?lang=en-us)>`_.

---------

#. Download the Pylians3 code from the repository

#. In a new conda environment (``conda create --name Pylians_env python=3.x.x``), install the above dependency packages in this new environment.

#. ``conda activate Pylians_env``

#. Using homebrew, install llvm and libomp by running:

.. code-block:: bash
		
   brew install llvm
   brew install libomp

5. navigate to the Pylians3/library directory. In there, open setup.py in a text editor. Delete every instance of '-fopenmp'

#. in a terminal, while in Pylians3/library, type ``python setup.py install``

#. to test the installation, navigate to Pylians3 directory, and run ```python Tests/import_libraries.py```

These instructions have been tested on High Sierra.

Many thanks to Alexander Gough for this!
