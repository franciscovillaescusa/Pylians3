*****
Linux
*****

Pylians3 make use of these packages:

- numpy
- scipy
- h5py
- pyfftw
- mpi4py
- cython
- openmp
 
We recommend installing all packages, with the exception of openmp, with `anaconda <https://www.anaconda.com/download/?lang=en-us)>`_.

.. warning::

   When facing problems installing openmp, its functionality can be disabled by removing the -fopenmp flags in the library/setup.py file.


Pylians3 can be installed in two ways:


Option 1
--------

.. code-block:: bash

   cd library
   python setup.py install


Option 2
--------

.. code-block:: bash

   cd library
   python setup.py build

the compiled libraries and scripts will be located in build/lib.XXX, where XXX depends on your machine. E.g. build/lib.linux-x86_64-3.7 or build/lib.macosx-10.7-x86_64-3.7

Add that folder to your PYTHONPATH in ~/.bashrc

.. code-block:: bash
   
   export PYTHONPATH=$PYTHONPATH:$HOME/Pylians3/library/build/lib.linux-x86_64-3.7

We recommend using this method since you will know exactly where the libraries are. If you want to uninstall Pylians3 and have used this option, just delete build folder.

.. note::

   Note that Pylians3 works with python3. In some systems, you may need to use ``python3`` instead of ``python`` when compiling the libraries.


-----

To verify that the installation was successful do

.. code-block:: bash

   python Tests/import_libraries.py

If no output is produced, everything went fine.
