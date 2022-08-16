***
Mac
***

.. code-block:: bash

   python -m pip install Pylians

Pylians make use of these packages

- ``numpy``
- ``scipy``
- ``h5py``
- ``pyfftw``
- ``cython``

that would be installed automatically by pip when invoking the above command. Note that Pylians also requires a working openmp environment. This needs to be installed separately before invoking the pip command.

.. note::

   Pylians works for both python2 and python3. Depending on what your python is, it will install accordingly. Sometimes, python3 needs to ne invoked explicitly as ``python3``. If so, install as ``python3 -m pip install Pylians``.

To upgrade to a newer version type:

.. code-bash::

   python -m pip install --upgrade Pylians

If you experience problems with the installation try to install the development version as detailed below.   

---------

The development version can be installed as follows:

#. Download the Pylians3 code from the repository: ``git clone 

#. Create a new conda environment: ``conda create --name Pylians_env python=3.x.x``)

#. Activate the environment: ``conda activate Pylians_env``

#. Install the required dependency packages

   - ``numpy``
   - ``scipy``
   - ``h5py``
   - ``pyfftw``
   - ``cython``

#. Using homebrew, install llvm and libomp by running:

.. code-block:: bash
		
   brew install llvm
   brew install libomp

#. navigate to the Pylians3 directory. In there, open setup.py in a text editor. Delete every instance of ``-fopenmp``

#. in a terminal, while in Pylians3, type ``python setup.py install``

#. to test the installation execute ``python Tests/import_libraries.py``

These instructions have been tested on High Sierra.

Many thanks to Alexander Gough for this!

.. note::

   For M1 mac users the instructions are similar but:

   - verify that clang version is >=13
   - replace ``-march=native`` by ``mcpu=apple-m1``
   - execute ``CC=clang python setup.py install``

   Thanks for Valerio Marra for this!
