============
Installation
============

.. tabs::

   .. tab:: Linux

      .. tabs::

	 .. tab:: Stable

	    .. code-block:: bash

	       python -m pip install Pylians

	 .. tab:: Development

	    .. code-block:: bash

	       git clone https://github.com/franciscovillaescusa/Pylians3.git
	       cd Pylians3
	       python -m pip install .

	    To verify that the installation was successful do

	    .. code-block:: bash
		      
	       python Tests/import_libraries.py

	    If no output is produced, everything went fine.

	    .. note::

	       You may need to add specific compilation flags for your system if the above procedure fails. For instance, for a Power9 system such as CINECA Marconi 100 you need to add these compilations flags to ``extra_compilation_args``: ``'-mcpu=powerpc64le'`` and ``'-mtune=powerpc64le'``.


   .. tab:: Mac

      .. tabs::

	 .. tab:: Stable

	    .. code-block:: bash
		      
	       python -m pip install Pylians

	    If this does not work on your machine try to install the development version.

	    
	 .. tab:: Development

	    The main problem installing Pylians on a mac is due to openmp. So these instructions will remove that functionality.

	    #. Download Pylians3 code from the repository:

	       .. code-block:: bash
	       
		  git clone https://github.com/franciscovillaescusa/Pylians3.git

	    #. Create a new conda environment and activate it

	       .. code-block:: bash
	       
		  conda create --name Pylians_env python=3.x.x
		  conda activate Pylians_env

	       3.x.x should be changed with your python version, e.g. 3.9.7
	     
	    #. Install llvm and libomp using homebrew

	       .. code-block:: bash
		      
		  brew install llvm
		  brew install libomp

	    #. Navigate to the Pylians3 directory. Open ``setup.py`` in a text editor. Delete every instance of ``-fopenmp``

	    #. In a terminal, while in Pylians3, type

	       .. code-block:: bash
	       
		  python -m pip install .

	    #. Test the installation by executing

	       .. code-block:: bash
	       
		  python Tests/import_libraries.py

	       If no output is produced, everything went fine.

	    These instructions have been tested on High Sierra. Many thanks to Alexander Gough for this!

	    .. note::

	       For M1 mac users the instructions are similar but:

	       - verify that clang version is >=13
	       - replace ``-march=native`` by ``mcpu=apple-m1``
	       - execute ``CC=clang python setup.py install``

	       Thanks for Valerio Marra for this!


.. note::

   Pylians works for both python2 and python3. Depending on your python version, it will install accordingly. Sometimes, python3 needs to be invoked explicitly as ``python3``. If so, install as ``python3 -m pip install Pylians``

.. warning::

   When facing problems installing Pylians due to openmp, its functionality can be disabled by removing the -fopenmp flags in the ``setup.py`` file (see Instructions for Mac development version). This requires the installation in the development mode.
   

Dependencies
------------
      
Pylians make use of these packages

- ``numpy``
- ``scipy``
- ``h5py``
- ``pyfftw``
- ``cython``

that would be installed automatically when invoking the above pip command. Note that Pylians also requires a working openmp environment. This needs to be installed separately before invoking the pip command. If there are conflicts with old versions, try to upgrade to the latest versions of the required packages, e.g.

.. code-block:: bash

   python -m pip install --upgrade numpy
   

Upgrade
-------

To upgrade to the latest version

.. tabs::

   .. tab:: Stable

      .. code-block:: bash

	 python -m pip install --upgrade Pylians

   .. tab:: Development

      .. code-block:: bash

	 cd Pylians3
	 git pull
	 python -m pip install .
	 
      In a Mac, if having problems with openmp remove the instances in the ``setup.py`` file before executing ``python -m pip install .``
