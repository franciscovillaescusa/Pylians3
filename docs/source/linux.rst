*****
Linux
*****

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

.. code-block::

   python -m pip install --upgrade Pylians

---------
   

The development version can be installed as follows:

.. code-block:: bash

   git clone https://github.com/franciscovillaescusa/Pylians3.git
   cd Pylians3
   python setup.py install

To verify that the installation was successful do

.. code-block:: bash

   python Tests/import_libraries.py

If no output is produced, everything went fine.   
  
.. note::

   When facing problems installing Pylians due to openmp, its functionality can be disabled by removing the -fopenmp flags in the ``setup.py`` file.

.. note::

   You may need to add specific compilation flags for your system in the case the above procedure fails. For instance, for a Power9 system such as CINECA Marconi 100 you will need to add these compilations flags to  ``extra_compilation_args``: ``'-mcpu=powerpc64le'`` and ``'-mtune=powerpc64le'``.


