**********
Bispectrum
**********

Pylians can compute bispectra of 3D density fields (total matter, CDM, gas, halos, galaxies...etc). The ingredients needed are:

- ``delta``. This is the density contrast field. It should be a 3 dimensional float numpy array such ``delta = np.zeros((grid, grid, grid), dtype=np.float32)``. See :ref:`density_fields` on how to compute  density fields using Pylians.
- ``BoxSize``. Size of the periodic box. The units of the output bispectrum depend on this. 
- ``k1``. The wavenumber of the first side of the considered triangle. Use units in correspondence with ``BoxSize``.
- ``k2``. The wavenumber of the second size of the considered triangle. Use units in correspondence with ``BoxSize``.
- ``theta``. This is a numpy array containing the different angles between ``k1`` and ``k2``.  
- ``MAS``. Mass-assignment scheme used to generate the density field, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the density field has not been generated with any of these set it to ``'None'``.
- ``threads``. The bispectrum code is openmp parallelized. Set this to the maximum number of cpus per node.

Pylians computes the amplitude of the bispectrum, and reduced bispectrum, for triangles that have sides ``k1`` and ``k2`` and different considered angles between them.

.. code-block:: python
		
   import numpy as np
   import Pk_library as PKL

   BoxSize = 1000.0 #Size of the density field in Mpc/h
   k1      = 0.5    #h/Mpc
   k2      = 0.6    #h/Mpc
   MAS     = 'CIC'
   threads = 1
   theta   = np.linspace(0, np.pi, 25) #array with the angles between k1 and k2

   # compute bispectrum
   BBk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS, threads)
   Bk  = BBk.B     #bispectrum
   Qk  = BBk.Q     #reduced bispectrum
   k   = BBk.k     #k-bins for power spectrum
   Pk  = BBk.Pk    #power spectrum
