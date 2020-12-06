********************
Correlation function
********************

Pylians can be used to efficiently compute correlation functions of a generic field (e.g. total matter, CDM, gas, halos, neutrinos, CDM+gas, galaxies...etc). The ingredients needed are:

- ``delta``. This is the density contrast field. It should be a 3 dimensional float numpy array such ``delta = np.zeros((grid, grid, grid), dtype=np.float32)``. See :ref:`density_fields` on how to compute  density fields using Pylians.
- ``BoxSize``. Size of the periodic box. The units of the output power spectrum depend on this.
- ``MAS``. Mass-assignment scheme used to generate the density field, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the density field has not been generated with any of these set it to ``'None'``.
- ``axis``. Axis along which compute the quadrupole, hexadecapole. If the field is in real-space set ``axis=0``. If the field is in redshift-space set ``axis=0``, ``axis=1`` or ``axis=2`` if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively.
- ``threads``. This routine is openmp parallelized. Set this to the maximum number of cores per node available in your machine.

An example on how to use the routine is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # CF parameters
   BoxSize = 1000.0 #Mpc/h
   MAS     = 'CIC'
   threads = 16
   axis    = 0

   # compute the correlation function
   CF     = PKL.Xi(delta, BoxSize, MAS, axis, threads)
   r      = CF.r3D #radii in Mpc/h
   xi0    = CF.xi[:,0]  #correlation function (monopole)
   xi2    = CF.xi[:,1]  #correlation function (quadrupole)
   xi4    = CF.xi[:,2]  #correlation function (hexadecapole)
   Nmodes = CF.Nmodes3D #number of modes

This routine uses a FFT approach that allows a very computationally efficient calculation of the correlation function. However, if the number density of the tracers is very low (i.e. the density field is very sparse) this function may produce strange results. In this case it is better to use the traditional Landy-Szalay routine also available in Pylians.
