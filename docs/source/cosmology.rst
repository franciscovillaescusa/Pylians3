*********
Cosmology
*********

Pylians provide a set of routines to carry out simple cosmological calculations.

Comoving distance
-----------------

The comoving distance to redshift ``z`` can be computed as:

.. code-block:: python
		
   import cosmology_library as CL

   z       = 1.0
   Omega_m = 0.3175
   Omega_l = 0.6825

   # compute the comoving distance to redshift z in Mpc/h
   r = CL.comoving_distance(z, Omega_m, Omega_l)  #Mpc/h


Linear growth factor
--------------------

The linear growth factor to redshift ``z`` can be computed as:

.. code-block:: python
		
   import cosmology_library as CL

   z       = 1.0
   Omega_m = 0.3175
   Omega_l = 0.6825

   # compute the linear growth factor
   D = CL.linear_growth_factor(z, Omega_m, Omega_l) 

Halofit
-------
   
From a linear power spectrum at z=0, Pylians can find the non-linear matter power spectrum halofit by Takahashi 2012 as

.. code-block::  python
		 
   import numpy as np
   import cosmology_library as CL

   z       = 1.0
   Omega_m = 0.3175
   Omega_l = 0.6825

   # read the linear power spectrum
   k_lin, Pk_lin = np.loadtxt('my_Pk_file_z=0.txt', unpack=True)

   # find the non-linear power spectrum from halofit
   Pk_nl = CL.Halofit_12(Omega_m, Omega_l, z, k_lin, Pk_lin) 

