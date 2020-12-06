***********************
Gaussian density fields
***********************

Pylians provide a few routines to generate Gaussian density fields either in 2D or 3D. The ingredients needed are:

- ``grid``. The generated Gaussian density field will have grid x grid pixels in 2D or grid x grid x grid voxels in 3D.
- ``k`` . 1D float32 numpy array containing the k-values of the input power spectrum.
- ``Pk``. 1D float32 numpy array containing the Pk-values of the input power spectrum.
- ``Rayleigh_sampling``. Where Rayleigh sampling the modes amplitudes when generating the Gaussian field. If ``Rayleigh_sampling=0`` the Gaussian field will not have cosmic variance. Set ``Rayleigh_sampling=1`` for standard Gaussian density fields.
- ``seed``. Integer for the random seed of the map.
- ``BoxSize``. Size of the region over which to generate the field. Units should be compatible with those of ``Pk``.
- ``threads``. Number of openmp threads. Only used when FFT the field from Fourier space to configuration space.
- ``verbose``. Whether output some information.

An example on how to generate these fields in 2D and 3D is this:
  
.. code-block:: python
		
   import numpy as np
   import density_field_library as DFL

   grid              = 128    #grid size
   BoxSize           = 1000.0 #Mpc/h
   seed              = 1      #value of the initial random seed
   Rayleigh_sampling = 0      #whether sampling the Rayleigh distribution for modes amplitudes
   threads           = 1      #number of openmp threads
   verbose           = True   #whether to print some information 

   # read power spectrum; k and Pk have to be floats, not doubles 
   k, Pk = np.loadtxt('my_Pk.txt', unpack=True)
   k, Pk = k.astype(np.float32), Pk.astype(np.float32)

   # generate a 2D Gaussian density field
   df_2D = DFL.gaussian_field_2D(grid, k, Pk, Rayleigh_sampling, seed, 
		                 BoxSize, threads, verbose)

   # generate a 3D Gaussian density field
   df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed, 
		                 BoxSize, threads, verbose)
