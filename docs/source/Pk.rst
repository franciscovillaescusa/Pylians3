**************
Power spectrum
**************

Pylians provide several routines to compute power spectra, that we outline now.


.. _auto-Pk: 

Auto-power spectrum
-------------------

The ingredients needed to compute the auto-power spectra are:

- ``delta``. This is the density, overdensity or density contrast field. It should be a 3 dimensional float numpy array such ``delta = np.zeros((grid, grid, grid), dtype=np.float32)``. See :ref:`density_fields` on how to compute  density fields using Pylians.
- ``BoxSize``. Size of the periodic box. The units of the output power spectrum depend on this.
- ``axis``. Axis along which compute the quadrupole, hexadecapole and the 2D power spectrum. If the field is in real-space set ``axis=0``. If the field is in redshift-space set ``axis=0``, ``axis=1`` or ``axis=2`` if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively. 
- ``MAS``. Mass-assignment scheme used to generate the density field, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the density field has not been generated with any of these set it to ``'None'``. This is used to correct for the MAS when computing the power spectrum.
- ``threads``. Number of openmp threads to be used.
- ``verbose``. Whether print information on the status/progress of the calculation: True or False

An example on how to compute the power spectrum is this:

.. code-block:: python
		
   import Pk_library as PKL

   # compute power spectrum
   Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)

   # Pk is a python class containing the 1D, 2D and 3D power spectra, that can be retrieved as

   # 1D P(k)
   k1D      = Pk.k1D      
   Pk1D     = Pk.Pk1D     
   Nmodes1D = Pk.Nmodes1D  

   # 2D P(k)
   kpar     = Pk.kpar    
   kper     = Pk.kper
   Pk2D     = Pk.Pk2D
   Nmodes2D = Pk.Nmodes2D

   # 3D P(k)
   k       = Pk.k3D
   Pk0     = Pk.Pk[:,0] #monopole
   Pk2     = Pk.Pk[:,1] #quadrupole
   Pk4     = Pk.Pk[:,2] #hexadecapole
   Pkphase = Pk.Pkphase #power spectrum of the phases
   Nmodes  = Pk.Nmodes3D

.. note::

   The 1D power spectrum is computed as :math:`P_{\rm 1D}(k_\parallel)=\int \frac{d^2\vec{k}_\bot}{(2\pi)^2}P_{\rm 3D}(k_\parallel,k_\bot)`. This shouldn't be confused with the traditional 3D power spectrum.


Cross-power spectrum
--------------------

Pylians also provides routines to compute the auto- and cross-power spectrum of multiple fields. For instance, to compute the auto- and cross-power spectra of two fields, ``delta1`` and ``delta2``:

.. code-block:: python
		
   import Pk_library as PKL

   Pk = PKL.XPk([delta1,delta2], BoxSize, axis, MAS=['CIC','CIC'], threads)

A description of the variables ``BoxSize``, ``axis``, ``MAS`` and ``threads`` can be found in :ref:`auto-Pk`. As with the auto-power spectrum, ``delta1`` and ``delta2`` need to be 3D float numpy arrays. ``Pk`` is a python class that contains all the following information

.. code-block:: python
		
   # 1D P(k)
   k1D      = Pk.k1D
   Pk1D_1   = Pk.Pk1D[:,0]  #field 1
   Pk1D_2   = Pk.Pk1D[:,1]  #field 2
   Pk1D_X   = Pk.PkX1D[:,0] #field 1 - field 2 cross 1D P(k)
   Nmodes1D = Pk.Nmodes1D

   # 2D P(k)
   kpar     = Pk.kpar
   kper     = Pk.kper
   Pk2D_1   = Pk.Pk2D[:,0]  #2D P(k) of field 1
   Pk2D_2   = Pk.Pk2D[:,1]  #2D P(k) of field 2
   Pk2D_X   = Pk.PkX2D[:,0] #2D cross-P(k) of fields 1 and 2
   Nmodes2D = Pk.Nmodes2D

   # 3D P(k)
   k      = Pk.k3D
   Pk0_1  = Pk.Pk[:,0,0]  #monopole of field 1
   Pk0_2  = Pk.Pk[:,0,1]  #monopole of field 2
   Pk2_1  = Pk.Pk[:,1,0]  #quadrupole of field 1
   Pk2_2  = Pk.Pk[:,1,1]  #quadrupole of field 2
   Pk4_1  = Pk.Pk[:,2,0]  #hexadecapole of field 1
   Pk4_2  = Pk.Pk[:,2,1]  #hexadecapole of field 2
   Pk0_X  = Pk.XPk[:,0,0] #monopole of 1-2 cross P(k)
   Pk2_X  = Pk.XPk[:,1,0] #quadrupole of 1-2 cross P(k)
   Pk4_X  = Pk.XPk[:,2,0] #hexadecapole of 1-2 cross P(k)
   Nmodes = Pk.Nmodes3D

The ``XPk`` function can be used for more than two fields, e.g.

.. code-block:: python
		
   BoxSize = 1000.0 #Mpc/h
   axis    = 0
   MAS     = ['CIC','NGP','TSC','None']
   threads = 16

   Pk = PKL.XPk([delta1,delta2,delta3,delta4], BoxSize, axis, MAS, threads)
   

Marked-power spectrum
---------------------

The above routines can be used for standard fields or for marked fields. The script below shows an example of how to compute a marked power spectrum where each particle is weighted by its mean density within a radius of 10 Mpc/h (see :ref:`smoothing` to see how to smooth a field).

.. code-block:: python

   import numpy as np
   import MAS_library as MASL
   import Pk_library as PKL
   import smoothing_library as SL

   ################################ INPUT ######################################
   # parameters to construct density field
   grid    = 512    #grid size
   BoxSize = 1000   #Mpc/h
   MAS     = 'CIC'  #Cloud-in-Cell

   # parameters to smooth the field
   R       = 10.0      #Mpc/h; smoothing scale
   Filter  = 'Top-Hat' #filter
   threads = 1         #openmp threads

   # Pk parameters
   do_RSD  = False   #whether do redshift-space distortions
   axis    = 0       #axis along which place RSD
   verbose = True    #whether to print some information on the calculation progress
   #############################################################################


   ###### compute density field #######
   # define the array hosting the density constrast field
   delta = np.zeros((grid,grid,grid), dtype=np.float32)

   # read the particle positions
   pos = np.loadtxt('myfile.txt') #Mpc/h
   pos = pos.astype(np.float32)   #pos should be a numpy float array

   # compute number of particles in each voxel
   MASL.MA(pos,delta,BoxSize,MAS)

   # compute density contrast: delta = rho/<rho> - 1
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
   ####################################

   ######### smooth the field #########
   # compute FFT of the filter
   W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

   # smooth the field
   delta_smoothed = SL.field_smoothing(delta, W_k, threads)
   ####################################

   ########### find mark ##############
   # find the value of the smoothed density field in the position of each particle
   mark = np.zeros(pos.shape[0], dtype=np.float32)

   # find the value of the density field at the positions pos
   MASL.CIC_interp(delta, BoxSize, pos, mark)
   del delta # we dont need delta anymore; save some memory
   ####################################

   ######## compute marked Pk #########
   # construct a density field weighting each particle by its overdensity
   marked_field = np.zeros((grid,grid,grid), dtype=np.float32)
   MASL.MA(pos, marked_field, BoxSize, MAS, W=mark)

   # compute marked power spectrum
   MPk = PKL.Pk(marked_field, BoxSize, axis, MAS, threads, verbose)

   # save 3D marked Pk to file
   np.savetxt('My_marked_Pk.txt', np.transpose([MPk.k3D, MPk.Pk[:,0]]))
   ####################################
