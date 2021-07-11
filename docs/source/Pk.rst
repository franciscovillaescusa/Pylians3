**************
Power spectrum
**************

Pylians provide several routines to compute power spectra, that we outline now.


3D
------------

Pylians provide routines to compute different power spectra for 3 dimensional fields.

.. _auto-Pk: 

Auto-power spectrum
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

Pylians also provides routines to compute the auto- and cross-power spectrum of multiple fields. For instance, to compute the auto- and cross-power spectra of two fields, ``delta1`` and ``delta2``:

.. code-block:: python
		
   import Pk_library as PKL

   Pk = PKL.XPk([delta1,delta2], BoxSize, axis, MAS=['CIC','CIC'], threads=1)

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

   
Gadget snapshots
~~~~~~~~~~~~~~~~

Pylians provides the routine ``Pk_Gadget`` that simplifies the computation of auto-power spectra from Gadget snapshots. The arguments of that routine are these:

- ``snapshot``. The name of the Gadget snapshot (supports format I, II and hdf5 files). If you have multiple files per snapshot, just use the prefix. For instance, if you have files as ``snapdir_004/snap_004.0.hdf5``, ``snapdir_004/snap_004.1.hdf5``, ``snapdir_004/snap_004.2.hdf5``...etc, use ``snapdir_004/snap_004``. For single files, you can use either the prefix or the full name.
- ``grid``. The routine will compute the density field on a regular grid with grid x grid x grid voxels. This will basically determine the size of the Nyquist frequency in the power spectrum calculation.
- ``particle_type``. The particle types to be used; this routine supports several types. For instance [1] for dark matter, [2] for neutrinos, [4] for stars. It can also be several of them, e.g. [1,2] for dark matter + neutrinos.
- ``do_RSD``. Whether move particles to redshift-space and compute power spectrum in redshift-space.
- ``axis``. Axis along which place the redshift-space distortions. Only matters if ``do_RSD = True``.
- ``cpus``. Number of openmp threads to be used in the calculation.
- ``folder_out``. Folder where to write the results. If set to ``None``, results will be written in the current folder.

An example of how to use this routine is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # parameters
   snapshot      = '/home/Paco/Quijote/Snapshots/fiducial/34/snapdir_004/snap_004' #snapshot name
   grid          = 512    #grid size
   particle_type = [1]    #use dark matter [1]
   do_RSD        = True   #move particles to redshift-space and calculate Pk in redshift-space
   axis          = 1      #RSD placed along the y-axis
   cpus          = 8      #number of openmp threads
   folder_out    = '/home/Paco/Quijote/Pk/fiducial/34' #folder where to write results

   # compute power spectrum of the snapshot
   PKL.Pk_Gadget(snapshot, grid, particle_type, do_RSD, axis, cpus, folder_out)

Calling the routine will compute the auto-power spectrum of the different particle types and their cross-power spectra (for multiple particle types). It will write files for the different auto- and cross-power spectra. The format of the files will be ``k Pk0 Pk2 Pk4 Nmodes``, where ``k`` is the wavenumber in units of h/Mpc (if snapshot is in kpc/h units), ``Pk0``, ``Pk2``, and ``Pk4`` are the monopole, quadrupole, and hexadecapole in units of (Mpc/h)^3 and ``Nmodes`` is the number of modes inside each k-bin.
   

Marked-power spectrum
~~~~~~~~~~~~~~~~~~~~~

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

Velocity power spectrum
~~~~~~~~~~~~~~~~~~~~~~~

Pylians provides a routine, ``Pk_theta`` that computes the power spectrum of the divergence of a 3D velocity field: :math:`P_{\theta \theta}`, where :math:`\theta=\vec{\nabla}\cdot\vec{V}`. The arguments of the routine are these:

- ``Vx``. A 3D numpy float32 array containing the x component of the 3D velocity field, e.g. ``Vx = np.zeros((128,128,128), dtype=np.float32)``.
- ``Vy``. A 3D numpy float32 array containing the y component of the 3D velocity field.
- ``Vz``. A 3D numpy float32 array containing the z component of the 3D velocity field.
- ``BoxSize``. The size of the simulation box. Units here will determine units of output.
  - ``axis``. Axis along which compute the quadrupole, hexadecapole for the theta Pk. If the velocities are in real-space set ``axis=0``. If the velocities are in redshift-space set ``axis=0``, ``axis=1`` or ``axis=2`` if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively. 
- ``MAS``. Mass-assignment scheme used to generate the velocity field, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the velocity field has not been generated with any of these set it to ``'None'``. This is used to correct for the MAS when computing the power spectrum.
- ``threads``. Number of openmp threads to be used.

An example of how to use this routine is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # parameters
   BoxSize = 1000.0 #Mpc/h
   axis    = 0      #velocity fields in real-space; this variable is not relevant in real-space
   MAS     = 'CIC'  #mass-assignment scheme used to create the velocity field
   threads = 20     #number of openmp threads to be used

   # compute the theta auto-power spectrum
   k, Pk, Nmodes = PKL.Pk_theta(Vx,Vy,Vz,BoxSize,axis,MAS,threads)

   # k will be in h/Mpc units. Pk will have (km/s)^2*(Mpc/h)^3 considering that the velocity field is in km/s

.. warning::

   One of the well known problems of computing the velocity power spectrum is empty voxels. When constructing the velocity field, it may happen that no particles reside within (or around) a given voxel. In this case, the velocity field is not well defined. In general, a zero velocity is assigned to that voxel, but that could be a very wrong assumption: for instance, inside voids the the number of particle/galaxy tracers may be low, but the underlying velocity field may be very different to 0. Thus, when using this routine, it is important to make convergence tests (e.g. using different grid sizes for the velocity field) to study the extent and/or presence of this problem.

Momentum power spectrum
~~~~~~~~~~~~~~~~~~~~~~~

Differently to the velocity field, the momentum field, :math:`\vec{p}=\rho \vec{V}`, is well-defined everywhere (even in voxels where there are no particles). Pylians provides the routine ``XPk_dv`` that computes :math:`P_{\delta\delta}`, :math:`P_{\tilde{\theta}\tilde{\theta}}`, and :math:`P_{\delta\tilde{\theta}}`, where :math:`\delta=\rho/\bar{\rho}-1` and :math:`\tilde{\theta}=\vec{\nabla}\cdot(1+\delta)\vec{V}`. The arguments of the function are these:

- ``delta``. A 3D numpy float32 array containing the value of the density constrast in each voxel.
- ``Vx``. A 3D numpy float32 array containing the x component of the 3D velocity field, e.g. ``Vx = np.zeros((128,128,128), dtype=np.float32)``.
- ``Vy``. A 3D numpy float32 array containing the y component of the 3D velocity field.
- ``Vz``. A 3D numpy float32 array containing the z component of the 3D velocity field.
- ``BoxSize``. The size of the simulation box. Units here will determine units of output.
  - ``axis``. Axis along which compute the quadrupole, hexadecapole for the theta Pk. If the velocities are in real-space set ``axis=0``. If the velocities are in redshift-space set ``axis=0``, ``axis=1`` or ``axis=2`` if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively. 
- ``MAS``. Mass-assignment scheme used to generate the velocity field, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the velocity field has not been generated with any of these set it to ``'None'``. This is used to correct for the MAS when computing the power spectrum.
- ``threads``. Number of openmp threads to be used.

An example on how to use this routine is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # parameters
   BoxSize  = 1000.0  #Mpc/h
   axis     = 0       #no RSD
   MAS      = 'CIC'   #it assumes the density constrast and velocities have been generated with the same MAS
   threads  = 2       #number of openmp threads

   # compute the density constrast and momentum auto- and cross-power spectra
   # k will have units of h/Mpc
   # Pk_dd will contain the standard power spectrum in (Mpc/h)^3
   # Pk_tt will be the momentum auto-power spectrum, defined as above, and with units of (km/s)^2*(Mpc/h)^3 in units of velocity field are (km/s)
   # Pk_dt will be the density-momentum cross-power spectrum with (km/s)*(Mpc/h)^3 units if velocity field has (km/s) units
   k, Pk_dd, Pk_tt, Pk_dt, Nmodes = PKL.XPk_dv(delta, Vx, Vy, Vz, BoxSize, axis, MAS, threads)
   

Binned power spectrum
~~~~~~~~~~~~~~~~~~~~~

Sometimes we may want to compare the power spectrum measured in a simulation versus the theoretical one (e.g. the linear power spectrum). On large scales, the number of modes will be small, so the binning used to compute the power spectrum becomes important when comparing simulations versus theory. Pylians provides the routine ``expected_Pk`` that will take a power spectrum and will bin it in the same way as is done with the simulations, so a comparison k by k is appropiate.

The ingredients needed are:

- ``k_in``. This is an array with the values of k. 
- ``Pk_in``. This is an array with the values of the power spectrum at ``k_in``. 
- ``BoxSize``. Size of the simulation. If you want to bin the input power spectrum in the same way as the power spectrum measured from a simulation with 1000 Mpc/h, then set ``BoxSize = 1000.0``. This parameter determines the fundamental frequency.
- ``grid``. The routine will bin the power spectrum according to a mesh with grid x grid x grid voxels. This parameters determines the Nyquist frequency.
- ``bins``. The routine will read the input Pk and interpolate it to the k-values sampled in the regular grid. It is desirable to first interpolate the input Pk into a finer 1D mesh to avoid larger errors in the interpolation. This parameter sets the number of bins for that. The more the better, but something around 1000-5000 should be enough.

An example is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # value of the parameters
   f_in    = 'my_linear_Pk.txt'  #input power spectrum
   BoxSize = 1000.0 #Mpc/h       #same of box to compute the binned Pk
   grid    = 256                 #compute binned Pk using a mesh with grid^3 voxels
   bins    = 2000                #number of bins to interpolate the input Pk
   
   # read input power spectrum
   k_in, Pk_in = np.loadtxt(f_in, unpack=True)
   
   # get binned Pk: returns k, power spectrum and number of modes in each k-bin
   k, Pk, Nmodes = PKL.expected_Pk(k_in, Pk_in, BoxSize, grid, bins)


2D
----

The routines Pylians provide to compute power spectra for 2 dimensional (images/planes) are these:

Auto-power spectrum
~~~~~~~~~~~~~~~~~~~

Pylians can also compute auto-power spectra of images/planes through the ``Pk_plane`` routine. The ingredients needed are:

- ``delta``. This should be a 2D numpy float32 array, like ``delta = np.zeros((128,128), dtype=np.float32)``.
- ``BoxSize``. The size of the plane.
- ``MAS``. Mass-assignment scheme used to generate the 2D density field, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the density field has not been generated with any of these, set it to ``'None'``. This is used to correct for the MAS when computing the power spectrum.
- ``threads``. Number of openmp threads to be used in the calculation.

An example of how to utilize this function is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # parameters
   grid    = 128     #the map will have grid^2 pixels
   BoxSize = 1000.0  #Mpc/h
   MAS     = 'None'  #MAS used to create the image; 'NGP', 'CIC', 'TSC', 'PCS' o 'None'
   threads = 1       #number of openmp threads

   # create an empty image
   delta = np.zeros((grid,grid), dtype=np.float32)

   # compute the Pk of that image
   Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads) 

   # get the attributes of the routine
   k      = Pk2D.k      #k in h/Mpc
   Pk     = Pk2D.Pk     #Pk in (Mpc/h)^2
   Nmodes = Pk2D.Nmodes #Number of modes in the different k bins
   
   
Cross-power spectrum
~~~~~~~~~~~~~~~~~~~~

Pylians provide the routine ``XPk_plane`` to compute cross-power spectrum between two images. The ingredients needed are:

delta1, delta2, BoxSize, MAS1=None, MAS2=None, threads=1):

- ``delta1``. A 2D numpy float32 array containing the data of the first image.
- ``delta2``. A 2D numpy float32 array containing the data of the second image.
- ``BoxSize``. Size of the plane. Note that the size of both images should be the same.
- ``MAS1``. The MAS (mass assignment scheme) employed to construct the first image, if any. Possible options are ``'NGP'``, ``'CIC'``, ``'TSC'``, ``'PCS'``.  If the density field has not been generated with any of these, set it to ``'None'``. This is used to correct for the MAS when computing the power spectrum.
- ``MAS2``. Same as ``MAS1`` but for the second image.
- ``threads``. Number of openmp threads to use.

An example of how to utilize this routine is this:

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # parameters
   BoxSize = 1000.0 #Mpc/h
   MAS1    = 'CIC'
   MAS2    = 'None'
   threads = 1

   # compute cross-power spectrum between two images
   XPk2D = PKL.XPk_plane(delta1, delta2, BoxSize, MAS1, MAS2, threads)

   # get the attributes of the routine
   k      = XPk2D.k        #k in h/Mpc
   Pk     = XPk2D.Pk       #auto-Pk of the two maps in (Mpc/h)^2
   Pk1    = Pk[:,0]        #auto-Pk of the first map in (Mpc/h^2)
   Pk2    = Pk[:,1]        #auto-Pk of the second map in (Mpc/h^2)
   XPk    = XPk2D.XPk      #cross-Pk in (Mpc/h)^2
   r      = XPk2D.r        #cross-correlation coefficient
   Nmodes = XPk2D.Nmodes   #number of modes in each k-bin
   
