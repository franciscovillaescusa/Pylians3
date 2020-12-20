************
Construction
************

Pylians provides the routine ``MA`` to construct 2D and 3D density fields from the positions of particles. That routine can also construct marked fields, by weigthing each particle according to some weigh. The arguments of that routine are these:

- ``pos``. The positions of the particles, either in 2D or 3D. It should be a numpy float32 array; e.g. in 3D should be something like ``pos = np.zeros((1000,3), dtyp=np.float32)``.
- ``field``. This is a numpy float32 array in either 2D or 3D that will contain the density field.
- ``BoxSize``. Size of the cubic region (in 3D) or the rectangular plane (2D).
- ``MAS``. Mass-assignment scheme used to deposit particles mass to the grid. Options are: ``'NGP'`` (nearest grid point), ``'CIC'`` (cloud-in-cell), ``'TSC'`` (triangular-shape cloud), ``'PCS'`` (piecewise cubic spline). For most applications ``'CIC'`` is enough.
- ``W``. The weight associated to each particle, if any. If no weights used, set it ``None``.
- ``verbose``. Whether to print some information on the progress.

.. Note::

   If you want to construct a field in redshift-space, you will need the particle positions in redshift-space. See :ref:`RSD` on how move particles, halos, galaxies...etc, from real to redshift-space.

We now provide examples on how to use this routine:

Density field in 3D
~~~~~~~~~~~~~~~~~~~

This example shows how to compute the density constrast field from the positions of particles in 3D.

.. code-block:: python

   import numpy as np
   import MAS_library as MASL

   # number of particles
   Np = 128**3
   
   # density field parameters
   grid    = 128    #the 3D field will have grid x grid x grid voxels
   BoxSize = 1000.0 #Mpc/h ; size of box
   MAS     = 'CIC'  #mass-assigment scheme
   verbose = True   #print information on progress

   # particle positions in 3D
   pos = np.random.random((Np,3)).astype(np.float32)*BoxSize

   # define 3D density field
   delta = np.zeros((grid,grid,grid), dtype=np.float32)

   # construct 3D density field
   MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

   # at this point, delta contains the effective number of particles in each voxel
   # now compute overdensity and density constrast
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

After the last line, ``delta`` contains the density constrast field, defined as :math:`\delta(x)=\rho(x)/\bar{\rho}-1`, where :math:`\rho(x)` is the value of the density field at position :math:`x`.
   

Density field in 2D
~~~~~~~~~~~~~~~~~~~

This example shows how to compute the density constrast field from the positions of particles in 2D.

.. code-block:: python

   import numpy as np
   import MAS_library as MASL

   # number of particles
   Np = 256**2
   
   # density field parameters
   grid    = 256    #the 2D field will have grid x grid pixels
   BoxSize = 1000.0 #Mpc/h ; size of box
   MAS     = 'TSC'  #mass-assigment scheme
   verbose = True   #print information on progress

   # particle positions in 2D
   pos = np.random.random((Np,2)).astype(np.float32)*BoxSize

   # define 2D density field
   delta = np.zeros((grid,grid), dtype=np.float32)

   # construct 2D density field
   MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

   # at this point, delta contains the effective number of particles in each pixel
   # now compute overdensity and density constrast
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

After the last line, ``delta`` contains the density constrast field, defined as :math:`\delta(x)=\rho(x)/\bar{\rho}-1`, where :math:`\rho(x)` is the value of the density field at position :math:`x`.

Gas density field in 3D
~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to construct a gas density field in 3D, where the position of the particles, together with their associated gas masses are used.


.. code-block:: python

   import numpy as np
   import MAS_library as MASL

   # number of particles
   Np = 128**3
   
   # density field parameters
   grid    = 128    #the 3D field will have grid x grid x grid voxels
   BoxSize = 1000.0 #Mpc/h ; size of box
   MAS     = 'CIC'  #mass-assigment scheme
   verbose = True   #print information on progress

   # particle positions in 3D
   pos = np.random.random((Np,3)).astype(np.float32)*BoxSize

   # gas masses of the particles (masses goes from 0 to 1)
   mass = np.random.random(Np).astype(np.float32) #Msun/h
   
   # define 3D density field
   delta = np.zeros((grid,grid,grid), dtype=np.float32)

   # construct 3D density field
   MASL.MA(pos, delta, BoxSize, MAS, W=mass, verbose=verbose)

   # at this point, delta contains the effective gas mass in each voxel
   # now compute overdensity and density constrast
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

After the last line, ``delta`` contains the gas density constrast field, defined as :math:`\delta_{\rm g}(x)=\rho_{\rm g}(x)/\bar{\rho}_{\rm g}-1`, where :math:`\rho_{\rm g}(x)` is the value of the gas density field at position :math:`x`.

.. note::

   Marked density fields (see e.g. `this paper <https://arxiv.org/abs/2001.11024>`_) can be constructed by using the considered mark as a weigh for every particle or galaxy.

Gadget snapshots
~~~~~~~~~~~~~~~~

Pylians provides the routine ``density_field_gadget`` to facilitate the construction of density fields from Gadget simulations (both N-body and hydrodynamic). The arguments of the routine are these:
 
snapshot_fname, ptypes, dims, MAS='CIC',
                         do_RSD=False, axis=0, verbose=True

- ``snapshot``. The name of the Gadget snapshot (supports format I, II and hdf5 files). If you have multiple files per snapshot, just use the prefix. For instance, if you have files as ``snapdir_004/snap_004.0.hdf5``, ``snapdir_004/snap_004.1.hdf5``, ``snapdir_004/snap_004.2.hdf5``...etc, use ``snapdir_004/snap_004``. For single files, you can use either the prefix or the full name.
- ``ptypes``. The particle types to be used; this routine supports several types. For instance [1] for dark matter, [2] for neutrinos, [4] for stars. It can also be several of them, e.g. [1,2] for dark matter + neutrinos.
- ``grid``. The routine will compute the density field on a regular grid with grid x grid x grid voxels. 
- ``MAS``. Mass-assignment scheme used to deposit particles mass to the grid. Options are: ``'NGP'`` (nearest grid point), ``'CIC'`` (cloud-in-cell), ``'TSC'`` (triangular-shape cloud), ``'PCS'`` (piecewise cubic spline). For most applications ``'CIC'`` is enough.
- ``do_RSD``. Whether to construct the density field in real- or redshift-space. If ``True``, the particle positions will be displaced to redshift-space and the density fields will be constructed from the particle positions in redshift-space.
- ``axis``. Axis along which place the redshift-space distortions. Only matters if ``do_RSD = True``.
- ``verbose``. Whether to print some information on the routine progress.

An example of how to use this routine is this:

.. code-block:: python

   import numpy as np
   import MAS_library as MASL
   
   # parameters
   snapshot = '/home/Paco/Quijote/Snapshots/Mnu_p/34/snapdir_003/snap_003'
   ptypes   = [1,2]  #dark matter + neutrinos
   grid     = 512    #regular grid will have grid x grid x grid voxels
   MAS      = 'CIC'  #mass-assignment scheme
   do_RSD   = False  #whether to create the density field in redshift-space
   axis     = 0      #axis along which place RSD. Not used if do_RSD = False
   verbose  = True   #whether print information on the progress

   # construct density field
   delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis, verbose)

   # delta contains the value of the density field in each voxel. In order to construct the
   # overdensity, or the density constrast field do:
   delta /= np.mean(delta, dtype=np.float64) #overdensity field (rho/<rho>)
   delta -= 1.0                              #density constrast field (rho/<rho> - 1)
   
