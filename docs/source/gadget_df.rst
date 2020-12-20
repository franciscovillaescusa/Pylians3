**************************
N-body simulations: Gadget
**************************

Pylians provides the routine ``density_field_gadget`` that simplifies the construction of 3D density fields from Gadget snapshots.

.. Note::

   This routine should also work smoothly with AREPO & GIZMO snapshots.

The arguments of this routine are:

- ``snapshot``. This is the name of the gadget snapshot. Pylians supports formats 1, 2 and hdf5. Set it as ``'snap_001'``, even if the files are ``'snap_001.0'``, ``'snap_001.1'``, ... or ``'snap_001.0.hdf5'``, ``'snap_001.1.hdf5'``.
- ``grid``. The constructed density field will be a 3D float numpy array with :math: grid*^3`` voxels. The larger this number the higher the resolution, but more memory will be used.
- ``ptypes``. Particle type over which compute the density field. It can be individual types, ``[0]`` (gas), ``[1]`` (cold dark matter), ``[2]`` (neutrinos), ``[3]`` (particle type 3), ``[4]`` (stars), ``[5]`` (black holes), or combinations. E.g. ``[0,1]`` (gas+cold dark matter), ``[0,4]`` (gas+stars), ``[0,1,2,4]`` (gas+CDM+neutrinos+stars). For all components (total matter) use ``[0,1,2,3,4,5]`` or ``[-1]``.
- ``MAS``. Mass-assignment scheme used to deposit particles mass to the grid. Options are: ``'NGP'`` (nearest grid point), ``'CIC'`` (cloud-in-cell), ``'TSC'`` (triangular-shape cloud), ``'PCS'`` (piecewise cubic spline). For most applications ``'CIC'`` is enough.
- ``do_RSD``. If ``True``, particles positions will be moved to redshift-space along the ``axis`` axis.
- ``axis``. Axis along which redshift-space distortions will be implemented (only needed if ``do_RSD=True``): 0, 1 or 2 for x-axis, y-axis or z-axis, respectively.
- ``verbose``. Whether to print some information on the routine progress.

This is an example of how to use this routine:

.. code-block:: python
		
   import numpy as np
   import MAS_library as MASL

   snapshot = 'snapdir_010/snap_010'  #snapshot name
   grid     = 512                     #grid size
   ptypes   = [1,2]                   #CDM + neutrinos
   MAS      = 'CIC'                   #Cloud-in-Cell
   do_RSD   = False                   #dont do redshif-space distortions
   axis     = 0                       #axis along which place RSD; not used here
   verbose  = True   #whether print information on the progress

   # Compute the effective number of particles/mass in each voxel
   delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis, verbose)

   # compute density contrast: delta = rho/<rho> - 1
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
