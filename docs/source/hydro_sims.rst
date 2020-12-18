************************
Hydrodynamic simulations
************************

2 dimensions
------------

Imagine that you have a collection of particles at some positions ``pos`` and each particle represents either a sphere (for SPH simulations) or a voronoi cell (for moving mesh simulations). If you would like to use a more physical mass assignment scheme than NGP, CIC...etc, Pylians provide several routines to deal with these situations and create 2D density fields. We now describe the available routines.


Sampling with tracers
~~~~~~~~~~~~~~~~~~~~~

The routine ``voronoi_NGP_2D`` is designed to take as input particle positions (voronoi cells) in 2D, masses and radii from moving mesh hydrodynamic simulations and compute the density field in a 2D region. This routine works as follows. It considers each particle as a uniform circle and associate the mass on it to the grid itself. It achieves that by splitting the circle into ``r_divisions`` shells that have the same area. Then, it associates to each shell a number of ``particles_per_cell`` that are distributed equally in angle. Finally, each of those subparticles belonging to the initial circle, is associated to a grid cell using the NGP mass assignment scheme. Note that this routine can be very computationally expensive if each particle is subsampled with many subparticles. The ingredients needed are:

- ``density``. This is the 2D density field that the routine will fill up. It should be a double numpy array.
- ``pos``. These are the positions of the particles, either in 2D or 3D. Should be float numpy array.
- ``mass``. This is a 1D array with the masses (or other property) of the particles. Should be a float numpy array.
- ``radii``. This is a 1D float numpy array with the radii of the particles. If only volume is available, radii can be computed as 4*pi/3*R^3 = Volume.
- ``x_min``, ``y_min`` & ``BoxSize``. The routine will compute the density field in a region with coordinates [``x_min:x_min+BoxSize`` , ``y_min:y_min+BoxSize``]. Units should be Mpc/h.
- ``particles_per_cell``. Total number of particles to subsample each particle (voronoi cell).
- ``r_divisions``. Number of circular shells to use to subsample each particle (voronoi cell)
- ``periodic``. Whether use periodic boundary conditions for the considered region.

.. code-block:: python
		
   import numpy as np
   import MAS_library as MASL

   x_min, y_min, BoxSize = 0.0, 0.0, 25.0 #Mpc/h; origin and size of considered region
   grid               = 512               #size of grid
   particles_per_cell = 1000              #total number of tracers to assign to each particle
   r_divisions        = 7                 #number of radial divisions
   periodic           = True              #whether the considered region has periodic conditions

   # define the 2D density field
   density = np.zeros((grid,grid), dtype=np.float64)

   # compute 2D density field from particle positions, radii and masses
   MASL.voronoi_NGP_2D(density, pos, mass, radii, x_min, y_min, BoxSize,
		       particles_per_cell, r_divisions, periodic)


Column density: voronoi cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This routine is designed to take as input particle positions (voronoi cells) in 3D, masses and radii from moving mesh hydrodynamic simulations and compute the column density field in a 2D region. Note that the difference with respect to the above routine is that in this case we compute the projected mass density, not the density itself (as above). This routine works as follows. It considers each particle/cell as a uniform sphere. It then takes a regular grid with the same dimensions as ``density``, and in each grid cell computes the projected density of all particles contributing to that line-of-sight. Notice that in this case mass conservation is not fullfilled, as in each grid cell a single line-of-sight is considered. The ingredients needed are:

- ``density``. This is the 2D density field that the routine will fill up. It should be a double numpy array.
- ``pos``. These are the positions of the particles, either in 2D or 3D. Should be float numpy array.
- ``mass``. This is a 1D array with the masses (or other property) of the particles. Should be a float numpy array.
- ``radius``. This is a 1D float numpy array with the radii of the particles. If only volume is available, radii can be computed as 4*pi/3*R^3 = Volume.
- ``x_min``, ``y_min`` & ``BoxSize``. The routine will compute the density field in a region with coordinates [``x_min:x_min+BoxSize`` , ``y_min:y_min+BoxSize``]. Units should be Mpc/h.
- ``axis_x``, ``axis_y``. Integers to select the axes along which made the projection: 0(X), 1(Y) or 2(Z).
- ``periodic``. Whether use periodic boundary conditions for the considered region.
- ``verbose``. Whether show information on the computation.

.. code-block:: python
		
   import numpy as np
   import MAS_library as MASL

   x_min, y_min, BoxSize = 0.0, 0.0, 25.0 #Mpc/h; origin and size of the considered region
   axis_x, axis_y     = 0, 1              #0(X), 1(Y), 2(Z)
   grid               = 512               #grid size
   periodic           = True              #whether the considered region has periodic conditions
   
   # define the array hosting the 2D field
   density = np.zeros((grid,grid), dtype=np.float64)

   # compute the density field
   MASL.voronoi_RT_2D(density, pos, mass, radius, x_min, y_min,
		      axis_x, axis_y, BoxSize, periodic, verbose=True)

.. Note::

   More detailed scripts can be found `here <https://camels.readthedocs.io/en/latest/images.html>`_. 
		   

Column density: SPH
~~~~~~~~~~~~~~~~~~~

This routine is basically the same as the above, but instead of assuming uniform spheres, uses the SPH kernel as its internal density profile. The ingredients needed are:

- ``density``. This is the 2D density field that the routine will fill up. It should be a double numpy array.
- ``pos``. These are the positions of the particles, either in 2D or 3D. Should be float numpy array.
- ``mass``. This is a 1D array with the masses (or other property) of the particles. Should be a float numpy array.
- ``radius``. This is a 1D float numpy array with the radii of the particles. If only volume is available, radii can be computed as 4*pi/3*R^3 = Volume.
- ``x_min``, ``y_min`` & ``BoxSize``. The routine will compute the density field in a region with coordinates [``x_min:x_min+BoxSize`` , ``y_min:y_min+BoxSize``]. Units should be Mpc/h.
- ``axis_x``, ``axis_y``. Integers to select the axes along which made the projection: 0(X), 1(Y) or 2(Z).
- ``periodic``. Whether use periodic boundary conditions for the considered region.
- ``verbose``. Whether show information on the computation.

.. code-block:: python
		
   import numpy as np
   import MAS_library as MASL

   x_min, y_min, BoxSize = 0.0, 0.0, 25.0 #Mpc/h; origin and size of considered region
   axis_x, axis_y     = 0, 1              #0(X), 1(Y), 2(Z)
   grid               = 512               #grid size
   periodic           = True              #whether the considered region has periodic conditions

   # define the array hosting the 2D field
   density = np.zeros((grid,grid), dtype=np.float64)

   # compute the density field
   MASL.SPH_RT_2D(density, pos, mass, radius, x_min, y_min,
		  axis_x, axis_y, BoxSize, periodic, verbose=True)

   
		  
3 dimensions
------------

In hydrodynamic simulations, gas is usually modelled as spheres or voronoi cells. In this case, instead of using the standard mass assignment schemes such as NPG, CIC or TSC, it is better to associate these spheres to the regular grid. We recomment using this code to achieve this:

`voxelize <https://github.com/leanderthiele/voxelize>`_
