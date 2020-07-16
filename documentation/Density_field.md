# Density field

Here we describe other routines Pylians incorporate to work with density fields.

### Density at particle positions

Imagine that a 3D density field ```field``` is given. The value of that field at some positions ```pos``` can be calculated using the routine ```CIC_interp```. That routine uses the CIC scheme to interpolate the density field into the particle positions. The ingredients are

- ```field```. This is the density field. Should be a 3D float numpy array
- ```BoxSize```. Size of the density field. Units in Mpc/h
- ```pos```. These are the positions of the particles. Units in Mpc/h
- ```density_interpolated```. This is a 1D array with the value of the density field at the particle positions. Should be a numpy float array: ```density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)```.

```python
import MAS_library as MASL

MASL.CIC_interp(field, BoxSize, pos, density_interpolated)
```

### Hydrodynamic interpolations

Imagine that you have a collection of particles at some positions ```pos``` and each particle represents either a sphere (for SPH simulations) or a voronoi cell (for moving mesh simulations). If you would like to use a more physical mass assignment scheme than NGP, CIC...etc, Pylians provide several routines to deal with these situations. We now describe the available routines.

#### ```voronoi_NGP_2D```

This routine is designed to take as input particle positions (voronoi cells) in 2D, masses and radii from moving mesh hydrodynamic simulations and compute the density field in a 2D region. This routine works as follows. It considers each particle as a uniform circle and associate the mass on it to the grid itself. It achieves that by splitting the circle into ```r_divisions``` shells that have the same area. Then, it associates to each shell a number of ```particles_per_cell``` that are distributed equally in angle. Finally, each of those subparticles belonging to the initial circle, is associated to a grid cell using the NGP mass assignment scheme. We notice that this routine can be very computationally expensive if each particle is subsampled with many subparticles. The ingredients needed are:

- ```density```. This is the 2D density field that the routine will fill up. It should be a double numpy array.
- ```pos```. These are the positions of the particles, either in 2D or 3D. Should be float numpy array.
- ```mass```. This is a 1D array with the masses (or other property) of the particles. Should be a float numpy array.
- ```radii```. This is a 1D float numpy array with the radii of the particles. If only volume is available, radii can be computed as 4*pi/3*R^3 = Volume.
- ```x_min```, ```y_min``` & ```BoxSize```. The routine will compute the density field in a region with coordinates [```x_min:x_min+BoxSize``` , ```y_min:y_min+BoxSize```]. Units should be Mpc/h.
- ```particles_per_cell```. Total number of particles to subsample each particle (voronoi cell).
- ```r_divisions```. Number of circular shells to use to subsample each particle (voronoi cell)
- ```periodic```. Whether use periodic boundary conditions for the considered region.

```python
import numpy as np
import MAS_library as MASL

x_min, y_min, BoxSize = 0.0, 0.0, 25.0 #Mpc/h
grid               = 512
particles_per_cell = 1000
r_divisions        = 7
periodic           = True

density = np.zeros((grid,grid), dtype=np.float64)

MASL.voronoi_NGP_2D(density, pos, mass, radii, x_min, y_min, BoxSize,
	            particles_per_cell, r_divisions, periodic)
```

#### ```voronoi_RT_2D```

This routine is designed to take as input particle positions (voronoi cells) in 3D, masses and radii from moving mesh hydrodynamic simulations and compute the column density field in a 2D region. Notice that the difference with respect to the above routine is that in this case we compute the projected mass density, not the density itself (as above). This routine works as follows. It considers each particle/cell as a uniform sphere. It then takes a regular grid with the same dimensions as ```density```, and in each grid cell computes the projected density of all particles contributing to that line-of-sight. Notice that in this case mass conservation is not fullfilled, as in each grid cell a single line-of-sight is considered. The ingredients needed are:

- ```density```. This is the 2D density field that the routine will fill up. It should be a double numpy array.
- ```pos```. These are the positions of the particles, either in 2D or 3D. Should be float numpy array.
- ```mass```. This is a 1D array with the masses (or other property) of the particles. Should be a float numpy array.
- ```radius```. This is a 1D float numpy array with the radii of the particles. If only volume is available, radii can be computed as 4*pi/3*R^3 = Volume.
- ```x_min```, ```y_min``` & ```BoxSize```. The routine will compute the density field in a region with coordinates [```x_min:x_min+BoxSize``` , ```y_min:y_min+BoxSize```]. Units should be Mpc/h.
- ```axis_x```, ```axis_y```. Integers to select the axes along which made the projection: 0(X), 1(Y) or 2(Z).
- ```periodic```. Whether use periodic boundary conditions for the considered region.
- ```verbose```. Whether show information on the computation.

```python
import numpy as np
import MAS_library as MASL

x_min, y_min, BoxSize = 0.0, 0.0, 25.0 #Mpc/h
axis_x, axis_y     = 0, 1 #0(X), 1(Y), 2(Z)
grid               = 512
periodic           = True

density = np.zeros((grid,grid), dtype=np.float64)

MASL.voronoi_RT_2D(density, pos, mass, radius, x_min, y_min,
                   axis_x, axis_y, BoxSize, periodic, verbose=True)
```

#### ```SPH_RT_2D```

This routine is basically the same as the above, but instead of assuming uniform spheres, uses the SPH kernel as its internal density profile. The ingredients needed are:

- ```density```. This is the 2D density field that the routine will fill up. It should be a double numpy array.
- ```pos```. These are the positions of the particles, either in 2D or 3D. Should be float numpy array.
- ```mass```. This is a 1D array with the masses (or other property) of the particles. Should be a float numpy array.
- ```radius```. This is a 1D float numpy array with the radii of the particles. If only volume is available, radii can be computed as 4*pi/3*R^3 = Volume.
- ```x_min```, ```y_min``` & ```BoxSize```. The routine will compute the density field in a region with coordinates [```x_min:x_min+BoxSize``` , ```y_min:y_min+BoxSize```]. Units should be Mpc/h.
- ```axis_x```, ```axis_y```. Integers to select the axes along which made the projection: 0(X), 1(Y) or 2(Z).
- ```periodic```. Whether use periodic boundary conditions for the considered region.
- ```verbose```. Whether show information on the computation.

```python
import numpy as np
import MAS_library as MASL

x_min, y_min, BoxSize = 0.0, 0.0, 25.0 #Mpc/h
axis_x, axis_y     = 0, 1 #0(X), 1(Y), 2(Z)
grid               = 512
periodic           = True

density = np.zeros((grid,grid), dtype=np.float64)

MASL.SPH_RT_2D(density, pos, mass, radius, x_min, y_min,
               axis_x, axis_y, BoxSize, periodic, verbose=True)
```
