# Voids

Pylians3 can be used to identify voids in a generic density field; e.g. matter, CDM+baryons, neutrinos, halos, galaxies, HI...etc.

### Method

Pylians3 uses the spherical overdensity void finder described in [Banerjee & Dalal 2016](https://ui.adsabs.harvard.edu/abs/2016JCAP...11..015B). We provide an example on how spherical voids can be easily identified with the void finder routine available in Pylians3. The script can be found in ```Pylians3/Voids/spheres_test.py```. That code generates random spheres with density profiles given by delta(r) = -1*(1-(r/R)^3) in a given cosmological volume. Note that for this density profile, the average overdensity at the void radius, R, is -0.5. The script then identifies the voids in that density field and finally plot the results. A figure like this should be obtained:

<p align="center"><img src="Spheres_test.png" alt="voids_test" width="700"/></p>

The left panel shows the projected density field of the generated random uniform spheres. The right panel displays the projected field of the identified voids. Note that, visually, density profiles look different in the two cases because the void finder set to 1 every voxel that belongs to a void, while the random spheres follow the above density profiles. The code also outputs the positions and radii of the generated and identified spheres.

### Void finder

The ingredients needed to identify voids in Pylians3 are:

- ```delta```. This is the overdensity field. It should be a 3 dimensional float numpy array such ```delta = np.zeros((grid, grid, grid), dtype=np.float32)```. See [density field](#density_field) on how to compute  density fields using Pylians.
- ```BoxSize```. Size of the periodic box. The units of the output power spectrum depend on this.
- ```threshold ```. The routine will identify voids with mean overdensity ```(1+threshold)```. This value is typically -0.7 or -0.8, but can be higher (e.g. -0.5), depending on your needs. 
- ```Radii```. This is a ```np.float32``` 1-dimension array containing the radii of the voids to identify. It doesn't need to be sorted. In general, the minimum void size should be ~4x-5x the grid size. E.g. if you have box of 1000 Mpc/h and a grid with ```1000^3``` cells, the minimum void size should be of 4-5 Mpc/h. It is better to choose the Radii such as their are multiples of the grid size. For the previous examples this will be good: ```Radii = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 34, 37, 40, 45, 50, 55], dtype=np.float32)```. 
- ```threads1```. The void finder routine is openmp parallelized. Set this to the maximum number of cpus per node.
- ```threads2```. Some routines are slower using all available cores. for those, we use a smaller number of cores. This number is typically 4 at most.
- ```void_field```. The routine can return a 3-dimension field filled with 0 (no void) and 1 (void) from the identified voids. If you want this set ```void_field=True```.

The void finder routine works as follows:

```python
import numpy as np
import void_library as VL

# parameters of the void finder
BoxSize    = 1000.0 #Mpc/h
threshold  = -0.7
Radii      = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                33, 35, 37, 39, 41, 44, 47, 50, 53, 56], dtype=np.float32) #Mpc/h
threads1   = 16
threads2   = 4
void_field = False

# identify voids
V = VL.void_finder(delta, BoxSize, threshold, Radii, threads1, threads2, void_field=void_field)
void_pos    = V.void_pos    #positions of the void centers
void_radius = V.void_radius #radius of the voids
VSF_R       = V.Rbins       #bins in radius for VSF(void size function)
VSF         = V.void_vsf    #VSF (#voids/volume/dR)
if void_field:  void_field  = V.void_field
```
