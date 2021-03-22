* # [Pylians](#Pylians)
    - ### [Requisites](#Requisites)
    - ### [Installation](#Installation)
* # [Density field](#density_field)
* # [Power spectrum](#auto_Pk)
    - ### [Auto-power spectrum](#auto_Pk)
    - ### [Cross-power spectrum](#cross_Pk)
* # [Correlation function](#CF)
* # [Bispectrum](#Bk)
* # [Voids](#Voids_P)
* # [Plots](#Plots_df)
* # [Cosmology](#Cosmology_P)
* # [Redshift space distortions](#RSD)
* # [Gaussian density fields](#Gaussian_fields_P)
* # [Integrals](#Integrals_P)
* # [Smooth fields](#Smooth)
* # [Miscellaneous](#Miscellaneous_P)
* # [Contact](#Contact_P)




# <a id="Pylians"></a> Pylians

Pylians stands for **Py**thon **li**braries for the **a**nalysis of **n**umerical **s**imulations. They are a set of python libraries, written in python, cython and C, whose purposes is to facilitate the analysis of numerical simulations (both N-body and hydro). Among other things, they can be used to:

- Compute density fields
- Compute power spectra
- Compute bispectra
- Compute correlation functions
- Identify voids
- Populate halos with galaxies using an HOD
- Apply HI+H2 corrections to the output of hydrodynamic simulations
- Make 21cm maps
- Compute DLAs column density distribution functions
- Plot density fields and make movies

[Pylians](https://en.wikipedia.org/wiki/Nestor_(mythology)) were the native or inhabitant of the Homeric town of Pylos. 

## <a id="Requisites"></a> Requisites
- numpy
- scipy
- h5py
- pyfftw
- mpi4py
- cython
- openmp
 
We recommend installing the first packages with [anaconda](https://www.anaconda.com/download/?lang=en-us). 

## <a id="Installation"></a> Installation

Pylians3 can be installed in two different ways:

1.)
```python
cd library
python setup.py build
```
the compiled libraries and scripts will be located in build/lib.XXX, where XXX depends on your machine. E.g. build/lib.linux-x86_64-3.7 or build/lib.macosx-10.7-x86_64-3.7

Add that folder to your PYTHONPATH in ~/.bashrc
```sh
export PYTHONPATH=$PYTHONPATH:$HOME/Pylians3/library/build/lib.linux-x86_64-3.7
```
2.) 
```python
cd library
python setup.py install
```

We recommend using the first method since you will know exactly where the libraries are. If you want to uninstall Pylians3 and have used the first option, just delete build folder.

To verify that the installation was successful, do
```python
python Tests/import_libraries.py
```
If no output is produced, everything went fine. On MACs, sometimes is hard to compile the libraries, because they require openmp and may not be trivial to install it. If you are experience problems with this, we recommend that you open the ```setup.py``` file and remove all ```openmp``` instances.

Keep in mind that in some systems, python 3 should be executed as ```python3```, instead of just ```python```.

## Usage
We provide some examples on how to use the library for different purposes.

## <a id="density_field"></a> Density field

Pylians provide routines to compute density fields from gadget snapshots. The ingredients needed are:

- ```snapshot```. This is the name of the gadget snapshot. Pylians supports formats 1, 2 and hdf5. Set it as ```'snap_001'```, even if the files are ```'snap_001.0'```, ```'snap_001.1'```, ... or ```'snap_001.0.hdf5'```, ```'snap_001.1.hdf5'```.
- ```grid```. The constructed density field will be a 3D float numpy array with ```grid**3``` cells. The larger this number the higher the resolution, but more memory will be used.
- ```ptypes```. Particle type over which compute the density field. It can be individual types, `[0]` (gas), `[1]` (cold dark matter), `[2]` (neutrinos), `[3]` (particle type 3), `[4]` (stars), `[5]` (black holes), or combinations. E.g. ```[0,1]``` (gas+cold dark matter), ```[0,4]``` (gas+stars), ```[0,1,2,4]``` (gas+CDM+neutrinos+stars). For all components (total matter) use ```[0,1,2,3,4,5]``` or ```[-1]```.
- ```MAS```. Mass-assignment scheme used to deposit particles mass to the grid. Options are: ```'NGP'``` (nearest grid point), ```'CIC'``` (cloud-in-cell), ```'TSC'``` (triangular-shape cloud), ```'PCS'``` (piecewise cubic spline). For most applications ```'CIC'``` is enough.
- ```do_RSD```. If ```True```, particles positions will be moved to redshift-space along the ```axis``` axis. 
- ```axis```. Axis along which redshift-space distortions will be implemented (only needed if ```do_RSD=True```): 0, 1 or 2 for x-axis, y-axis or z-axis, respectively. 

An example is as follows

```python
import numpy as np
import MAS_library as MASL

snapshot = 'snapdir_010/snap_010'
grid     = 512  
ptypes   = [1] 
MAS      = 'CIC' 
do_RSD   = False
axis     = 0 

delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
```
```delta``` contains the number density field of CDM. To compute its overdensity do

```python
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
```

Pylians can also be used to compute the density field of a set of particles that are not in gadget format. An example is this

```python
import numpy as np
import MAS_library as MASL

# input parameters
grid    = 512  
BoxSize = 1000 #Mpc/h
MAS     = 'CIC'

# define the array hosting the density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)

# read the particle positions
pos = np.loadtxt('myfile.txt') #Mpc/h 
pos = pos.astype(np.float32)   #pos should be a numpy float array

# compute density field
MASL.MA(pos,delta,BoxSize,MAS)

# compute overdensity field
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0 
```

Pylians can also compute density fields where each particle/halo has a different weight. This can be useful in the case where the observed field is constructed in that way, e.g. the 21cm field is built from gas particles weighting each of them by its HI mass.

```python
import numpy as np
import MAS_library as MASL

# input parameters
grid    = 512  
BoxSize = 1000 #Mpc/h
MAS     = 'CIC'

# define the array hosting the density field
delta = np.zeros((grid,grid,grid), dtype=np.float32)

# read the particle positions
pos     = np.loadtxt('myfile.txt')   #Mpc/h 
pos     = pos.astype(np.float32)     #pos should be a numpy float array
weights = np.loadtxt('weights.txt')  #weights of the particles
weights = weights.astype(np.float32) #weights should be a numpy float array

# compute density field taking into account the particle weights
MASL.MA(pos,delta,BoxSize,MAS,W=weights)

# compute overdensity field
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0 
```

Pylians provide more routines to work/create density fields. See [here](Density_field.md).


## <a id="auto_Pk"></a> Power spectrum
The ingredients needed to compute the power spectrum are:

- ```delta```. This is the density or overdensity field. It should be a 3 dimensional float numpy array such ```delta = np.zeros((grid, grid, grid), dtype=np.float32)```. See [density field](#density_field) on how to compute  density fields using Pylians.
- ```BoxSize```. Size of the periodic box. The units of the output power spectrum depend on this.
- ```axis```. Axis along which compute the quadrupole, hexadecapole and the 2D power spectrum. If the field is in real-space set ```axis=0```. If the field is in redshift-space set ```axis=0```, ```axis=1``` or ```axis=2``` if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively. 
- ```MAS```. Mass-assignment scheme used to generate the density field, if any. Possible options are ```'NGP'```, ```'CIC'```, ```'TSC'```, ```'PCS'```.  If the density field has not been generated with any of these set it to ```'None'```.
- ```threads```. Number of openmp threads to be used.
- ```verbose```. Whether print information on the status/progress of the calculation: True or False

The power spectrum can be computed as:

```python
import Pk_library as PKL

Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)
```

```Pk``` is a python class containing the 1D, 2D and 3D power spectra, that can be retrieved with

```python
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
```

### <a id="cross_Pk"></a> Cross-power spectrum

Pylians can be used to compute the auto- and cross-power spectrum of multiple fields. For instance, to compute the auto- and cross-power spectra of two overdensity fields, ```delta1``` and ```delta2```:

```python
import Pk_library as PKL

Pk = PKL.XPk([delta1,delta2], BoxSize, axis, MAS=['CIC','CIC'], threads)
```

A description of the variables ```BoxSize```, ```axis```, ```MAS``` and ```threads``` can be found in the [power spectrum](#auto_Pk) section. As with the auto-power spectrum, ```delta1``` and ```delta2``` need to be 3D float numpy arrays. ```Pk``` is a python class that contains all the following information

```python
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
```

The ```XPk``` function can be used for more than two fields, e.g.

```python
BoxSize = 1000.0 #Mpc/h
axis    = 0
MAS     = ['CIC','NGP','TSC','None']
threads = 16

Pk = PKL.XPk([delta1,delta2,delta3,delta4], BoxSize, axis, MAS, threads)
```

## <a id="CF"></a> Correlation function
Pylians can be used to efficiently compute correlation functions of a generic field (e.g. total matter, CDM, gas, halos, neutrinos, CDM+gas, galaxies...etc). The ingredients needed are:
- ```delta```. This is the overdensity field. It should be a 3 dimensional float numpy array such ```delta = np.zeros((grid, grid, grid), dtype=np.float32)```. See [density field](#density_field) on how to compute  density fields using Pylians.
- ```BoxSize```. Size of the periodic box. The units of the output power spectrum depend on this.
- ```MAS```. Mass-assignment scheme used to generate the density field, if any. Possible options are ```'NGP'```, ```'CIC'```, ```'TSC'```, ```'PCS'```.  If the density field has not been generated with any of these set it to ```'None'```.
- ```axis```. Axis along which compute the quadrupole, hexadecapole. If the field is in real-space set ```axis=0```. If the field is in redshift-space set ```axis=0```, ```axis=1``` or ```axis=2``` if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively.
- ```threads```. This routine is openmp parallelized. Set this to the maximum number of cores per node available in your machine.

An example on how to use the routine is this:
```python
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
```

This routine uses a FFT approach that allows a very computationally efficient calculation of the correlation function. However, if the number density of the tracers is very low (i.e. the density field is very sparse) this function may produce strange results. In this case it is better to use the traditional Landy-Szalay routine also available in Pylians.

## <a id="Bk"></a>Bispectrum
Pylians can compute bispectra of 3D density fields (total matter, CDM, gas, halos, galaxies...etc). The ingredients needed are:
- ```delta```. This is the overdensity field. It should be a 3 dimensional float numpy array such ```delta = np.zeros((grid, grid, grid), dtype=np.float32)```. See [density field](#density_field) on how to compute  density fields using Pylians.
- ```BoxSize```. Size of the periodic box. The units of the output bispectrum depend on this. 
- ```k1```. The wavenumber of the first side of the considered triangle. Use units in correspondence with ```BoxSize```.
- ```k2```. The wavenumber of the second size of the considered triangle. Use units in correspondence with ```BoxSize```.
- ```theta```. This is a numpy array containing the different angles between ```k1``` and ```k2```.  
- ```MAS```. Mass-assignment scheme used to generate the density field, if any. Possible options are ```'NGP'```, ```'CIC'```, ```'TSC'```, ```'PCS'```.  If the density field has not been generated with any of these set it to ```'None'```.
- ```threads```. The bispectrum code is openmp parallelized. Set this to the maximum number of cpus per node.

Pylians computes the amplitude of the bispectrum, and reduced bispectrum, for triangles that have sides ```k1``` and ```k2``` and different considered angles between them.

```python
import numpy as np
import Pk_library as PKL

BoxSize = 1000.0 #Size of the density field in Mpc/h
k1      = 0.5    #h/Mpc
k2      = 0.6    #h/Mpc
MAS     = 'CIC'
threads = 1
theta   = np.linspace(0, np.pi, 25) #array with the angles between k1 and k2

# compute bispectrum
Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS, threads)
Bk = Bk.B     #bispectrum
Qk = Bk.Q     #reduced bispectrum
k  = Bk.k     #k-bins for power spectrum
Pk = Bk.Pk    #power spectrum
```

## <a id="Voids_P"></a> Voids
Pylians can be used to identify voids in a generic density field (e.g. total matter, CDM, gas, halos, neutrinos, CDM+gas, galaxies...etc). The ingredients needed are:
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

For more details on the void finder see [here](Voids.md).

## <a id="Plots_df"></a> Plots

Pylians provide a set of routines to quickly make plots of density fields of simulations. For instance, to generate the density field of a slice of a Gadget N-body snapshot, the ingredients needed are:
- ```snapshot```. This is the name of the snapshot. For instance, ```snapdir_004/snap_004```. Note that only the prefix is needed. If e.g. ```snapdir_004``` is a folder that contains multiple subfiles, ```snapdir_004/snap_004.0```, ```snapdir_004/snap_004.1```...etc, ```snapshot``` should be set to ```snapdir_004/snap_004```.
- ```x_min```, ```x_max```, ```y_min```, ```y_max```, ```z_min```, ```z_max```. These are the coordinates of the considered region.
- ```grid```. Resolution of the image, that will have grid^2 pixels.
- ```ptypes```. Particle type over which compute the density field. It can be individual types, `[0]` (gas), `[1]` (cold dark matter), `[2]` (neutrinos), `[3]` (particle type 3), `[4]` (stars), `[5]` (black holes), or combinations. E.g. ```[0,1]``` (gas+cold dark matter), ```[0,4]``` (gas+stars), ```[0,1,2,4]``` (gas+CDM+neutrinos+stars). For all components (total matter) use ```[0,1,2,3,4,5]``` or ```[-1]```.
- ```plane```. Plane over which project the region. Can be ```'XY'```, ```'XZ'``` or ```'YZ'```.
- ```MAS```. Mass-assignment scheme used to generate the density field. Possible options are ```'NGP'```, ```'CIC'```, ```'TSC'```, ```'PCS'```.
- ```save_df```. Whether save the density field or not. It can be useful to save the field, and don't have to recompute it, when only changing the colors, overdensities values...etc.

One example on how to use Pylians to plots the density field of a snapshot is this
```python
import numpy as np
import plotting_library as PL
from pylab import *
from matplotlib.colors import LogNorm


#snapshot name
snapshot = '/mnt/ceph/users/fvillaescusa/Quijote/Snapshots/latin_hypercube_HR/0/snapdir_004/snap_004'

# density field parameters
x_min, x_max = 0.0, 500.0
y_min, y_max = 0.0, 500.0
z_min, z_max = 0.0, 20.0
grid         = 1024
ptypes       = [1]   # 0-Gas, 1-CDM, 2-NU, 4-Stars; can deal with several species
plane        = 'XY'  #'XY','YZ' or 'XZ'
MAS          = 'PCS' #'NGP', 'CIC', 'TSC', 'PCS' 
save_df      = True  #whether save the density field into a file

# image parameters
fout            = 'Image.png'
min_overdensity = 0.5      #minimum overdensity to plot
max_overdensity = 50.0    #maximum overdensity to plot
scale           = 'log' #'linear' or 'log'
cmap            = 'hot'


# compute 2D overdensity field
dx, x, dy, y, overdensity = PL.density_field_2D(snapshot, x_min, x_max, y_min, y_max,
                                                z_min, z_max, grid, ptypes, plane, MAS, save_df)


# plot density field
print('\nCreating the figure...')
fig = figure()    #create the figure
ax1 = fig.add_subplot(111) 

ax1.set_xlim([x, x+dx])  #set the range for the x-axis
ax1.set_ylim([y, y+dy])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #y-axis label

if min_overdensity==None:  min_overdensity = np.min(overdensity)
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

if scale=='linear':
    cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                     extent=[x, x+dx, y, y+dy], interpolation='bicubic',
                     vmin=min_overdensity,vmax=max_overdensity)
else:
    cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                     extent=[x, x+dx, y, y+dy], interpolation='bicubic',
                     norm = LogNorm(vmin=min_overdensity,vmax=max_overdensity))

cbar = fig.colorbar(cax)
cbar.set_label(r"$\rho/\bar{\rho}$",fontsize=20)
savefig(fout, bbox_inches='tight')
close(fig)
```

The above script, on one of the [Quijote simulations](https://github.com/franciscovillaescusa/Quijote-simulations) produces the following image:

<p align="center"><img src="LSS.png" alt="voids_test" width="700"/></p>


## <a id="Cosmology_P"></a> Cosmology

Pylians provide a set of routines to carry out simple cosmological calculations.

The comoving distance and the linear growth factor to redshift ```z``` can be computed as

```python
import cosmology_library as CL

z       = 1.0
Omega_m = 0.3175
Omega_l = 0.6825

r = CL.comoving_distance(z, Omega_m, Omega_l)  #Mpc/h
D = CL.linear_growth_factor(z, Omega_m, Omega_l) 
```

From a linear power spectrum at z=0, Pylians can find the non-linear matter power spectrum halofit by Takahashi 2012 as

```python
import numpy as np
import cosmology_library as CL

z       = 1.0
Omega_m = 0.3175
Omega_l = 0.6825

k_lin, Pk_lin = np.loadtxt('my_Pk_file_z=0.txt', unpack=True)

Pk_nl = CL.Halofit_12(Omega_m, Omega_l, z, k_lin, Pk_lin) 
```


## <a id="RSD"></a> Redshift space distortions

Pylians provide a simple routine to displace particle positions from real-space to redshift-space. The ingredients needed are:

- ```pos```. This is an array with the co-moving positions of the particles. Should be float numpy array. Notice that this array will be overwritten with the positions of the particles in redshift-space. So if you want to keep the positions of the original particles, is better to pass a copy of this array: e.g. ```pos_RSD = np.copy(pos)```. Units should be Mpc/h.
- ```vel```. This is an array with the peculiar velocities of the particles. Should be a float numpy array. Units should be km/s
- ```BoxSize```. Size of the simulation box. Units should be Mpc/h
- ```Hubble```. Value of the Hubble constant at redshift ```redshift```. Units should be (km/s)/(Mpc/h).
- ```redshift```. The considered redshift.
- ```axis```. Redshift-space distortions are going to be place along the x-(axis=0), y-(axis=1) or z-(axis=2) axis.

```python
import redshift_space_library as RSL

RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)
```

## <a id="Gaussian_fields_P"></a> Gaussian density fields

Pylians provide a few routines to generate Gaussian density fields either in 2D or 3D. The ingredients needed are:

- ```grid```. The generated Gaussian density field will have grid x grid pixels.
- ```k``` . 1D float32 numpy array containing the k-values of the input power spectrum.
- ```Pk```. 1D float32 numpy array containing the Pk-values of the input power spectrum.
- ```Rayleigh_sampling```. Where Rayleigh sampling the modes amplitudes when generating the Gaussian field. If ```Rayleigh_sampling=0``` the Gaussian field will not have cosmic variance. Set ```Rayleigh_sampling=1``` for standard Gaussian density fields.
- ```seed```. Integer for the random seed of the map.
- ```BoxSize```. Size of the region over which to generate the field. Units should be compatible with those of ```Pk```.
- ```threads```. Number of openmp threads. Only used when FFT the field from Fourier space to configuration space.
- ```verbose```. Whether output some information.

```python
import numpy as np
import density_field_library as DFL

grid              = 128
BoxSize           = 1000.0 #Mpc/h
seed              = 1
Rayleigh_sampling = 0
threads           = 1
verbose           = True

# read power spectrum; k and Pk have to be floats, not doubles 
k, Pk = np.loadtxt('my_Pk.txt', unpack=True)
k, Pk = k.astype(np.float32), Pk.astype(np.float32)

# generate a 2D Gaussian density field
df_2D = DFL.gaussian_field_2D(grid, k, Pk, Rayleigh_sampling, seed, 
     			      BoxSize, threads, verbose)

# generate a 3D Gaussian density field
df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed, 
       			      BoxSize, threads, verbose)
```


## <a id="Integrals_P"></a> Integrals
            
Pylians provide routines to carry out numerical integrals in a more efficient way than scipy integrate. The philosophy is that to compute the integral $\int_a^b f(x)dx$ the user passes the integrator two arrays, one with some values of x between a and b and another with the values of f(x) at those positions. The integrator will interpolate internally the input data to evaluate the function at an arbitrary position x. Pylians implements in c the fortran odeint function and wrap in python through cython.

For instance, to compute $\int_0^5 (3x^3+2x+5) dx$ one would do

```python
import numpy as np
import integration_library as IL

# integral value, its limits and precision parameters
yinit = np.zeros(1, dtype=np.float64) 
x1    = 0.0
x2    = 5.0
eps   = 1e-8 
h1    = 1e-10 
hmin  = 0.0   

# integral method and integrand function
function = 'linear'
bins     = 1000
x        = np.linspace(x1, x2, bins)
y        = 3*x**2 + 2*x + 5

I = IL.odeint(yinit, x1, x2, eps, h1, hmin, x, y, function, verbose=True)[0]
```

The value of integral is stored in ```I```. The ```odeint``` routine needs the following ingredients:

- ```yinit```. The value of the integral is stored in this variable. Should be a 1D double numpy array with one single element equal to 0. If several integrals are being computed sequentially this variable need to be declared for each integral.
- ```x1```. Lower limit of the integral.
- ```x2```. Upper limit of the integral.
- ```eps```. Maximum local relative error tolerated. Typically set its value to be 1e8-1e10 lower than the value of the integral. Verify the convergence of the results by studying the dependence of the integral on this number.
- ```h1```. Initial guess for the first time step (can not be 0).
- ```hmin```. Minimum allower time step (can be 0).
- ```verbose```. Set it to ```True``` to print some information on the integral computation.

There are two main methods to carry out the integral, depending on how the interpolation is performed. 

- ```function = 'linear'```. The function is evaluated by interpolating linearly the input values of x and y=f(x).
    - ```x```. 1D double numpy array containing the input, equally spaced, values of x. 
    - ```y```. 1D double numpy array containing the values of y=f(x) at the ```x``` array positions.
 
- ```function = 'log'```. The function is evaluated by interpolating logaritmically the input values of x and y=f(x).
    - ```x```. 1D double numpy array containing the input, equally spaced in log, values of log10(x). 
    - ```y```. 1D double numpy array containing the values of y=f(x) at the ```x``` array positions.

An example of using the log-interpolation to compute the integral $\int_1^2 e^x dx$ is this

```python
import numpy as np
import integration_library as IL

# integral value, its limits and precision parameters
yinit = np.zeros(1, dtype=np.float64) 
x1    = 1.0
x2    = 2.0
eps   = 1e-10 
h1    = 1e-12 
hmin  = 0.0   

# integral method and integrand function
function = 'log'
bins     = 1000
x        = np.logspace(np.log10(x1), np.log10(x2), bins)
y        = np.exp(x)

I = IL.odeint(yinit, x1, x2, eps, h1, hmin, np.log10(x), y,
              function, verbose=True)[0]
```

Be careful when using the log-interpolation since the code will crash if a zero or negative value is encounter. 

The user can create its own function if he/she does not want to evaluate the integrand through interpolations. This function has to be placed in the file library/integration_library/integration_library.pyx (see linear and sigma functions as examples). After that, a new function call has to be created in the function odeint of that file (see linear and log as examples).

## <a id="Smooth"></a>Smooth Fields

Pylians provide routines to smooth fields with several filters. The ingredients needed are:

- ```field```. This is a 3D float numpy array that contains the input field to be smoothed.
- ```BoxSize```. This is the size of the box with the input density field. 
- ```R```. This is the smoothing scale.
- ```grid```. This is the grid size of the input field, i.e. ```field.shape[0]```.
-  ```threads```. Number of openmp threads to be used.
-  ```Filter```. Filter to use. ```'Top-Hat'``` or ```'Gaussian'```.
-  ```W_k```. This is a 3D complex64 numpy array containing the Fourier-transform of the filter. Notice that when smoothing a discrete field, like the one stored on a regular grid, the Fourier-transform of the filter need to be computed in the same way as the for the field, i.e. through DFT instead of FT.

An example is this

```python
import smoothing_library as SL

BoxSize = 75.0 #Mpc/h
R       = 5.0  #Mpc.h
grid    = field.shape[0]
Filter  = 'Top-Hat'
threads = 28

W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
field_smoothed = SL.field_smoothing(field, W_k, threads)
```


## <a id="Miscellaneous_P"></a>Miscellaneous

See this [page](miscellaneous.md) for a collection of different useful things.

## <a id="Contact_P"></a>Contact

For comments, problems, bugs... etc you can reach me at [villaescusa.francisco@gmail.com](mailto:villaescusa.francisco@gmail.com).
