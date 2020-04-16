# Miscellaneous
  
Here we present a set of different things that we found useful in the past

* ## [mpi4py](#mpi4py_P)
* ## [Checksums](#Checksums_P)
* ## [h5py](#h5py_P)
* ## [argparse](#argparse_P)
* ## [cython](cython_P)
* ## [bash](#bash_P)
* ## [Emacs](#Emacs_P)
* ## [Slurm](#Slurm_P)
* ## [fit](#fit_P)

## <a id="mpi4py_P"></a> mpi4py

Sometimes one needs to execute the same lines of code over many different cases. In this case, mpi can be used for a trivial parallelization. Below is an example on how to print a line over 1000 numbers with different cpus

```python
from mpi4py import MPI
import numpy as np

###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

files = 10000

# find the numbers that each cpu will work with
numbers = np.where(np.arange(files)%nprocs==myrank)[0]

# main loop; each cpu only works on its subset
for i in numbers:
    print('Cpu %3d working with number %4d'%(myrank,i))
```

Sometimes each cpu does a part of a calculation, and we want to join the results of all the cpus into a single core. This joining can be adding, multiplying...etc the results of each core. Imagine that you have an array that need to be filled up with numbers. Instead of having a single cpu filling up the array, we want different cpus to do a fraction of the work and fill up the array but only with their partial numbers. The way to sum all partial results to obtain the full array is this:

```python
from mpi4py import MPI
import numpy as np

###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

partial_array = np.zeros(nprocs) #array with the partial results of each cpu
total_array   = np.zeros(nprocs) #array with the overall results

partial_array[myrank] = myrank #each cpu fill their array elements

comm.Reduce(partial_array, total_array, root=0)

print(myrank, partial_array, total_array)
```

Sometimes it is needed to make sure that all cores are at the same point in a code. For that, use

```python
comm.Barrier()
```

## <a id="Checksums_P"></a> Checksums

When transfering large amounts of data, or very important data, among machines, it is important to verify that the integrity of the data transfered. Checksums can be used for this. Say you have a folder that want to transfer from San Diego to Princeton. The way to do that is as follows. In the San Diego machine type:

```sh
cd my_folder/
find -type f \! -name SHA224SUMS -exec sha224sum \{\} \+ > SHA224SUMS
```

The above command will create a file called SHA224SUMS with the checksums of all files in that folder. Transfer the folder to the Princeton machine (including the file SHA224SUMS). Check the integrity of the data by executing the following command in the Princeton machine:

```sh
sha224sum -c SHA224SUMS --quiet
```

If nothing is printed out, the data has been properly transfered.

## <a id="h5py_P"></a> h5py

Some examples to remind myself the syntax of h5py:

```python
import numpy as np
import h5py

z = 3.0
a = np.arange(10)

# write a hdf5 file
f = h5py.File('my_file_z=%.3f.hdf5'%z, 'w')
f.create_dataset('Mass', data=a)
f.close()

# read hdf5 file
f = h5py.File('M_HI_new_75_1820_z=%.3f.hdf5'%z, 'r')
M_HI = f['M_HI'][:]
M    = f['Mass'][:]
R    = f['R'][:]
f.close()
```

## <a id="argparse_P"></a> argparse

Some examples on how to use this library:

```python
# python argparse.py /home/villa/ -snapnums 0 1 2 3 --long_ids
# for help type: python argparse.py -h
import argparse
import numpy as np
import sys,os

parser = argparse.ArgumentParser(description="description of routine")

# non-optional arguments
parser.add_argument("snapdir", help="folder where the groups_XXX folder is")

# optional arguments
parser.add_argument("-cx1", type=int, default=0, 
                    help="column x in file 1, default 0")
parser.add_argument("-snapnums", nargs='+', type=int, help="groups number")

parser.add_argument("--swap", dest="swap", action="store_true", default=False, 
                    help="False by default. Set --swap for True")

parser.add_argument("--SFR", dest="SFR", action="store_true", default=False, 
                    help="False by default. Set --SFR for True")

parser.add_argument("--long_ids", dest="long_ids", action="store_true", 
                    default=False, help="False by default. Set --long_ids for True")

args = parser.parse_args()

print(args.snapdir)
print(args.snapnums)
print(args.swap)
print(args.SFR)
print(args.long_ids)
```

## <a id="cython_P"></a> cython

Sometimes it is very useful to visualize which pieces of the code are fully translated into c and which still have some part of python. Typing

```sh
cython -a my_file.pyx
```

will generate a html file with different intensities in yellow: from white (pure C) to dark yellow (full python).

## <a id="bash_P"></a> bash

Here are some examples on how to deal with if and loops in bash

```sh
#!/bin/bash

dims=512
do_RSD=1
axis=0

#name of the folders containing the snapshots
snapshot_folder=('/scratch/villa/SAM/CDM/'       \
                 '/scratch2/villa/SAM/NU0.3/'    \
                 '/scratch2/villa/SAM/NU0.3s8/'  \
                 '/scratch2/villa/SAM/NU0.6/'    \
                 '/scratch2/villa/SAM/NU0.6s8/'  \
                 '/scratch/villa/SAM/CDM/'       \
                 '/scratch/villa/SAM/CDM/')

#root of the files containing the galaxy catalogues
root_catalogue=('LG_NCDM_' \
                'LG_NU03_' \
                'LG_N3s8_' \
                'LG_NU06_' \
                'LG_N6s8_' \
                'LG_DC_N_' \
                'LG_SA_N_')

#name of the folders containing the 2PCF files
root_f_out=('0.0/2PCF_RS_gal_0.0_z='      \
            '0.3/2PCF_RS_gal_0.3_z='      \
            '0.3s8/2PCF_RS_gal_0.3s8_z='  \
            '0.6/2PCF_RS_gal_0.6_z='      \
            '0.6s8/2PCF_RS_gal_0.6s8_z='  \
            '0.0_DC/2PCF_RS_gal_0.0_z='   \
            '0.0_SA/2PCF_RS_gal_0.0_z=')

#redshifts
z=('3.06.dat' '2.07.dat' '0.99.dat' '0.51.dat' '0.00.dat')

#snapshots number corresponding to the above redshifts
suffix_snapshot_CDM=('snap_026' 'snap_031' 'snap_040' 'snap_047' 'snap_062')
suffix_snapshot=('snapdir_026/snap_026' \
                 'snapdir_031/snap_031' \
                 'snapdir_040/snap_040' \
                 'snapdir_047/snap_047' \
                 'snapdir_062/snap_062')


#do a loop over the different cosmologies
for i in ${!snapshot_folder[*]}
do

    #do a loop over the different redshifts for the same cosmology
    for j in ${!z[*]}
    do
	
	if [ "$i" == "0" -o "$i" -ge "5" ]; then
	    snapshot_fname=${snapshot_folder[$i]}${suffix_snapshot_CDM[$j]}
	else
	    snapshot_fname=${snapshot_folder[$i]}${suffix_snapshot[$j]}
	fi

	f_cat=${root_catalogue[$i]}${z[$j]}
	f_out=${root_f_out[$i]}${z[$j]}

	echo $snapshot_fname
	echo $f_cat
	echo $f_out

	mpirun -np 8 python correlation_function.py $snapshot_fname $f_cat $f_out $do_RSD $axis

    done
    echo ' '
done
```

```bash
#!/bin/bash


#WAYS TO LOOP OVER THE ELEMENTS OF AN ARRAY

array=('first' 'second' 'third' 'fourth') #array

#print the number of elements in the array
printf 'Number of elements in the array = %d \n\n' ${#array[*]}

#print the elements in the array
echo '###### array elements ######'
for element in ${array[*]}
do
    echo $element
done
echo '############################'

#print the elements in the array and their order
printf '\n###### array elements ######\n'
for i in ${!array[*]}
do
    printf 'element %d = %s\n' $i ${array[$i]}
done
printf '############################\n\n'

echo ${array[*]}   #print all the elements of the array
echo ${!array[*]}  #print the indexes of the array
echo ${#array[*]}  #print the number of elements in the array
```


## <a id="Emacs_P"></a> Emacs

Sometimes, when using a new machine, the emacs files do not have the dimensions we want, e.g. the do not fit well in the screen. This can be fixed by specifying directly the geometry when calling emacs

```sh
emacs --geometry=88x37 library.py
```

Thus, once the geometry that best fit the screen is found, it is enough to add this line to the ~/.bashrc file:

```sh
alias emacs="emacs --geometry=88x37"
```

## <a id="Slurm_P"></a> Slurm

When I want to run a job in interactive mode I use this command to request 2 nodes and 5 cores per node for 24 hours

```bash
salloc -N 2 --ntasks-per-node 5 -t 24:00:00
```

This will allocate 10 cores. In order to run your code do

```bash
srun -n 10 python my_mpi_code.py
```

For the usage of 1 GPU in interactive mode I do:

```bash
salloc -N 1 -p gpu --gres=gpu:1 -c 1 --mem=100GB
srun --pty bash -i 
```

## <a id="fit_P"></a> fit

An example on how to make fits using python and emcee

```python
# This scripts is an example of a MCMC fit. We generate points from a linear law
# y = a*x+b and we add noise to them. We then try to obtain the values of a and b
# from the data

import numpy as np
import emcee,corner
from scipy.optimize import minimize
import sys,os

# This function returns the predicted y-values for the particular model considered
# x --------> array with the x-values of the data
# theta ----> array with the value of the model parameters [a,b]
def model_theory(x,theta):
    a,b = theta
    return a*x+b


# This functions returns the log-likelihood for the theory model
# theta -------> array with parameters to fit [a,b]
# x,y,dy ------> data to fit
def lnlike_theory(theta,x,y,dy):
    # put priors here
    if (theta[0]>1000.0) or (theta[0]<-1000.0) or (theta[1]>1000.0) \
            or (theta[1]<-1000.0):
        return -np.inf
    else:
        y_model = model_theory(x,theta)
        chi2 = -np.sum(((y-y_model)/dy)**2,dtype=np.float64)
        return chi2


#################################### INPUT ####################################
x_min, x_max = 0.0, 10.0 #minimum and maximum value for the x-axis
points = 100  #number of points to use in the fit
sigma  = 1.0  #standard deviation for the errors
theta_model = [3.0, 5.0]  #real value of the parameters
theta0      = [0.0, 0.0]  #intial guess of the value of the parameters

# MCMC parameters
nwalkers   = 100
chain_pts  = 10000
f_contours = 'Ellipses.pdf' 
###############################################################################

# find number of degrees of freedom and parameter space dimension
ndim = len(theta_model);  ndof = points - ndim


####################### GENERATE DATA #############################
# generate noise data from underlying model y=x
x  = np.linspace(x_min, x_max, points)
dy = np.random.randn(points)*sigma
y  = theta_model[0]*x + theta_model[1] + dy  # y = a*x + b + epsilon

# save data to file
np.savetxt('data.txt',np.transpose([x,y])) 
####################################################################


######################## FIT DATA WITH NUMPY #######################
chi2_func = lambda *args: -lnlike_theory(*args)
best_fit  = minimize(chi2_func,theta0,
                     args=(x,y,dy),method='Powell')
theta_best_fit = best_fit["x"]
chi2 = chi2_func(theta_best_fit,x,y,dy)*1.0/ndof
    
a, b  = theta_best_fit
print('\n##### NUMPY fit results #####')
print('a    = %7.3e'%a)
print('b    = %7.3e'%b)
print('chi2 = %7.3f\n'%chi2)
#####################################################################


########################## FIT WITH MCMC ############################
# starting point of each MCMC chain: small ball around best fit or estimate
alpha = 0.1 #you can play with this number
pos = [theta_best_fit*(1.0+alpha*np.random.randn(ndim)) for i in range(nwalkers)]

# run MCMC
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlike_theory,args=(x,y,dy))
sampler.run_mcmc(pos,chain_pts);  del pos

# get the points in the MCMC chains
samples = sampler.chain[:,300:,:].reshape((-1,ndim))
sampler.reset()

# make the plot and save it to file
fig = corner.corner(samples,truths=theta_model,labels=[r"$a$",r"$b$"])
fig.savefig(f_contours)

# compute best fit and associated error
a, b, =  map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
             zip(*np.percentile(samples, [16, 50, 84],axis=0)))
del samples, sampler
                                      
# compute chi^2 of best fit
theta_mcmc = [a[0],b[0]]
chi2 = chi2_func(theta_mcmc,x,y,dy)/ndof

# show results of MCMC
print('\n##### MCMC fit results #####')
print('a    = %7.3e + %.2e - %.2e'%(a[0],a[1],a[2]))
print('b    = %7.3e + %.2e - %.2e'%(b[0],b[1],b[2]))
print('chi2 = %7.3f'%chi2)

# save best-fit model to file
y_best_fit = model_theory(x,theta_mcmc)
np.savetxt('Best-fit.txt',np.transpose([x,y_best_fit]))
#####################################################################
```
