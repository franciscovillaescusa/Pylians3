# Miscellaneous
  
Here we present a set of different things that we found useful in the past

* ## [mpi4py](#mpi4py_P)
* ## [Checksums](#Checksums_P)
* ## [Emacs](#Emacs_P)
* ## [Slurm](#Slurm_P)

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