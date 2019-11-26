from mpi4py import MPI
import numpy as np
import sys,os

###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


############## trivial parallelization ###############
files = 10000

# find the numbers that each cpu will work with
numbers = np.where(np.arange(files)%nprocs==myrank)[0]

for i in numbers:
    print('Cpu %3d working with number %4d'%(myrank,i))
######################################################

comm.Barrier()

################### reduction ########################
a   = np.zeros(nprocs) #array with the value of each cpu
a_R = np.zeros(nprocs) #array with the reduction

a[myrank] = myrank #each cpu fill their array elements

comm.Reduce(a,a_R,root=0)

print(myrank,a,a_R)
######################################################
