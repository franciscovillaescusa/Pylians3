# This routine is designed to read in parallel Gadget snapshots.
# so far it can read POS, VEL and ID blocks. It automatically takes into 
# account whether IDs are uint32 or uint64

from mpi4py import MPI
import numpy as np
import readsnap
import sys,os
import time as Time

###### MPI DEFINITIONS ######                                    
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()



def read_block(filename, block, parttype, physical_velocities=True,
               verbose=False):

    if parttype not in [0,1,2,3,4,5]:
        print('Routine can only read parttypes in [0,1,2,3,4,5,6]')
        sys.exit()

    # find the format,swap, total number of particles and subfiles
    head = readsnap.snapshot_header(filename)
    files, format, swap = head.filenum, head.format, head.swap
    nall, time, redshift = head.nall, head.time, head.redshift;  del head
    if myrank==0 and verbose:
        print('Reading snapshot with %d cores'%nprocs)
        print('Number of subfiles = %d'%files)

    # find the files each cpu reads
    subfiles = np.array_split(np.arange(files),nprocs)[myrank]

    # find the total number of particles to read
    Nall_local = np.int64(0)
    for i in subfiles:
        head = readsnap.snapshot_header(filename+'.%d'%i)
        Nall_local += head.npart[parttype]
    del head

    if verbose:
        print('core %03d reading %03d files: [%03d-%03d] %9d particles'\
            %(myrank,len(subfiles),subfiles[0],subfiles[-1],Nall_local))

    # check that all particles are read
    Nall = comm.reduce(Nall_local, op=MPI.SUM, root=0)
    if myrank==0 and Nall!=nall[parttype]:
        print('Read %d particles while expected %d'%(Nall,nall[parttype]))
        sys.exit()

    # find the data type
    if block=="POS ":  
        dt = np.dtype((np.float32,3))
        block_num = 2
    elif block=="VEL ":  
        dt = np.dtype((np.float32,3))
        block_num = 3
    elif block=="ID  ":  
        dt = np.uint32
        block_num = 4
    else:
        print('Block not found!');  sys.exit()

    # define the data array
    if myrank==0:  data = np.empty(Nall,       dtype=dt)
    else:          data = np.empty(Nall_local, dtype=dt)

    # do a loop over all subfiles
    offset_array = 0;  start = Time.time()
    for i in subfiles:

        # find subfile name and number of particles in it
        curfilename = filename+'.%d'%i
        head        = readsnap.snapshot_header(curfilename)
        npart       = head.npart 
        particles   = npart[parttype]

        offset_species = np.zeros(6,np.int64)
        allpartnum = np.int64(0)
        for j in range(6):
            offset_species[j] = allpartnum
            allpartnum += npart[j]

        # find the offset and the size of the block (for all particle types)
        offset_block,blocksize = readsnap.find_block(curfilename,format,swap,
                                                     block,block_num)

        # if long IDs change dt to np.uint64
        if i==subfiles[0] and block=="ID  ":
            if blocksize==np.dtype(dt).itemsize*allpartnum*2:
                dt = np.uint64

        # read file
        f = open(curfilename, 'rb')
        f.seek(offset_block + offset_species[parttype]*np.dtype(dt).itemsize, 
               os.SEEK_CUR)
        curdat = np.fromfile(f, dtype=dt, count=particles)
        f.close()

        if swap:  curdat.byteswap(True)

        data[offset_array:offset_array+particles] = curdat
        offset_array += particles
    if verbose:
        print('%d: Time to read files = %.2f'%(myrank,Time.time()-start))


    # slaves send master the particles read
    if myrank>0:
        comm.send(Nall_local, dest=0, tag=1) #number of particles read
        comm.Send(data,       dest=0, tag=2) #property read (pos,vel,ID..)
        return 0 #put 0 to avoid problems when reading pos and /1e3 #Mpc/h
    
    # master collect all information from slaves and return the array
    else:
        offset = Nall_local
        # do a loop over all slaves
        start = Time.time()
        for i in range(1,nprocs):
            npart = comm.recv(source=i, tag=1)
            comm.Recv(data[offset:offset+npart], source=i, tag=2)
            if verbose:
                print('Time to transfer files = %.2f'%(Time.time()-start))
            offset += npart

        if physical_velocities and block=="VEL " and redshift!=0:
            data *= np.sqrt(time)

        return data

