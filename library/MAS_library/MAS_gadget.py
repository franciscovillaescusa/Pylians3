import numpy as np 
import time,sys,os
import readsnap,readgadget
import redshift_space_library as RSL
import MAS_library as MASL

def density_field_gadget(snapshot_fname, ptypes, dims, MAS='CIC',
                         do_RSD=False, axis=0, verbose=True): 

    start = time.time()
    if verbose:  print('\nComputing density field of particles',ptypes)

    # declare the array hosting the density field
    density = np.zeros((dims, dims, dims), dtype=np.float32)

    # read relevant paramaters on the snapshot
    head     = readgadget.header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Masses   = head.massarr*1e10 #Msun/h
    Nall     = head.nall;  Ntotal = np.sum(Nall,dtype=np.int64)
    filenum  = head.filenum
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    fformat  = head.format
    Hubble   = head.Hubble


    if ptypes==[-1]:    ptypes = [0, 1, 2, 3, 4, 5]
    if len(ptypes)==1:  single_component=True
    else:               single_component=False

    # do a loop over all files
    num = 0.0
    for i in range(filenum):

        # find the name of the sub-snapshot
        if filenum==1:       snapshot = snapshot_fname
        else:                snapshot = snapshot_fname+'.%d'%i
        if fformat=='hdf5':  snapshot = snapshot+'.hdf5'

        # find the local particles in the sub-snapshot
        head  = readgadget.header(snapshot)
        npart = head.npart

        # do a loop over all particle types
        for ptype in ptypes:

            if npart[ptype]==0:  continue

            # read positions in Mpc/h
            pos = readgadget.read_field(snapshot, "POS ", ptype)/1e3
            #pos = readsnap.read_block(snapshot,"POS ",parttype=ptype)/1e3

            # read velocities in km/s and move particles to redshift-space
            if do_RSD:
                vel = readgadget.read_field(snapshot, "VEL ", ptype)
                #vel = readsnap.read_block(snapshot,"VEL ",parttype=ptype)
                RSL.pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis)
                del vel

            # compute density field. If multicomponent, read/find masses
            if Masses[ptype]!=0:
                if single_component:
                    MASL.MA(pos, density, BoxSize, MAS) 
                    num += pos.shape[0]
                else:
                    mass = np.ones(npart[ptype], dtype=np.float32)*Masses[ptype]
                    MASL.MA(pos, density, BoxSize, MAS, W=mass)
                    num += np.sum(mass, dtype=np.float64)
            else:
                mass = readgadget.read_field(snapshot, "MASS", ptype)*1e10
                #mass = readsnap.read_block(snapshot,"MASS",
                #        parttype=ptype)*1e10 #Msun/h
                MASL.MA(pos, density, BoxSize, MAS, W=mass)
                num += np.sum(mass, dtype=np.float64)

    if verbose:
        print('%.8e should be equal to\n%.8e'\
            %(np.sum(density, dtype=np.float64), num))
        print('Time taken = %.2f seconds'%(time.time()-start))

    return np.asarray(density)















