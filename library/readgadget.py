# This library is designed to read Gadget format I, format II and hdf5 files
import numpy as np
import readsnap
import sys,os,h5py

# find snapshot name and format
def fname_format(snapshot):
    if os.path.exists(snapshot):
        if snapshot[-4:]=='hdf5':  filename, fformat = snapshot, 'hdf5'
        else:                      filename, fformat = snapshot, 'binary'
    elif os.path.exists(snapshot+'.0'):
        filename, fformat = snapshot+'.0', 'binary'
    elif os.path.exists(snapshot+'.hdf5'):
        filename, fformat = snapshot+'.hdf5', 'hdf5'
    elif os.path.exists(snapshot+'.0.hdf5'):
        filename, fformat = snapshot+'.0.hdf5', 'hdf5'
    else:  raise Exception('File not found!')
    return filename,fformat


# This class reads the header of the gadget file
class header:
    def __init__(self, snapshot):

        filename, fformat = fname_format(snapshot)

        if fformat=='hdf5':
            f             = h5py.File(filename, 'r')
            self.time     = f['Header'].attrs[u'Time']
            self.redshift = f['Header'].attrs[u'Redshift']
            self.boxsize  = f['Header'].attrs[u'BoxSize']
            self.filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
            self.omega_m  = f['Header'].attrs[u'Omega0']
            self.omega_l  = f['Header'].attrs[u'OmegaLambda']
            self.hubble   = f['Header'].attrs[u'HubbleParam']
            self.massarr  = f['Header'].attrs[u'MassTable']
            self.npart    = f['Header'].attrs[u'NumPart_ThisFile']
            self.nall     = f['Header'].attrs[u'NumPart_Total']
            self.cooling  = f['Header'].attrs[u'Flag_Cooling']
            self.format   = 'hdf5'
            f.close()

        else:        
            head = readsnap.snapshot_header(filename)
            self.time     = head.time
            self.redshift = head.redshift
            self.boxsize  = head.boxsize
            self.filenum  = head.filenum
            self.omega_m  = head.omega_m
            self.omega_l  = head.omega_l
            self.hubble   = head.hubble
            self.massarr  = head.massarr
            self.npart    = head.npart
            self.nall     = head.nall
            self.cooling  = head.cooling
            self.format   = head.format

        # km/s/(Mpc/h)
        self.Hubble = 100.0*np.sqrt(self.omega_m*(1.0+self.redshift)**3+self.omega_l)


# This function reads a block of an individual file of a gadget snapshot
def read_field(snapshot, block, ptype):

    filename, fformat = fname_format(snapshot)
    head              = header(filename)

    if fformat=="binary":
        return readsnap.read_block(filename, block, parttype=ptype)
    else:
        prefix = 'PartType%d/'%ptype
        f = h5py.File(filename, 'r')
        if   block=="POS ":  suffix = "Coordinates"
        elif block=="MASS":  suffix = "Masses"
        elif block=="ID  ":  suffix = "ParticleIDs"
        elif block=="VEL ":  suffix = "Velocities"
        else: raise Exception('block not implemented in readgadget!')
        array = f[prefix+suffix][:];  f.close()

        if block=="VEL ":  array *= np.sqrt(head.time)
        if block=="POS " and array.dtype==np.float64:
            array = array.astype(np.float32)

        return array

# This function reads a block from an entire gadget snapshot (all files)
# it can read several particle types at the same time. 
# ptype has to be a list. E.g. ptype=[1], ptype=[1,2], ptype=[0,1,2,3,4,5]
def read_block(snapshot, block, ptype, verbose=False):

    # find the format of the file and read header
    filename, fformat = fname_format(snapshot)
    head    = header(filename)    
    Nall    = head.nall
    filenum = head.filenum

    # find the total number of particles to read
    Ntotal = 0
    for i in ptype:
        Ntotal += Nall[i]

    # find the dtype of the block
    if   block=="POS ":  dtype=np.dtype((np.float32,3))
    elif block=="VEL ":  dtype=np.dtype((np.float32,3))
    elif block=="MASS":  dtype=np.float32
    elif block=="ID  ":  dtype=read_field(filename, block, ptype[0]).dtype
    else: raise Exception('block not implemented in readgadget!')

    # define the array containing the data
    array = np.zeros(Ntotal, dtype=dtype)


    # do a loop over the different particle types
    offset = 0
    for pt in ptype:

        # format I or format II Gadget files
        if fformat=="binary":
            array[offset:offset+Nall[pt]] = \
                readsnap.read_block(snapshot, block, pt, verbose=verbose)
            offset += Nall[pt]

        # single files (either binary or hdf5)
        elif filenum==1:
            array[offset:offset+Nall[pt]] = read_field(snapshot, block, pt)
            offset += Nall[pt]

        # multi-file hdf5 snapshot
        else:

            # do a loop over the different files
            for i in range(filenum):
                
                # find the name of the file to read
                filename = '%s.%d.hdf5'%(snapshot,i)

                # read number of particles in the file and read the data
                npart = header(filename).npart[pt]
                array[offset:offset+npart] = read_field(filename, block, pt)
                offset += npart   

    if offset!=Ntotal:  raise Exception('not all particles read!!!!')
            
    return array
