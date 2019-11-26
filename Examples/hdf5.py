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
