# This script tests the performance and results of the MASL and CIC libraries
import numpy as np
import MAS_library as MASL
import CIC_library as CIC
import sys,os,time


dims, BoxSize = 256, 1.0
dims2,dims3  = dims**2*1.0, dims**3*1.0

pos_3D = np.random.random((dims**3,3)).astype(np.float32)
pos_2D = np.random.random((dims**2,2)).astype(np.float32)
W      = np.ones(dims**3,dtype=np.float32)
print('positions and weights created')

# do a loop over the different MAS schemes
for MAS in ['NGP','CIC','TSC','PCS']:

    # do a loop over 3D and 2D arrays
    for pos,vals,elements in zip([pos_3D,pos_2D],
                                 [(dims,dims,dims),(dims,dims)],
                                 [dims3,dims2]):

        # do the calculation w/ and w/o weights
        for W in [None,W]:

            delta = np.zeros(vals, dtype=np.float32)
            if W is None:  MASL.MA(pos,delta,BoxSize,MAS)
            else:          MASL.MA(pos,delta,BoxSize,MAS,W)
            if abs(np.sum(delta, dtype=np.float64)/elements-1.0)<1e-5:
                print('Test passed!!!')
            else:
                print('Test not passed!!!');  sys.exit()



#### old CIC library ####
delta = np.zeros(dims**3, dtype=np.float32)
start = time.time()
CIC.NGP_serial(pos_3D,dims,BoxSize,delta)
print('Time taken = %.3f seconds'%(time.time()-start))

delta = np.zeros(dims**3, dtype=np.float32)
start = time.time()
CIC.CIC_serial(pos_3D,dims,BoxSize,delta)
print('Time taken = %.3f seconds'%(time.time()-start))

delta = np.zeros(dims**3, dtype=np.float32)
start = time.time()
CIC.TSC_serial(pos_3D,dims,BoxSize,delta)
print('Time taken = %.3f seconds'%(time.time()-start))
#########################




