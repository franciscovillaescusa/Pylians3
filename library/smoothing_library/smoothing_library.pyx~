
import numpy as np
import sys,os,time
cimport numpy as np
cimport cython
from cython.parallel import prange,parallel
from libc.math cimport sqrt,pow,sin,cos,log,log10,fabs,round,exp
from libc.stdlib cimport malloc, free
from cython.parallel import prange,parallel
import Pk_library as PKL


DEF PI=3.141592653589793

############################################################################
# This routine places the filter on a grid and returns its FT
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef FT_filter(float BoxSize, float R, int dims, Filter, int threads):

    cdef int i,j,k,i1,j1,k1,middle,d2
    cdef float R2, R_grid, factor
    cdef double normalization
    cdef float[:,:,::1] field
    cdef np.complex64_t[:,:,::1] field_k

    if Filter not in ['Top-Hat','Gaussian']:
        raise Exception('Filter %s not implemented!'%Filter)

    middle = dims/2
    normalization = 0.0
    R_grid = (R*dims/BoxSize)
    R2 = R_grid**2

    # define the field array
    field = np.zeros((dims,dims,dims), dtype=np.float32)

    ###### Top-Hat ######
    if Filter=='Top-Hat':
        for i in prange(dims, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - dims

            for j in xrange(dims):
                j1 = j
                if j1>middle:  j1 = j1 - dims

                for k in xrange(dims):
                    k1 = k
                    if k1>middle:  k1 = k1 - dims

                    d2 = i1*i1 + j1*j1 + k1*k1
                    if d2<=R2:  
                        field[i,j,k] = 1.0
                        normalization += 1.0

    ###### Gaussian ######
    if Filter=='Gaussian':
        for i in prange(dims, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - dims

            for j in xrange(dims):
                j1 = j
                if j1>middle:  j1 = j1 - dims

                for k in xrange(dims):
                    k1 = k
                    if k1>middle:  k1 = k1 - dims

                    d2 = i1*i1 + j1*j1 + k1*k1
                
                    factor = exp(-d2/(2.0*R2))
                    field[i,j,k] = factor
                    normalization += factor

    # normalize the field
    for i in prange(dims, nogil=True):
        for j in xrange(dims):
            for k in xrange(dims):
                field[i,j,k] = field[i,j,k]/normalization
    
    # FT the field
    field_k = PKL.FFT3Dr_f(np.asarray(field), threads)
    return np.asarray(field_k)

############################################################################
# This routine smooths a field with a given filter and returns the smoothed 
# field. Inputs are the field, the FT of the filter and the number of threads
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef field_smoothing(field, np.complex64_t[:,:,::1] filter_k, int threads):

    cdef int i,j,k,dims,middle
    cdef np.complex64_t[:,:,::1] field_k

    # check that dimensions are the same
    dims = field.shape[0];  middle = dims/2
    if field.shape[0]!=filter_k.shape[0]:
        raise Exception('field and filter have different grids!!!')

    ## compute FFT of the field (change this for double precision) ##
    field_k = PKL.FFT3Dr_f(field,threads) 

    # do a loop over the independent modes.
    for i in prange(dims, nogil=True):
        for j in xrange(dims):
            for k in xrange(middle+1): #k=[0,1,..,middle] --> kz>0
                field_k[i,j,k] = field_k[i,j,k]*filter_k[i,j,k]
                                       
    # Fourier transform back
    return PKL.IFFT3Dr_f(field_k,threads)
