
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
cpdef FT_filter(float BoxSize, float R, int dims, Filter, int threads, 
                float kmin=0, float kmax=0):

    cdef int i,j,l,i1,j1,l1,middle,d2
    cdef float R2, R_grid, factor, k, kF
    cdef double normalization
    cdef float[:,:,::1] field
    cdef np.complex64_t[:,:,::1] field_k

    if Filter not in ['Top-Hat','Gaussian', 'Top-Hat-k']:
        raise Exception('Filter %s not implemented!'%Filter)

    middle        = dims//2
    normalization = 0.0
    R_grid        = (R*dims/BoxSize)
    R2            = R_grid**2
    kF            = 2.0*np.pi/BoxSize

    ###### Top-Hat ######
    if Filter=='Top-Hat':
        field = np.zeros((dims,dims,dims), dtype=np.float32)
        for i in prange(dims, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - dims

            for j in range(dims):
                j1 = j
                if j1>middle:  j1 = j1 - dims

                for l in range(dims):
                    l1 = l
                    if l1>middle:  l1 = l1 - dims

                    d2 = i1*i1 + j1*j1 + l1*l1
                    if d2<=R2:  
                        field[i,j,l] = 1.0
                        normalization += 1.0

    ###### Gaussian ######
    if Filter=='Gaussian':
        field = np.zeros((dims,dims,dims), dtype=np.float32)
        for i in prange(dims, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - dims

            for j in range(dims):
                j1 = j
                if j1>middle:  j1 = j1 - dims

                for l in range(dims):
                    l1 = l
                    if l1>middle:  l1 = l1 - dims

                    d2 = i1*i1 + j1*j1 + l1*l1
                
                    factor = exp(-d2/(2.0*R2))
                    field[i,j,l] = factor
                    normalization += factor

    ##### Top-Hat k #####
    if Filter=='Top-Hat-k':
        field_k = np.zeros((dims,dims,middle+1), dtype=np.complex64)
        for i in prange(dims, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - dims

            for j in range(dims):
                j1 = j
                if j1>middle:  j1 = j1 - dims

                for l in range(middle+1):
                    l1 = l
                    if l1>middle:  l1 = l1 - dims

                    # get the wavenumber of the considered mode
                    k = kF*sqrt(i1*i1 + j1*j1 + l1*l1)

                    # just put to 0 the modes outside kmin < k < kmax
                    if k>=kmin and k<kmax:  field_k[i,j,l] = 1.0
                    else:                   field_k[i,j,l] = 0.0

        # dont mess with DC mode
        field_k[0,0,0] = 1.0

        # now to determine the normalization of this filter, go to real-space
        field = PKL.IFFT3Dr_f(np.asarray(field_k), threads)
        normalization = np.sum(field, dtype=np.float64)

    # normalize the field
    for i in prange(dims, nogil=True):
        for j in range(dims):
            for l in range(dims):
                field[i,j,l] = field[i,j,l]/normalization
    
    # FT the field
    field_k = PKL.FFT3Dr_f(np.asarray(field), threads)
    return np.asarray(field_k)

############################################################################
# This routine places the filter on a 2D grid and returns its FT
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef FT_filter_2D(float BoxSize, float R, int grid, Filter, int threads, 
                   float kmin=0, float kmax=0):

    cdef int i,j,i1,j1,middle,d2
    cdef float R2, R_grid, factor, kF, kN, k 
    cdef double normalization
    cdef float[:,::1] field
    cdef np.complex64_t[:,::1] field_k

    if Filter not in ['Top-Hat','Gaussian', 'Top-Hat-k']:
        raise Exception('Filter %s not implemented!'%Filter)

    middle = grid//2
    normalization = 0.0
    R_grid = (R*grid/BoxSize)
    R2 = R_grid**2
    kF = 2.0*np.pi/BoxSize          #fundamental frequency
    kN = 2.0*np.pi/BoxSize*(grid/2) #Nyquist frequency

    ###### Top-Hat ######
    if Filter=='Top-Hat':
        field = np.zeros((grid,grid), dtype=np.float32)
        for i in prange(grid, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - grid

            for j in range(grid):
                j1 = j
                if j1>middle:  j1 = j1 - grid

                d2 = i1*i1 + j1*j1
                if d2<=R2:  
                    field[i,j] = 1.0
                    normalization += 1.0

    ###### Gaussian ######
    if Filter=='Gaussian':
        field = np.zeros((grid,grid), dtype=np.float32)
        for i in prange(grid, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - grid

            for j in range(grid):
                j1 = j
                if j1>middle:  j1 = j1 - grid

                d2 = i1*i1 + j1*j1
                
                factor = exp(-d2/(2.0*R2))
                field[i,j] = factor
                normalization += factor

    ##### Top-Hat k #####
    if Filter=='Top-Hat-k':
        field_k = np.zeros((grid,middle+1), dtype=np.complex64)
        for i in prange(grid, nogil=True):
            i1 = i
            if i1>middle:  i1 = i1 - grid

            for j in range(middle+1):
                j1 = j
                if j1>middle:  j1 = j1 - grid

                # get the wavenumber of the considered mode
                k = kF*sqrt(i1*i1 + j1*j1)

                # just put to 0 the modes outside kmin < k < kmax
                if k>=kmin and k<kmax:  field_k[i,j].real = 1.0
                else:                   field_k[i,j].real = 0.0

        # dont mess with DC mode
        field_k[0,0].real = 1.0
                
        # now to determine the normalization of this filter, go to real-space
        field = PKL.IFFT2Dr_f(np.asarray(field_k), threads)
        normalization = np.sum(field, dtype=np.float64)

    # normalize the field
    for i in prange(grid, nogil=True):
        for j in range(grid):
            field[i,j] = field[i,j]/normalization
    
    # FT the field
    field_k = PKL.FFT2Dr_f(np.asarray(field), threads)
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
    dims = field.shape[0];  middle = dims//2
    if field.shape[0]!=filter_k.shape[0]:
        raise Exception('field and filter have different grids!!!')

    ## compute FFT of the field (change this for double precision) ##
    field_k = PKL.FFT3Dr_f(field,threads) 

    # do a loop over the independent modes.
    for i in prange(dims, nogil=True):
        for j in range(dims):
            for k in range(middle+1): #k=[0,1,..,middle] --> kz>0
                field_k[i,j,k] = field_k[i,j,k]*filter_k[i,j,k]
                                       
    # Fourier transform back
    return PKL.IFFT3Dr_f(field_k,threads)

############################################################################
# This routine smooths a 2D field with a given filter and returns the smoothed 
# field. Inputs are the 2D field, the FT of the filter and the number of threads
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef field_smoothing_2D(field, np.complex64_t[:,::1] filter_k, int threads):

    cdef int i,j,grid,middle
    cdef np.complex64_t[:,::1] field_k

    # check that dimensions are the same
    grid = field.shape[0];  middle = grid//2
    if field.shape[0]!=filter_k.shape[0]:
        raise Exception('field and filter have different grids!!!')

    ## compute FFT of the field (change this for double precision) ##
    field_k = PKL.FFT2Dr_f(field,threads) 

    # do a loop over the independent modes.
    for i in prange(grid, nogil=True):
        for j in range(middle+1): #k=[0,1,..,middle] --> kz>0
            field_k[i,j] = field_k[i,j]*filter_k[i,j]
                                       
    # Fourier transform back
    return PKL.IFFT2Dr_f(field_k,threads)
