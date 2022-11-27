import numpy as np 
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,cos,floor,fabs
cimport MAS_c as MASC
import Pk_library as PKL
import MAS_library as MASL

# This routine computes the gravitational potential of a 3D density field
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef potential(delta, float Omega_m, float a, MAS='CIC', threads=1):

    cdef int kxx, kyy, kzz, kx, ky, kz, grid, middle, MAS_index
    cdef double green_func, prefact
    cdef double MAS_corr[3]
    ####### change this for double precision ######
    cdef float MAS_factor
    cdef np.complex64_t[:,:,::1] delta_k, phi_k
    ###############################################

    # get dimensions of the grid
    grid = delta.shape[0];  middle = grid//2
    MAS_index = PKL.MAS_function(MAS)

    ## compute FFT of the field (change this for double precision) ##
    delta_k = PKL.FFT3Dr_f(delta, threads)

    # define the potential field
    phi_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)

    # do a loop over the independent modes.
    # compute k,k_par,k_per, mu for each mode. k's are in kF units
    start2 = time.time();  prefact = np.pi/grid
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        MAS_corr[0] = PKL.MAS_correction(prefact*kx,MAS_index)
        
        for kyy in range(grid):
            ky = (kyy-grid if (kyy>middle) else kyy)
            MAS_corr[1] = PKL.MAS_correction(prefact*ky,MAS_index)

            for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                kz = (kzz-grid if (kzz>middle) else kzz)
                MAS_corr[2] = PKL.MAS_correction(prefact*kz,MAS_index)  

                # avoid DC mode
                if (kx==0 and ky==0 and kz==0):  continue

                # correct modes amplitude for MAS
                MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                # compute Green function
                green_func = -3.0*Omega_m/(8.0*a)/(sin(kx*prefact)**2 + sin(ky*prefact)**2 + sin(kz*prefact)**2)

                # multiply by Green function
                phi_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*green_func

    return PKL.IFFT3Dr_f(phi_k,threads)



# This routine computes the tidal tensor of a 3D density field
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
class tidal_tensor:
    def __init__(self, delta, float Omega_m, float a, MAS='CIC', threads=1):

        cdef int kxx, kyy, kzz, kx, ky, kz, grid, middle, MAS_index
        cdef double green_func, prefact
        cdef float complex phi_k
        cdef double MAS_corr[3]
        ####### change this for double precision ######
        cdef float MAS_factor
        cdef np.complex64_t[:,:,::1] delta_k, T00_k, T01_k, T02_k, T11_k, T12_k, T22_k
        ###############################################

        # get dimensions of the grid
        grid = delta.shape[0];  middle = grid//2
        MAS_index = PKL.MAS_function(MAS)

        # define the different tidal tensor arrays
        T00_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)
        T01_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)
        T02_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)
        T11_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)
        T12_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)
        T22_k = np.zeros((grid,grid,middle+1), dtype=np.complex64)

        ## compute FFT of the field (change this for double precision) ##
        delta_k = PKL.FFT3Dr_f(delta, threads)

        # do a loop over all modes
        prefact = np.pi/grid
        for kxx in range(grid):
            kx = (kxx-grid if (kxx>middle) else kxx)
            MAS_corr[0] = PKL.MAS_correction(prefact*kx,MAS_index)
        
            for kyy in range(grid):
                ky = (kyy-grid if (kyy>middle) else kyy)
                MAS_corr[1] = PKL.MAS_correction(prefact*ky,MAS_index)

                for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-grid if (kzz>middle) else kzz)
                    MAS_corr[2] = PKL.MAS_correction(prefact*kz,MAS_index)  

                    # avoid DC mode
                    if (kx==0 and ky==0 and kz==0):  continue

                    # correct modes amplitude for MAS
                    MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                    # compute Green function
                    green_func = -3.0*Omega_m/(8.0*a)/(sin(kx*prefact)**2 + sin(ky*prefact)**2 + sin(kz*prefact)**2)

                    # get each component of the tidal tensor
                    phi_k              = delta_k[kxx,kyy,kzz]*green_func
                    T00_k[kxx,kyy,kzz] = phi_k*kx*kx
                    T01_k[kxx,kyy,kzz] = phi_k*kx*ky
                    T02_k[kxx,kyy,kzz] = phi_k*kx*kz
                    T11_k[kxx,kyy,kzz] = phi_k*ky*ky
                    T12_k[kxx,kyy,kzz] = phi_k*ky*kz
                    T22_k[kxx,kyy,kzz] = phi_k*kz*kz

                
        self.T00 = PKL.IFFT3Dr_f(T00_k,threads)
        self.T01 = PKL.IFFT3Dr_f(T01_k,threads)
        self.T02 = PKL.IFFT3Dr_f(T02_k,threads)
        self.T11 = PKL.IFFT3Dr_f(T11_k,threads)
        self.T12 = PKL.IFFT3Dr_f(T12_k,threads)
        self.T22 = PKL.IFFT3Dr_f(T22_k,threads)



