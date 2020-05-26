import numpy as np
import time,sys
cimport numpy as np
cimport cython 
from libc.stdlib cimport rand,srand
from libc.math cimport log,sqrt,sin,cos
import Pk_library as PKL

cdef extern from "limits.h":
    int INT_MAX
    int RAND_MAX


def theta(np.ndarray[np.complex128_t, ndim=3] Vx_k,Vy_k,Vz_k,
          int dims):

    #define the theta(k) array
    cdef np.ndarray[np.complex128_t,ndim=3] theta_k
    theta_k = np.zeros((dims,dims,dims),dtype=np.complex128)

    cdef int kx,ky,kz,ii,jj,kk,ii_m,jj_m,kk_m
    cdef int middle = dims//2

    for ii in range(dims):
        kx = (ii-dims if (ii>middle) else ii)
        for jj in range(dims):
            ky = (jj-dims if (jj>middle) else jj)
            for kk in range(dims):
                kz = (kk-dims if (kk>middle) else kk)
                
                theta_k[ii,jj,kk]=1j*(Vx_k[ii,jj,kk]*kx+
                                      Vy_k[ii,jj,kk]*ky+
                                      Vz_k[ii,jj,kk]*kz)
    return theta_k

    


def delta_k(int dims, precision,
            np.ndarray[double, ndim=1] kf,
            np.ndarray[double, ndim=1] Pkf, 
            Rayleigh_sampling, do_sirko, seed):

    start = time.clock()

    #define the delta(k) array
    dt = {'single':np.complex64, 'double':np.complex128}

    #initialize the random generator
    srand(seed)
    cdef int kf_elements = len(kf)
    
    #define the delta(k) array
    cdef np.ndarray[np.complex128_t,ndim=3] delta_k
    delta_k = np.zeros((dims,dims,dims),dtype=np.complex128)

    #define the Vx(k), Vy(k) and Vz(k) arrays
    cdef np.ndarray[np.complex128_t,ndim=3] Vx_k, Vy_k, Vz_k
    Vx_k = np.zeros((dims,dims,dims),dtype=np.complex128)
    Vy_k = np.zeros((dims,dims,dims),dtype=np.complex128)
    Vz_k = np.zeros((dims,dims,dims),dtype=np.complex128)

    cdef int kx,ky,kz,ii,jj,kk,ii_m,jj_m,kk_m
    cdef int lmin,lmax,l,middle = dims//2
    cdef float kmod,Pk
    cdef double phase,amplitude

    cdef float prefac1 = 2.0*np.pi/float(INT_MAX)
    cdef float constant1 = 1.0/float(INT_MAX)

    cdef double real_part,imag_part 
    cdef np.complex128_t  zero = 0.0+1j*0.0

    #we make a loop over the indexes of the matrix delta_k(ii,jj,kk)
    #but the vector k is given by \vec{k}=(kx,ky,kz)
    for ii in range(dims//2+1):
        kx = (ii-dims if (ii>middle) else ii)
        for jj in range(dims):
            ky = (jj-dims if (jj>middle) else jj)
            for kk in range(dims):
                kz = (kk-dims if (kk>middle) else kk)

                if (ii==0 and jj==0 and kk==0):
                    continue

                #find the value of |k| of the mode
                kmod = sqrt(kx*kx + ky*ky + kz*kz)
                
                #interpolate to compute P(|k|)
                lmin = 0;  lmax = kf_elements-1
                while (lmax-lmin>1):
                    l = (lmin+lmax)//2
                    if kf[l]<kmod:  lmin = l
                    else:           lmax = l
                Pk = ((Pkf[lmax]-Pkf[lmin])/(kf[lmax]-kf[lmin])*\
                          (kmod-kf[lmin]))+Pkf[lmin]           

                #generate the mode random phase and amplitude
                phase     = prefac1*rand()
                amplitude = constant1*rand()
                while (amplitude==0.0):   amplitude = constant1*rand()
                if Rayleigh_sampling:     amplitude = sqrt(-log(amplitude))
                else:                     amplitude = 1.0
                amplitude *= sqrt(Pk)
                
                #notice that delta(k), as defined here is dimensionless since 
                #P(k) has no units here. In reality delta(k) should have units
                #of volume. Also the velocities Vx_k,Vy_k and Vz_k are 
                #dimensionless
                real_part = amplitude*cos(phase)
                imag_part = amplitude*sin(phase)

                #fill the upper plane of the delta_k array
                if delta_k[ii,jj,kk]==zero:
                    value = (real_part+1j*imag_part)
                    delta_k[ii,jj,kk] = value
                    Vx_k   [ii,jj,kk] = value*1j*kx/kmod**2
                    Vy_k   [ii,jj,kk] = value*1j*ky/kmod**2
                    Vz_k   [ii,jj,kk] = value*1j*kz/kmod**2

                #fill the bottom plane of the delta_k array
                #if kx>0 then ii=kx and -kx corresponds to ii=dims-kx
                #if kx<0 then ii=dims-kx and -kx corresponds to ii=kx
                ii_m = (dims-kx if (kx>0) else -kx)
                jj_m = (dims-ky if (ky>0) else -ky)
                kk_m = (dims-kz if (kz>0) else -kz)
                if delta_k[ii_m,jj_m,kk_m]==zero:
                    value = (real_part-1j*imag_part)
                    delta_k[ii_m,jj_m,kk_m] = value
                    Vx_k   [ii_m,jj_m,kk_m] = value*(-1j)*kx/kmod**2
                    Vy_k   [ii_m,jj_m,kk_m] = value*(-1j)*ky/kmod**2
                    Vz_k   [ii_m,jj_m,kk_m] = value*(-1j)*kz/kmod**2


    #modes in the corners must be real since they have to be equal to themselves
    k_im = np.array([1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0), sqrt(3.0)])
    k_im *= middle;   Pk_im = np.interp(np.log10(k_im),np.log10(kf),Pkf)
    if Rayleigh_sampling:
        amplitudes = np.random.random(len(k_im))
        amplitudes[np.where(amplitudes==0.0)[0]] = 1e-20
        amplitudes = np.sqrt(-np.log(amplitudes))
    else:   amplitudes = np.ones(len(k_im))
    amplitudes = np.sqrt(Pk_im)

    delta_k_im = amplitudes.astype(dt[precision])
    
    delta_k[middle,0,0]           = delta_k_im[0]
    delta_k[0,middle,0]           = delta_k_im[1]
    delta_k[0,0,middle]           = delta_k_im[2]
    delta_k[middle,middle,0]      = delta_k_im[3]
    delta_k[0,middle,middle]      = delta_k_im[4]
    delta_k[middle,0,middle]      = delta_k_im[5]
    delta_k[middle,middle,middle] = delta_k_im[6];  #del k_im, Pk_im

    Vx_k[middle,0,0]           = delta_k_im[0]*middle/k_im[0]**2
    Vx_k[0,middle,0]           = delta_k_im[1]*0.0/k_im[1]**2
    Vx_k[0,0,middle]           = delta_k_im[2]*0.0/k_im[2]**2
    Vx_k[middle,middle,0]      = delta_k_im[3]*middle/k_im[3]**2
    Vx_k[0,middle,middle]      = delta_k_im[4]*0.0/k_im[4]**2
    Vx_k[middle,0,middle]      = delta_k_im[5]*middle/k_im[5]**2
    Vx_k[middle,middle,middle] = delta_k_im[6]*middle/k_im[6]**2

    Vy_k[middle,0,0]           = delta_k_im[0]*0.0/k_im[0]**2
    Vy_k[0,middle,0]           = delta_k_im[1]*middle/k_im[1]**2
    Vy_k[0,0,middle]           = delta_k_im[2]*0.0/k_im[2]**2
    Vy_k[middle,middle,0]      = delta_k_im[3]*middle/k_im[3]**2
    Vy_k[0,middle,middle]      = delta_k_im[4]*middle/k_im[4]**2
    Vy_k[middle,0,middle]      = delta_k_im[5]*0.0/k_im[5]**2
    Vy_k[middle,middle,middle] = delta_k_im[6]*middle/k_im[6]**2

    Vz_k[middle,0,0]           = delta_k_im[0]*0.0/k_im[0]**2
    Vz_k[0,middle,0]           = delta_k_im[1]*0.0/k_im[1]**2
    Vz_k[0,0,middle]           = delta_k_im[2]*middle/k_im[2]**2
    Vz_k[middle,middle,0]      = delta_k_im[3]*0.0/k_im[3]**2
    Vz_k[0,middle,middle]      = delta_k_im[4]*middle/k_im[4]**2
    Vz_k[middle,0,middle]      = delta_k_im[5]*middle/k_im[5]**2
    Vz_k[middle,middle,middle] = delta_k_im[6]*middle/k_im[6]**2


    #Vx_k[middle,0,0]           = 0.0
    #Vx_k[0,middle,0]           = 0.0
    #Vx_k[0,0,middle]           = 0.0
    #Vx_k[middle,middle,0]      = 0.0
    #Vx_k[0,middle,middle]      = 0.0
    #Vx_k[middle,0,middle]      = 0.0
    #Vx_k[middle,middle,middle] = 0.0

    #Vy_k[middle,0,0]           = 0.0
    #Vy_k[0,middle,0]           = 0.0
    #Vy_k[0,0,middle]           = 0.0
    #Vy_k[middle,middle,0]      = 0.0
    #Vy_k[0,middle,middle]      = 0.0
    #Vy_k[middle,0,middle]      = 0.0
    #Vy_k[middle,middle,middle] = 0.0

    #Vz_k[middle,0,0]           = 0.0
    #Vz_k[0,middle,0]           = 0.0
    #Vz_k[0,0,middle]           = 0.0
    #Vz_k[middle,middle,0]      = 0.0
    #Vz_k[0,middle,middle]      = 0.0
    #Vz_k[middle,0,middle]      = 0.0
    #Vz_k[middle,middle,middle] = 0.0


    #set the amplitude of the DC mode
    if do_sirko:   delta_k[0,0,0] = sqrt(Pkf[0])
    else:          delta_k[0,0,0] = 0.0

    time_taken = time.clock()-start
    print('delta(k) field generated\ntime taken =',time_taken,'seconds\n')
    return delta_k,Vx_k,Vy_k,Vz_k


# This function is used to generate mocks with perfectly known 2pt CF
def CF_mocks(int dims, precision,
             np.ndarray[double, ndim=1] rf,
             np.ndarray[double, ndim=1] xif, 
             seed):

    start = time.clock()

    #define the delta(k) array
    dt = {'single':np.complex64, 'double':np.complex128}

    #initialize the random generator
    srand(seed)
    cdef int r_elements = len(rf)
    
    #define the xi(r) array
    cdef np.ndarray[np.complex128_t,ndim=3] delta_k
    delta_k = np.zeros((dims,dims,dims),dtype=np.complex128)

    cdef int kx,ky,kz,ii,jj,kk,ii_m,jj_m,kk_m
    cdef int lmin,lmax,l,middle = dims//2
    cdef float kmod,Pk
    cdef double phase,amplitude

    cdef float prefac1 = 2.0*np.pi/float(INT_MAX)
    cdef float constant1 = 1.0/float(INT_MAX)

    cdef double real_part,imag_part 
    cdef np.complex128_t  zero = 0.0+1j*0.0

    #we make a loop over the indexes of the matrix delta_k(ii,jj,kk)
    #but the vector k is given by \vec{k}=(kx,ky,kz)
    for ii in range(dims//2+1):
        kx = (ii-dims if (ii>middle) else ii)
        for jj in range(dims):
            ky = (jj-dims if (jj>middle) else jj)
            for kk in range(dims):
                kz = (kk-dims if (kk>middle) else kk)

                if (ii==0 and jj==0 and kk==0):
                    continue

                #find the value of |k| of the mode
                kmod = sqrt(kx*kx + ky*ky + kz*kz)
                
                #interpolate to compute P(|k|)
                lmin = 0;  lmax = r_elements-1
                while (lmax-lmin>1):
                    l = (lmin+lmax)//2
                    if rf[l]<kmod:  lmin = l
                    else:           lmax = l
                xi_interp = ((xif[lmax]-xif[lmin])/(rf[lmax]-rf[lmin])*\
                                 (kmod-rf[lmin]))+xif[lmin]           

                #generate the mode random phase and amplitude
                phase     = 0.0  #prefac1*rand()
                #amplitude = constant1*rand()
                #while (amplitude==0.0):   amplitude = constant1*rand()
                #if Rayleigh_sampling:     amplitude = sqrt(-log(amplitude))
                #else:                     amplitude = 1.0
                amplitude = xi_interp

                #notice that delta(k), as defined here is dimensionless since 
                #P(k) has no units here. In reality delta(k) should have units
                #of volume. Also the velocities Vx_k,Vy_k and Vz_k are 
                #dimensionless
                real_part = amplitude*cos(phase)
                imag_part = amplitude*sin(phase)

                #fill the upper plane of the delta_k array
                if delta_k[ii,jj,kk]==zero:
                    value = (real_part+1j*imag_part)
                    delta_k[ii,jj,kk] = value

                #fill the bottom plane of the delta_k array
                #if kx>0 then ii=kx and -kx corresponds to ii=dims-kx
                #if kx<0 then ii=dims-kx and -kx corresponds to ii=kx
                ii_m = (dims-kx if (kx>0) else -kx)
                jj_m = (dims-ky if (ky>0) else -ky)
                kk_m = (dims-kz if (kz>0) else -kz)
                if delta_k[ii_m,jj_m,kk_m]==zero:
                    value = (real_part-1j*imag_part)
                    delta_k[ii_m,jj_m,kk_m] = value


    #modes in the corners must be real since they have to be equal to themselves
    k_im = np.array([1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0), sqrt(3.0)])
    k_im *= middle;   Pk_im = np.interp(k_im,rf,xif)
    #if Rayleigh_sampling:
    #    amplitudes = np.random.random(len(k_im))
    #    amplitudes[np.where(amplitudes==0.0)[0]] = 1e-20
    #    amplitudes = np.sqrt(-np.log(amplitudes))
    #else:   amplitudes = np.ones(len(k_im))
    amplitudes = np.ones(len(k_im))
    amplitudes = Pk_im
    print(amplitudes)


    delta_k_im = amplitudes.astype(dt[precision])
    
    delta_k[middle,0,0]           = delta_k_im[0]
    delta_k[0,middle,0]           = delta_k_im[1]
    delta_k[0,0,middle]           = delta_k_im[2]
    delta_k[middle,middle,0]      = delta_k_im[3]
    delta_k[0,middle,middle]      = delta_k_im[4]
    delta_k[middle,0,middle]      = delta_k_im[5]
    delta_k[middle,middle,middle] = delta_k_im[6];  #del k_im, Pk_im

    #set the amplitude of the DC mode
    #if do_sirko:   delta_k[0,0,0] = sqrt(Pkf[0])
    #else:          delta_k[0,0,0] = 0.0
    delta_k[0,0,0] = 0.0

    time_taken = time.clock()-start
    print('delta(k) field generated\ntime taken =',time_taken,'seconds\n')
    return delta_k




#####################################################################################
#####################################################################################
# grid ------------------> the code will return an image with grid x grid pixels
# kf --------------------> array with the values of k for Pk
# Pkf -------------------> array wiht the values of Pk
# Rayleigh_sampling -----> whether Rayleight sampling modes amplitude or not
# seed ------------------> random seed to generate the map
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def gaussian_field_2D(int grid, float[:] kf, float[:] Pkf, int Rayleigh_sampling, 
                      int seed, float BoxSize, threads, verbose=False):

    cdef int k_bins, kxx, kyy, kx, ky, kxx_m, kyy_m, middle, lmin, lmax, l
    cdef float kmod, Pk, phase, amplitude, real_part, imag_part 
    cdef np.complex64_t  zero = 0.0 + 1j*0.0
    cdef np.complex64_t[:,:] delta_k

    cdef float prefac1 = 2.0*np.pi/float(RAND_MAX)
    cdef float prefac2 = 2.0*np.pi/BoxSize
    cdef float constant1 = 1.0/float(RAND_MAX)

    start = time.time()
    k_bins, middle = len(kf), grid//2

    # define the density field in Fourier space
    delta_k = np.zeros((grid, middle+1), dtype=np.complex64)

    # initialize the random generator
    srand(seed)
    
    # we make a loop over the indexes of the matrix delta_k(kxx,kyy)
    # but the vector k is given by \vec{k}=(kx,ky)
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        kxx_m = (grid-kx if (kx>0) else -kx) #index corresponding to -kx

        for kyy in range(middle+1):
            ky = (kyy-grid if (kyy>middle) else kyy)

            # find the value of |k| of the mode
            kmod = sqrt(kx*kx + ky*ky)*prefac2
                
            # interpolate to compute P(|k|)
            lmin = 0;  lmax = k_bins-1
            while (lmax-lmin>1):
                l = (lmin+lmax)//2
                if kf[l]<kmod:  lmin = l
                else:           lmax = l
            Pk = ((Pkf[lmax]-Pkf[lmin])/(kf[lmax]-kf[lmin])*\
                  (kmod-kf[lmin]))+Pkf[lmin]           
            Pk = Pk*(grid**2/BoxSize)**2 #remove units. Density field shouldnt have units

            #generate the mode random phase and amplitude
            phase     = prefac1*rand()
            amplitude = constant1*rand()
            while (amplitude==0.0):   amplitude = constant1*rand()
            if Rayleigh_sampling==1:  amplitude = sqrt(-log(amplitude))
            else:                     amplitude = 1.0
            amplitude *= sqrt(Pk)
                
            # get real and imaginary parts
            real_part = amplitude*cos(phase)
            imag_part = amplitude*sin(phase)

            # fill the upper plane of the delta_k array
            if delta_k[kxx,kyy]==zero:
                delta_k[kxx,kyy] = real_part + 1j*imag_part

                # fill the bottom plane of the delta_k array
                # we do this ONLY if we fill up the upper plane
                # we need to satisfy delta(-k) = delta*(k)
                # k=(kx,ky)---> -k=(-kx,-ky). For ky!=0 or ky!=middle
                # the vector -k is not in memory, so we dont care
                # thus, we only care when ky==0 or ky==middle
                if ky==0 or ky==middle: #for these points: -ky=ky
                    if delta_k[kxx_m,kyy]==zero:
                        delta_k[kxx_m,kyy] = real_part - 1j*imag_part
                    if kxx_m==kxx:  #when k=-k delta(k) should be real
                        delta_k[kxx,kyy] = amplitude + 1j*0.0

    # force this in case input Pk doesnt go to k=0
    delta_k[0,0] = 0.0

    """
    # This is just to save the values of delta(k) to a file
    # mainly for debugging purposes
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)

        for kyy in range(middle+1):
            ky = (kyy-grid if (kyy>middle) else kyy)

            # find the value of |k| of the mode
            kmod = sqrt(kx*kx + ky*ky)*prefac2

            f = open('borrar.txt','a')
            f.write('%.3e %.3e\n'%(kmod,sqrt(delta_k[kxx,kyy].real**2 + \
                                             delta_k[kxx,kyy].imag**2)))
            f.close()
    """

    time_taken = time.time()-start
    if verbose:  
        print('delta(k) field generated\ntime taken = %.5f seconds\n'%time_taken)
    return PKL.IFFT2Dr_f(delta_k, threads)
#####################################################################################
#####################################################################################

#####################################################################################
#####################################################################################
# grid ------------------> the code will return a cube with grid x grid x grid voxels
# kf --------------------> array with the values of k for Pk
# Pkf -------------------> array wiht the values of Pk
# Rayleigh_sampling -----> whether Rayleight sampling modes amplitude or not
# seed ------------------> random seed to generate the map
# BoxSize ---------------> Size of the periodic box. Units in agreement with Pk
# threads ---------------> number of openmp threads, only for FFT
# verbose ---------------> print information about the status of the calculation
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def gaussian_field_3D(int grid, float[:] kf, float[:] Pkf, int Rayleigh_sampling, 
                      int seed, float BoxSize, threads, verbose=False):

    cdef int k_bins, kxx, kyy, kzz, kx, ky, kz, kxx_m, kyy_m, kzz_m
    cdef int middle, lmin, lmax, l
    cdef float kmod, Pk, phase, amplitude, real_part, imag_part 
    cdef np.complex64_t zero = 0.0 + 1j*0.0
    cdef np.complex64_t[:,:,:] delta_k

    cdef float phase_prefac = 2.0*np.pi/float(RAND_MAX)
    cdef float k_prefac     = 2.0*np.pi/BoxSize
    cdef float inv_max_rand = 1.0/float(RAND_MAX)

    start = time.time()
    k_bins, middle = len(kf), grid//2

    # define the density field in Fourier space
    delta_k = np.zeros((grid, grid, middle+1), dtype=np.complex64)

    # initialize the random generator
    srand(seed)
    
    # we make a loop over the indexes of the matrix delta_k(kxx,kyy,kzz)
    # but the vector k is given by \vec{k}=(kx,ky,kz)
    for kxx in range(grid):
        kx = (kxx-grid if (kxx>middle) else kxx)
        kxx_m = (grid-kx if (kx>0) else -kx) #index corresponding to -kx

        for kyy in range(grid):
            ky = (kyy-grid if (kyy>middle) else kyy)
            kyy_m = (grid-ky if (ky>0) else -ky) #index corresponding to -ky

            for kzz in range(middle+1):
                kz = (kzz-grid if (kzz>middle) else kzz)

                # find the value of |k| of the mode
                kmod = sqrt(kx*kx + ky*ky + kz*kz)*k_prefac
                
                # interpolate to compute P(|k|)
                lmin = 0;  lmax = k_bins-1
                while (lmax-lmin>1):
                    l = (lmin+lmax)//2
                    if kf[l]<kmod:  lmin = l
                    else:           lmax = l
                Pk = ((Pkf[lmax]-Pkf[lmin])/(kf[lmax]-kf[lmin])*\
                      (kmod-kf[lmin]))+Pkf[lmin]           
                Pk = Pk*(grid**2/BoxSize)**3 #remove units

                #generate the mode random phase and amplitude
                phase     = phase_prefac*rand()
                amplitude = inv_max_rand*rand()
                while (amplitude==0.0):   amplitude = inv_max_rand*rand()
                if Rayleigh_sampling==1:  amplitude = sqrt(-log(amplitude))
                else:                     amplitude = 1.0
                amplitude *= sqrt(Pk)
                
                # get real and imaginary parts
                real_part = amplitude*cos(phase)
                imag_part = amplitude*sin(phase)

                # fill the upper plane of the delta_k array
                if delta_k[kxx,kyy,kzz]==zero:
                    delta_k[kxx,kyy,kzz] = real_part + 1j*imag_part

                    # fill the bottom plane of the delta_k array
                    # we do this ONLY if we fill up the upper plane
                    # we need to satisfy delta(-k) = delta*(k)
                    # k=(kx,ky,kz)---> -k=(-kx,-ky,-kz). For kz!=0 or kz!=middle
                    # the vector -k is not in memory, so we dont care
                    # thus, we only care when kz==0 or kz==middle
                    if kz==0 or kz==middle: #for these points: -kz=kz
                        if delta_k[kxx_m,kyy_m,kzz]==zero:
                            delta_k[kxx_m,kyy_m,kzz] = real_part - 1j*imag_part
                        if kxx_m==kxx and kyy_m==kyy: #when k=-k delta(k) should be real
                            delta_k[kxx,kyy,kzz] = amplitude + 1j*0.0

    # force this in case input Pk doesnt go to k=0
    delta_k[0,0,0] = zero

    time_taken = time.time()-start
    if verbose:  
        print('delta(k) field generated\ntime taken = %.5f seconds\n'%time_taken)
    return PKL.IFFT3Dr_f(delta_k, threads)
#####################################################################################
#####################################################################################
