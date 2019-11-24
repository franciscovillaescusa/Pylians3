import numpy as np
import time,sys
cimport numpy as np
cimport cython 
from libc.stdlib cimport rand,srand
from libc.math cimport log,sqrt,sin,cos

cdef extern from "limits.h":
    int INT_MAX



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
