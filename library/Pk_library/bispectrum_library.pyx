import numpy as np 
import time,sys,os
import pyfftw
import Pk_library as PKL
cimport numpy as np
cimport cython
#from cython.parallel import prange
from libc.math cimport sqrt,pow,sin,log10,abs
from libc.stdlib cimport malloc, free

 

# This function implement the MAS correction to modes amplitude
#@cython.cdivision(False)
#@cython.boundscheck(False)
cdef inline double MAS_correction(double x, int MAS_index):
    return (1.0 if (x==0.0) else pow(x/sin(x),MAS_index))

# This function computes the bispectrum of a triangle with k1, k2 and theta.
# theta can be an array and the output will be the bispectrum for the different
# values of theta
# delta --------> 3D density field
# BoxSize ------> size of the simulation box
# k1 -----------> wavenumber of k1
# k2 -----------> wavenumber of k2
# theta --------> array with the theta values. Set [theta0] for only one element
# MAS ----------> Mass assignment scheme used to compute density field
# threads ------> number of threads to compute the FFTs
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
class Bk:
    def __init__(self, delta, BoxSize, k1, k2, theta, MAS='CIC', threads=1):
                 
        start = time.time()
        cdef int kxx, kyy, kzz, kx, ky, kz, MAS_index
        cdef int dims, middle, bins, i, j
        cdef unsigned long ID, dims2
        cdef list numbers
        cdef double k, prefact, pairs
        cdef double MAS_corr[3]
        cdef np.ndarray[np.float64_t, ndim=1] k_min, k_max
        cdef np.ndarray[np.float64_t, ndim=1] B, Q, triangles, kall, Pk
        ####### change this for double precision ######
        cdef float MAS_factor
        cdef np.complex64_t[:,:,::1] delta_k, delta1_k, delta2_k, delta3_k
        cdef np.complex64_t[:,:,::1] I1_k, I2_k, I3_k
        cdef np.float32_t[:,:,::1] delta1, delta2, delta3, I1, I2, I3
        ###############################################

        # find dimensions of delta: we assume is a (dims,dims,dims) array
        # determine the different frequencies and the MAS_index
        print('\nComputing bispectrum of the field...')
        dims = len(delta);  middle = dims//2;  dims2 = dims*dims
        kF,kN,kmax_par,kmax_per,kmax = PKL.frequencies(BoxSize,dims)
        MAS_index = PKL.MAS_function(MAS)

        # find the number of bins in theta. Define B, k_min, k_max arrays values of k3
        bins      = theta.shape[0]
        B         = np.zeros(bins,   dtype=np.float64)
        Q         = np.zeros(bins,   dtype=np.float64)
        triangles = np.zeros(bins,   dtype=np.float64)
        k_min     = np.zeros(bins+2, dtype=np.float64)
        k_max     = np.zeros(bins+2, dtype=np.float64)
        k_all     = np.zeros(bins+2, dtype=np.float64)
        Pk        = np.zeros(bins+2, dtype=np.float64)
        k3        = np.sqrt((k2*np.sin(theta))**2 + (k2*np.cos(theta)+k1)**2)
        k_all[0] = k1;  k_all[1] = k2;  k_all[2:] = k3

        # define the structure hosting the IDs of the cells within the spherical
        # shells. numbers[1] will be a list containing the IDs = dims^2*i+dims*j+k
        # of the cells that contribute to delta2_k. Find k_min and k_max for each k
        numbers = [];  numbers.append([]);  numbers.append([])
        k_min[0], k_max[0] = (k1-kF)/kF, (k1+kF)/kF
        k_min[1], k_max[1] = (k2-kF)/kF, (k2+kF)/kF
        for i in range(bins):  
            numbers.append([])
            k_min[i+2], k_max[i+2] = (k3[i]-kF)/kF, (k3[i]+kF)/kF
         
        ## compute FFT of the field (change this for double precision) ##
        delta_k = PKL.FFT3Dr_f(delta,threads)
        #################################

        # do a loop over the independent modes.
        # compute k,k_par,k_per, mu for each mode. k's are in kF units
        prefact = np.pi/dims
        for kxx in range(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)
        
            for kyy in range(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

                for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)  

                    # kz=0 and kz=middle planes are special
                    """
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue
                    """

                    # correct modes amplitude for MAS
                    MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*MAS_factor

                    # find the ID of the cell
                    ID = dims2*kxx + dims*kyy + kzz
                    
                    # compute |k| of the mode and its integer part
                    k = sqrt(kx*kx + ky*ky + kz*kz)
                    for i in range(bins+2):
                        if k>=k_min[i] and k<k_max[i]:
                            numbers[i].append(ID)

        # fill the delta1_k array and compute delta1
        delta1_k = np.zeros((dims, dims, dims//2+1), dtype=np.complex64)
        I1_k     = np.zeros((dims, dims, dims//2+1), dtype=np.complex64)
        for i in range(len(numbers[0])):
            ID = numbers[0][i]
            kxx, kyy, kzz = ID//dims2, (ID%dims2)//dims, (ID%dims2)%dims
            delta1_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]
            I1_k[kxx,kyy,kzz]     = 1.0
        delta1 = PKL.IFFT3Dr_f(delta1_k, threads);  del delta1_k
        I1     = PKL.IFFT3Dr_f(I1_k,     threads);  del I1_k

        # compute Pk(k1)
        Pk[0],pairs = 0.0, 0.0
        for kxx in range(dims):        
            for kyy in range(dims):
                for kzz in range(dims):
                    Pk[0] += (delta1[kxx,kyy,kzz]*delta1[kxx,kyy,kzz])
                    pairs += (I1[kxx,kyy,kzz]*I1[kxx,kyy,kzz])
        Pk[0] = (Pk[0]/pairs)*(BoxSize/dims**2)**3

        # fill the delta2_k array and compute delta2
        delta2_k = np.zeros((dims, dims, dims//2+1), dtype=np.complex64)
        I2_k     = np.zeros((dims, dims, dims//2+1), dtype=np.complex64)
        for i in range(len(numbers[1])):
            ID = numbers[1][i]
            kxx, kyy, kzz = ID//dims2, (ID%dims2)//dims, (ID%dims2)%dims
            delta2_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]
            I2_k[kxx,kyy,kzz]     = 1.0
        delta2 = PKL.IFFT3Dr_f(delta2_k, threads);  del delta2_k
        I2     = PKL.IFFT3Dr_f(I2_k,     threads);  del I2_k
        
        # compute Pk(k2)
        Pk[1],pairs = 0.0, 0.0
        for kxx in range(dims):        
            for kyy in range(dims):
                for kzz in range(dims):
                    Pk[1] += (delta2[kxx,kyy,kzz]*delta2[kxx,kyy,kzz])
                    pairs += (I2[kxx,kyy,kzz]*I2[kxx,kyy,kzz])
        Pk[1] = (Pk[1]/pairs)*(BoxSize/dims**2)**3

        # fill the delta3_k array for the different theta bins
        for j in range(bins):
            delta3_k = np.zeros((dims, dims, dims//2+1), dtype=np.complex64)
            I3_k     = np.zeros((dims, dims, dims//2+1), dtype=np.complex64)
            for i in range(len(numbers[j+2])):
                ID = numbers[j+2][i]
                kxx, kyy, kzz = ID//dims2, (ID%dims2)//dims, (ID%dims2)%dims
                delta3_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]
                I3_k[kxx,kyy,kzz]     = 1.0
            delta3 = PKL.IFFT3Dr_f(delta3_k, threads);  del delta3_k
            I3     = PKL.IFFT3Dr_f(I3_k,     threads);  del I3_k

            # compute Pk(k3)
            Pk[j+2],pairs = 0.0, 0.0
            for kxx in range(dims):        
                for kyy in range(dims):
                    for kzz in range(dims):
                        Pk[j+2] += (delta3[kxx,kyy,kzz]*delta3[kxx,kyy,kzz])
                        pairs += (I3[kxx,kyy,kzz]*I3[kxx,kyy,kzz])
            Pk[j+2] = (Pk[j+2]/pairs)*(BoxSize/dims**2)**3

            # make the final sum and save bispectrum in B
            B[j] = 0.0
            for kxx in range(dims):        
                for kyy in range(dims):
                    for kzz in range(dims):
                        B[j] += (delta1[kxx,kyy,kzz]*delta2[kxx,kyy,kzz]*delta3[kxx,kyy,kzz])
                        triangles[j] += (I1[kxx,kyy,kzz]*I2[kxx,kyy,kzz]*I3[kxx,kyy,kzz])
            B[j] = (B[j]/triangles[j])*(BoxSize**2/dims**3)**3                
            Q[j] = B[j]/(Pk[0]*Pk[1]+Pk[0]*Pk[j+2]+Pk[1]*Pk[j+2])
    
        self.B         = B
        self.triangles = triangles
        self.Q         = Q
        self.k         = k_all
        self.Pk        = Pk
        print('Time to compute bispectrum = %.2f'%(time.time()-start))


# F2 kernel
def F2(k1_vec, k2_vec):
    k1_mod = np.sqrt(np.dot(k1_vec, k1_vec))
    k2_mod = np.sqrt(np.dot(k2_vec, k2_vec))
    ctheta = np.dot(k1_vec, k2_vec)/(k1_mod*k2_mod)
    return 5.0/7.0 + 1.0/2.0*ctheta*(k1_mod/k2_mod + k2_mod/k1_mod) + 2.0/7.0*ctheta**2
        

# This routine computes the tree-level bispectrum given k1,k2 and the linear Pk
def Bispectrum_theory(k,Pk,k1,k2):

    bins = 50
    B = np.zeros(bins, dtype=np.float64)

    thetas = np.linspace(0, np.pi, bins)
    k1_vec = np.array([0, 0, k1])

    Pk1 = np.interp(np.log(k1), np.log(k), Pk)
    Pk2 = np.interp(np.log(k2), np.log(k), Pk)

    for i,theta in enumerate(thetas):
        k2_vec = np.array([0,  k2*np.sin(theta), k2*np.cos(theta)])
        k3_vec = np.array([0, -k2*np.sin(theta),-k2*np.cos(theta)-k1])
        k3 = np.sqrt(np.dot(k3_vec, k3_vec))
        Pk3 = np.interp(np.log(k3), np.log(k), Pk)
        
        F2_12 = F2(k1_vec, k2_vec)
        F2_13 = F2(k1_vec, k3_vec)
        F2_23 = F2(k2_vec, k3_vec)

        B[i] = 2.0*Pk1*Pk2*F2_12 + 2.0*Pk1*Pk3*F2_13 + 2.0*Pk2*Pk3*F2_23
        print(F2_12, F2_13, F2_23)
        print(k1,k2,k3)
        print(Pk1,Pk2,Pk3)
        print(i,'\n')

    return thetas,B
        
        
