import numpy as np 
import time,sys,os
cimport numpy as np
cimport cython
from cython.parallel import prange
from scipy.special import erfcinv,erfc

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)


cpdef PMinkowski_slice(np.ndarray[np.float32_t, ndim=3] delta_slices, int grid, np.ndarray[np.float32_t, ndim=1] thresholds, int threads):
    cdef int i,j,tt,len_thres,delta_min_bool
    cdef np.float32_t t
    cdef np.ndarray[np.float64_t, ndim=2] MFs
    cdef np.ndarray[np.int32_t, ndim=1] n
    cdef np.ndarray[np.float32_t, ndim=3] ds
    cdef np.ndarray[np.float32_t, ndim=2] ds_min
    cdef np.ndarray[np.float32_t, ndim=2] ds_max
    
    len_thres = len(thresholds)
    MFs       = np.zeros((len_thres,5),dtype=np.float64)
    n         = np.zeros(4,dtype=np.int32)
    ds        = np.zeros((8,grid,grid),dtype=np.float32)
    ds_min    = np.zeros((grid,grid),dtype=np.float32)
    ds_max    = np.zeros((grid,grid),dtype=np.float32)
    
    ds[0]  = delta_slices[0]
    ds[1]  = delta_slices[1]
    ds[2]  = np.roll(ds[0],-1,axis=0)
    ds[3]  = np.roll(ds[1],-1,axis=0)
    ds[4]  = np.roll(ds[0],-1,axis=1)
    ds[5]  = np.roll(ds[1],-1,axis=1)
    ds[6]  = np.roll(ds[2],-1,axis=1)
    ds[7]  = np.roll(ds[3],-1,axis=1)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)
    
    for tt in prange(len_thres,nogil=True,num_threads=threads):
        t = thresholds[tt]
        for i in range(grid):
            for j in range(grid):
                if t<ds_min[i,j]:
                    MFs[tt,1] += 1
                elif t<ds_max[i,j]:   
                    n[3] = ds[0,i,j]>t
                    n[2] = ((ds[0,i,j]>t or ds[1,i,j]>t)+
                            (ds[0,i,j]>t or ds[2,i,j]>t)+
                            (ds[0,i,j]>t or ds[4,i,j]>t))
                    n[1] = ((ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t)+
                            (ds[0,i,j]>t or ds[2,i,j]>t or ds[4,i,j]>t or ds[6,i,j]>t)+
                            (ds[0,i,j]>t or ds[4,i,j]>t or ds[1,i,j]>t or ds[5,i,j]>t))
                    n[0] = (ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t or
                            ds[4,i,j]>t or ds[5,i,j]>t or ds[6,i,j]>t or ds[7,i,j]>t)
                    MFs[tt,1] +=     n[3]
                    MFs[tt,2] += (-3*n[3] + n[2]) *2/9
                    MFs[tt,3] += ( 3*n[3]-2*n[2] + n[1]) *2/9
                    MFs[tt,4] += ( - n[3] + n[2] - n[1] + n[0])
        
    return MFs

################################################################################
################################################################################
# This routine computes the Minkowski functionals of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# thresholds --------> density threshold above which the excursion set is defined
# threads -----> number of threads (OMP) used 
# verbose------> whether print some information on the status/progress

class PMFs:
    def __init__(self,delta,BoxSize,thres_type,thres_low,thres_high,thres_bins,threads=1,verbose=True):

        start = time.time()
        cdef long long n, dims_x, dims_y, dims_z
        cdef int len_thres,tt,sli
        cdef np.float32_t t, miu_sigma
        cdef double miu, sigma, a, l
        ###############################################
        cdef np.ndarray[np.float32_t, ndim=1]   thresholds
        cdef np.ndarray[np.float64_t, ndim=2] MFs_from_slice
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D 

        # find dimensions of delta: we assume is a (dims,dims,dims) array
        if verbose:  print('\nComputing Minkowski functionals of the field...')
        dims_x,dims_y,dims_z = delta.shape
        n = dims_x*dims_y*dims_z
        a = BoxSize/dims_x
        len_thres = thres_bins+1

        ## Normalize the field ##
        miu     = np.mean(delta,dtype=np.float64)
        sigma   = np.std(delta,dtype=np.float64)
        miu_sigma = np.float32(miu/sigma)
        delta   = (delta - miu)/sigma
        delta   = np.concatenate((delta,delta[0:1,:,:]),axis=0)
        #################################
        # define arrays containing the Minkowski functionals.
        thresholds     = np.zeros(len_thres,dtype=np.float32)
        MFs3D          = np.zeros((len_thres,5), dtype=np.float64)
        MFs_from_slice = np.zeros((len_thres,5), dtype=np.float64)
        
        if thres_type=='rho':  
            for tt in range(len_thres):
                thresholds[tt] = (thres_low+tt*(thres_high-thres_low)/thres_bins)*miu_sigma
                MFs3D[tt,0] = thresholds[tt]
        elif thres_type=='nu':
            for tt in range(len_thres):
                thresholds[tt] = (thres_low+tt*(thres_high-thres_low)/thres_bins)
                MFs3D[tt,0] = thresholds[tt]
        else:
            print("thres_type = 'rho' or 'nu' ")
            return 0
   
        #calculate the MFs of slices of the density field and add up 
        for sli in range(dims_x):
            MFs_from_slice=PMinkowski_slice(delta[sli:sli+2,:,:],dims_x,thresholds,threads)
            for tt in range(len_thres):
                MFs3D[tt,1] += MFs_from_slice[tt,1]
                MFs3D[tt,2] += MFs_from_slice[tt,2] 
                MFs3D[tt,3] += MFs_from_slice[tt,3] 
                MFs3D[tt,4] += MFs_from_slice[tt,4] 

        a=BoxSize/dims_x
        l=dims_x**3
        MFs3D = np.multiply(MFs3D,np.array([1,1/l,1/(l*a),1/(l*a*a),1/(l*a*a*a)]))
    
        self.MFs3D = np.asarray(MFs3D)

        if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################
################################################################################


cpdef PMinkowski_slice_nuf(np.ndarray[np.float32_t, ndim=3] delta_slices, int grid, np.ndarray[np.float32_t, ndim=1] thresholds,int threads):
    cdef int i,j,tt,len_thres,delta_min_bool
    cdef np.float32_t t
    cdef np.ndarray[np.float64_t, ndim=2] MFs
    cdef np.ndarray[np.int32_t, ndim=2] n
    cdef np.ndarray[np.float32_t, ndim=3] ds
    cdef np.ndarray[np.float32_t, ndim=2] ds_min
    cdef np.ndarray[np.float32_t, ndim=2] ds_max
    
    len_thres = len(thresholds)
    MFs       = np.zeros((len_thres,5),dtype=np.float64)
    n         = np.zeros((4,len_thres),dtype=np.int32)
    ds        = np.zeros((8,grid,grid),dtype=np.float32)
    ds_min    = np.zeros((grid,grid),dtype=np.float32)
    ds_max    = np.zeros((grid,grid),dtype=np.float32)
    
    ds[0]  = delta_slices[0]
    ds[1]  = delta_slices[1]
    ds[2]  = np.roll(ds[0],-1,axis=0)
    ds[3]  = np.roll(ds[1],-1,axis=0)
    ds[4]  = np.roll(ds[0],-1,axis=1)
    ds[5]  = np.roll(ds[1],-1,axis=1)
    ds[6]  = np.roll(ds[2],-1,axis=1)
    ds[7]  = np.roll(ds[3],-1,axis=1)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)
    
    for tt in prange(len_thres,nogil=True,num_threads=threads):
        t = thresholds[tt]
        for i in range(grid):
            for j in range(grid):
                if t<ds_min[i,j]:
                    MFs[tt,1] += 1
                elif t<ds_max[i,j]:   
                    n[3,tt] = ds[0,i,j]>t
                    n[2,tt] = ((ds[0,i,j]>t or ds[1,i,j]>t)+
                               (ds[0,i,j]>t or ds[2,i,j]>t)+
                               (ds[0,i,j]>t or ds[4,i,j]>t))
                    n[1,tt] = ((ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t)+
                               (ds[0,i,j]>t or ds[2,i,j]>t or ds[4,i,j]>t or ds[6,i,j]>t)+
                               (ds[0,i,j]>t or ds[4,i,j]>t or ds[1,i,j]>t or ds[5,i,j]>t))
                    n[0,tt] = (ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t or
                               ds[4,i,j]>t or ds[5,i,j]>t or ds[6,i,j]>t or ds[7,i,j]>t)
                    
                    MFs[tt,1] +=     n[3,tt]
                    MFs[tt,2] += (-3*n[3,tt] + n[2,tt]) *2/9
                    MFs[tt,3] += ( 3*n[3,tt]-2*n[2,tt] + n[1,tt]) *2/9
                    MFs[tt,4] += ( - n[3,tt] + n[2,tt] - n[1,tt] + n[0,tt])
        
    return MFs

################################################################################
################################################################################
# This routine computes the Minkowski functionals of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# thresholds --------> density threshold above which the excursion set is defined
# threads -----> number of threads (OMP) used 
# verbose------> whether print some information on the status/progress

class PMFs_nuf:
    def __init__(self,delta,BoxSize,thres_low,thres_high,thres_bins,threads=1,verbose=True):

        start = time.time()
        cdef long long n, dims_x, dims_y, dims_z
        cdef int len_thres,tt,sli
        cdef np.float32_t t, miu_sigma
        cdef double miu, sigma, a, l
        ###############################################
        cdef np.ndarray[np.float32_t, ndim=1]   thresholds
        cdef np.ndarray[np.float32_t, ndim=1]   thres_frac
        cdef np.ndarray[np.float64_t, ndim=2] MFs_from_slice
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D 

        # find dimensions of delta: we assume is a (dims,dims,dims) array
        if verbose:  print('\nComputing Minkowski functionals of the field...')
        dims_x,dims_y,dims_z = delta.shape
        n = dims_x*dims_y*dims_z
        a = BoxSize/dims_x
        len_thres = thres_bins+1

        ## Normalize the field ##
        miu     = np.mean(delta,dtype=np.float64)
        sigma   = np.std(delta,dtype=np.float64)
        miu_sigma = np.float32(miu/sigma)
        delta   = (delta - miu)/sigma
        delta_sorted = np.sort(delta,axis=None)
        delta   = np.concatenate((delta,delta[0:1,:,:]),axis=0)
        #################################
        # define arrays containing the Minkowski functionals.
        thresholds     = np.zeros(len_thres,dtype=np.float32)
        MFs3D          = np.zeros((len_thres,5), dtype=np.float64)
        MFs_from_slice = np.zeros((len_thres,5), dtype=np.float64)

     
        # Get threshold values for threshold choice nu_f
        thresholds = np.linspace(thres_low,thres_high,num=len_thres,dtype=np.float32)
        thres_frac = erfc(thresholds)/2
        len_delta_sorted = len(delta_sorted)
        thres_index= len_delta_sorted*(1-thres_frac)
        thresholds = delta_sorted[np.clip(thres_index.astype(np.int32),0,len_delta_sorted-1)]
        for tt in range(len_thres):MFs3D[tt,0] = thresholds[tt]
   
        #calculate the MFs of slices of the density field and add up 
        for sli in range(dims_x):
            MFs_from_slice=PMinkowski_slice_nuf(delta[sli:sli+2,:,:],dims_x,thresholds,threads)
            for tt in range(len_thres):
                MFs3D[tt,1] += MFs_from_slice[tt,1]
                MFs3D[tt,2] += MFs_from_slice[tt,2] 
                MFs3D[tt,3] += MFs_from_slice[tt,3] 
                MFs3D[tt,4] += MFs_from_slice[tt,4] 

        a=BoxSize/dims_x
        l=dims_x**3
        MFs3D = np.multiply(MFs3D,np.array([1,1/l,1/(l*a),1/(l*a*a),1/(l*a*a*a)]))
    
        self.MFs3D = np.asarray(MFs3D)

        if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################
################################################################################


def PMinkowski_slice_mask(np.ndarray[np.float32_t, ndim=3] delta_slices, np.ndarray[np.float32_t, ndim=3] mask_slices, np.float32_t min_sky_weight, int grid, np.ndarray[np.float32_t, ndim=1] thresholds, int threads):
    cdef int i,j,tt,len_thres,delta_min_bool,sl # The length of cells with eight points all in mask
    cdef np.float32_t t
    cdef np.ndarray[np.float64_t, ndim=2] MFs
    cdef np.ndarray[np.int32_t, ndim=1] n
    cdef np.ndarray[np.float32_t, ndim=3] ds
    cdef np.ndarray[np.float32_t, ndim=2] ds_min
    cdef np.ndarray[np.float32_t, ndim=2] ds_max
    cdef np.ndarray[np.float32_t, ndim=3] ms
    cdef np.ndarray[np.float32_t, ndim=2] ms_min
    
    sl        = 0 # The length of cells with eight points all in mask
    len_thres = len(thresholds)
    MFs       = np.zeros((len_thres,5),dtype=np.float64)
    n         = np.zeros(4,dtype=np.int32)
    ds        = np.zeros((8,grid,grid),dtype=np.float32)
    ds_min    = np.zeros((grid,grid),dtype=np.float32)
    ds_max    = np.zeros((grid,grid),dtype=np.float32)
    ms        = np.zeros((8,grid,grid),dtype=np.float32)
    ms_min    = np.zeros((grid,grid),dtype=np.float32)
    
    ds[0]  = delta_slices[0]
    ds[1]  = delta_slices[1]
    ds[2]  = np.roll(ds[0],-1,axis=0)
    ds[3]  = np.roll(ds[1],-1,axis=0)
    ds[4]  = np.roll(ds[0],-1,axis=1)
    ds[5]  = np.roll(ds[1],-1,axis=1)
    ds[6]  = np.roll(ds[2],-1,axis=1)
    ds[7]  = np.roll(ds[3],-1,axis=1)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)
    
    ms[0]  = mask_slices[0]
    ms[1]  = mask_slices[1]
    ms[2]  = np.roll(ms[0],-1,axis=0)
    ms[3]  = np.roll(ms[1],-1,axis=0)
    ms[4]  = np.roll(ms[0],-1,axis=1)
    ms[5]  = np.roll(ms[1],-1,axis=1)
    ms[6]  = np.roll(ms[2],-1,axis=1)
    ms[7]  = np.roll(ms[3],-1,axis=1)
    ms_min = np.min(ms,axis=0)
    
    for i in range(grid):
        for j in range(grid):
            if ms_min[i,j]>min_sky_weight:
                sl = sl + 1
                for tt in range(len_thres):
                    t = thresholds[tt]
                    if t<=ds_min[i,j]:
                        MFs[tt,1] += 1
                    elif t<ds_max[i,j]:   
                        n[3] = ds[0,i,j]>t
                        n[2] = ((ds[0,i,j]>t or ds[1,i,j]>t)+
                                (ds[0,i,j]>t or ds[2,i,j]>t)+
                                (ds[0,i,j]>t or ds[4,i,j]>t))
                        n[1] = ((ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t)+
                                (ds[0,i,j]>t or ds[2,i,j]>t or ds[4,i,j]>t or ds[6,i,j]>t)+
                                (ds[0,i,j]>t or ds[4,i,j]>t or ds[1,i,j]>t or ds[5,i,j]>t))
                        n[0] = (ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t or
                                ds[4,i,j]>t or ds[5,i,j]>t or ds[6,i,j]>t or ds[7,i,j]>t)
                        MFs[tt,1] +=     n[3]
                        MFs[tt,2] += (-3*n[3] + n[2]) *2/9
                        MFs[tt,3] += ( 3*n[3]-2*n[2] + n[1]) *2/9
                        MFs[tt,4] += ( - n[3] + n[2] - n[1] + n[0])
        
    return (MFs,sl)

################################################################################
################################################################################
# This routine computes the Minkowski functionals of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# thresholds --------> density threshold above which the excursion set is defined
# threads -----> number of threads (OMP) used 
# verbose------> whether print some information on the status/progress

class PMFs_mask:
    def __init__(self,delta,mask,min_sky_weight,BoxSize,thres_type,thres_low,thres_high,thres_bins,threads=1,verbose=True):

        start = time.time()
        cdef long long n, dims_x, dims_y, dims_z
        cdef int len_thres,tt,sli,sl #sl is num of cells with eight points all in mask, sli is the sli-th slice
        cdef np.float32_t t, miu_sigma
        cdef double miu, sigma, a, l
        ###############################################
        cdef np.ndarray[np.float32_t, ndim=1]   thresholds
        cdef np.ndarray[np.float64_t, ndim=2] MFs_from_slice
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D 

        # find dimensions of delta: we assume is a (dims,dims,dims) array
        if verbose:  print('\nComputing Minkowski functionals of the field...')
        dims_x,dims_y,dims_z = delta.shape
        n = dims_x*dims_y*dims_z
        a = BoxSize/dims_x
        l = 0
        len_thres = thres_bins+1

        ## Normalize the field ##
        mask_pix= mask>min_sky_weight
        miu     = np.mean(delta[mask_pix],dtype=np.float64)
        sigma   = np.std(delta[mask_pix],dtype=np.float64)
        miu_sigma = np.float32(miu/sigma)
        delta[mask_pix] = (delta[mask_pix] - miu)/sigma
        delta   = np.concatenate((delta,delta[0:1,:,:]),axis=0)
        mask    = np.concatenate((mask,mask[0:1,:,:]),axis=0)
        #################################
        # define arrays containing the Minkowski functionals.
        thresholds     = np.zeros(len_thres,dtype=np.float32)
        MFs3D          = np.zeros((len_thres,5), dtype=np.float64)
        MFs_from_slice = np.zeros((len_thres,5), dtype=np.float64)

        if thres_type=='rho':  
            for tt in range(len_thres):
                thresholds[tt] = (thres_low+tt*(thres_high-thres_low)/thres_bins)*miu_sigma
                MFs3D[tt,0] = thresholds[tt]
        elif thres_type=='nu':
            for tt in range(len_thres):
                thresholds[tt] = (thres_low+tt*(thres_high-thres_low)/thres_bins)
                MFs3D[tt,0] = thresholds[tt]
        else:
            print("thres_type = 'rho' or 'nu' ")
            return 0
   
        #calculate the MFs of slices of the density field and add up 
        for sli in range(dims_x):
            MFs_from_slice, sl = PMinkowski_slice_mask(delta[sli:sli+2,:,:],mask[sli:sli+2,:,:],min_sky_weight,dims_x,thresholds,threads)
            l = l + sl
            for tt in range(len_thres):
                MFs3D[tt,1] += MFs_from_slice[tt,1]
                MFs3D[tt,2] += MFs_from_slice[tt,2] 
                MFs3D[tt,3] += MFs_from_slice[tt,3] 
                MFs3D[tt,4] += MFs_from_slice[tt,4] 

        a=BoxSize/dims_x
#         l=dims_x**3
        MFs3D = np.multiply(MFs3D,np.array([1,1/l,1/(l*a),1/(l*a*a),1/(l*a*a*a)]))
    
        self.MFs3D = np.asarray(MFs3D)

        if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################
################################################################################

def PMinkowski_slice_mask_nuf(np.ndarray[np.float32_t, ndim=3] delta_slices, np.ndarray[np.float32_t, ndim=3] mask_slices, np.float32_t min_sky_weight,int grid, np.ndarray[np.float32_t, ndim=1] thresholds, int threads):
    cdef int i,j,tt,len_thres,delta_min_bool,sl # The length of cells with eight points all in mask
    cdef np.float32_t t
    cdef np.ndarray[np.float64_t, ndim=2] MFs
    cdef np.ndarray[np.int32_t, ndim=1] n
    cdef np.ndarray[np.float32_t, ndim=3] ds
    cdef np.ndarray[np.float32_t, ndim=2] ds_min
    cdef np.ndarray[np.float32_t, ndim=2] ds_max
    cdef np.ndarray[np.float32_t, ndim=3] ms
    cdef np.ndarray[np.float32_t, ndim=2] ms_min
    
    sl        = 0 # The length of cells with eight points all in mask
    len_thres = len(thresholds)
    MFs       = np.zeros((len_thres,5),dtype=np.float64)
    n         = np.zeros(4,dtype=np.int32)
    ds        = np.zeros((8,grid,grid),dtype=np.float32)
    ds_min    = np.zeros((grid,grid),dtype=np.float32)
    ds_max    = np.zeros((grid,grid),dtype=np.float32)
    ms        = np.zeros((8,grid,grid),dtype=np.float32)
    ms_min    = np.zeros((grid,grid),dtype=np.float32)
    
    ds[0]  = delta_slices[0]
    ds[1]  = delta_slices[1]
    ds[2]  = np.roll(ds[0],-1,axis=0)
    ds[3]  = np.roll(ds[1],-1,axis=0)
    ds[4]  = np.roll(ds[0],-1,axis=1)
    ds[5]  = np.roll(ds[1],-1,axis=1)
    ds[6]  = np.roll(ds[2],-1,axis=1)
    ds[7]  = np.roll(ds[3],-1,axis=1)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)
    
    ms[0]  = mask_slices[0]
    ms[1]  = mask_slices[1]
    ms[2]  = np.roll(ms[0],-1,axis=0)
    ms[3]  = np.roll(ms[1],-1,axis=0)
    ms[4]  = np.roll(ms[0],-1,axis=1)
    ms[5]  = np.roll(ms[1],-1,axis=1)
    ms[6]  = np.roll(ms[2],-1,axis=1)
    ms[7]  = np.roll(ms[3],-1,axis=1)
    ms_min = np.min(ms,axis=0)
    
    for i in range(grid):
        for j in range(grid):
            if ms_min[i,j]>min_sky_weight:
                sl = sl + 1
                for tt in range(len_thres):
                    t = thresholds[tt]
                    if t<=ds_min[i,j]:
                        MFs[tt,1] += 1
                    elif t<ds_max[i,j]:   
                        n[3] = ds[0,i,j]>t
                        n[2] = ((ds[0,i,j]>t or ds[1,i,j]>t)+
                                (ds[0,i,j]>t or ds[2,i,j]>t)+
                                (ds[0,i,j]>t or ds[4,i,j]>t))
                        n[1] = ((ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t)+
                                (ds[0,i,j]>t or ds[2,i,j]>t or ds[4,i,j]>t or ds[6,i,j]>t)+
                                (ds[0,i,j]>t or ds[4,i,j]>t or ds[1,i,j]>t or ds[5,i,j]>t))
                        n[0] = (ds[0,i,j]>t or ds[1,i,j]>t or ds[2,i,j]>t or ds[3,i,j]>t or
                                ds[4,i,j]>t or ds[5,i,j]>t or ds[6,i,j]>t or ds[7,i,j]>t)
                        MFs[tt,1] +=     n[3]
                        MFs[tt,2] += (-3*n[3] + n[2]) *2/9
                        MFs[tt,3] += ( 3*n[3]-2*n[2] + n[1]) *2/9
                        MFs[tt,4] += ( - n[3] + n[2] - n[1] + n[0])
        
    return (MFs,sl)

################################################################################
################################################################################
# This routine computes the Minkowski functionals of a density field
# delta -------> 3D density field: (dims,dims,dims) numpy array
# BoxSize -----> size of the cubic density field
# thresholds --------> density threshold above which the excursion set is defined
# threads -----> number of threads (OMP) used 
# verbose------> whether print some information on the status/progress

class PMFs_mask_nuf:
    def __init__(self,delta,mask,min_sky_weight,BoxSize,thres_low,thres_high,thres_bins,threads=1,verbose=True):

        start = time.time()
        cdef long long n, dims_x, dims_y, dims_z
        cdef int len_thres,len_delta_sorted,tt,sli,sl #sl is num of cells with 8 points all in mask, sli is the sli-th slice
        cdef np.float32_t t, miu_sigma
        cdef double miu, sigma, a, l
        ###############################################
        cdef np.ndarray[np.float32_t, ndim=1]   thresholds
        cdef np.ndarray[np.float32_t, ndim=1]   thres_frac
        cdef np.ndarray[np.float64_t, ndim=2] MFs_from_slice
        cdef np.ndarray[np.float64_t, ndim=2] MFs3D 

        # find dimensions of delta: we assume is a (dims,dims,dims) array
        if verbose:  print('\nComputing Minkowski functionals of the field...')
        dims_x,dims_y,dims_z = delta.shape
        n = dims_x*dims_y*dims_z
        a = BoxSize/dims_x
        l = 0
        len_thres = thres_bins+1

        ## Normalize the field ##
        mask_pix= mask>min_sky_weight
        miu     = np.mean(delta[mask_pix],dtype=np.float64)
        sigma   = np.std(delta[mask_pix],dtype=np.float64)
        miu_sigma = np.float32(miu/sigma)
        delta[mask_pix] = (delta[mask_pix] - miu)/sigma
        delta_sorted = np.sort(delta[mask_pix],axis=None) # we need delta_sorted for thresholds
        delta   = np.concatenate((delta,delta[0:1,:,:]),axis=0)
        mask    = np.concatenate((mask,mask[0:1,:,:]),axis=0)
        #################################
        # define arrays containing the Minkowski functionals.
        thresholds     = np.zeros(len_thres,dtype=np.float32)
        MFs3D          = np.zeros((len_thres,5), dtype=np.float64)
        MFs_from_slice = np.zeros((len_thres,5), dtype=np.float64)

        # Get threshold values for threshold choice nu_f
        thresholds = np.linspace(thres_low,thres_high,num=len_thres,dtype=np.float32)
        thres_frac = erfc(thresholds)/2
        len_delta_sorted = len(delta_sorted)
        thres_index= len_delta_sorted*(1-thres_frac)
        thresholds = delta_sorted[np.clip(thres_index.astype(np.int32),0,len_delta_sorted-1)]
        for tt in range(len_thres):MFs3D[tt,0] = thresholds[tt]
   
        #calculate the MFs of slices of the density field and add up 
        for sli in range(dims_x):
            MFs_from_slice, sl = PMinkowski_slice_mask_nuf(delta[sli:sli+2,:,:],mask[sli:sli+2,:,:],min_sky_weight,dims_x,thresholds,threads)
            l = l + sl
            for tt in range(len_thres):
                MFs3D[tt,1] += MFs_from_slice[tt,1]
                MFs3D[tt,2] += MFs_from_slice[tt,2] 
                MFs3D[tt,3] += MFs_from_slice[tt,3] 
                MFs3D[tt,4] += MFs_from_slice[tt,4] 

        a=BoxSize/dims_x
#         l=dims_x**3
        MFs3D = np.multiply(MFs3D,np.array([1,1/l,1/(l*a),1/(l*a*a),1/(l*a*a*a)]))
    
        self.MFs3D = np.asarray(MFs3D)

        if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################
################################################################################


