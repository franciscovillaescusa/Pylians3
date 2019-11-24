# The purpose of this library is to take an unsorted set of particles
# positions and return a structure that can be be used to quickly find
# particles in a given region of the space
from __future__ import print_function
import numpy as np 
import time,sys,os
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,log10,abs,exp,log,log10
from cpython cimport bool


# This routine reads an array with particle positions and computes the associated
# index of it:  index = dims2*i + dims*j + k
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef indexes_3D_cube(np.float32_t[:,:] pos, float BoxSize, float cell_size):

    cdef long particles, dims2, i
    cdef int dims, ii, jj, kk
    cdef np.int64_t[:] index
    cdef float factor

    # find the total number of halos in the catalogue
    particles = pos.shape[0]
    dims = <int>(BoxSize/cell_size) 
    dims2 = dims*dims
    factor = dims*1.0/BoxSize

    # each particle is characterized by its index = dims2*ii + dims*jj + kk
    # index_sorted is just the index array sorted
    index = np.zeros(particles, dtype=np.int64)

    # for each halo find its index
    for i in range(particles):
        ii = <int>(pos[i,0]*factor)
        if ii==dims:  ii = 0
        jj = <int>(pos[i,1]*factor)
        if jj==dims:  jj = 0
        kk = <int>(pos[i,2]*factor)
        if kk==dims:  kk = 0

        index[i] = ii*dims2 + jj*dims + kk

    return np.asarray(index)

# This routine takes as input an array with the positions of particles
# and returns an array with the particle positions sorted together with
# an array that specifies how many particles are in each cell. We assume
# periodic boundary conditions
# pos ----------> positions of the particles
# Boxsize ------> size of the box
# cell_size ----> size of the cells to organize the data
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
class sort_3D_pos:
    def __init__(self, np.float32_t[:,:] pos, float BoxSize, float cell_size,
            bool return_offset=True, bool return_indexes=True):

        print('Sorting particles... ',end='');  start = time.time()
        cdef long particles, dims2, dims3, i, number
        cdef int dims, ii, jj, kk
        cdef np.int64_t[:] index, index_sorted, offset, indexes
        cdef np.float32_t[:,:] pos_sorted
        cdef float factor

        # find the total number of halos in the catalogue
        particles = pos.shape[0]
        dims = <int>(BoxSize/cell_size) 
        dims2, dims3 = dims*dims, dims*dims*dims
        factor = dims*1.0/BoxSize

        # each particle is characterized by its index = dims2*ii + dims*jj + kk
        # index_sorted is just the index array sorted
        index        = np.zeros(particles,   dtype=np.int64)
        index_sorted = np.zeros(particles,   dtype=np.int64)

        # for each halo find its index
        for i in range(particles):
            ii = <int>(pos[i,0]*factor)
            if ii==dims:  ii = 0
            jj = <int>(pos[i,1]*factor)
            if jj==dims:  jj = 0
            kk = <int>(pos[i,2]*factor)
            if kk==dims:  kk = 0

            index[i] = ii*dims2 + jj*dims + kk

        # find the positions where the array index will be sorted
        # e.g. index = [1 0 4 9 3 2] ---> indexes = [1 0 5 4 2 3]
        # so index[indexes] = [0 1 2 3 4 9] 
        indexes = np.argsort(index)

        if return_indexes:  self.indexes = np.asarray(indexes)

        # get the new arrays with the sorted positions and the 
        # sorted indexes
        pos_sorted   = np.empty((particles,3), dtype=np.float32)
        index_sorted = np.empty(particles,     dtype=np.int64)
        for i in range(particles):
            number = indexes[i]
            pos_sorted[i,0] = pos[number,0]
            pos_sorted[i,1] = pos[number,1]
            pos_sorted[i,2] = pos[number,2]
            index_sorted[i] = index[number]
        del index
        self.pos_sorted = np.asarray(pos_sorted)

        # obtain the offset array that stores where the particles
        # in each cell begin and end
        if return_offset:
            offset = np.zeros(dims3+1, dtype=np.int64) - 1 #put -1 to empty cells 
            number = index_sorted[0]
            offset[index_sorted[0]] = 0
            for i in range(particles):
                if index_sorted[i]!=number:
                    offset[index_sorted[i]] = i
                    number = index_sorted[i]
            offset[dims3] = particles

            number = offset[dims3]
            for i in range(dims3,-1, -1):
                if offset[i]==-1:  offset[i] = number
                else:              number    = offset[i]

            self.offset = np.asarray(offset)

        print('Done : %.3f seconds'%(time.time()-start))



# This routine takes two arrays with positions as inputs. It computes the
# distances between the first array to the second one keeping only the
# closest distances
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef nearby_particles(np.float32_t[:,:] pos1, np.float32_t[:,:] pos2, 
    np.float32_t[:] radii2, np.int64_t[:] indexes_part, float BoxSize, float cell_size):

    print('Finding nearby halos...');  start = time.time()
    cdef np.float32_t[:,:] pos1_sorted, pos2_sorted
    cdef np.float32_t[:] radii2_sorted
    cdef np.int64_t[:] offset1, offset2, indexes1, indexes2
    cdef long i, j, particles1, particles2, index, dims2, particles_counted
    cdef float x, y, z, dist, factor, dx, dy, dz
    cdef int dims, ii, jj, kk, iii, jjj, kkk, i_index, j_index, k_index
    cdef list halo_numbers

    # find the number of particles in each array
    particles1 = pos1.shape[0]
    particles2 = pos2.shape[0]

    # find the total number of halos in the catalogue
    dims   = <int>(BoxSize/cell_size) 
    dims2  = dims*dims
    factor = dims*1.0/BoxSize

    # sort the positions of pos1 and pos2
    pos1_sorted, offset1, indexes1 = sort_3D_pos(pos1, BoxSize, cell_size)
    pos2_sorted, offset2, indexes2 = sort_3D_pos(pos2, BoxSize, cell_size)

    radii2_sorted = np.empty(radii2.shape[0], dtype=np.float32)
    for i in range(radii2.shape[0]):
        radii2_sorted[i] = radii2[indexes2[i]]

    halo_numbers = []
    for i in range(particles1):
        halo_numbers.append([])

    # do a loop over the particles2
    particles_counted = 0
    for i in range(particles1):
        x, y, z = pos1_sorted[i,0], pos1_sorted[i,1], pos1_sorted[i,2]

        ii = <int>(pos1_sorted[i,0]*factor)
        jj = <int>(pos1_sorted[i,1]*factor)
        kk = <int>(pos1_sorted[i,2]*factor)
        
        for iii in range(ii-1, ii+2):
            i_index = (iii+dims)%dims
            for jjj in range(jj-1, jj+2):
                j_index = (jjj+dims)%dims
                for kkk in range(kk-1, kk+2):
                    k_index = (kkk+dims)%dims

                    index = dims2*i_index + dims*j_index + k_index
                    particles_counted += (offset2[index+1] - offset2[index])

                    for j in range(offset2[index], offset2[index+1]):
                        dx = pos2_sorted[j,0]-x
                        if dx<-BoxSize/2.0:  dx += BoxSize
                        if dx>BoxSize/2.0:   dx -= BoxSize

                        dy = pos2_sorted[j,1]-y
                        if dy<-BoxSize/2.0:  dy += BoxSize
                        if dy>BoxSize/2.0:   dy -= BoxSize

                        dz = pos2_sorted[j,2]-z
                        if dz<-BoxSize/2.0:  dz += BoxSize
                        if dz>BoxSize/2.0:   dz -= BoxSize

                        dist = dx*dx + dy*dy + dz*dz

                        if dist<(radii2_sorted[j]*radii2_sorted[j]*1.3) or dist<0.100:
                            halo_numbers[i].append(indexes_part[indexes2[j]])

    print('Time taken = %.3f seconds'%(time.time()-start))
    print('Particles counted in = %ld'%particles_counted)

    return halo_numbers, indexes1







