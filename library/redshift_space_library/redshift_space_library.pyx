import numpy as np 
import time,sys,os 
cimport numpy as np
cimport cython
#from cython.parallel import prange
from libc.math cimport sqrt,pow,sin,log10,abs
from libc.stdlib cimport malloc, free

############## AVAILABLE ROUTINES ##############
#pos_redshift_space
################################################



###############################################################################
#This routine receives the positions of the particles in configuration-space
#and return them in redshift-space
#pos --------------------> float array with the positions of the particles
#vel --------------------> float array with the velocities of the particles
#BoxSize ----------------> size of the simulation box
#Hubble -----------------> value of H(z) in (km/s)/(Mpc/h)
#redshift ---------------> redshift
#axis -------------------> axis along which perform the RSD
#The routines just uses: s = r + (1+z)/H(z)*v
#@cython.cdivision(True) never set this as % in python is different to c
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef void pos_redshift_space(float[:,::1] pos, float[:,::1] vel,
                       float BoxSize, float Hubble, float redshift, int axis):

    cdef long particles,i
    cdef float factor

    particles = pos.shape[0]
    factor    = (1.0 + redshift)/Hubble

    for i in range(particles):
        pos[i,axis] = pos[i,axis] + vel[i,axis]*factor

        #neutrinos can cross the box multiple times. Use % for the boundary
        if pos[i,axis]>BoxSize or pos[i,axis]<0.0:
            pos[i,axis] = (pos[i,axis]+BoxSize)%BoxSize 
###############################################################################

################################ old routine ##################################
"""
def pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis):
    #transform coordinates to redshift space
    delta_y=(vel[:,axis]/Hubble)*(1.0+redshift)  #displacement in Mpc/h
    pos[:,axis]+=delta_y #add distorsion to position of particle in real-space
    del delta_y

    #take care of the boundary conditions
    #beyond=np.where(pos[:,axis]>BoxSize)[0]; pos[beyond,axis]-=BoxSize
    #beyond=np.where(pos[:,axis]<0.0)[0];     pos[beyond,axis]+=BoxSize
    #del beyond

    # for neutrinos it could happen that in redshift-space their position
    # is further than 2 times size of the box, thus instead of doing
    # pos = pos+BoxSize or pos = pos - BoxSize we can just do pos = pos%BoxSize
    beyond = np.where((pos[:,axis]>BoxSize) | (pos[:,axis]<0.0))[0]
    pos[beyond,axis] = np.mod(pos[beyond,axis],BoxSize);  del beyond
"""
###############################################################################
