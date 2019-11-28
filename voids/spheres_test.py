# This script takes an uniform density field and places random spheres
# of random sizes with a profile of delta(r)=-1*(1-(r/R)^3)
# Then it identifies the voids using the void finder. Finally, it plots
# the average density field across the entire box of the input and 
# recovered void field
import numpy as np 
import sys,os,time
import void_library as VL
from pylab import *
from matplotlib.colors import LogNorm

############################### INPUT #####################################
BoxSize = 1000.0 #Mpc/h
Nvoids  = 10     #number of random voids 
dims    = 512    #grid resolution to find voids

threshold = -0.5 #for delta(r)=-1*(1-(r/R)^3)

Rmax = 200.0   #maximum radius of the input voids
Rmin = 20.0    #minimum radius of the input voids
bins = 50      #number of radii between Rmin and Rmax to find voids

threads1 = 16 #openmp threads
threads2 = 4

f_out = 'Spheres_test.png'
###########################################################################


# create density field with random spheres
V = VL.random_spheres(BoxSize, Rmin, Rmax, Nvoids, dims)
delta = V.delta

# find voids
Radii = np.logspace(np.log10(Rmin), np.log10(Rmax), bins+1, dtype=np.float32)
V2 = VL.void_finder(delta, BoxSize, threshold, Radii, 
                    threads1, threads2, void_field=True)
delta2 = V2.in_void


# print the positions and radius of the generated voids
pos1 = V.void_pos
R1   = V.void_radius
pos2 = V2.void_pos
R2   = V2.void_radius

print('          X       Y       Z       R')
for i in range(Nvoids):
        dx = pos1[i,0]-pos2[:,0]
        dx[np.where(dx>BoxSize/2.0)] -= BoxSize
        dx[np.where(dx<-BoxSize/2.0)]+= BoxSize

        dy = pos1[i,1]-pos2[:,1]
        dy[np.where(dy>BoxSize/2.0)] -= BoxSize
        dy[np.where(dy<-BoxSize/2.0)]+= BoxSize

        dz = pos1[i,2]-pos2[:,2]
        dz[np.where(dz>BoxSize/2.0)] -= BoxSize
        dz[np.where(dz<-BoxSize/2.0)]+= BoxSize

        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        index = np.where(d==np.min(d))[0]
        pos, R = pos2[index][0], R2[index][0]

        print('\nVoid %02d'%i)
        print("Actual:     %6.2f  %6.2f  %6.2f  %6.2f"\
                %(pos1[i,0], pos1[i,1], pos1[i,2], R1[i]))
        print("Identified: %6.2f  %6.2f  %6.2f  %6.2f"\
                %(pos[0],    pos[1],    pos[2],   R))


############# plot results #############
fig = figure(figsize=(15,7))
ax1,ax2 = fig.add_subplot(121), fig.add_subplot(122) 

# plot the density field of the random spheres
ax1.imshow(np.mean(delta[:,:,:],axis=0),
    cmap=get_cmap('nipy_spectral'),origin='lower',
    vmin=-1, vmax=0.0,
    extent=[0, BoxSize, 0, BoxSize])

# plot  the void field identified by the void finder
ax2.imshow(np.mean(delta2[:,:,:],axis=0),
    cmap=get_cmap('nipy_spectral_r'),origin='lower',
    vmin=0, vmax=1.0,
    extent=[0, BoxSize, 0, BoxSize])

savefig(f_out, bbox_inches='tight')
show()
#########################################






