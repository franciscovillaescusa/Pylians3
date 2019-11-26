# This script is an example on how to run the void finder on a Gadget snapshot
import numpy as np
import readgadget
import MAS_library as MASL
import void_library as VL

################################# INPUT ######################################
snapshot = '/simons/scratch/fvillaescusa/pdf_information/fiducial/0/snapdir_004/snap_004'

# density field parameters
grid = 768
MAS  = 'PCS'

# void finder parameters
Radii      = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
                       29, 31, 33, 35, 37, 39, 41], dtype=np.float32) #Mpc/h
threshold  = -0.7
threads1   = 16
threads2   = 4
void_field = True
##############################################################################

# read snapshot head and obtain BoxSize, Omega_m and Omega_L
head    = readgadget.header(snapshot)
BoxSize = head.boxsize/1e3  #Mpc/h

# read particle positions
pos = readgadget.read_block(snapshot,"POS ",ptype=1)/1e3 #Mpc/h

# compute density field
delta = np.zeros((grid, grid, grid), dtype=np.float32)
MASL.MA(pos, delta, BoxSize, MAS)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

# find the voids
V = VL.void_finder(delta, BoxSize, threshold, Radii, 
                   threads1, threads2, void_field=void_field)

# void properties
void_pos    = V.void_pos     #Mpc/h
void_radius = V.void_radius  #Mpc/h
VSF_R       = V.Rbins        #VSF bins in radius
VSF         = V.void_vsf     #VSF
if void_field:
    void_field = V.void_field  #array with 0. Cells belonging to voids have 1


