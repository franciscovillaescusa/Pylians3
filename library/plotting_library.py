import numpy as np
import readsnap,readgadget
import MAS_library as MASL
import sys,os

# This library contains scripts that can be used to plot density field

# This routine finds the name of the density field
def density_field_name(snapshot_fname, x_min, x_max, y_min, y_max, 
                       z_min, z_max, dims, ptypes, plane, MAS):

    # this is a number that describes the ptypes used.
    # ptypes = [1] -----> part_num = 2
    # ptypes = [0,1] ---> part_num = 3
    # ptypes = [0,4] ---> part_num = 17
    part_num = np.sum(2**np.array(ptypes))

    # name of the density field
    f_df = 'density_field_%.3f_%.3f_%.3f_%.3f_%.3f_%.3f_%d_%d_%s_%s_%s.npy'\
           %(x_min, x_max, y_min, y_max, z_min, z_max, dims, part_num, 
             plane, MAS, snapshot_fname[-3:])

    return f_df

# This routine computes the coordinates of the density field square 
def geometry(snapshot_fname, plane, x_min, x_max, y_min, y_max, z_min, z_max):

    # read snapshot head and obtain BoxSize
    head    = readgadget.header(snapshot_fname)
    BoxSize = head.boxsize/1e3 #Mpc/h                    

    plane_dict = {'XY':[0,1], 'XZ':[0,2], 'YZ':[1,2]}

    # check that the plane is square
    if plane=='XY':
        length1 = x_max-x_min;  length2 = y_max-y_min;  depth = z_max-z_min 
        offset1 = x_min;        offset2 = y_min
    elif plane=='XZ':
        length1 = x_max-x_min;  length2 = z_max-z_min;  depth = y_max-y_min 
        offset1 = x_min;        offset2 = z_min
    else:
        length1 = y_max-y_min;  length2 = z_max-z_min;  depth = x_max-x_min 
        offset1 = y_min;        offset2 = z_min
    if length1!=length2:
        print('Plane has to be a square!!!'); sys.exit()
    BoxSize_slice = length1

    return length1, offset1, length2, offset2, depth, BoxSize_slice


# This routine reads the positions (and masses) of the particle type choosen
# and computes the 2D density field using NGP, CIC, TSC or PCS
def density_field_2D(snapshot_fname, x_min, x_max, y_min, y_max, z_min, z_max,
                     dims, ptypes, plane, MAS, save_density_field):
    
    plane_dict = {'XY':[0,1], 'XZ':[0,2], 'YZ':[1,2]}

    # read snapshot head and obtain BoxSize, filenum...
    head     = readgadget.header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h                    
    Nall     = head.nall
    Masses   = head.massarr*1e10 #Msun/h                  
    filenum  = head.filenum
    redshift = head.redshift

    # find the geometric values of the density field square
    len_x, off_x, len_y, off_y, depth, BoxSize_slice = \
            geometry(snapshot_fname, plane, x_min, x_max, y_min, y_max, 
                    z_min, z_max)

    # compute the mean density in the box
    if len(ptypes)==1 and Masses[ptypes[0]]!=0.0:
        single_specie = True
    else:
        single_specie = False

    # define the density array
    overdensity = np.zeros((dims,dims), dtype=np.float32)

    # do a loop over all subfiles in the snapshot
    total_mass, mass_slice = 0.0, 0.0;  renormalize_2D = False
    for i in range(filenum):

        # find the name of the subfile
        snap = snapshot_fname + '.%d'%i

        # in the last snapshot we renormalize the field
        if i==filenum-1:  renormalize_2D = True

        # do a loop over 
        for ptype in ptypes:

            # read the positions of the particles in Mpc/h
            pos = readgadget.read_field(snap,"POS ",ptype)/1e3

            if single_specie:  total_mass += len(pos)

            # keep only with the particles in the slice
            indexes = np.where((pos[:,0]>x_min) & (pos[:,0]<x_max) &
                               (pos[:,1]>y_min) & (pos[:,1]<y_max) &
                               (pos[:,2]>z_min) & (pos[:,2]<z_max) )
            pos = pos[indexes]

            # renormalize positions
            pos[:,0] -= x_min;  pos[:,1] -= y_min;  pos[:,2] -= z_min

            # project particle positions into a 2D plane
            pos = pos[:,plane_dict[plane]]

            # read the masses of the particles in Msun/h
            if not(single_specie):
                mass = readgadget.read_field(snap,"MASS",ptype)*1e10
                total_mass += np.sum(mass, dtype=np.float64)
                mass = mass[indexes]
                MASL.MA(pos, overdensity, BoxSize_slice, MAS=MAS, W=mass,
                        renormalize_2D=renormalize_2D)
            else:
                mass_slice += len(pos)
                MASL.MA(pos, overdensity, BoxSize_slice, MAS=MAS, W=None,
                        renormalize_2D=renormalize_2D)

    print('Expected mass = %.7e'%mass_slice)
    print('Computed mass = %.7e'%np.sum(overdensity, dtype=np.float64))

    # compute mean density in the whole box
    mass_density = total_mass*1.0/BoxSize**3 #(Msun/h)/(Mpc/h)^3 or #/(Mpc/h)^3

    print('mass density = %.5e'%mass_density)

    # compute the volume of each cell in the density field slice
    V_cell = BoxSize_slice**2*depth*1.0/dims**2  #(Mpc/h)^3

    # compute the mean mass in each cell of the slice
    mean_mass = mass_density*V_cell #Msun/h or #

    # compute overdensities
    overdensity /= mean_mass
    print(np.min(overdensity),'< rho/<rho> <',np.max(overdensity))

    # in our convention overdensity(x,y), while for matplotlib is
    # overdensity(y,x), so we need to transpose the field
    overdensity = np.transpose(overdensity)

    # save density field to file
    f_df = density_field_name(snapshot_fname, x_min, x_max, y_min, y_max, 
                              z_min, z_max, dims, ptypes, plane, MAS)
    if save_density_field:  np.save(f_df, overdensity)

    return overdensity
