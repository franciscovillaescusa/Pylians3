import numpy as np
import redshift_space_library as RSL
import MAS_library as MASL
import smoothing_library as SL
import units_library as UL
import MFs_library as MFL
import sys,os

###############################################################################
# This routine computes the Minkowski functionals of either a single species or of all
# species from a Gadget routine
# snapshot_fname -----------> name of the Gadget snapshot
# ptype --------------------> scalar: 0-GAS, 1-CDM, 2-NU, 4-Stars, -1:ALL
# MAS  ---------------------> Mass assignment scheme
# R_G  ---------------------> Smoothing scale
# Filter -------------------> Smoothing filter, Gaussian window function is recommended 
# dims ---------------------> Total number of cells is dims^3 to compute MFs
# BoxSize ------------------> The simulation box size
# do_RSD -------------------> MFs in redshift-space (True) or real-space (False)
# axis ---------------------> axis along which move particles in redshift-space
# thres_low ----------------> Lower bound of the threshold range
# thres_high ---------------> Higher bound of the threshold range
# thres_bins ---------------> Number of threshold bins
# folder_out ---------------> directory where to save the output
# threads ------------------> Number of threads to compute Minkowski functionals

def MFs_comp(snapshot,ptype,MAS,R_G,Filter,dims,BoxSize,do_RSD,axis,thres_low,thres_high,thres_bins,folder_out,threads=10):

    delta = MASL.density_field_gadget(snapshot,ptype,dims,MAS,do_RSD,axis)
    R     = np.float32(R_G)  #Mpc.h
    W_k   = SL.FT_filter(BoxSize, R, dims, Filter, threads)
    delta = SL.field_smoothing(delta, W_k, threads)
    

    # compute the MFs and save results to file
    MFs = MFL.MFs(delta,BoxSize,thres_low,thres_high,thres_bins,threads);  del delta
    np.savetxt(fout,MFs)