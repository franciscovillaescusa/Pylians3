from mpi4py import MPI
import numpy as np
import sys,os,time
import healpy as HP
from pylab import *
import cosmology_library as CL
import scipy.fftpack
import scipy.integrate as si
import scipy.special as ss
from scipy.optimize import minimize
import emcee, corner
import math
import gc

###### MPI DEFINITIONS ######                                         
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

###############################################################################
# These functions return the value of Omega_HI(z),b_HI and b_21cm for the model
def Omega_HI_model(z):
    return 4e-4*(1.0+z)**0.6

def HI_bias_model(z):
    return 0.904 + 0.135*(1.0+z)**1.696

def bias_21cm_model(z,Omega_m,Omega_L,h):
    Dz      = CL.linear_growth_factor(z,Omega_m,Omega_L)[0]
    HI_bias = HI_bias_model(z)
    mean_T  = 190.0*(1.0+z)**2/np.sqrt(Omega_m*(1.0+z)**3+Omega_L)*\
        Omega_HI_model(z)*h #mK
    b_21cm = Dz*mean_T*HI_bias
    return b_21cm
###############################################################################

###############################################################################
# This routine returns the dictionary survey that contains the parameters of
# the survey such as mean redshift, BoxSize, dims...etc
# bins_min,bins_max --------> consider maps between [bins_min,bins_max]
# nuTable ------------------> file with frequencies and redshifts of maps
# cosmo --------------------> dictionary with cosmological parameters
# instrument ---------------> dictionary with instrument parameters
def get_survey_params(bin_min,bin_max,nuTable,cosmo,instrument):

    # compute value of dims for the 
    dims = bin_max - bin_min + 1

    # read the frequencies and redshifts of the maps
    num, nui, nuf, zi, zf = np.loadtxt(nuTable,unpack=True)

    # compute the comoving distance to the different maps
    d_co = np.zeros(len(num),dtype=np.float32);  z = 0.5*(zi+zf)
    for i in range(len(z)):  
        d_co[i] = CL.comoving_distance(z[i],cosmo['Omega_m'],
                                       cosmo['Omega_L'])  #Mpc/h

    # for the maps selected compute BoxSize and mean distance among maps
    # notice that maps begin with 1, while z and d_co array begin with 0
    BoxSize = d_co[bin_min-1]-d_co[bin_max-1]  #Mpc/h
    print('pixel separation = %.3f Mpc/h'%(BoxSize/dims))
    print('BoxSize =',BoxSize,'Mpc/h')

    # compute maps mean frequency, Bandwidth, wavelength and angular resolution
    nu_map     = 0.5*(nui+nuf)              #MHz
    Bnu_map    = nuf-nui                    #MHz
    lambda_map = 3e8/(nu_map*1e6)           #meters
    beam_fwhm  = lambda_map/instrument['D'] #radians

    # compute the mean redshift of the survey
    mean_z = 0.5*(z[bin_min-1]+z[bin_max-1]);  print('survey <z> = %.3f'%mean_z)

    survey = {'dims':dims,       'BoxSize':BoxSize,       'mean_z':mean_z, 
              'Bnu_map':Bnu_map, 'lambda_map':lambda_map, 'beam_fwhm':beam_fwhm,
              'z_map':z,         'nu_map':nu_map}

    return survey
###############################################################################

###############################################################################
# This routine computes the expected noise level and variance from the 
# instrument temperature
# cosmo ------------------------> dictionary with cosmological parameters
# survey -----------------------> dictionary with survey parameters
# instrument -------------------> dictionary with instrument parameters
# z ----------------------------> mean redshift of the maps
# Tsystem ----------------------> system temperature at maps mean redshift
def noise_level_variance(cosmo,survey,instrument,z,Tsystem):

    # value of H(z) in km/s/(Mpc/h)
    Hz = 100.0*np.sqrt(cosmo['Omega_m']*(1.0+z)**3 + cosmo['Omega_L'])
    
    # mean brightness temperature at z in mK
    Tb = 190.0*(100.0*(1.0+z)**2/Hz)*Omega_HI_model(z)*cosmo['h'] 

    # compute noise level and variance in noise for auto-correlations
    noise_level = 0.5*0.21e-3*(1.0+z)**2/Hz*(Tsystem/Tb)**2/\
        (survey['t_pixel']*instrument['n_dish'])  #Mpc/h
    sigma_noise = noise_level/np.sqrt(survey['pixels'])
    survey['noise_level_auto'] = noise_level
    survey['sigma_noise_auto'] = sigma_noise

    # correct above numbers for cross-correlations. Total time is the same for
    # auto and cross-correlations
    sigma_noise /= np.sqrt(2.0)  #Mpc/h 
    noise_level = 0.0            #Mpc/h
    survey['noise_level_cross'] = noise_level
    survey['sigma_noise_cross'] = sigma_noise
###############################################################################

###############################################################################
# This routine computes the value of sigma_T for the 21cm IM pixels
# It returns arrays with the system temperature and sigma_T for the different 
# maps and the time per pixel
# cosmo ------------------> dictionary with cosmological parameters
# survey -----------------> dictionary with survey parameters
# instrument -------------> dictionary with instrument parameters
def sigmaT(survey,instrument):

    T_instrument = instrument['T_instrument']  #K
    n_dish       = instrument['n_dish']

    # compute the system temperature 
    T_sky    = 60*(survey['nu_map']/300.)**(-2.5) #Atmosphere temperature
    T_system = (T_instrument + T_sky)*1e3  #Total system temperature in mK

    # compute sigma_T. pixels is total number of pixels in whole sky. If survey
    # is covering a fraction fsky, the number of pixels surveyed is fsky*pixels
    t_pixel = survey['time_total']*3600.0/(survey['fsky']*survey['pixels']) #sec
    sigma_T = T_system/np.sqrt(2.0*survey['Bnu_map']*1e6*t_pixel*n_dish) #mK

    survey['T_system'] = T_system
    survey['t_pixel']  = t_pixel
    survey['sigma_T']  = sigma_T
###############################################################################

###############################################################################
# This routine computes the radial P(k) and xi(r) in each patch of the sky.
# The number of patchs on the sky is given by pixels/subpixels. This routine
# can be used to compute the mean P(k) and xi(r) in the whole sky and to 
# estimate the JK errors
# cosmo ------------------> dictionary with cosmological parameters
# survey -----------------> dictionary with survey parameters
# instrument -------------> dictionary with instrument parameters
# delta_k2 ---------------> structure with values of delta^2(k) for all los
def clustering_region(cosmo,survey,instrument,delta_k2,fmask,do_JK):
    
    # read mask
    mask = HP.read_map(fmask)

    dims = survey['dims'];  BoxSize   = survey['BoxSize']
    #subpixels = survey['subpixels'];  

    # compute the values of k and r of the modes for the 1D P(k) and xi(r)
    modes   = np.arange(dims,dtype=np.float64);  middle = dims//2
    indexes = np.where(modes>middle)[0];  modes[indexes] = modes[indexes]-dims
    k = modes*(2.0*np.pi/BoxSize)  #k in h/Mpc
    r = modes*(BoxSize*1.0/dims)   #r in Mpc/h
    k = np.absolute(k);  r = np.absolute(r)  #just take the modulus
    del indexes, modes

    # define the k-bins and r-bins
    k_bins = np.linspace(0,dims//2,dims//2+1)*(2.0*np.pi/BoxSize)
    r_bins = np.linspace(0,dims//2,dims//2+1)*(BoxSize*1.0/dims)

    # compute the number of modes and the average number-weighted value of k,r
    k_modes = np.histogram(k,bins=k_bins)[0]
    k_bin   = np.histogram(k,bins=k_bins,weights=k)[0]/k_modes
    r_modes = np.histogram(r,bins=r_bins)[0]
    r_bin   = np.histogram(r,bins=r_bins,weights=r)[0]/r_modes


    # compute P(k) over the lines of sight not masked out
    if do_JK==False:

        # find the lines of sight over which compute the radial P(k)
        indexes_los = np.where(mask==1.0)[0]
        
        ############################## Pk(k) ###############################
        # find the delta_k2 matrix of the region
        delta_k2_region = delta_k2[:,indexes_los]

        # take all LOS and compute the average value for each mode
        delta_k2_stacked = np.mean(delta_k2_region,dtype=np.float64,axis=1)

        # compute the 1D P(k)
        Pk_mean = np.histogram(k,bins=k_bins,weights=delta_k2_stacked)[0]
        Pk_mean = Pk_mean/(BoxSize*k_modes);  del delta_k2_stacked

        ############################## xi(r) ###############################
        # Fourier transform P(k) to obtain xi(r)
        xi = scipy.fftpack.fftn(delta_k2_region,overwrite_x=True,axes=(0,))
        del delta_k2_region

        # take all LOS and compute the average value for each mode
        xi_stacked = np.mean(xi,dtype=np.float64,axis=1);  del xi

        # compute the 1D xi(r)
        xi_mean = np.histogram(r,bins=r_bins,weights=xi_stacked)[0]
        xi_mean = xi_mean/r_modes;  del xi_stacked, indexes_los
        
        # no JK errors are computed here
        Pk_var = np.zeros(len(Pk_mean),dtype=np.float32)
        xi_var = np.zeros(len(xi_mean),dtype=np.float32)
        
        return [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var]


    # compute the number of maps to estimate variance with jackknife
    regions = survey['pixels']//subpixels
    survey['regions'] = regions

    # for each region compute the 1D P(k) and xi(r)
    Pk_region = np.zeros((regions,len(k_modes)),dtype=np.float32)
    xi_region = np.zeros((regions,len(r_modes)),dtype=np.float32)
    for i in range(regions):

        print('Computing P_1D(k) and xi_1D(r) of region',i,'out of',regions)

        # find the indexes of the region
        indexes_region = np.arange(i*subpixels,(i+1)*subpixels)
    
        ############################## Pk(k) ###############################
        # find the delta_k2 matrix of the region
        delta_k2_region = delta_k2[:,indexes_region]

        # take all LOS and compute the average value for each mode
        delta_k2_stacked = np.mean(delta_k2_region,dtype=np.float64,axis=1)

        # compute the 1D P(k)
        Pk = np.histogram(k,bins=k_bins,weights=delta_k2_stacked)[0]
        Pk = Pk/(BoxSize*k_modes);  Pk_region[i] = Pk;  del delta_k2_stacked

        ############################## xi(r) ###############################
        # Fourier transform P(k) to obtain xi(r)
        xi = scipy.fftpack.fftn(delta_k2_region,overwrite_x=True,axes=(0,))
        del delta_k2_region

        # take all LOS and compute the average value for each mode
        xi_stacked = np.mean(xi,dtype=np.float64,axis=1);  del xi

        # compute the 1D xi(r)
        xi = np.histogram(r,bins=r_bins,weights=xi_stacked)[0]
        xi = xi/r_modes;  xi_region[i] = xi;  del xi_stacked
    del indexes_region    

    # do a loop removing one region and compute P(k) and xi(r)
    Pk_jack = np.zeros((regions,len(k_modes)),dtype=np.float64)
    xi_jack = np.zeros((regions,len(r_modes)),dtype=np.float64)
    indexes = np.arange(regions)
    for i in range(regions):

        # find the indexes of all regions other than the deleted one
        indexes_to_consider = np.delete(indexes,i)

        # compute the average P(k) and xi(r) after removing the region
        Pk_jack[i] = np.mean(Pk_region[indexes_to_consider],axis=0)
        xi_jack[i] = np.mean(xi_region[indexes_to_consider],axis=0)

    # from all jackknife P(k) and xi(r) compute mean and variance
    Pk_mean = np.sum(Pk_jack,axis=0,dtype=np.float64)/regions
    xi_mean = np.sum(xi_jack,axis=0,dtype=np.float64)/regions

    Pk_var  = np.sqrt(np.sum((Pk_jack-Pk_mean)**2,axis=0,dtype=np.float64)*\
                          (regions-1.0)/regions)
    xi_var  = np.sqrt(np.sum((xi_jack-xi_mean)**2,axis=0,dtype=np.float64)*\
                          (regions-1.0)/regions)
    del Pk_jack, xi_jack, indexes

    return [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var]
###############################################################################
###############################################################################
# This routine reads the cosmo,cosmo+noise and cosmo+noise+fg maps and computes 
# the 1D P(k) taking into account the mask
# root_map --------> array containing the roots of the 5 different maps
# fmask -----------> array containing the files with the different masks
def clustering_1D_all(root_map,nuTable,D,Omega_m,Omega_L,h,n_side_map,
                      bin_min,bin_max,T_instrument,time_total,fsky,n_dish,
                      fmask,f_out):

    # define cosmology dictionary
    cosmo = {'Omega_m':Omega_m, 'Omega_L':Omega_L, 'h':h}

    # define instrument dictionary
    instrument = {'D':D, 'n_dish':n_dish, 'T_instrument':T_instrument}

    # obtain some parameters of the survey
    survey = get_survey_params(bin_min,bin_max,nuTable,cosmo,instrument)
    survey['time_total'] = time_total
    survey['fsky']       = fsky
    dims = survey['dims'];  BoxSize = survey['BoxSize']
    
    # define the data structure: create five structures for maps: 1 for maps
    # with cosmo, 2 for maps with cosmo+noise and 2 for maps with cosmo+noise+fg
    pixels = HP.nside2npix(n_side_map)  #number of pixels in a map with n_side
    delta1 = np.zeros((dims,pixels),dtype=np.float32) #cosmo
    delta2 = np.zeros((dims,pixels),dtype=np.float32) #cosmo+noise1
    delta3 = np.zeros((dims,pixels),dtype=np.float32) #cosmo+noise2
    delta4 = np.zeros((dims,pixels),dtype=np.float32) #cosmo+fg+noise1
    delta5 = np.zeros((dims,pixels),dtype=np.float32) #cosmo+fg+noise2
    print('Number of pixels in the maps =',pixels,'\n')
    survey['pixels'] = pixels

    # compute T_system and sigma_T for each map and the time per pixel
    sigmaT(survey,instrument);  sigma_T = survey['sigma_T']  #mK
                                       
    # compute expected noise level and variance from system noise
    Tsystem_z = np.interp(survey['mean_z'],survey['z_map'][::-1],
                          survey['T_system'][::-1]) #mK
    noise_level_variance(cosmo,survey,instrument,survey['mean_z'],Tsystem_z)


    # read the maps and smooth them
    for i in range(bin_min, bin_max+1):
        
        # read maps
        suffix = str(i).zfill(3) + '.fits'
        map1 = HP.read_map(root_map[0]+suffix, nest=False, verbose=False)
        map2 = HP.read_map(root_map[1]+suffix, nest=False, verbose=False)
        map3 = HP.read_map(root_map[2]+suffix, nest=False, verbose=False)
        map4 = HP.read_map(root_map[3]+suffix, nest=False, verbose=False)
        map5 = HP.read_map(root_map[4]+suffix, nest=False, verbose=False)
        
        # compute map mean temperature, find delta_T and fill data matrix
        meanT = np.mean(map1,dtype=np.float64);  delta1[i-bin_min] = map1-meanT
        meanT = np.mean(map2,dtype=np.float64);  delta2[i-bin_min] = map2-meanT
        meanT = np.mean(map3,dtype=np.float64);  delta3[i-bin_min] = map3-meanT
        meanT = np.mean(map4,dtype=np.float64);  delta4[i-bin_min] = map4-meanT
        meanT = np.mean(map5,dtype=np.float64);  delta5[i-bin_min] = map5-meanT
        del map1,map2,map3,map4,map5


    # Fourier transform only along the given axis
    print('\nFFT the overdensity fields along axis 0')
    delta1_k = scipy.fftpack.fftn(delta1,overwrite_x=True,axes=(0,))
    delta1_k *= (BoxSize/dims);  del delta1

    delta2_k = scipy.fftpack.fftn(delta2,overwrite_x=True,axes=(0,))
    delta2_k *= (BoxSize/dims);  del delta2

    delta3_k = scipy.fftpack.ifftn(delta3,overwrite_x=True,axes=(0,))
    delta3_k *= BoxSize;         del delta3

    delta4_k = scipy.fftpack.fftn(delta4,overwrite_x=True,axes=(0,))
    delta4_k *= (BoxSize/dims);  del delta4

    delta5_k = scipy.fftpack.ifftn(delta5,overwrite_x=True,axes=(0,))
    delta5_k *= BoxSize;         del delta5

    # compute delta^2(k) for auto-correlation w/o noise and for 
    # cross-correlation with noise
    delta2_k_auto   = np.absolute(delta1_k)**2;    del delta1_k
    delta2_k_cross1 = np.real(delta2_k*delta3_k);  del delta2_k,delta3_k
    delta2_k_cross2 = np.real(delta4_k*delta5_k);  del delta4_k,delta5_k
    
    # define suffix of output files
    suffix = '%d-%d_z=%.2f_D=%.0f_t=%.0f_nside=%d.dat'\
        %(bin_min,bin_max,survey['mean_z'],D,time_total,n_side_map)

    ######################## cosmo ##########################
    # compute mean P(k) and xi(r) and determine variance with JK
    [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var] = \
        clustering_region(cosmo,survey,instrument,delta2_k_auto,
                          fmask[0],do_JK=False);  del delta2_k_auto
                          
    # save results to file ignoring the DC mode
    f1 = f_out+'/Pk_cosmo_'+suffix;  f2 = f_out+'/xi_cosmo_'+suffix
    np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:],Pk_var[1:]]))
    np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:],xi_var[1:]]))

    #################### cosmo+noise #######################
    # compute mean P(k) and xi(r) and determine variance with JK
    [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var] = \
        clustering_region(cosmo,survey,instrument,delta2_k_cross1,
                          fmask[0],do_JK=False);  del delta2_k_cross1
                          
    # save results to file ignoring the DC mode
    f1 = f_out+'/Pk_noise_'+suffix;  f2 = f_out+'/xi_noise_'+suffix
    np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:],Pk_var[1:]]))
    np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:],xi_var[1:]]))

    ################## cosmo+noise+fg ######################
    # compute mean P(k) and xi(r) and determine variance with JK
    [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var] = \
        clustering_region(cosmo,survey,instrument,delta2_k_cross2,
                          fmask[1],do_JK=False);  del delta2_k_cross2
                          
    # save results to file ignoring the DC mode
    f1 = f_out+'/Pk_fg_'+suffix;  f2 = f_out+'/xi_fg_'+suffix
    np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:],Pk_var[1:]]))
    np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:],xi_var[1:]]))
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# This routine reads the healpix maps by David Alonso and computes the
# radial 21cm P(k) and xi(r).
# root_map -----------------------> 
# nuTable ------------------------> file with frequencies and redshifts of maps
# D ------------------------------> diameter of the antenna in m
# n_dish -------------------------> number of antennae
# T_instrument -------------------> temperature of the antenna in K
# time_total ---------------------> total observing time in hours
# fsky ---------------------------> fraction of the sky observed 
# Omega_m ------------------------> value of Omega_m
# Omega_L ------------------------> value of Omega_l
# h ------------------------------> value of h
# n_side_map ---------------------> 
# subpixels ----------------------> 
# bin_min ------------------------> 
# bin_max ------------------------> 
# f_out --------------------------> 
def clustering_1D(root_map,nuTable,D,Omega_m,Omega_L,h,n_side_map,subpixels,
                  bin_min,bin_max,T_instrument,time_total,fsky,n_dish,
                  fmask,do_JK,f_out):

    # define cosmology dictionary
    cosmo = {'Omega_m':Omega_m, 'Omega_L':Omega_L, 'h':h}

    # define instrument dictionary
    instrument = {'D':D, 'n_dish':n_dish, 'T_instrument':T_instrument}

    # obtain some parameters of the survey
    survey = get_survey_params(bin_min,bin_max,nuTable,cosmo,instrument)
    survey['time_total'] = time_total
    survey['fsky']       = fsky
    survey['subpixels']  = subpixels 
    dims = survey['dims'];  BoxSize = survey['BoxSize']

    # compute the growth factor at the mean redshift
    Dz = CL.linear_growth_factor(survey['mean_z'],Omega_m,Omega_L);  
    cosmo['Dz'] = Dz;  print('D(<z>=%.3f) = %.3f'%(survey['mean_z'],Dz));  del Dz
    
    # define the data structure: create three structures for maps: one for maps
    # with no noise, and two for maps with noise for cross-correlations
    pixels = HP.nside2npix(n_side_map)  #number of pixels in a map with n_side
    delta1 = np.zeros((dims,pixels),dtype=np.float32) #no noise
    delta2 = np.zeros((dims,pixels),dtype=np.float32) #with noise
    delta3 = np.zeros((dims,pixels),dtype=np.float32) #with noise
    print('Number of pixels in the maps =',pixels,'\n')
    survey['pixels'] = pixels

    # compute T_system and sigma_T for each map and the time per pixel
    sigmaT(survey,instrument);  sigma_T = survey['sigma_T']  #mK
                                       
    # compute expected noise level and variance from system noise
    Tsystem_z = np.interp(survey['mean_z'],survey['z_map'][::-1],
                          survey['T_system'][::-1]) #mK
    noise_level_variance(cosmo,survey,instrument,survey['mean_z'],Tsystem_z)


    # read the maps and smooth them
    for i in range(bin_min, bin_max+1):

        # get map name and read it
        print('working with map:',i)
        f_map = root_map    + str(i).zfill(3) + '.fits';  
        #map   = HP.read_map(f_map,nest=True,verbose=False)
        map   = HP.read_map(f_map,nest=False,verbose=False)
        map   = HP.smoothing(map,fwhm=survey['beam_fwhm'][i],regression=False)
        map   = HP.reorder(map,r2n=True)  #change ordering from ring to nest
        map   = HP.ud_grade(map,nside_out=n_side_map,
                            order_in='NESTED',order_out='NESTED')

        # plot to make sure that submaps are coherent
        #for i in range(pixels/subpixels):
        #    map2 = np.copy(map);  map2[i*subpixels:(i+1)*subpixels] = 0 
        #    HP.mollview(map2,nest=True)
        #show()

        # make three copies of the map: 1 without noise and 2 with noise
        # maps begin from 1, while sigma_T array begin from 0
        map1 = np.copy(map)                                                #mK
        map2 = np.copy(map);  map2 += sigma_T[i-1]*np.random.randn(pixels) #mK
        map3 = np.copy(map);  map3 += sigma_T[i-1]*np.random.randn(pixels) #mK

        # compute map mean temperature, find delta_T and fill data matrix
        mean_Tb1 = np.mean(map1,dtype=np.float64) #mK
        mean_Tb2 = np.mean(map2,dtype=np.float64) #mK
        mean_Tb3 = np.mean(map3,dtype=np.float64) #mK
        map1 = map1 - mean_Tb1;  delta1[i-bin_min] = map1
        map2 = map2 - mean_Tb2;  delta2[i-bin_min] = map2 
        map3 = map3 - mean_Tb3;  delta3[i-bin_min] = map3
        print('z = %.3f ----> <Tb> = %.4f mK'%(survey['z_map'][i],mean_Tb1))
        print('%.3f < delta_T(%d) < %.3f\n'%(np.min(map1),i,np.max(map1)))
        del map1,map2,map3,map


    # Fourier transform only along the given axis
    print('\nFFT the overdensity field along axis 0')
    delta1_k = scipy.fftpack.fftn(delta1,overwrite_x=True,axes=(0,))
    delta1_k *= (BoxSize/dims);  del delta1

    delta2_k = scipy.fftpack.fftn(delta2,overwrite_x=True,axes=(0,))
    delta2_k *= (BoxSize/dims);  del delta2

    delta3_k = scipy.fftpack.ifftn(delta3,overwrite_x=True,axes=(0,))
    delta3_k *= BoxSize;         del delta3

    # compute delta^2(k) for auto-correlation w/o noise and for 
    # cross-correlation with noise
    delta2_k_auto  = np.absolute(delta1_k)**2;    del delta1_k
    delta2_k_cross = np.real(delta2_k*delta3_k);  del delta2_k,delta3_k
    
    # define suffix of output files
    suffix = '%d-%d_z=%.2f_D=%.0f_t=%.0f_nside=%d.dat'\
        %(bin_min,bin_max,survey['mean_z'],D,time_total,n_side_map)

    #################### auto-correlation ########################
    # compute mean P(k) and xi(r) and determine variance with JK
    [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var] = \
        clustering_region(cosmo,survey,instrument,delta2_k_auto,fmask,do_JK)
    del delta2_k_auto
                          
    # save results to file ignoring the DC mode
    f1 = f_out+'/Pk_new_cosmo_'+suffix;  f2 = f_out+'/xi_new_cosmo_'+suffix
    np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:],Pk_var[1:]]))
    np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:],xi_var[1:]]))

    #################### cross-correlation #######################
    # compute mean P(k) and xi(r) and determine variance with JK
    [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var] = \
        clustering_region(cosmo,survey,instrument,delta2_k_cross,fmask,do_JK)
    del delta2_k_cross
                          
    # save results to file ignoring the DC mode
    f1 = f_out+'/Pk_new_noise_'+suffix;  f2 = f_out+'/xi_new_noise_'+suffix
    np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:],Pk_var[1:]]))
    np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:],xi_var[1:]]))
    
    print('Noise level = %.3f Mpc/h'%survey['noise_level_cross'])
    print('sigma noise = %.3e Mpc/h'%survey['sigma_noise_cross'])
###############################################################################
###############################################################################


###############################################################################
###############################################################################
def clustering_1D_fg(root_map1,root_map2,nuTable,D,Omega_m,Omega_L,h,n_side_map,
                     subpixels,bin_min,bin_max,T_instrument,time_total,fsky,
                     n_dish,method,n_fog,fmask,do_JK,f_out):
                  
    # define cosmology dictionary
    cosmo = {'Omega_m':Omega_m, 'Omega_L':Omega_L, 'h':h}

    # define instrument dictionary
    instrument = {'D':D, 'n_dish':n_dish, 'T_instrument':T_instrument}

    # obtain some parameters of the survey
    survey = get_survey_params(bin_min,bin_max,nuTable,cosmo,instrument)
    survey['time_total'] = time_total
    survey['fsky']       = fsky
    survey['subpixels']  = subpixels 
    dims = survey['dims'];  BoxSize = survey['BoxSize']

    # compute the growth factor at the mean redshift
    Dz = CL.linear_growth_factor(survey['mean_z'],Omega_m,Omega_L);  
    cosmo['Dz'] = Dz;  print('D(<z>=%.3f) = %.3f'%(survey['mean_z'],Dz));  del Dz
    
    # define the data structure: create three structures for maps: one for maps
    # with no noise, and two for maps with noise for cross-correlations
    pixels = HP.nside2npix(n_side_map)  #number of pixels in a map with n_side
    delta1 = np.zeros((dims,pixels),dtype=np.float32) #map1
    delta2 = np.zeros((dims,pixels),dtype=np.float32) #map2
    print('Number of pixels in the maps =',pixels,'\n')
    survey['pixels'] = pixels

    # compute T_system and sigma_T for each map and the time per pixel
    sigmaT(survey,instrument)
    sigma_T = survey['sigma_T']  #mK
                                       
    # compute expected noise level and variance from system noise
    Tsystem_z = np.interp(survey['mean_z'],survey['z_map'][::-1],
                          survey['T_system'][::-1]) #mK
    noise_level_variance(cosmo,survey,instrument,survey['mean_z'],Tsystem_z)


    # read the maps and smooth them
    for i in range(bin_min, bin_max+1):

        # get map name and read it
        print('working with map:',i)
        f_map1 = root_map1 + str(i).zfill(3) + '.fits'
        f_map2 = root_map2 + str(i).zfill(3) + '.fits'  
        map1   = HP.read_map(f_map1,nest=False,verbose=False)
        map2   = HP.read_map(f_map2,nest=False,verbose=False)
        
        # compute map mean temperature, find delta_T and fill data matrix
        # when cleaning the foregrounds the map only contain T-<T>
        mean_T1 = np.mean(map1,dtype=np.float64)
        mean_T2 = np.mean(map2,dtype=np.float64)
        delta1[i-bin_min] = map1-mean_T1
        delta2[i-bin_min] = map2-mean_T2
        del map1,map2


    # Fourier transform only along the given axis
    print('\nFFT the overdensity field along axis 0')
    delta1_k = scipy.fftpack.fftn(delta1,overwrite_x=True,axes=(0,))
    delta1_k *= (BoxSize/dims);  del delta1

    delta2_k = scipy.fftpack.ifftn(delta2,overwrite_x=True,axes=(0,))
    delta2_k *= BoxSize;         del delta2

    # compute delta^2(k) for cross-correlation with noise
    delta2_k_cross = np.real(delta1_k*delta2_k);  del delta1_k,delta2_k
    
    #################### cross-correlation #######################
    # compute mean P(k) and xi(r) and determine variance with JK
    data = clustering_region(cosmo,survey,instrument,delta2_k_cross,fmask,do_JK)
    if do_JK:  [k_bin,r_bin,Pk_mean,xi_mean,Pk_var,xi_var] = data
    else:      [k_bin,r_bin,Pk_mean,xi_mean] = data
    del delta2_k_cross
                          
    # save results to file ignoring the DC mode
    suffix = '%d-%d_z=%.2f_D=%.0f_t=%.0f_nside=%d.dat'\
        %(bin_min,bin_max,survey['mean_z'],D,time_total,n_side_map)
    f1 = f_out+'/Pk_fg_'+method+'_'+n_fog+'_'+suffix
    f2 = f_out+'/xi_fg_'+method+'_'+n_fog+'_'+suffix
    if do_JK:
        np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:],Pk_var[1:]]))
        np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:],xi_var[1:]]))
    else:
        np.savetxt(f1,np.transpose([k_bin[1:],Pk_mean[1:]]))
        np.savetxt(f2,np.transpose([r_bin[1:],xi_mean[1:]]))
    
    print('Noise level = %.3f Mpc/h'%survey['noise_level_cross'])
    print('sigma noise = %.3e Mpc/h'%survey['sigma_noise_cross'])
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# This routine carry out a fit to the P(k) data in f_in, over the interval
# [kmin,kmax] using as template the P(k) in f_w and f_nw (which are the mean 
# over 100 realizations with wiggles and no-wiggles). The routine requires an
# initial guess of the parameters (theta0)
# f_w ---------> file with the P(k)_wiggles template
# f_nw --------> file with the P(k)_no_wiggles template
# f_in --------> file with P(k) to fit
# kmin,kmax ---> k-range used for the fit
# theta0 ------> initial guess of best fit parameters
# ndim --------> number of free parameters to fit
# nwalkers ----> number of MCMC chains
# chain_pts ---> number of points in each MCMC chain
# fout --------> name of file to save MCMC plots
def fitting(f_w,f_nw,f_in,kmin,kmax,theta0,ndim,nwalkers,chain_pts,fout):
    
    # sanity check
    if ndim!=len(theta0):
        print('number of free parameters is not %d!!!'%ndim);  sys.exit()

    # read templates and realization
    kw,Pkw,dPkw    = np.loadtxt(f_w,unpack=True)
    knw,Pknw,dPknw = np.loadtxt(f_nw,unpack=True)
    k,Pk,dPk       = np.loadtxt(f_in,unpack=True)

    # keep only with data in the given k-range
    indexes  = np.where((k>kmin) & (k<kmax))[0]
    k,Pk,dPk = k[indexes],Pk[indexes],dPkw[indexes] #k,Pk and dPk of data to fit

    # compute number of degrees of freedom
    ndof = len(indexes)-len(theta0);  del indexes

    ###### find best fit parameters and minimum chi2 ######
    chi2_func = lambda *args: -lnlike_sim_template(*args)
    best_fit  = minimize(chi2_func,theta0,args=(k,Pk,dPk,kw,Pkw,knw,Pknw),
                         method='Powell')
    theta_best_fit = best_fit["x"]
    chi2 = chi2_func(theta_best_fit,k,Pk,dPk,kw,Pkw,knw,Pknw)*1.0/ndof
    
    alpha, b, p0, p1, p2 = theta_best_fit
    print('alpha = %10.3e'%alpha)
    print('b     = %10.3e'%b)
    print('p0    = %10.3e'%p0)
    print('p1    = %10.3e'%p1)
    print('p2    = %10.3e'%p2)
    print('chi2  = %10.3f\n'%chi2)
    #######################################################

    ###################### run MCMC #######################
    # starting point of each MCMC chain: small ball around best fit or estimate
    pos = [theta_best_fit+0.2*np.random.randn(ndim) for i in range(nwalkers)]

    # run MCMC
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlike_sim_template,
                                args=(k,Pk,dPk,kw,Pkw,knw,Pknw))      
    sampler.run_mcmc(pos,chain_pts)
    sampler.reset()

    # get the points in the MCMC chains
    samples = sampler.chain[:,500:,:].reshape((-1,ndim))
    sampler.reset()

    # make the plot and save it to file
    fig = corner.corner(samples,truths=theta0,
                        labels=[r"$\alpha$",r"$b$",r"$p_0$",r"$p_1$",r"$p_2$"])
    fig.savefig(fout)

    # compute best fit and associated error
    alpha, b, p0, p1, p2 =  map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(samples, [16, 50, 84],
                                                   axis=0)))
                                      
    # compute chi^2 of best fit
    theta = [alpha[0],b[0],p0[0],p1[0],p2[0]]
    chi2 = chi2_func(theta,k,Pk,dPk,kw,Pkw,knw,Pknw)/ndof

    print('alpha = %10.3e + %.2e - %.2e'%(alpha[0],alpha[1],alpha[2]))
    print('b     = %10.3e + %.2e - %.2e'%(b[0],b[1],b[2]))
    print('p0    = %10.3e + %.2e - %.2e'%(p0[0],p0[1],p0[2]))
    print('p1    = %10.3e + %.2e - %.2e'%(p1[0],p1[1],p1[2]))
    print('p2    = %10.3e + %.2e - %.2e'%(p2[0],p2[1],p2[2]))
    print('chi2  = %10.3f'%chi2)

    # compute the P(k) of the model
    Pk = model(kw,theta,kw,Pkw,knw,Pknw)

    return alpha,b,p0,p1,p2,chi2,Pk
###############################################################################
###############################################################################



###############################################################################
###############################################################################
# This function makes a fit to the data, inputed as (k_real,Pk_real,dPk_real)
# using a theoretical template
def fitting_theory(theta0,k_real,Pk_real,dPk_real,kw,Pkw,Pknw,beta,Radii,
                   nwalkers,chain_pts,f_contours):

    # compute number of parameters and degrees of freedom
    ndim = len(theta0);  ndof = len(Pk_real) - ndim

    ###### find best fit parameters and minimum chi2 ######
    chi2_func = lambda *args: -lnlike_theory(*args)
    best_fit  = minimize(chi2_func,theta0,
                         args=(k_real,Pk_real,dPk_real,
                               kw,Pkw,Pknw,beta,Radii),method='Powell')
    theta_best_fit = best_fit["x"]
    chi2 = chi2_func(theta_best_fit,k_real,Pk_real,dPk_real,
                     kw,Pkw,Pknw,beta,Radii)*1.0/ndof
    
    alpha, b, R, p0, p1, p2 = theta_best_fit
    print('alpha = %10.3e'%alpha)
    print('b     = %10.3e'%b)
    print('R     = %10.3e'%R)
    print('p0    = %10.3e'%p0)
    print('p1    = %10.3e'%p1)
    print('p2    = %10.3e'%p2)
    #print('p3    = %10.3e'%p3)
    print('chi2  = %10.3f\n'%chi2)
    #######################################################                 

    ###################### run MCMC #######################
    # starting point of each MCMC chain: small ball around best fit or estimate
    pos = [theta_best_fit*(1.0+0.1*np.random.randn(ndim)) for i in range(nwalkers)]

    # run MCMC
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlike_theory,
                                    args=(k_real,Pk_real,dPk_real,
                                          kw,Pkw,Pknw,beta,Radii))
    sampler.run_mcmc(pos,chain_pts);  del pos

    # get the points in the MCMC chains
    samples = sampler.chain[:,6000:,:].reshape((-1,ndim))
    sampler.reset()

    # make the plot and save it to file
    fig = corner.corner(samples,truths=theta0,labels=[r"$\alpha$",
                                                      r"$b$",r"$R$",
                                                      r"$p_0$",r"$p_1$",
                                                      r"$p_2$"])
    fig.savefig(f_contours)

    # compute best fit and associated error
    alpha, b, R, p0, p1, p2 =  map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                   zip(*np.percentile(samples, [16, 50, 84],
                                                      axis=0)))
    del samples, sampler
                                      
    # compute chi^2 of best fit
    theta_mcmc = [alpha[0],b[0],R[0],p0[0],p1[0],p2[0]]
    chi2 = chi2_func(theta_mcmc,k_real,Pk_real,dPk_real,kw,Pkw,Pknw,
                     beta,Radii)*1.0/ndof
                     

    # show results of MCMC
    print('alpha = %10.3e + %.2e - %.2e'%(alpha[0],alpha[1],alpha[2]))
    print('b     = %10.3e + %.2e - %.2e'%(b[0],b[1],b[2]))
    print('R     = %10.3e + %.2e - %.2e'%(R[0],R[1],R[2]))
    print('p0    = %10.3e + %.2e - %.2e'%(p0[0],p0[1],p0[2]))
    print('p1    = %10.3e + %.2e - %.2e'%(p1[0],p1[1],p1[2]))
    print('p2    = %10.3e + %.2e - %.2e'%(p2[0],p2[1],p2[2]))
    #print('p3    = %10.3e + %.2e - %.2e'%(p3[0],p3[1],p3[2]))
    print('chi2  = %10.3f'%chi2)

    # compute best fit model for data and save results to file
    Pk_model = model_theory(k_real,theta_mcmc,kw,Pkw,Pknw,beta,Radii)
    np.savetxt('Best_fit.txt',np.transpose([k_real,Pk_model]))

    # compute best fit model without the polynomial and save results to file
    theta_mcmc = [alpha[0], b[0], R[0], 0.0, 0.0, 0.0]
    Pk_model = model_theory(k_real,theta_mcmc,kw,Pkw,Pknw,beta,Radii)
    np.savetxt('Best_fit_zero_pol.txt',np.transpose([k_real,Pk_model]))

    return alpha,b,R,p0,p1,p2,chi2
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# This function computes P(k) in k values using as template the mean P(k)
# from the simulations
# P_model(k) = b*[(P_lin(k/a)-P_nw(k/a)) + P_nw(k)] + (p0*k + p1 + p2/k)
def model_sim_template(k,theta,kw,Pkw,knw,Pknw):
    alpha,b,p = theta[0],theta[1],theta[2:]
    Pnw  = np.interp(k,knw,Pknw)
    Pwa  = np.interp(k/alpha,kw,Pkw)
    Pnwa = np.interp(k/alpha,knw,Pknw)
    pol  = p[0]*k + p[1] + p[2]/k
    return b*((Pwa-Pnwa)+Pnw) + pol

# This functions returns the log-likelihood for the model with the sim template
def lnlike_sim_template(theta,k,Pk,dPk,kw,Pkw,knw,Pknw,beta):
    if (theta[0]<=0) or (theta[0]>5.0) or (theta[1]<=0):
        return -np.inf
    else:
        Pk_model = model(k,theta,kw,Pkw,knw,Pknw)
        return -np.sum(((Pk-Pk_model)/dPk)**2,dtype=np.float64)

# This function computes P(k) in k values using the theoretical template
# P_model(k) = b*[(P_w(k/a)-P_nw(k/a)) + P_nw(k)] + (p0*k + p1 + p2/k)
def model_theory(k,theta,kw,Pkw,Pknw,beta,Radii):

    alpha, bias, R, p = theta[0], theta[1], theta[2], theta[3:]        

    Rmin = Radii[0];  Rmax = Radii[-1];  R_bins = len(Radii)
    imin = int((R-Rmin)*(R_bins-1.0)/(Rmax-Rmin)) #index of closest array
    if imin==R_bins:      imin = R_bins-1
    if imin==(R_bins-1):  imax = imin
    else:                 imax = imin + 1

    if imin==imax:  
        print('error!!!!');  sys.exit()

    Rdown = (Rmax-Rmin)/(R_bins-1.0)*1.0*imin + Rmin
    Rup   = (Rmax-Rmin)/(R_bins-1.0)*1.0*imax + Rmin

    Pkw_model  = (Pkw[imax]-Pkw[imin])/(Rup-Rdown)*(R-Rdown) + Pkw[imin]
    Pknw_model = (Pknw[imax]-Pknw[imin])/(Rup-Rdown)*(R-Rdown) + Pknw[imin]

    Pnw  = np.interp(k,       kw, Pknw_model)
    Pwa  = np.interp(k/alpha, kw, Pkw_model)
    Pnwa = np.interp(k/alpha, kw, Pknw_model)

    return bias**2*((Pwa-Pnwa)+Pnw) + p[0]*k + p[1] + p[2]/k

# This functions returns the log-likelihood for the theory model
# theta -------> array with parameters to fit [alpha,bias,R,p0,p1,p2]
# k,Pk,dPk ----> data to fit
# kw ----------> array containing the values of k of the templates
# Pkw ---------> matrix with values of P_wiggles(k) for different R values
# Pknw --------> matrix with values of P_no_wiggles(k) for different R values
# beta --------> value of beta
def lnlike_theory(theta,k,Pk,dPk,kw,Pkw,Pknw,beta,Radii):
    # put priors here
    if (theta[0]<0.8) or (theta[0]>1.2) or (theta[1]<=0.0) \
            or (theta[2]>=Radii[-1]) or (theta[2]<=Radii[0]):
            #or (theta[6]<0.0):
        return -np.inf
    else:
        Pk_model = model_theory(k,theta,kw,Pkw,Pknw,beta,Radii)
        chi2 = -np.sum(((Pk-Pk_model)/dPk)**2,dtype=np.float64)
        return chi2

###############################################################################
###############################################################################


# This routine computes the mean and variance of the results
def mean_variance_Pk(f_3D_Pk,nuTable,nside,bin_min,bin_max,D,time_total,
                     realizations,Omega_m,Omega_L,h,root_fout,do_nw,do_fg):

    # compute Gaussian errors
    mean_z,k_g,dPk_g = Gaussian_1D_errors(f_3D_Pk,nuTable,bin_min,bin_max,
                                          Omega_m,Omega_L,h,D)

    # read the frequencies and redshifts of the maps
    num, nui, nuf, zi, zf = np.loadtxt(nuTable,unpack=True);  z = 0.5*(zi+zf)

    # compute the mean redshift of the survey
    mean_z = 0.5*(z[bin_min-1]+z[bin_max-1]);  print('survey <z> = %.3f'%mean_z)

    suffix = '%d-%d_z=%.2f_D=%.0f_t=%.0f_nside=%d.dat'\
        %(bin_min,bin_max,mean_z,D,time_total,nside)

    Pk_cosmo, Pk_noise, Pk_nw_cosmo, Pk_fg = [],[],[],[]
    for i in np.arange(1,1+realizations):
        
        f1 = root_fout+str(i)+'/Pk_cosmo_'+suffix
        f2 = root_fout+str(i)+'/Pk_noise_'+suffix
        if do_nw:  f3 = root_fout+'nw_'+str(i)+'/Pk_cosmo_'+suffix
        if do_fg:  f4 = root_fout+str(i)+'/Pk_fg_'+suffix

        k,Pk,dumb = np.loadtxt(f1,unpack=True);  Pk_cosmo.append(Pk)
        k,Pk,dumb = np.loadtxt(f2,unpack=True);  Pk_noise.append(Pk)
        if do_nw:
            k,Pk,dumb = np.loadtxt(f3,unpack=True);  Pk_nw_cosmo.append(Pk)
        if do_fg:  k,Pk,dumb = np.loadtxt(f4,unpack=True);  Pk_fg.append(Pk)
        
    Pk_cosmo = np.array(Pk_cosmo);        Pk_noise = np.array(Pk_noise)
    Pk_nw_cosmo = np.array(Pk_nw_cosmo);  Pk_fg = np.array(Pk_fg)

    f = open(root_fout+'mean_Pk_cosmo_'+suffix,'w')
    g = open(root_fout+'mean_Pk_noise_'+suffix,'w')
    if do_nw:  h = open(root_fout+'mean_Pk_nw_cosmo_'+suffix,'w')
    if do_fg:  l = open(root_fout+'mean_Pk_fg_'+suffix,'w')
    for i in range(len(k)):
        f.write(str(k[i])+' '+str(np.mean(Pk_cosmo[:,i]))+' '+\
                    str(np.std(Pk_cosmo[:,i]))+' '+str(dPk_g[i])+'\n')
        g.write(str(k[i])+' '+str(np.mean(Pk_noise[:,i]))+' '+\
                    str(np.std(Pk_noise[:,i]))+'\n')
        if do_nw:
            h.write(str(k[i])+' '+str(np.mean(Pk_nw_cosmo[:,i]))+' '+\
                        str(np.std(Pk_nw_cosmo[:,i]))+'\n')
        if do_fg:
            l.write(str(k[i])+' '+str(np.mean(Pk_fg[:,i]))+' '+\
                        str(np.std(Pk_fg[:,i]))+'\n')
    f.close();  g.close()
    if do_nw: h.close()
    if do_fg: l.close()    

###############################################################################
###############################################################################

# compute P_1D(k) = \int_0^oo k_perp P_3D(k_par,k_perp) e^(-k_perp^2R^2) dk_perp
def Pk_1D_integrand(y,x,k,Pk,k_par,beta,R):
    k_mod     = math.sqrt(k_par**2 + x**2)
    Pk_interp = np.interp(np.log(k_mod),np.log(k),Pk)
    aux       = math.exp(-(x*R)**2 - 4.0*k_mod**2)
    #aux       = math.exp(-(x*R)**2)
    return Pk_interp*x*(1.0+beta*(k_par/k_mod)**2)**2*aux

# This function computes the 1D P(k) within [kmin,kmax] from the 3D one 
# taking into account the value of beta and R
def Pk_1D_model(kmin,kmax,bins_1D,k,Pk,beta,R):
    k_1D  = np.logspace(np.log10(kmin),np.log10(kmax),bins_1D)
    Pk_1D = np.zeros(bins_1D,dtype=np.float64)
    for i,k_par in enumerate(k_1D):
        yinit = [0.0];  k_limits = [0.0,np.sqrt(k[-1]**2-k_par**2)]
        Pk_1D[i] = si.odeint(Pk_1D_integrand,yinit,k_limits,
                             args=(k,Pk,k_par,beta,R),
                             mxstep=100000,rtol=1e-10,atol=1e-12,
                             h0=1e-10)[1][0]/(2*np.pi)
    return [k_1D,Pk_1D]


# This is the main function used to fit the results of the simulations to the
# theoretical template
def fit_function(nuTable,bin_min,bin_max,model,kmin,kmax,k_bins,R_bins,
                 D,time_total,nside,Omega_m,Omega_L,h,f_3D_Pkw,f_3D_Pknw,
                 nwalkers,chain_pts,realizations):
                 

    # read frequencies and redshifts of the maps and find z_min,z_max and <z>
    num, nui, nuf, zi, zf = np.loadtxt(nuTable,unpack=True);  z = 0.5*(zi+zf)
    z_min = z[bin_max-1];  z_max = z[bin_min-1];  mean_z = 0.5*(z_min + z_max)

    # file to save results
    suffix = '%d-%d_z=%.2f_D=%.0f_t=%.0f_nside=%d.dat'\
        %(bin_min,bin_max,mean_z,D,time_total,nside)
    fout_partial = 'MCMC_'+model+'_'+suffix+'_'+str(myrank)

    # compute the value of the HI bias and 21cm bias
    HI_bias = HI_bias_model(mean_z)
    b_21cm  = bias_21cm_model(mean_z,Omega_m,Omega_L,h)

    # compute redshift-space distortion parameter beta
    Omega_mz = Omega_m*(1.0+mean_z)**3/(Omega_m*(1.0+mean_z)**3+Omega_L)
    beta     = (Omega_mz**0.54545454)/HI_bias

    # compute maximum and minimum comoving angular smoothings: use fwhm factor
    fwhm = 2.0*np.sqrt(2.0*np.log(2.0))  #fwhm = sigma*2*sqrt(2*log(2))
    Rmin = (0.21*(1.0+z_min)/D/fwhm)*CL.comoving_distance(z_min,Omega_m,Omega_L)
    Rmax = (0.21*(1.0+z_max)/D/fwhm)*CL.comoving_distance(z_max,Omega_m,Omega_L)

    # verbose
    if myrank==0:
        print('Expected bias = %.3f'%b_21cm)
        print('angular smoothing within [%.2f,%.2f] Mpc/h'%(Rmin,Rmax))
        print('mean angular smoothing = %.2f Mpc/h'%(0.5*(Rmin+Rmax)))

    # compute/read Pkw and Pknw matrices
    f_data = 'data_kmin=%.3f_kmax=%.3f_kbins=%d_Rbins=%d_'\
        %(kmin,kmax,k_bins,R_bins)+suffix
    if os.path.exists(f_data+'.npy'):
        kw,Radii,Pkw,Pknw = np.load(f_data+'.npy')
    else:
        # read the CAMB 3D P(k) with and without wiggles
        k3Dw, Pk3Dw  = np.loadtxt(f_3D_Pkw, unpack=True)
        k3Dnw,Pk3Dnw = np.loadtxt(f_3D_Pknw,unpack=True)

        Pkw    = np.zeros((R_bins,k_bins),dtype=np.float64)
        Pknw   = np.zeros((R_bins,k_bins),dtype=np.float64)
        Pkw_d  = np.zeros((R_bins,k_bins),dtype=np.float64)
        Pknw_d = np.zeros((R_bins,k_bins),dtype=np.float64)
        Radii  = np.linspace(Rmin,Rmax,R_bins)
        bins   = np.where((np.arange(R_bins)%nprocs)==myrank)[0]
        for i in bins:
            print(i)
            kw,Pkw_d[i]  = \
                Pk_1D_model(kmin,kmax,k_bins,k3Dw,Pk3Dw,beta,Radii[i])
            kw,Pknw_d[i] = \
                Pk_1D_model(kmin,kmax,k_bins,k3Dnw,Pk3Dnw,beta,Radii[i])
        comm.Barrier()
        for i in range(R_bins):
            comm.Reduce(Pkw_d[i],  Pkw[i],  op=MPI.SUM, root=0)   
            comm.Reduce(Pknw_d[i], Pknw[i], op=MPI.SUM, root=0)
        del Pkw_d,Pknw_d,k3Dw,Pk3Dw,k3Dnw,Pk3Dnw;  
        if myrank==0:  np.save(f_data,[kw,Radii,Pkw,Pknw])
        Pkw = comm.Bcast(Pkw,root=0);  Pknw = comm.Bcast(Pknw,root=0)
        
        
    # these are the realizations each cpu work on
    numbers = np.where((np.arange(realizations)%nprocs)==myrank)[0]+1

    # do a loop over all realizations
    f = open(fout_partial,'w')
    for i in numbers:

        print('Working with realization',i)

        # Power spectrum of the realization and errors
        folder           = 'results/'+str(i)
        f_Pk_errors      = 'results/mean_Pk_'+model+'_'+suffix
        f_Pk_realization = folder+'/Pk_'+model+'_'+suffix
        f_contours       = 'results/contours_'+model+' '+suffix[:-3]+'png'

        # read 1D P(k) of the realization and error
        k_real,Pk_real,dumb = np.loadtxt(f_Pk_realization,unpack=True)
        data                = np.loadtxt(f_Pk_errors,unpack=False)
        dPk_real            = data[:,2]

        # keep only data in k-range: for the fit we use k_real,Pk_real,dPk_real
        indexes  = np.where((k_real>kmin) & (k_real<kmax))[0]
        k_real   = k_real[indexes];  Pk_real = Pk_real[indexes]
        dPk_real = dPk_real[indexes];  del indexes

        # initial guess for the fit:  theta0 = [alpha, bias, R, p0, p1, p2]
        theta0 = [1.0, b_21cm, 0.5*(Rmin+Rmax), 0.0, 0.0, 0.0]

        # do the fitting and compute error on parameters
        data = fitting_theory(theta0,k_real,Pk_real,dPk_real,kw,Pkw,Pknw,
                              beta,Radii,nwalkers,chain_pts,f_contours)
        gc.collect()
                                  
        alpha,b,R,p0,p1,p2,chi2 = data
        del k_real,Pk_real,dPk_real,dumb

        # save results to file
        f.write(str(i)+' '+\
                str(alpha[0])+' '+str(alpha[1])+' '+str(alpha[2])+' '+\
                str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+\
                str(R[0])+' '+str(R[1])+' '+str(R[2])+' '+\
                str(p0[0])+' '+str(p0[1])+' '+str(p0[2])+' '+\
                str(p1[0])+' '+str(p1[1])+' '+str(p1[2])+' '+\
                str(p2[0])+' '+str(p2[1])+' '+str(p2[2])+' '+\
                #str(p3[0])+' '+str(p3[1])+' '+str(p3[2])+' '+\
                str(chi2)+'\n')  
    f.close()

    # synchronize all processes here
    comm.Barrier()

    # merge the results from partial files
    if myrank==0:

        # concatenate all files and remove them
        fout = 'MCMC_'+model+'_'+suffix;  string = ''
        for i in range(nprocs):
            string += (fout+'_'+str(i)+' ')
        os.system('cat '+string+' > '+fout);  os.system('rm '+string)
    
        # read data from concatenated file, sort it and rewrite it
        f = open(fout,'r'); data = []
        for line in f.readlines():  
            data.append(line.split())
        f.close();  data = sorted(data,key=lambda x:int(x[0]))
        np.savetxt(fout,data,fmt='%s')

    # synchronize all processes here
    comm.Barrier()
###############################################################################
###############################################################################

# I = \int_0^oo dk_perp k_perp P^2(k_par,k_perp)
def Pk_1D_error(y,x,k,Pk,k_par,b_21,beta,R_smooth):
    k_mod = np.sqrt(x**2 + k_par**2)
    Pk_kpar_kperp = b_21**2*(1.0+beta*(k_par/k_mod)**2)**2*np.interp(k_mod,k,Pk)
    W = np.exp(-x**2*R_smooth**2/2.0)
    return x*(W**2*Pk_kpar_kperp)**2

# This routine computes the Gaussian errors on the 1D power spectrum 
def Gaussian_1D_errors(f_3D_Pk,nuTable,bin_min,bin_max,Omega_m,Omega_L,h,D):

    # read the 3D P(k) 
    k_3D,Pk_3D = np.loadtxt(f_3D_Pk,unpack=True)

    # read the frequencies and redshifts of the maps
    num, nui, nuf, zi, zf = np.loadtxt(nuTable,unpack=True);  z = 0.5*(zi+zf)

    # compute the mean redshift of the survey
    mean_z = 0.5*(z[bin_min-1]+z[bin_max-1]);  print('survey <z> = %.2f'%mean_z)

    # compute the comoving distance to the survey mean redshift
    d_co = CL.comoving_distance(mean_z,Omega_m,Omega_L)  #Mpc/h
    print('comoving distance to <z> = %.2f -------> %.2f Mpc/h'%(mean_z,d_co))
    
    # compute the HI bias and Tb at <z>
    Omega_HI = Omega_HI_model(mean_z)
    b_HI     = HI_bias_model(mean_z)                     #HI bias
    b_21     = bias_21cm_model(mean_z,Omega_m,Omega_L,h) #21cm bias

    # compute instrument angular resolution at <z> and smoothing radius 
    theta    = 0.21*(1.0+mean_z)/D/2.355
    R_smooth = d_co*theta
    print('Smoothing angular scale at <z> = %.2f = %.2f Mpc/h'%(mean_z,R_smooth))

    # compute the redshift-space distortion parameters beta
    Omega_mz = Omega_m*(1.0+mean_z)**3/(Omega_m*(1.0+mean_z)**3+Omega_L)
    beta     = Omega_mz**0.54545454/b_HI

    # compute maximum and minimum comoving distance and survey volune
    d_max = CL.comoving_distance(z[bin_min-1],Omega_m,Omega_L) #Mpc/h
    d_min = CL.comoving_distance(z[bin_max-1],Omega_m,Omega_L) #Mpc/h
    V     = 4.0/3.0*np.pi*(d_max**3-d_min**3)                  #(Mpc/h)^3

    # define the k-bins
    dims    = bin_max - bin_min + 1
    BoxSize = d_max-d_min  #Mpc/h
    k_bins  = np.linspace(0,dims//2,dims//2+1)*(2.0*np.pi/BoxSize)

    # compute the values of k and r for the 1D P(k) and xi(r)
    modes   = np.arange(dims,dtype=np.float64);  middle = dims//2
    indexes = np.where(modes>middle)[0];  modes[indexes] = modes[indexes]-dims
    k = modes*(2.0*np.pi/BoxSize)  #k in h/Mpc
    k = np.absolute(k);  #just take the modulus
    del indexes, modes

    # compute the number of modes and the average number-weighted value of k,r
    k_modes = np.histogram(k,bins=k_bins)[0]
    k_bin   = np.histogram(k,bins=k_bins,weights=k)[0]/k_modes

    # for each k_par make David's integral
    dPk = np.zeros(len(k_bin),dtype=np.float32)
    for i,k_par in enumerate(k_bin):
        k_perp_max = np.sqrt(k_3D[-1]**2-k_par**2)
        yinit = [0.0];  k_limits = [0, k_perp_max]
        delta_k = k_bins[i+1]-k_bins[i]
        dPk[i] = si.odeint(Pk_1D_error,yinit,k_limits,
                           args=(k_3D,Pk_3D,k_par,b_21,beta,R_smooth),
                           mxstep=10000,rtol=1e-12,atol=1e-15,
                           h0=1e-15)[1][0]/(delta_k*V)
        dPk[i] = np.sqrt(dPk[i])

    # avoid fundamental frequency
    k_bin = k_bin[1:];  dPk = dPk[1:]
    
    return [mean_z,k_bin,dPk]
###############################################################################
###############################################################################
