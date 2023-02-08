import numpy as np 
import time,sys,os,h5py,hdf5plugin
import readsnapHDF5 as rs
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,log10,abs,exp,log,rint
import readsnap#, groupcat
import units_library as UL
import MAS_library as MASL
import units_library as UL


################################# UNITS #####################################
cdef double rho_crit = (UL.units()).rho_crit 

cdef double yr    = 3.15576e7   #seconds
cdef double km    = 1e5         #cm
cdef double Mpc   = 3.0856e24   #cm
cdef double kpc   = 3.0856e21   #cm
cdef double Msun  = 1.989e33    #g
cdef double Ymass = 0.24        #helium mass fraction
cdef double mH    = 1.6726e-24  #proton mass in grams
cdef double gamma = 5.0/3.0     #ideal gas
cdef double kB    = 1.3806e-26  #gr (km/s)^2 K^{-1}
cdef double nu0   = 1420.0      #21-cm frequency in MHz
cdef double muH   = 2.3e-24     #gr
cdef double pi    = np.pi 
#############################################################################

# This routine computes the self-shielding parameters of Rahmati et al.
# and the amplitude of the UV background at a given redshift together with
def Rahmati_parameters(redshift, TREECOOL_file, Gamma_UVB=None, fac=1.0,
                       verbose=False):

    # Rahmati et. al. 2013 self-shielding parameters (table A1)
    z_t       = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00])
    n0_t      = np.array([-2.94,-2.29,-2.06,-2.13,-2.23,-2.35]); n0_t=10**n0_t
    alpha_1_t = np.array([-3.98,-2.94,-2.22,-1.99,-2.05,-2.63])
    alpha_2_t = np.array([-1.09,-0.90,-1.09,-0.88,-0.75,-0.57])
    beta_t    = np.array([1.29, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_t       = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])

    # compute the self-shielding parameters at the redshift of the N-body
    n0      = np.interp(redshift, z_t, n0_t)
    alpha_1 = np.interp(redshift, z_t, alpha_1_t)
    alpha_2 = np.interp(redshift, z_t, alpha_2_t)
    beta    = np.interp(redshift, z_t, beta_t)
    f       = np.interp(redshift, z_t, f_t)
    if verbose:
        print('n0 = %e\nalpha_1 = %2.3f\nalpha_2 = %2.3f\nbeta = %2.3f\n'\
            %(n0,alpha_1,alpha_2,beta) + 'f = %2.3f'%f)
        
    # find the value of the photoionization rate
    if Gamma_UVB is None:
        data = np.loadtxt(TREECOOL_file); logz=data[:,0]; Gamma_UVB=data[:,1]
        Gamma_UVB=np.interp(np.log10(1.0+redshift),logz,Gamma_UVB); del data
        if verbose:  print('Gamma_UVB(z=%2.2f) = %e s^{-1}'%(redshift,Gamma_UVB))
                     
    # Correct to reproduce the Lya forest mean flux
    Gamma_UVB /= fac

    return n0, alpha_1, alpha_2, beta, f, Gamma_UVB




# This routine computes the HI/H fraction of star-forming particles in the 
# Illustris-TNG simulation
# rho ---------------> star-forming part. densities in h^2 Msun/Mpc^3 (comoving)
# SPH ---------------> star-forming part radii in Mpc/h (comoving)
# metals ------------> star-forming part metallicities in solar units
# TREECOOL_file -----> file containing the UV strength
# Gamma -------------> value of the UVB if TREECOOL_file not provided
# fac ---------------> correction to the UVB to reproduce <F> of the Lya forest
# correct_H2 --------> whether to correct for H2
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def Rahmati_HI_Illustris(np.float32_t[:] rho, np.float32_t[:] SPH, 
                         np.float32_t[:] metals, redshift, h,
                         TREECOOL_file, Gamma=None, fac=1.0, 
                         correct_H2=False, verbose=False):
                         
    cdef long i, particles
    cdef float prefact, prefact2, nH, f_H2
    cdef double n0, alpha_1, alpha_2, beta, f, Gamma_UVB, chi, tau_c
    cdef double T, Lambda, alpha_A, Lambda_T, Gamma_phot, A, B, C, s
    cdef double sigma
    cdef np.float32_t[:] nH0

    # find the values of the self-shielding parameters and the UV background
    n0, alpha_1, alpha_2, beta, f, Gamma_UVB = \
        Rahmati_parameters(redshift, TREECOOL_file, Gamma, fac, verbose)
                      
    # find the number of star-forming particles and set their temperatures
    particles = rho.shape[0]
    T         = 1e4 #K

    # compute the case A recombination rate
    Lambda  = 315614.0/T
    alpha_A = 1.269e-13*pow(Lambda,1.503)\
        /pow(1.0 + pow((Lambda/0.522),0.47),1.923) #cm^3/s

    # Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
    Lambda_T = 1.17e-10*sqrt(T)*exp(-157809.0/T)/(1.0+sqrt(T/1e5)) #cm^3/s

    # prefactor to change densities from h^2 Msun/Mpc^3 to atoms/cm^{-3}
    prefact  = 0.76*h**2*Msun/Mpc**3/mH*(1.0+redshift)**3                   

    # prefactor to change surface densities from h Msun/Mpc^2 to g/cm^2
    prefact2 = (1.0+redshift)**2*h*Msun/Mpc**2 

    # define HI/H fraction
    nH0 = np.zeros(particles, dtype=np.float32)

    # do a loop over all particles
    for i in range(particles):

        # compute particle density in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
        nH = rho[i]*prefact
        
        # compute the photoionization rate
        Gamma_phot = Gamma_UVB
        Gamma_phot *= ((1.0 - f)*pow(1.0 + pow(nH/n0,beta), alpha_1) +\
                           f*pow(1.0 + nH/n0, alpha_2))

        # compute the coeficients A,B and C to calculate the HI/H fraction
        A = alpha_A + Lambda_T
        B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
        C = alpha_A

        # compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
        nH0[i] = (B-sqrt(B*B-4.0*A*C))/(2.0*A)

        ##### correct for H2 using KMT #####
        if correct_H2:
            chi   = 0.756*(1.0 + 3.1*metals[i]**0.365)           #dimensionless
            sigma = rho[i]*SPH[i]*prefact2                       #g/cm^2
            tau_c = sigma*(metals[i]*1e-21)/muH                  #dimensionless 
            s     = log(1.0 + 0.6*chi + 0.01*chi**2)/(0.6*tau_c) #dimensionless
            if s<2.0:  f_H2 = 1.0 - 0.75*s/(1.0 + 0.25*s)
            else:      f_H2 = 0.0
            if f_H2<0.0:  f_H2 = 0.0
            nH0[i] = nH0[i]*(1.0-f_H2)
            
    return np.asarray(nH0)

#########################################################################    
# This routine reads a single Illustris snapshot subfile and returns 
# particle positions and HI masses
cpdef HI_mass_from_Illustris_snap(snapshot, TREECOOL_file):

    f = h5py.File(snapshot, 'r')

    # read redshift and h
    redshift = f['Header'].attrs[u'Redshift']
    h        = f['Header'].attrs[u'HubbleParam']

    # read pos, radii, densities, HI/H and masses of gas particles 
    pos  = (f['PartType0/Coordinates'][:]/1e3).astype(np.float32)
    MHI  = f['PartType0/NeutralHydrogenAbundance'][:]
    mass = f['PartType0/Masses'][:]*1e10  #Msun/h
    rho  = f['PartType0/Density'][:]*1e19 #(Msun/h)/(Mpc/h)^3
    SFR  = f['PartType0/StarFormationRate'][:]
    indexes = np.where(SFR>0.0)[0];  del SFR
            
    # find the metallicity of star-forming particles
    metals = f['PartType0/GFM_Metallicity'][:]
    metals = metals[indexes]/0.0127

    # find densities of star-forming particles: units of h^2 Msun/Mpc^3
    Volume = mass/rho                            #(Mpc/h)^3
    radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
    rho    = rho[indexes]                        #h^2 Msun/Mpc^3
    Volume = Volume[indexes]                     #(Mpc/h)^3

    # find volume and radius of star-forming particles
    radii_SFR  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 
        
    # find HI/H fraction for star-forming particles
    MHI[indexes] = Rahmati_HI_Illustris(rho, radii_SFR, metals, 
                                        redshift, h, TREECOOL_file, 
                                        Gamma=None, fac=1, 
                                        correct_H2=True) #HI/H
    MHI *= (0.76*mass)
    f.close()
    
    return pos, MHI

#########################################################################    
"""
# This routine identifies the offsets for halos and galaxies
# and perform several checks
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef test(f_offset, snapshot_root, num):

    cdef int i, j, Number
    cdef long offset, offset_halo, particles
    cdef np.int64_t[:] offset_galaxies, 
    cdef np.int64_t[:] end_halos, end_all_galaxies
    cdef np.int64_t[:] offset_halos, offset_subhalos
    cdef np.int32_t[:] lens_halos, subhalos_num, lens_subhalos

    # read halos and subhalos offset
    f = h5py.File(f_offset, "r")
    offset_halos    = f['Group/SnapByType'][:,0]
    offset_subhalos = f['Subhalo/SnapByType'][:,0]
    f.close()

    # read number of particles in halos and subhalos and number of subhalos
    halos = groupcat.loadHalos(snapshot_root, num, 
                               fields=['GroupLenType','GroupNsubs'])
    lens_halos   = halos['GroupLenType'][:,0]  
    subhalos_num = halos['GroupNsubs']
    subhalos = groupcat.loadSubhalos(snapshot_root, num, 
                                     fields=['SubhaloLenType','SubhaloMass'])
    lens_subhalos = subhalos['SubhaloLenType'][:,0]

    # define the array containing the beginning and ends of the halos
    end_halos = np.zeros(lens_halos.shape[0], dtype=np.int64)

    # find the offsets of the halos
    particles = 0
    for i in range(lens_halos.shape[0]):
        if offset_halos[i]!=particles:
            raise Exception('Offset are wrong!!')
        particles += lens_halos[i]
        end_halos[i] = particles
    del offset_halos


    # define the array hosting the offset of galaxies
    end_all_galaxies = np.zeros(lens_halos.shape[0], dtype=np.int64)
    offset_galaxies  = np.zeros(lens_subhalos.shape[0], dtype=np.int64)

    # do a loop over all halos
    Number, offset, offset_halo = 0, 0, 0
    for i in range(lens_halos.shape[0]):

        offset = offset_halo
        end_all_galaxies[i] = offset

        # do a loop over all galaxies belonging to the halo
        for j in range(subhalos_num[i]):
            offset_galaxies[Number] = offset
            end_all_galaxies[i] += lens_subhalos[Number]
            offset += lens_subhalos[Number];  Number += 1

        if (end_all_galaxies[i]-offset_halo)>lens_halos[i]:
            raise Exception('More particles in galaxies than in halo!!!')

        offset_halo += lens_halos[i]

    # check that galaxy offsets are the same
    for i in range(offset_galaxies.shape[0]):
        if offset_galaxies[i]!=offset_subhalos[i]:
            raise Exception('Offset of subhalos are different!!!')

    return np.asarray(end_halos),np.asarray(end_all_galaxies)
"""
#########################################################################    

# This routine computes the HI 
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef M_HI_counter(np.float32_t[:] NHI, np.float32_t[:] mass, 
                   np.float64_t[:] M_HI, np.float64_t[:] M_HI_gal, 
                   np.int64_t[:] end_halos, np.int64_t[:] end_all_galaxies, 
                   long Number, long start, long end, long end_gal,
                   long halo_num, done):

    cdef long j, particles, num_halos
    
    # find the number of particles to iterate over
    particles = NHI.shape[0]
    num_halos = M_HI.shape[0]

    # do a loop over all particles
    for j in range(particles):
        if Number>end:
            halo_num += 1
            if halo_num<num_halos:  
                start   = end
                end     = end_halos[halo_num]
                end_gal = end_all_galaxies[halo_num]
            else:
                done = True;  break

        # if particle is in galaxy add to M_HI_gal
        if Number>=start and Number<end_gal:
            M_HI_gal[halo_num] += 0.76*NHI[j]*mass[j]
                    
        M_HI[halo_num] += 0.76*NHI[j]*mass[j]
        Number += 1

    return Number, start, end, end_gal, halo_num, done
#######################################################################

#######################################################################
# This routine computes the HI mass within halos and the HI mass within
# halos that it is in galaxies
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef M_HI_halos_galaxies(np.int64_t[:] pars, done,
                          np.int32_t[:] halo_len, np.int32_t[:] gal_len,
                          np.int32_t[:] gal_in_halo,
                          np.float32_t[:] mass_ratio, np.float32_t[:] MHI,
                          np.float64_t[:] M_HI, np.float64_t[:] M_HI_gal):

    cdef long Number            = pars[0]
    cdef long start_h           = pars[1]
    cdef long end_h             = pars[2]
    cdef long start_g           = pars[3]
    cdef long end_g             = pars[4]
    cdef long halo_num          = pars[5]
    cdef long gal_num           = pars[6]
    cdef long gal_in_halo_local = pars[7]
    cdef long i, particles, num_halos, num_galaxies
    
    # find the number of particles to iterate over
    particles    = MHI.shape[0]
    num_halos    = M_HI.shape[0]
    num_galaxies = gal_len.shape[0]

    # do a loop over all particles
    for i in range(particles):

        # if particle belongs to new halo change variables
        if Number>=end_h:

            # accout for missing galaxies (should be galaxies with 0 particles)
            if Number>=end_g and gal_in_halo_local<gal_in_halo[halo_num]-1:
                gal_num += 1
                gal_in_halo_local += 1
                while gal_len[gal_num]==0 and \
                        gal_in_halo_local<gal_in_halo[halo_num]-1:
                    gal_num += 1
                    gal_in_halo_local += 1
            
            #if gal_num!=np.sum(gal_in_halo[:halo_num+1], dtype=np.int64)-1:
            #    print 'Numbers differ!!!!'
            #    print halo_num
            #    print gal_num
            #    print np.sum(gal_in_halo[:halo_num+1], dtype=np.int64)
                
            # update halo variables
            halo_num += 1

            if halo_num>=num_halos:
                # check that all galaxies have been counted
                if gal_num!=np.sum(gal_in_halo, dtype=np.int64)-1:
                    print('gal_num  = %ld'%gal_num)
                    print('galaxies = %ld'%(np.sum(gal_in_halo, dtype=np.int64)-1))
                    raise Exception("Finished without counting all galaxies")
                done = True;  break

            #if halo_num<num_halos:
                #start_h = end_h
                #end_h   = end_h + halo_len[halo_num]
            while halo_len[halo_num]==0 and halo_num<num_halos:
                gal_num  += gal_in_halo[halo_num]
                halo_num += 1
            start_h = end_h
            end_h   = end_h + halo_len[halo_num]

            # restart galaxy variables
            if gal_num<num_galaxies and gal_in_halo[halo_num]>0:
                gal_in_halo_local = 0
                gal_num += 1
                start_g = start_h
                end_g   = start_h + gal_len[gal_num]

        # if particle belongs to new galaxy change variables
        if Number>=end_g and gal_in_halo_local<gal_in_halo[halo_num]-1:
            gal_num += 1
            gal_in_halo_local += 1
            while gal_len[gal_num]==0 and \
                    gal_in_halo_local<gal_in_halo[halo_num]-1:
                gal_num += 1
                gal_in_halo_local += 1
            start_g = end_g
            end_g   = end_g + gal_len[gal_num]

        # if particle is inside galaxy add it to M_HI_gal
        if Number>=start_g and Number<end_g and mass_ratio[gal_num]>0.1:
            M_HI_gal[halo_num] += MHI[i]

        # for halos always add the HI mass
        M_HI[halo_num] += MHI[i]
        Number += 1

    pars[0], pars[1], pars[2]          = Number, start_h, end_h 
    pars[3], pars[4], pars[5], pars[6] = start_g, end_g, halo_num, gal_num
    pars[7]                            = gal_in_halo_local

    return done

#######################################################################
# This routine computes the HI mass within halos and the HI mass within
# halos that it is in galaxies
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef M_HI_halos_gal_cen_sat(np.int64_t[:] pars, done,
                             np.int32_t[:] halo_len, np.int32_t[:] gal_len,
                             np.int32_t[:] gal_in_halo,
                             np.float32_t[:] mass_ratio, np.float32_t[:] MHI,
                             np.float64_t[:] M_HI, 
                             np.float64_t[:] M_HI_gal,
                             np.float64_t[:] M_HI_cen, 
                             np.float64_t[:] M_HI_sat):

    cdef long Number            = pars[0]
    cdef long start_h           = pars[1]
    cdef long end_h             = pars[2]
    cdef long start_g           = pars[3]
    cdef long end_g             = pars[4]
    cdef long halo_num          = pars[5]
    cdef long gal_num           = pars[6]
    cdef long gal_in_halo_local = pars[7]
    cdef long i, particles, num_halos, num_galaxies
    
    # find the number of particles to iterate over
    particles    = MHI.shape[0]
    num_halos    = M_HI.shape[0]
    num_galaxies = gal_len.shape[0]

    # do a loop over all particles
    for i in range(particles):

        # if particle belongs to new halo change variables
        if Number>=end_h:

            # accout for missing galaxies (should be galaxies with 0 particles)
            if Number>=end_g and gal_in_halo_local<gal_in_halo[halo_num]-1:
                gal_num += 1
                gal_in_halo_local += 1
                while gal_len[gal_num]==0 and \
                        gal_in_halo_local<gal_in_halo[halo_num]-1:
                    gal_num += 1
                    gal_in_halo_local += 1
            
            #if gal_num!=np.sum(gal_in_halo[:halo_num+1], dtype=np.int64)-1:
            #    print 'Numbers differ!!!!'
            #    print halo_num
            #    print gal_num
            #    print np.sum(gal_in_halo[:halo_num+1], dtype=np.int64)
                
            # update halo variables
            halo_num += 1

            if halo_num>=num_halos:
                # check that all galaxies have been counted
                if gal_num!=np.sum(gal_in_halo, dtype=np.int64)-1:
                    print('gal_num  = %ld'%gal_num)
                    print('galaxies = %ld'%(np.sum(gal_in_halo, dtype=np.int64)-1))
                    raise Exception("Finished without counting all galaxies")
                done = True;  break

            #if halo_num<num_halos:
                #start_h = end_h
                #end_h   = end_h + halo_len[halo_num]
            while halo_len[halo_num]==0 and halo_num<num_halos:
                gal_num  += gal_in_halo[halo_num]
                halo_num += 1
            start_h = end_h
            end_h   = end_h + halo_len[halo_num]

            # restart galaxy variables
            if gal_num<num_galaxies and gal_in_halo[halo_num]>0:
                gal_in_halo_local = 0
                gal_num += 1
                start_g = start_h
                end_g   = start_h + gal_len[gal_num]

        # if particle belongs to new galaxy change variables
        if Number>=end_g and gal_in_halo_local<gal_in_halo[halo_num]-1:
            gal_num += 1
            gal_in_halo_local += 1
            while gal_len[gal_num]==0 and \
                    gal_in_halo_local<gal_in_halo[halo_num]-1:
                gal_num += 1
                gal_in_halo_local += 1
            start_g = end_g
            end_g   = end_g + gal_len[gal_num]

        # if particle is inside galaxy add it to M_HI_gal
        if Number>=start_g and Number<end_g and mass_ratio[gal_num]>0.1:
            M_HI_gal[halo_num] += MHI[i]
            if gal_in_halo_local==0:  M_HI_cen[halo_num] += MHI[i]
            else:                     M_HI_sat[halo_num] += MHI[i]

        # for halos always add the HI mass
        M_HI[halo_num] += MHI[i]
        Number += 1

    pars[0], pars[1], pars[2]          = Number, start_h, end_h 
    pars[3], pars[4], pars[5], pars[6] = start_g, end_g, halo_num, gal_num
    pars[7]                            = gal_in_halo_local

    return done

#######################################################################
# This routine computes the peculiar velocity of HI within halos
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef V_HI_halos(long[:] pars, done, int[:] halo_len, float[:,:] halo_vel,
                 float[:] MHI, float[:,:] vel,
                 double[:,::1] V_HI, double[:] M_HI):

    cdef long Number            = pars[0]
    cdef long start_h           = pars[1]
    cdef long end_h             = pars[2]
    cdef long halo_num          = pars[3]
    cdef long i, particles, num_halos
    cdef float dVx, dVy, dVz
    
    # find the number of particles to iterate over
    particles    = MHI.shape[0]
    num_halos    = M_HI.shape[0]

    # do a loop over all particles
    for i in range(particles):

        # if particle belongs to new halo change variables
        if Number>=end_h:
                
            # update halo variables
            halo_num += 1

            if halo_num>=num_halos:
                done = True;  break

            while halo_len[halo_num]==0 and halo_num<num_halos:
                halo_num += 1
            start_h = end_h
            end_h   = end_h + halo_len[halo_num]

        # for halos always add the HI mass
        M_HI[halo_num] += MHI[i]

        # compute (Vi-V_halo)^2*M_HIi
        V_HI[halo_num,0] += vel[i,0]*MHI[i]
        V_HI[halo_num,1] += vel[i,1]*MHI[i]
        V_HI[halo_num,2] += vel[i,2]*MHI[i]
        Number += 1

    pars[0], pars[1], pars[2], pars[3] = Number, start_h, end_h, halo_num
    return done
    
#######################################################################
# This routine computes the HI mass within halos and the HI velocity 
# dispersion within halos
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef sigma_HI_halos(long[:] pars, done,
                     int[:] halo_len, float[:,:] halo_vel,
                     float[:] MHI, float[:,:] vel,
                     double[:] sigma2_HI, double[:] M_HI):

    cdef long Number            = pars[0]
    cdef long start_h           = pars[1]
    cdef long end_h             = pars[2]
    cdef long halo_num          = pars[3]
    cdef long i, particles, num_halos
    cdef float dVx, dVy, dVz
    cdef double sigma2
    
    # find the number of particles to iterate over
    particles    = MHI.shape[0]
    num_halos    = M_HI.shape[0]

    # do a loop over all particles
    for i in range(particles):

        # if particle belongs to new halo change variables
        if Number>=end_h:
                
            # update halo variables
            halo_num += 1

            if halo_num>=num_halos:
                done = True;  break

            while halo_len[halo_num]==0 and halo_num<num_halos:
                halo_num += 1
            start_h = end_h
            end_h   = end_h + halo_len[halo_num]

        # for halos always add the HI mass
        M_HI[halo_num] += MHI[i]

        # compute (Vi-V_halo)^2*M_HIi
        dVx = vel[i,0]-halo_vel[halo_num,0]
        dVy = vel[i,1]-halo_vel[halo_num,1]
        dVz = vel[i,2]-halo_vel[halo_num,2]
        sigma2 = dVx*dVx + dVy*dVy + dVz*dVz
        sigma2_HI[halo_num] += (sigma2*MHI[i])

        Number += 1

    pars[0], pars[1], pars[2], pars[3] = Number, start_h, end_h, halo_num
    return done

###############################################################
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef HI_image(np.float32_t[:,:] pos, float x_min, float x_max, 
               float y_min, float y_max, float z_min, float z_max):

    start = time.time()
    cdef long i, particles
    cdef list indexes_list

    indexes_list = []
    particles = pos.shape[0]
    for i in range(particles):

        if pos[i,0]>=x_min and pos[i,0]<=x_max:
            if pos[i,1]>=y_min and pos[i,1]<=y_max:
                if pos[i,2]>=z_min and pos[i,2]<=z_max:
                    indexes_list.append(i)
                    
    print('Time taken = %.2f seconds'%(time.time()-start))
    return indexes_list


# This routine implements reads Gadget gas output and correct HI/H fractions
# to account for self-shielding and H2
# snapshot_fname -------> name of the N-body snapshot
# fac ------------------> factor to reproduce the mean Lya flux
# TREECOOL_file --------> TREECOOL file used in the N-body
# Gamma_UVB ------------> value of the UVB photoionization rate
# correct_H2 -----------> correct the HI/H fraction to account for H2
# if Gamma_UVB is set to None the value of the photoionization rate will be read
# from the TREECOOL file, otherwise it is used the Gamma_UVB value
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def Rahmati_HI_assignment(snapshot_fname, fac, TREECOOL_file, Gamma_UVB=None,
                          correct_H2=False, IDs=None, verbose=False):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    print('\nREADING SNAPSHOT PROPERTIES')
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Nall     = head.nall
    redshift = head.redshift

    # find the values of the self-shielding parameters and the UV background
    n0, alpha_1, alpha_2, beta, f, Gamma_UVB = \
        Rahmati_parameters(redshift, TREECOOL_file, Gamma_UVB, fac, verbose)

    #compute the HI/H fraction 
    nH0 = HI_from_UVB(snapshot_fname, Gamma_UVB, True, correct_H2,
                      f, n0, alpha_1, alpha_2, beta, SF_temperature=1e4)
    
    #read the gas masses
    mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10 #Msun/h

    #create the array M_HI and fill it
    M_HI = 0.76*mass*nH0;  del nH0,mass
    print('Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit))

    return M_HI



#This function compute the HI/H fraction given the photoionization rate and 
#the density of the gas particles
#snapshot_fname ------------> name of the N-body snapshot
#Gamma_UVB -----------------> value of the UVB photoionization rate
#self_shielding_correction -> apply (True) or not (False) the self-shielding
#correct_H2 ----------------> correct the HI masses to account for H2
#f -------------------------> parameter of the Rahmati et al model (see eq. A1)
#n0 ------------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_1 -------------------> parameter of the Rahmati et al model (see eq. A1)
#alpha_2 -------------------> parameter of the Rahmati et al model (see eq. A1)
#beta ----------------------> parameter of the Rahmati et al model (see eq. A1)
#SF_temperature ------------> associate temperature of star forming particles
#If SF_temperature is set to None it will compute the temperature of the SF
#particles from their density and internal energy
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def HI_from_UVB(snapshot_fname, double Gamma_UVB,
                self_shielding_correction=False,correct_H2=False,
                double f=0.0, double n0=0.0, double alpha_1=0.0,
                double alpha_2=0.0, double beta=0.0,
                double SF_temperature=0.0):

    cdef long gas_part,i 
    cdef double mean_mol_weight, yhelium, A, B, C, R_surf, P, prefact2
    cdef double nH, Lambda, alpha_A, Lambda_T, Gamma_phot, prefact, T
    cdef np.float32_t[:] rho,U,ne,SFR,nH0

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    head     = readsnap.snapshot_header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Nall     = head.nall
    Masses   = head.massarr*1e10 #Msun/h
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
    h        = head.hubble
    gas_part = Nall[0]

    # read density, internal energy, electron fraction and star formation rate
    # density units: h^2 Msun / Mpc^3
    yhelium = (1.0-0.76)/(4.0*0.76) 
    U   = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
    ne  = readsnap.read_block(snapshot_fname,"NE  ",parttype=0) #electron frac
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
    SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR

    # define HI/H fraction
    nH0 = np.zeros(gas_part, dtype=np.float32)

    #### Now compute the HI/H fraction following Rahmati et. al. 2013 ####
    prefact  = 0.76*h**2*Msun/Mpc**3/mH*(1.0+redshift)**3 
    prefact2 = (gamma-1.0)*(1.0+redshift)**3*h**2*Msun/kpc**3/kB 

    print('doing loop...');  start = time.time()    
    for i in range(gas_part):

        # compute particle density in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
        nH = rho[i]*prefact

        # compute particle temperature
        if SFR[i]>0.0:
            T = SF_temperature
            P = prefact2*U[i]*rho[i]*1e-9  #K/cm^3
        else:
            mean_mol_weight = (1.0+4.0*yhelium)/(1.0+yhelium+ne[i])
            T = U[i]*(gamma-1.0)*mH*mean_mol_weight/kB
        
        #compute the case A recombination rate
        Lambda  = 315614.0/T
        alpha_A = 1.269e-13*pow(Lambda,1.503)\
                  /pow(1.0 + pow((Lambda/0.522),0.47),1.923) #cm^3/s

        #Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
        Lambda_T = 1.17e-10*sqrt(T)*\
                   exp(-157809.0/T)/(1.0+sqrt(T/1e5)) #cm^3/s

        #compute the photoionization rate
        Gamma_phot = Gamma_UVB
        if self_shielding_correction:
            Gamma_phot *= ((1.0 - f)*pow(1.0 + pow(nH/n0,beta), alpha_1) +\
                           f*pow(1.0 + nH/n0, alpha_2))

        #compute the coeficients A,B and C to calculate the HI/H fraction
        A = alpha_A + Lambda_T
        B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
        C = alpha_A

        #compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
        nH0[i] = (B-sqrt(B*B-4.0*A*C))/(2.0*A)

        # correct for the presence of H2
        if correct_H2 and SFR[i]>0.0:
            R_surf = (P/1.7e4)**0.8
            nH0[i] = nH0[i]/(1.0 + R_surf)

    print('Time taken = %.3f s'%(time.time()-start))
    return np.asarray(nH0)
############################################################################


############################################################################
# This routine looks for halos around the selected halos
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef locate_nearby_halos(np.float32_t[:,:] halos_pos, 
        np.float32_t[:,:] selected_halos_pos, float BoxSize):

    print('Starting routine...');  start = time.time()
    cdef np.int64_t[:] index, index_sort, indexes, cube_indexes
    cdef int ii, jj, kk, dims2, iii, jjj, kkk
    cdef long all_halos, selected_halos, i, j, dims3
    cdef int dims, number
    cdef np.float32_t[:,:] halos_pos_new
    cdef float dist, x, y, z
    cdef np.int64_t[:] offset

    # find the total number of halos in the catalogue
    all_halos = halos_pos.shape[0]
    selected_halos = selected_halos_pos.shape[0]
    dims  = <int>(BoxSize/2.0) # we consider the max radius of a halo to be 2 Mpc/h
    dims2 = dims*dims
    dims3 = dims*dims*dims

    # for each halo compute its index
    index      = np.zeros(all_halos,   dtype=np.int64)-1 #initialize to -1
    index_sort = np.zeros(all_halos,   dtype=np.int64)
    cube_indexes = np.zeros(27, dtype=np.int64)

    # for each halo find its index
    for i in range(all_halos):
        ii = <int>(halos_pos[i,0]*dims/BoxSize)
        if ii==dims:  ii = 0
        jj = <int>(halos_pos[i,1]*dims/BoxSize)
        if jj==dims:  jj = 0
        kk = <int>(halos_pos[i,2]*dims/BoxSize)
        if kk==dims:  kk = 0

        index[i] = ii*dims2 + jj*dims + kk


    indexes = np.argsort(index)
    halos_pos_new = np.copy(halos_pos)
    print(index)

    for i in range(indexes.shape[0]):
        halos_pos_new[i,0] = halos_pos[indexes[i],0]
        halos_pos_new[i,1] = halos_pos[indexes[i],1]
        halos_pos_new[i,2] = halos_pos[indexes[i],2]
        index_sort[i]      = index[indexes[i]]


    # current_number = index_sort[0]
    # offset[index_sort[0]] = 0
    # for i in range(all_halos):
    #     if index_sort[i]!=current_number:
    #         offset[index_sort[i]] = i
    #         current_number = index_sort[i]
    # offset[all_halos] = dims3

    # current_number = dims3
    # for i in range(all_halos,-1, -1):
    #     if offset[i]==-1:  offset[i] = current_number
    #     else:              current_number = offset[i]


    # print index_sort

    for i in range(all_halos):
        if i%1000==0:
            print(i)
        x = halos_pos[i,0]
        y = halos_pos[i,1]
        z = halos_pos[i,2]
        for j in range(selected_halos):
            dist = sqrt((selected_halos_pos[j,0]-x)**2 + \
                        (selected_halos_pos[j,1]-y)**2 + \
                        (selected_halos_pos[j,2]-z)**2)

    # # do a loop over the selected halos
    # for i in range(selected_halos_pos.shape[0]):
    #     ii = <int>(selected_halos_pos[i,0]*dims/BoxSize)
    #     if ii==dims:  ii = 0
    #     jj = <int>(selected_halos_pos[i,1]*dims/BoxSize)
    #     if jj==dims:  jj = 0
    #     kk = <int>(selected_halos_pos[i,2]*dims/BoxSize)
    #     if kk==dims:  kk = 0
        
    #     number = 0
    #     for iii in range(ii-1, ii+2):
    #         for jjj in range(jj-1, jj+2):
    #             for kkk in range(kk-1, kk+2):
    #                 cube_indexes[number] = 

    #     cube_indexes

    #     for j in range(all_halos):

    #         if index

    print('Time taken = %.3f seconds'%(time.time()-start))


############################################################################
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef HI_profile(np.float32_t[:,:] halo_pos, np.float32_t[:] halo_R,
    np.int64_t[:] halo_id, np.float32_t[:,:] pos, np.float32_t[:] MHI, 
    np.int64_t[:] offset, np.float64_t[:,:] HI_mass_shell, 
    np.int64_t[:] part_in_halo, float BoxSize, float R1):
    
    start = time.time()
    cdef long i, j, particles, halos, index, index_box, begin, end
    cdef float x_h, y_h, z_h, dx, dy, dz, factor
    cdef float middle, R2, radius, radius2, R12
    cdef int dims, dims2, bins, bin, index_x, index_y, index_z, 
    cdef int ii, jj, kk, iii, jjj, kkk, index_x_min, index_x_max
    cdef int index_y_min, index_y_max, index_z_min, index_z_max

    particles = pos.shape[0]
    halos     = halo_pos.shape[0]
    bins      = HI_mass_shell.shape[1]
    dims      = <int>rint((offset.shape[0]-1)**(1.0/3.0))
    dims2     = dims*dims
    R12       = R1*R1
    middle    = BoxSize/2.0
    factor    = dims*1.0/BoxSize

    # do a loop over all halos
    for i in range(halos):

        x_h     = halo_pos[i,0]
        y_h     = halo_pos[i,1]
        z_h     = halo_pos[i,2]
        radius  = halo_R[i]
        radius2 = radius*radius
        #index   = halo_id[i]

        # in C, <int>(3.2)=3, but <int>(-0.3)=0, so add dims to avoid this
        index_x_min = <int>((x_h-radius)*factor+dims)
        index_x_max = <int>((x_h+radius)*factor+dims)
        index_y_min = <int>((y_h-radius)*factor+dims)
        index_y_max = <int>((y_h+radius)*factor+dims)
        index_z_min = <int>((z_h-radius)*factor+dims)
        index_z_max = <int>((z_h+radius)*factor+dims)

        #index_x = index/dims2
        #index_y = (index%dims2)/dims
        #index_z = (index%dims2)%dims

        #for ii in range(-1,2):
            #iii = (index_x + ii + dims)%dims

            #for jj in range(-1,2):
            #    jjj = (index_y + jj + dims)%dims

                 #for kk in range(-1,2):
                 #   kkk = (index_z + kk + dims)%dims
        for ii in range(index_x_min, index_x_max+1):
            iii = ii%dims

            for jj in range(index_y_min, index_y_max+1):
                jjj = jj%dims

                for kk in range(index_z_min, index_z_max+1):
                    kkk = kk%dims

                    index_box = dims2*iii + dims*jjj + kkk

                    begin = offset[index_box]
                    end   = offset[index_box+1]

                    for j in range(begin,end):

                        #x        = pos[j,0]
                        #y        = pos[j,1]
                        #z        = pos[j,2]
                        #MHI_part = MHI[j]

                        dx = abs(pos[j,0] - x_h)
                        if dx>middle:  dx = BoxSize - dx
                        if dx>radius:  continue

                        dy = abs(pos[j,1] - y_h)
                        if dy>middle:  dy = BoxSize - dy
                        if dy>radius:  continue

                        dz = abs(pos[j,2] - z_h)
                        if dz>middle:  dz = BoxSize - dz
                        if dz>radius:  continue

                        R2 = dx*dx + dy*dy + dz*dz
                        if R2<radius2:

                            if R2<=R12:  bin = 0
                            else:     
                                bin = <int>(log10(R2/R12)/log10(radius2/R12)*(bins-1))+1
                            HI_mass_shell[i,bin]  += MHI[j]
                            part_in_halo[i] += 1


    print('Time taken = %.2f seconds'%(time.time()-start))
######################################################################################

# This routine computes the HI profile of one signale halo using brute force
# This is used mainly for validation purposes
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef HI_profile_1halo_brute_force(np.float32_t[:] halo_pos, float radius,
    np.float32_t[:,:] pos, np.float32_t[:] MHI, 
    np.float64_t[:] HI_mass_shell, float BoxSize, float R1, np.int64_t[:] part_in_halo):
    
    
    start = time.time()
    cdef long j, particles
    cdef float x, y, z, x_h, y_h, z_h, dx, dy, dz
    cdef float MHI_part, middle, R2, radius2, R12
    cdef int bin, bins

    particles = pos.shape[0]
    bins      = HI_mass_shell.shape[0]
    R12       = R1*R1
    middle    = BoxSize/2.0

    x_h     = halo_pos[0]
    y_h     = halo_pos[1]
    z_h     = halo_pos[2]
    radius2 = radius*radius

    # do a loop over all particles
    for j in range(particles):

        x        = pos[j,0]
        y        = pos[j,1]
        z        = pos[j,2]
        MHI_part = MHI[j]

        dx = abs(x - x_h)
        if dx>middle:  dx = BoxSize - dx
        if dx>radius:  continue

        dy = abs(y - y_h)
        if dy>middle:  dy = BoxSize - dy
        if dy>radius:  continue

        dz = abs(z - z_h)
        if dz>middle:  dz = BoxSize - dz
        if dz>radius:  continue

        R2 = dx*dx + dy*dy + dz*dz
        if R2<radius2:

            if R2<=R12:  bin = 0
            else:     
                bin = <int>(log10(R2/R12)/log10(radius2/R12)*(bins-1))+1
            HI_mass_shell[bin]  += MHI_part
            part_in_halo[0] += 1


    print('Time taken = %.2f seconds'%(time.time()-start))



cpdef overlap_particles_halos(np.float32_t[:,:] halo_pos, float Rvir, float BoxSize,
        float x_min, float x_max, float y_min, float y_max, 
        float z_min, float z_max):

    cdef long halos, i
    cdef float x, y, z, x1, x2, y1, y2, z1, z2

    halos = halo_pos.shape[0]

    for i in range(halos):
        x = halo_pos[i,0]
        y = halo_pos[i,1]
        z = halo_pos[i,2]

        x1 = (x-Rvir)%BoxSize
        x2 = (x+Rvir)%BoxSize
        if x1<x_min or x1>x_max or x2<x_min or x2>x_max:
            return True

        y1 = (y-Rvir)%BoxSize
        y2 = (y+Rvir)%BoxSize
        if y1<y_min or y1>y_max or y2<y_min or y2>y_max:
            return True

        z1 = (z-Rvir)%BoxSize
        z2 = (z+Rvir)%BoxSize
        if z1<z_min or z1>z_max or z2<z_min or z2>z_max:
            return True

    return False

############################################################################
# This routine is used to compute the total HI mass within the radius of 
# dark matter halos.
# halo_pos ----------> sorted positions of the halos
# halo_R ------------> corresponding radii of the halos
# pos ---------------> sorted positions of the particles
# M_HI --------------> corresponding HI masses of the particles
# offset ------------> offset array with the begin,end information of each cell
# M_HI_halo ---------> array containing the total HI mass inside each halo
# BoxSize -----------> simulation box size. Needed for boundary conditions
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef void HI_mass_SO(np.float32_t[:,::1] halo_pos, np.float32_t[::1] halo_R,
    np.float32_t[:,::1] pos, np.float32_t[::1] MHI, 
    np.int64_t[::1] offset, np.float64_t[::1] M_HI_halo, float BoxSize):
    
    start = time.time()
    cdef long i, j, particles, halos, index, index_box, begin, end
    cdef float x_h, y_h, z_h, dx, dy, dz, factor
    cdef float middle, R2, radius, radius2
    cdef int dims, dims2, index_x_min, index_x_max
    cdef int index_y_min, index_y_max, index_z_min, index_z_max
    cdef int ii, jj, kk, iii, jjj, kkk

    particles = pos.shape[0]
    halos     = halo_pos.shape[0]
    dims      = <int>rint((offset.shape[0]-1)**(1.0/3.0))
    dims2     = dims*dims
    middle    = BoxSize/2.0
    factor    = dims*1.0/BoxSize

    # do a loop over all halos
    for i in range(halos):

        x_h     = halo_pos[i,0]
        y_h     = halo_pos[i,1]
        z_h     = halo_pos[i,2]
        radius  = halo_R[i]
        radius2 = radius*radius

        # in C, <int>(3.2)=3, but <int>(-0.3)=0, so add dims to avoid this
        index_x_min = <int>((x_h-radius)*factor+dims)
        index_x_max = <int>((x_h+radius)*factor+dims)
        index_y_min = <int>((y_h-radius)*factor+dims)
        index_y_max = <int>((y_h+radius)*factor+dims)
        index_z_min = <int>((z_h-radius)*factor+dims)
        index_z_max = <int>((z_h+radius)*factor+dims)

        # do a loop over the cell hosting the halo and its neighburgs
        for ii in range(index_x_min, index_x_max+1):
            iii = ii%dims

            for jj in range(index_y_min, index_y_max+1):
                jjj = jj%dims

                for kk in range(index_z_min, index_z_max+1):
                    kkk = kk%dims

                    index_box = dims2*iii + dims*jjj + kkk

                    # find the location of the particles belonging to the cell
                    # in the sorted array
                    begin = offset[index_box]
                    end   = offset[index_box+1]

                    # do a loop over all particles in the cell
                    for j in range(begin,end):

                        dx = abs(pos[j,0] - x_h)
                        if dx>middle:  dx = BoxSize - dx
                        if dx>radius:  continue

                        dy = abs(pos[j,1] - y_h)
                        if dy>middle:  dy = BoxSize - dy
                        if dy>radius:  continue

                        dz = abs(pos[j,2] - z_h)
                        if dz>middle:  dz = BoxSize - dz
                        if dz>radius:  continue

                        R2 = dx*dx + dy*dy + dz*dz
                        if R2<radius2:
                            M_HI_halo[i] += MHI[j]


    print('Time taken = %.2f seconds'%(time.time()-start))

# # This routine computes the cross-section of the DLAs
# @cython.boundscheck(False)
# @cython.cdivision(True)
# @cython.wraparound(False)
# def DLAs_cross_section_original(snapshot_root, snapnum, TREECOOL_file, resolution):

#     cdef int subfile_num
#     cdef long i, num_halos, begin, end, offset, start_h, end_h
#     # cdef np.float32_t[:,:] pos_h
#     # cdef np.float32_t[:] radii_h, MHI_h
#     cdef np.float64_t[:] M_HI_tot, cross_section
#     cdef np.int32_t[:] halo_len
#     # cdef np.float32_t[:] mass,SFR,metals,rho,Volume,

#     # find snapshot name and read header
#     snapshot = snapshot_root + 'snapdir_%03d/snap_%03d'%(snapnum, snapnum)
#     header   = rs.snapshot_header(snapshot)
#     redshift = header.redshift
#     BoxSize  = header.boxsize/1e3 #Mpc/h
#     filenum  = header.filenum
#     Omega_m  = header.omega0
#     Omega_L  = header.omegaL
#     h        = header.hubble

#     print '\nBoxSize         = %.1f Mpc/h'%BoxSize
#     print 'Number of files = %d'%filenum
#     print 'Omega_m         = %.3f'%Omega_m
#     print 'Omega_l         = %.3f'%Omega_L
#     print 'redshift        = %.3f'%redshift

#     # read number of particles in halos, their positions and masses
#     halos = groupcat.loadHalos(snapshot_root, snapnum, 
#             fields=['GroupLenType','GroupPos','GroupMass','Group_R_TopHat200'])
#     halo_len    = halos['GroupLenType'][:,0]  
#     halo_pos    = halos['GroupPos']/1e3
#     halo_mass   = halos['GroupMass']*1e10
#     halo_radius = halos['Group_R_TopHat200']/1e3
#     del halos

#     # factor to convert (Mpc/h)^{-2} to cm^{-2}
#     factor = h*(1.0+redshift)**2*\
#         (UL.units().Msun_g)/(UL.units().mH_g)/(UL.units().Mpc_cm)**2

#     # find the total number of halos
#     num_halos = halo_pos.shape[0]
#     print 'Found %d halos'%num_halos
#     M_HI_tot      = np.zeros(num_halos, dtype=np.float64)
#     cross_section = np.zeros(num_halos, dtype=np.float64)

#     # read the first subfile of the snapshot
#     subfile_num = 0
#     offset = 0
#     snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'\
#         %(snapnum, snapnum, subfile_num)
#     header = rs.snapshot_header(snapshot)
#     npart  = header.npart[0]

#     # find positions, HI masses and radii of the gas particles
#     pos  = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
#     pos  = pos.astype(np.float32)
#     MHI  = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
#     mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
#     SFR  = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
#     indexes = np.where(SFR>0.0)[0];  del SFR

#     # find the metallicity of star-forming particles
#     metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
#     metals = metals[indexes]/0.0127

#     # find densities of star-forming particles: units of h^2 Msun/Mpc^3
#     rho = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
#     Volume = mass/rho                            #(Mpc/h)^3
#     radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 

#     # find density and radius of star-forming particles
#     radii_SFR = radii[indexes]    
#     rho       = rho[indexes]

#     # find HI/H fraction for star-forming particles
#     MHI[indexes] = Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
#                                             h, TREECOOL_file, Gamma=None,
#                                             fac=1, correct_H2=True) #HI/H
#     MHI *= (0.76*mass)

#     # do a loop over all halos
#     begin_abs = 0
#     for i in range(num_halos):

#         # print i,subfile_num

#         # find where the halo starts and ends in the file
#         if i>0:  begin_abs += halo_len[i-1]
#         end     = begin_abs + halo_len[i]
#         start_h, end_h = 0, end-begin_abs
#         begin   = begin_abs

#         # if i==294689:
#         #     print begin_abs,end

#         # define arrays hosting positions, masses and radii of particles in halo
#         pos_h   = np.zeros((end-begin,3), dtype=np.float32)
#         radii_h = np.zeros(end-begin,     dtype=np.float32)
#         MHI_h   = np.zeros(end-begin,     dtype=np.float32)

#         # if halo is in current subfile fill it 
#         if end<=(offset+npart):
#             pos_h   = pos[begin-offset:end-offset]
#             radii_h = radii[begin-offset:end-offset]
#             MHI_h   = MHI[begin-offset:end-offset]
#             M_HI_tot[i] = np.sum(MHI_h, dtype=np.float64)

#             if len(pos_h)==0:  continue

#             # check if all particles can produce a single DLA
#             rho_h   = 3.0*MHI_h/(4.0*np.pi*radii_h**3)
#             NHI_max = factor*2.0*np.sum(rho_h*radii_h, dtype=np.float64)
#             if NHI_max<10**(20.3):  continue

#             if halo_radius[i]>0.0:
#                 dims = int(halo_radius[i]*1.0/resolution)
#             else:
#                 dims = 50

#             area = DLAs_cross_section_1halo(pos_h, radii_h, MHI_h, dims, 
#                 BoxSize, h, redshift)
#             cross_section[i] = area
#             if area>0.1:
#                 print i,area
#                 np.savetxt('borrar.dat',np.transpose([pos_h[:,0], pos_h[:,1], pos_h[:,2], radii_h, MHI_h]))  
#                 sys.exit()
#             continue
#             # f=open('cross_section.dat','a')
#             # f.write(str(halo_mass[i])+' '+str(area)+'\n')
#             # f.close()

#         else:
#             pos_h[0:offset+npart-begin]   = pos[begin-offset:]
#             radii_h[0:offset+npart-begin] = radii[begin-offset:]
#             MHI_h[0:offset+npart-begin]   = MHI[begin-offset:]
#             start_h = offset+npart-begin
#             begin = offset+npart

#         # do a loop over subfiles until all particles in the halo has been read
#         done = False
#         while not(done):
#             offset += npart;  subfile_num += 1;  print subfile_num,i
#             snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'\
#                 %(snapnum, snapnum, subfile_num)
#             header = rs.snapshot_header(snapshot)
#             npart  = header.npart[0]

#             # find positions, HI masses and radii of the gas particles
#             pos  = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
#             pos  = pos.astype(np.float32)
#             MHI  = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
#             mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
#             SFR  = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
#             indexes = np.where(SFR>0.0)[0];  del SFR

#             # find the metallicity of star-forming particles
#             metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
#             metals = metals[indexes]/0.0127

#             # find densities of star-forming particles: units of h^2 Msun/Mpc^3
#             rho = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
#             Volume = mass/rho                            #(Mpc/h)^3
#             radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 

#             # find density and radius of star-forming particles
#             radii_SFR = radii[indexes]    
#             rho       = rho[indexes]

#             # find HI/H fraction for star-forming particles
#             MHI[indexes] = Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
#                                                     h, TREECOOL_file, Gamma=None,
#                                                     fac=1, correct_H2=True) #HI/H
#             MHI *= (0.76*mass)

#             if end<=(offset+npart):
#                 pos_h[start_h:]   = pos[begin-offset:end-offset]
#                 radii_h[start_h:] = radii[begin-offset:end-offset]
#                 MHI_h[start_h:]   = MHI[begin-offset:end-offset]
#                 done = True
#                 M_HI_tot[i] = np.sum(MHI_h, dtype=np.float64)

#             else:
#                 pos_h[start_h:start_h+offset+npart-begin]   = pos[begin-offset:]
#                 radii_h[start_h:start_h+offset+npart-begin] = radii[begin-offset:]
#                 MHI_h[start_h:start_h+offset+npart-begin]   = MHI[begin-offset:]
#                 start_h = start_h+offset+npart-begin
#                 begin = offset+npart

#     return M_HI_tot, cross_section

"""
# This routine computes the cross-section of the DLAs
# resolution -----> spatial resolution in Mpc/h for the grid
# NHI_values -----> computes the cross-section for different NHI values
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def DLAs_cross_section(snapshot_root, snapnum, TREECOOL_file, resolution,
    NHI_values):

    cdef int subfile_num
    cdef long i, k, num_halos, begin, end, offset, start_h, end_h
    # cdef np.float32_t[:,:] pos_h
    # cdef np.float32_t[:] radii_h, MHI_h
    cdef np.float64_t[:] M_HI_tot, area 
    cdef np.int32_t[:] halo_len
    # cdef np.float64_t[:,:] cross_section
    # cdef np.float32_t[:] mass,SFR,metals,rho,Volume,

    # find snapshot name and read header
    snapshot = snapshot_root + 'snapdir_%03d/snap_%03d'%(snapnum, snapnum)
    header   = rs.snapshot_header(snapshot)
    redshift = header.redshift
    BoxSize  = header.boxsize/1e3 #Mpc/h
    filenum  = header.filenum
    Omega_m  = header.omega0
    Omega_L  = header.omegaL
    h        = header.hubble

    print('\nBoxSize         = %.1f Mpc/h'%BoxSize)
    print('Number of files = %d'%filenum)
    print('Omega_m         = %.3f'%Omega_m)
    print('Omega_l         = %.3f'%Omega_L)
    print('redshift        = %.3f'%redshift)

    # read number of particles in halos, their positions and masses
    halos = groupcat.loadHalos(snapshot_root, snapnum, 
            fields=['GroupLenType','GroupPos','GroupMass','Group_R_TopHat200'])
    halo_len    = halos['GroupLenType'][:,0]  
    halo_pos    = halos['GroupPos']/1e3
    halo_mass   = halos['GroupMass']*1e10
    halo_radius = halos['Group_R_TopHat200']/1e3
    del halos

    # factor to convert (Mpc/h)^{-2} to cm^{-2}
    factor = h*(1.0+redshift)**2*\
        (UL.units().Msun_g)/(UL.units().mH_g)/(UL.units().Mpc_cm)**2

    # find the total number of halos and define arrays contining MHI and sigma
    num_halos = halo_pos.shape[0]
    NHI_bins  = NHI_values.shape[0]
    print('Found %d halos'%num_halos)
    M_HI_tot      = np.zeros(num_halos,             dtype=np.float64)
    cross_section = np.zeros((num_halos, NHI_bins), dtype=np.float64)

    # do a loop over all halos
    subfile_num, offset, npart, begin_abs = -1, 0, 0, 0
    pos   = np.empty((0,3), dtype=np.float32)
    radii = np.empty(0,     dtype=np.float32)
    MHI   = np.empty(0,     dtype=np.float32)
    for i in range(num_halos):

        # find where the halo starts and ends in the file
        if i>0:  begin_abs += halo_len[i-1]
        end     = begin_abs + halo_len[i]
        start_h, end_h = 0, end-begin_abs
        begin   = begin_abs

        # define arrays hosting positions, masses and radii of particles in halo
        pos_h   = np.zeros((end-begin,3), dtype=np.float32)
        radii_h = np.zeros(end-begin,     dtype=np.float32)
        MHI_h   = np.zeros(end-begin,     dtype=np.float32)

        # if halo is in current subfile fill it 
        if end<=(offset+npart):
            pos_h   = pos[begin-offset:end-offset]
            radii_h = radii[begin-offset:end-offset]
            MHI_h   = MHI[begin-offset:end-offset]
            M_HI_tot[i] = np.sum(MHI_h, dtype=np.float64)

            if len(pos_h)==0:  continue

            # check if all particles can produce a single DLA
            rho_h   = 3.0*MHI_h/(4.0*np.pi*radii_h**3)
            NHI_max = factor*2.0*np.sum(rho_h*radii_h, dtype=np.float64)
            if NHI_max<10**(20.3):  continue

            if halo_radius[i]>0.0:  dims = int(halo_radius[i]*1.0/resolution)
            else:                   dims = 50

            cross_section[i] = DLAs_cross_section_1halo(pos_h, radii_h, MHI_h, dims, 
                BoxSize, h, redshift, NHI_values, NHI_bins)
            continue

        else:
            pos_h[0:offset+npart-begin]   = pos[begin-offset:]
            radii_h[0:offset+npart-begin] = radii[begin-offset:]
            MHI_h[0:offset+npart-begin]   = MHI[begin-offset:]
            start_h = offset+npart-begin
            begin = offset+npart

        # do a loop over subfiles until all particles in the halo has been read
        done = False
        while not(done):
            offset += npart;  subfile_num += 1;  print(subfile_num,i)
            snapshot = snapshot_root + 'snapdir_%03d/snap_%03d.%d'\
                %(snapnum, snapnum, subfile_num)
            header = rs.snapshot_header(snapshot)
            npart  = header.npart[0]

            # find positions, HI masses and radii of the gas particles
            pos  = rs.read_block(snapshot, 'POS ', parttype=0, verbose=False)/1e3
            pos  = pos.astype(np.float32)
            MHI  = rs.read_block(snapshot, 'NH  ', parttype=0, verbose=False)#HI/H
            mass = rs.read_block(snapshot, 'MASS', parttype=0, verbose=False)*1e10
            SFR  = rs.read_block(snapshot, 'SFR ', parttype=0, verbose=False)
            indexes = np.where(SFR>0.0)[0];  del SFR

            # find the metallicity of star-forming particles
            metals = rs.read_block(snapshot, 'GZ  ', parttype=0, verbose=False)
            metals = metals[indexes]/0.0127

            # find densities of star-forming particles: units of h^2 Msun/Mpc^3
            rho = rs.read_block(snapshot, 'RHO ', parttype=0, verbose=False)*1e19
            Volume = mass/rho                            #(Mpc/h)^3
            radii  = (Volume/(4.0*np.pi/3.0))**(1.0/3.0) #Mpc/h 

            # find density and radius of star-forming particles
            radii_SFR = radii[indexes]    
            rho       = rho[indexes]

            # find HI/H fraction for star-forming particles
            MHI[indexes] = Rahmati_HI_Illustris(rho, radii_SFR, metals, redshift, 
                                                    h, TREECOOL_file, Gamma=None,
                                                    fac=1, correct_H2=True) #HI/H
            MHI *= (0.76*mass)

            if end<=(offset+npart):
                pos_h[start_h:]   = pos[begin-offset:end-offset]
                radii_h[start_h:] = radii[begin-offset:end-offset]
                MHI_h[start_h:]   = MHI[begin-offset:end-offset]
                done = True
                M_HI_tot[i] = np.sum(MHI_h, dtype=np.float64)

                if len(pos_h)==0:  continue

                # check if all particles can produce a single DLA
                rho_h   = 3.0*MHI_h/(4.0*np.pi*radii_h**3)
                NHI_max = factor*2.0*np.sum(rho_h*radii_h, dtype=np.float64)
                if NHI_max<10**(20.3):  continue

                if halo_radius[i]>0.0:  dims = int(halo_radius[i]*1.0/resolution)
                else:                   dims = 50

                cross_section[i] = DLAs_cross_section_1halo(pos_h, radii_h, MHI_h, dims, 
                    BoxSize, h, redshift, NHI_values, NHI_bins)

            else:
                pos_h[start_h:start_h+offset+npart-begin]   = pos[begin-offset:]
                radii_h[start_h:start_h+offset+npart-begin] = radii[begin-offset:]
                MHI_h[start_h:start_h+offset+npart-begin]   = MHI[begin-offset:]
                start_h = start_h+offset+npart-begin
                begin = offset+npart

    return halo_mass, M_HI_tot, cross_section
"""

# This routine computes the DLAs cross-section of a single halo
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef DLAs_cross_section_1halo(np.float32_t[:,:] pos, np.float32_t[:] radii,
    np.float32_t[:] MHI, int dims, float BoxSize, float h, float redshift,
    np.float64_t[:] NHI_values, int NHI_bins):

    cdef int i, j, k, particles
    cdef float x_min, x_max, y_min, y_max, xW, yW, middle, BoxSize_region
    cdef double factor 
    cdef np.float64_t[:,:] NHI
    cdef np.float64_t[:] sigma, threshold

    particles = pos.shape[0]
    middle = BoxSize/2.0

    sigma     = np.zeros(NHI_bins, dtype=np.float64)
    threshold = np.zeros(NHI_bins, dtype=np.float64)

    # find the edges of the region
    x_min = np.min(pos[:,0]);  x_max = np.max(pos[:,0]);  xW = x_max-x_min
    y_min = np.min(pos[:,1]);  y_max = np.max(pos[:,1]);  yW = y_max-y_min

    if xW>BoxSize/2.0:
        for i in range(particles):
            if pos[i,0]<middle:  pos[i,0] += BoxSize
        x_min = np.min(pos[:,0]);  x_max = np.max(pos[:,0]);  xW = x_max-x_min
    
    if yW>BoxSize/2.0:
        for i in range(particles):
            if pos[i,1]<middle:  pos[i,1] += BoxSize
        y_min = np.min(pos[:,1]);  y_max = np.max(pos[:,1]);  yW = y_max-y_min

    BoxSize_region = max(yW, xW)

    # define array contining column densities
    NHI = np.zeros((dims,dims), dtype=np.float64)
    pos_arr = np.ascontiguousarray(pos[:,0:2])

    MASL.voronoi_RT_2D(np.asarray(NHI), pos_arr, 
        np.asarray(MHI), np.asarray(radii), x_min, y_min, 0, 1, BoxSize_region, 
        periodic=False, verbose=False)

    # convert (Msun/h)/(Mpc/h)^2 to cm^{-2}
    factor = h*(1.0+redshift)**2*\
        (UL.units().Msun_g)/(UL.units().mH_g)/(UL.units().Mpc_cm)**2
    for k in range(NHI_bins):
        threshold[k] = NHI_values[k]/factor

    for i in range(dims):
        for j in range(dims):
            for k in range(NHI_bins):
                if NHI[i,j]>threshold[k]:  sigma[k] += 1.0
                else:                      break
 
    for k in range(NHI_bins):
        sigma[k] = sigma[k]*BoxSize_region**2*1.0/(dims*dims)

    return np.asarray(sigma)
















