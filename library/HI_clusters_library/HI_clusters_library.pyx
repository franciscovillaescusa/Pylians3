#This library is used for the HI & clusters project

import numpy as np
import readsnap,readsnap2
import sys,os,time 
cimport numpy as np

################################# UNITS #####################################
rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3

yr    = 3.15576e7  #seconds
km    = 1e5        #cm
Mpc   = 3.0856e24  #cm
kpc   = 3.0856e21  #cm
Msun  = 1.989e33   #g
Ymass = 0.24       #helium mass fraction
mH    = 1.6726e-24 #proton mass in grams
gamma = 5.0/3.0    #ideal gas
kB    = 1.3806e-26 #gr (km/s)^2 K^{-1}
nu0   = 1420.0     #21-cm frequency in MHz

pi = np.pi
#############################################################################

################################ Rahmati ######################################
#Notice that in this implementation we set the temperature of the star-forming
#particles to 1e4 K, and use the Rahmati formula to compute the HI/H fraction.
#The routine computes and returns the HI/H fraction taking into account the
#HI self-shielding and the presence of molecular hydrogen.
#snapshot_fname ----------> file containing the snapshot
#TREECOOL_file -----------> file containing the TREECOOL file
#T_block -----------------> whether the snapshot contains the TEMP block or not
#Gamma_UVB ---------------> value of Gamma_UVB. None to read it from TREECOOL
#SF_temperature ----------> The value to set the temperature of the SF particles
#self_shielding_correction --> whether to correct HI fraction for self-shielding
#correct_H2 --------------> correct HI fraction for presence of H2:
#                           'BR': Blitz-Rosolowsky, 'THINGS', 'KMT' or 'None'
def Rahmati(snapshot_fname, TREECOOL_file, T_block=True, Gamma_UVB=None,
            SF_temperature=1e4, self_shielding_correction=True,
            correct_H2='BR'):

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

    # read/compute the temperature of the gas particles
    if T_block:
        T = readsnap.read_block(snapshot_fname,"TEMP",parttype=0) #K
    else:
        U = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #km/s
        T = U/1.5/(1.380658e-16/(0.6*1.672631e-24)*1e-10)         #K
    T = T.astype(np.float64) #to increase precision use float64 variables
    print('%.3e < T[K] < %.3e'%(np.min(T),np.max(T)))

    # read the density block: units of h^2 Msun/Mpc^3
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10/1e-9 
    print('%.3e < rho < %.3e'%(np.min(rho),np.max(rho)))

    # Rahmati et. al. 2013 self-shielding parameters (table A1)
    z_t       = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00])
    n0_t      = np.array([-2.94,-2.29,-2.06,-2.13,-2.23,-2.35]); n0_t=10**n0_t
    alpha_1_t = np.array([-3.98,-2.94,-2.22,-1.99,-2.05,-2.63])
    alpha_2_t = np.array([-1.09,-0.90,-1.09,-0.88,-0.75,-0.57])
    beta_t    = np.array([1.29, 1.21, 1.75, 1.72, 1.93, 1.77])
    f_t       = np.array([0.01, 0.03, 0.03, 0.04, 0.02, 0.01])

    # compute the self-shielding parameters at the redshift of the N-body
    n0      = np.interp(redshift,z_t,n0_t)
    alpha_1 = np.interp(redshift,z_t,alpha_1_t)
    alpha_2 = np.interp(redshift,z_t,alpha_2_t)
    beta    = np.interp(redshift,z_t,beta_t)
    f       = np.interp(redshift,z_t,f_t)
    print('n0 = %e\nalpha_1 = %2.3f\nalpha_2 = %2.3f\nbeta = %2.3f\nf = %2.3f'\
        %(n0,alpha_1,alpha_2,beta,f))

    # find the value of the photoionization rate
    if Gamma_UVB==None:
        data = np.loadtxt(TREECOOL_file);  logz = data[:,0]; 
        Gamma_UVB = data[:,1]
        Gamma_UVB = np.interp(np.log10(1.0+redshift),logz,Gamma_UVB);  del data
        print('Gamma_UVB(z=%2.2f) = %e s^{-1}'%(redshift,Gamma_UVB))


    # for star forming particle assign T=10^4 K
    if SF_temperature!=None and T_block==True:
        SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0) #SFR
        indexes = np.where(SFR>0.0)[0];  T[indexes] = SF_temperature
        del indexes,SFR; print('%.3e < T[K] < %.3e'%(np.min(T),np.max(T)))
    

    #### Now compute the HI/H fraction following Rahmati et. al. 2013 ####
    # compute densities in cm^{-3}. rho is in h^2 Msun/Mpc^3 units
    nH = 0.76*h**2*Msun/Mpc**3/mH*rho*(1.0+redshift)**3; #del rho
    nH = nH.astype(np.float64)

    #compute the case A recombination rate
    Lambda  = 315614.0/T
    alpha_A = 1.269e-13*Lambda**(1.503)\
        /(1.0+(Lambda/0.522)**(0.47))**(1.923) #cm^3/s
    alpha_A = alpha_A.astype(np.float64)

    #Compute Lambda_T (eq. A6 of Rahmati et. al. 2013)
    Lambda_T=1.17e-10*np.sqrt(T)*np.exp(-157809.0/T)/(1.0+np.sqrt(T/1e5))#cm^3/s
    Lambda_T=Lambda_T.astype(np.float64)

    #compute the photoionization rate
    Gamma_phot = Gamma_UVB
    if self_shielding_correction:
        Gamma_phot *=\
            ((1.0-f)*(1.0+(nH/n0)**beta)**alpha_1 + f*(1.0+nH/n0)**alpha_2)
    print('Gamma_phot = ',Gamma_phot)

    #compute the coeficients A,B and C to calculate the HI/H fraction
    A = alpha_A + Lambda_T
    B = 2.0*alpha_A + Gamma_phot/nH + Lambda_T
    C = alpha_A

    #compute the HI/H fraction (eq. A8 of Rahmati et. al. 2013)
    nH0 = (B-np.sqrt(B**2-4.0*A*C))/(2.0*A); del nH
    nH0 = nH0.astype(np.float32)

    #correct for the presence of H2
    if correct_H2 in ['BR','THINGS']:
        print('correcting HI/H to account for the presence of H2...')
        #compute the pression of the gas particles
        #h^2Msun/kpc^3
        rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10
        U   = readsnap.read_block(snapshot_fname,"U   ",parttype=0) #(km/s)^2
        P   = (gamma-1.0)*U*rho*(1.0+redshift)**3 #h^2 Msun/kpc^3*(km/s)^2
        P   = h**2*Msun/kpc**3*P                  #gr/cm^3*(km/s)^2
        P  /= kB                                 #K/cm^3
        del rho,U
 
        #assign H2 only to star forming particles
        SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
        indexes = np.where(SFR>0.0)[0]; del SFR

        #compute the H2/HI fraction
        R_surf = np.zeros(Nall[0],dtype=np.float32)
        if correct_H2   == 'BR':       R_surf = (P/3.5e4)**0.92
        elif correct_H2 == 'THINGS':   R_surf = (P/1.7e4)**0.8
        #R_surf[IDs]=1.0/(1.0+(35.0*(0.1/nH/0.76)**(gamma))**0.92)
        else:
            print('bad choice of correct_H2!!!'); sys.exit()

        #compute the corrected HI mass taking into account the H2
        nH0[indexes]=nH0[indexes]/(1.0+R_surf[indexes]); del indexes,R_surf
        #M_HI[indexes]=M_HI[indexes]*(1.0-R_surf[indexes]); del indexes,R_surf

    if correct_H2=='KMT':
        print('correcting HI/H to account for the presence of H2 using KMT...')
        H2_frac = H2_fraction(snapshot_fname)  #H2/NH = H2/(H2+HI)
        nH0 = nH0*(1.0-H2_frac)                #(NH/H)*(HI/NH)
        if np.any(nH0<0.0):
            print('HI/H cant be negative!!!!'); sys.exit()

    return nH0
##############################################################################

##############################################################################
#This routine computes and returns the metallicity of each gas particle,
#in units of the solar metallicity
def gas_metallicity(snapshot_fname):

    # read snapshot head and obtain BoxSize, Omega_m and Omega_L
    print('finding the metallicity of the gas particles')
    head  = readsnap.snapshot_header(snapshot_fname)
    Nall  = head.nall
    files = head.filenum

    # read the masses of the gas particles in 1e10 Msun/h
    mass = readsnap.read_block(snapshot_fname,"MASS",parttype=0)

    # define the metallicity array
    metallicity = np.zeros(Nall[0],dtype=np.float64)

    offset = 0
    for filenum in range(files):
        if files==1:  fname = snapshot_fname
        else:         fname = snapshot_fname+'.'+str(filenum)
        
        npart = readsnap.snapshot_header(fname).npart  #particles in that file

        block_found = False;  EoF = False;  f = open(fname,'rb')
        f.seek(0,2); last_position = f.tell(); f.seek(0,0)
        while (not(block_found) or not(EoF)):
        
            # read the three first elements and the blocksize
            format_type = np.fromfile(f,dtype=np.int32,count=1)[0]
            block_name  = f.read(4)
            dummy       = np.fromfile(f,dtype=np.float64,count=1)[0] 
            blocksize1  = np.fromfile(f,dtype=np.int32,count=1)[0]
            
            if block_name=='Zs  ':
                Z = np.fromfile(f,dtype=np.float32,count=npart[0]*15)
                Z = np.reshape(Z,(npart[0],15))
            
                metal = np.sum(Z[:,1:],axis=1,dtype=np.float64)\
                    /(mass[offset:offset+npart[0]])/0.0127
                metallicity[offset:offset+npart[0]] = metal;  offset += npart[0]

                f.seek(npart[4]*15*4,1);  block_found = True

            else:   f.seek(blocksize1,1)

            # read the size of the block and check that is the same number
            blocksize2 = np.fromfile(f,dtype=np.int32,count=1)[0]

            if blocksize2!=blocksize1:
                print('error!!!'); sys.exit()
        
            current_position = f.tell()
            if current_position == last_position:  EoF = True
        f.close()

    if offset!=Nall[0]:
        print('Not all files read!!!'); sys.exit()
        
    return metallicity
##############################################################################

##############################################################################
#This routine computes the H2 fraction in the gas particles. Only H2 is assigned
#to star-forming particles. We use the KMT formalism, see also Dave et al. 2013
def H2_fraction(snapshot_fname):

    # read snapshot header
    print('\nREADING SNAPSHOTS PROPERTIES')
    head     = readsnap.snapshot_header(snapshot_fname)
    Nall     = head.nall
    redshift = head.redshift
    h        = head.hubble

    # find the metallicity of the gas particles
    Z = gas_metallicity(snapshot_fname)

    Msun    = 1.99e33   #g
    kpc     = 3.0857e21 #cm
    sigma_d = Z*1e-21   #cm^2
    mu_H    = 2.3e-24   #g

    # read the density of the gas particles in h^2 Msun/kpc^3 in proper units
    rho = readsnap.read_block(snapshot_fname,"RHO ",parttype=0)*1e10 
    rho = rho*h**2*(1.0+redshift)**3  #Msun/kpc^3
 
    # read the SPH radii of the gas particles in kpc/h in proper units
    R = readsnap.read_block(snapshot_fname,"HSML",parttype=0) #ckpc/h
    R = (R/h)/(1.0+redshift)                                  #kpc

    # compute the gas surface density in g/cm^2
    sigma = (rho*R)*(Msun/kpc**2)  #g/cm^2
    del rho, R

    # read the star formation rate. Only star-forming particles will host H2
    SFR = readsnap.read_block(snapshot_fname,"SFR ",parttype=0)
    indexes = np.where((SFR>0.0) & (Z>0))[0]
    
    #define the molecular hydrogen fraction array
    f_H2 = np.zeros(Nall[0],dtype=np.float64)

    # compute the value of tau_c, chi and s of the star-forming particles
    tau_c = sigma[indexes]*(sigma_d[indexes]/mu_H)
    chi   = 0.756*(1.0 + 3.1*Z[indexes]**0.365)
    s     = np.log(1.0 + 0.6*chi + 0.01*chi**2)/(0.6*tau_c)

    f_H2[indexes] = 1.0 - 0.75*s/(1.0 + 0.25*s);  del indexes, tau_c, chi, s
    f_H2[np.where(f_H2<0.0)[0]] = 0.0 #avoid negative values in the H2_fraction

    return f_H2
##############################################################################


# Returns the snapshot_fname and halo_file for a given region and physics
class Region:
    def __init__(self,region,physics,snap_number):

        phys_dict   = {'CSF':'/CSF2015/', 'AGN':'/BH2015/'}
        root_folder = '/pico/home/userexternal/fvillaes/Dianoga/D'+\
            str(region)+phys_dict[physics]

        snapshot_fname = root_folder+'snap_'+snap_number
        halo_file      = root_folder+\
            'PostProcessing/PostProc_OldSetUp/snap_'+snap_number+'.glob'
        subfind_file   = root_folder+'Subfind/groups_'+snap_number+\
            '/sub_'+snap_number

        self.snapshot_fname = snapshot_fname
        self.halo_file      = halo_file
        self.subfind_file   = subfind_file



# this routine renumber the IDs
# IDs ------------> array with the "wrong" IDs to renumber
# bad_IDs --------> array with the sorted bad IDs of the simulation
# missing_IDs ----> array with the missed good IDs of the entire simulation
# there is a one-to-one correspondence between bad_IDs and missing_IDs,
# such as the ID to assign to a particle with a "wrong" ID bad_IDs[j] would be
# missing_IDs[j]. The routine, given a wrong ID, will look for the index j such
# as bad_IDs[j]=ID, and then assign a new ID which would be missing_IDs[j]
def rename_IDs(np.ndarray[np.uint32_t, ndim=1] IDs,
               np.ndarray[np.uint32_t, ndim=1] bad_IDs,
               np.ndarray[np.uint32_t, ndim=1] missing_IDs):

    cdef long i, j_min, j_max, j, value
    cdef np.ndarray[np.uint32_t, ndim=1] new_IDs

    length  = len(IDs);  length_bad = len(bad_IDs)
    new_IDs = np.zeros(length,dtype=IDs.dtype)  #define the new_IDs array

    for i in range(length):
        value = IDs[i]
        j_min = 0;  j_max = length_bad;  j = (j_min+j_max)//2
        while(value!=bad_IDs[j]):
            if value>bad_IDs[j]:   j_min = j
            else:                  j_max = j
            j = (j_min+j_max)//2
        new_IDs[i] = missing_IDs[j]

    return new_IDs


# this routine checks whether there is any repeated ID in the subfind file
# if there are repeated IDs it means that some particles are counted twice
# IDs ------------> array with the IDs to check
def repeated_IDs(np.ndarray[np.uint32_t, ndim=1] IDs):
    
    cdef np.ndarray[np.uint64_t, ndim=1] array
    cdef int i

    array = np.zeros(np.max(IDs)+1,dtype=np.uint64)

    for i in range(len(IDs)):
        array[IDs[i]] += 1

    if np.any(array>1):
        print('some particle is counted twice!!!\n');  sys.exit()
    else:
        print('no repeated IDs found\n')



# This routine takes all halos of a given region and find the galaxies within
# it. It also computes the HI mass of the particles belonging to those galaxies
# within R200 of the halo
# physics -----------------> 'CSF' or 'AGN'
# reg ---------------------> region over which perform the analysis
# TREECOOL_file -----------> TREECOOL file
def HI_in_galaxies(physics,reg,TREECOOL_file):

    #################################################
    # from the region obtain the snapshot fname and the halo file
    region         = Region(reg,physics)
    snapshot_fname = region.snapshot_fname
    halo_file      = region.halo_file
    subfind_file   = region.subfind_file

    # read halo file 
    data  = np.loadtxt(halo_file)
    pos_h = data[:,1:4]/1e3;  R_h  = data[:,17]/1e3;  R200    = data[:,24]/1e3
    M200  = data[:,18];       M500 = data[:,19];      M2500   = data[:,20]
    M_g   = data[:,13];       M_hg = data[:,14];      M_stars = data[:,15] 
    M_cg  = data[:,16];       contaminants = data[:,37]
    if np.any(contaminants!=0):
        print('contaminants founds!!!',contaminants);  sys.exit()
    #################################################

    #################################################
    # find the total number of particles in the simulation
    Nall   = readsnap.snapshot_header(snapshot_fname).nall
    Ntotal = np.sum(Nall,dtype=np.uint64)
    print('Total number of particles in the simulation =',Ntotal)

    # create a zero array and fill it with 1 using the normal IDs
    # We identify the missing IDs as the positions where the array is 0
    IDs_sorted = np.zeros(Ntotal,dtype=np.uint32)

    # find the IDs missing
    IDs = readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 #normalized
    #indexes = np.where((IDs>2150078190) & (IDs<2150078220))[0]
    #print IDs[indexes]
    print('Number of IDs =',len(IDs));  print(np.min(IDs),'< ID <',np.max(IDs))
    IDs_sorted[ IDs[np.where(IDs<Ntotal)[0]] ] = 1 #put 1 in normal ID positions
    missing_IDs = (np.where(IDs_sorted==0)[0]).astype(np.uint32)
    del IDs_sorted;  bad_IDs = IDs[np.where(IDs>=Ntotal)[0]];  del IDs
    #print 'bad IDs =',bad_IDs;  print 'missing IDs =',missing_IDs
    print('Number of particles with IDs longer than Ntotal =',len(missing_IDs))

    # sort the bad IDs
    bad_IDs = np.sort(bad_IDs);  print('bad IDs sorted =',bad_IDs,'\n')
    # we renumber the IDs, such as:
    # bad_IDs[0] --> missing_IDs[0]
    # bad_IDs[1] --> missing_IDs[1] ...etc
    #################################################

    #################################################
    # read subfind file
    groups_pos        = readsnap2.read_block(subfind_file,'GPOS')/1e3  #Mpc/h
    groups_M200       = readsnap2.read_block(subfind_file,'MCRI')*1e10 #Msun/h
    groups_R200       = readsnap2.read_block(subfind_file,'RCRI')/1e3  #Mpc/h
    #groups_M200       = readsnap2.read_block(subfind_file,'M200')*1e10 #Msun/h
    #groups_Mtot       = readsnap2.read_block(subfind_file,'MTOT')*1e10 #Msun/h
    galaxies_per_halo = readsnap2.read_block(subfind_file,'NSUB')
    galaxy_pos        = readsnap2.read_block(subfind_file,'SPOS')/1e3  #Mpc/h
    galaxy_number     = readsnap2.read_block(subfind_file,'SLEN')
    galaxy_offset     = readsnap2.read_block(subfind_file,'SOFF')
    galaxy_IDs        = readsnap2.read_block(subfind_file,'PID ')-1 #Normalized
    galaxy_mass       = readsnap2.read_block(subfind_file,'MSUB')*1e10  #Msun/h

    # renumber the IDs of the particles belonging to subhalos
    bad_IDs_gal_indexes = np.where(galaxy_IDs>=Ntotal)[0]
    bad_IDs_gal         = galaxy_IDs[bad_IDs_gal_indexes]
    if len(bad_IDs_gal)>0:
        galaxy_IDs[bad_IDs_gal_indexes] = \
            rename_IDs(bad_IDs_gal,bad_IDs,missing_IDs)
    del bad_IDs_gal_indexes, bad_IDs_gal
    print(np.min(galaxy_IDs),'< ID galaxy particles <',np.max(galaxy_IDs))

    # check that no IDs are repeated
    repeated_IDs(galaxy_IDs)
    #################################################

    #################################################
    # read the positions, masses and IDs of all the particles
    pos_part  = readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3  
    IDs_part  = readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1 
    mass_part = readsnap.read_block(snapshot_fname,"MASS",parttype=-1)*1e10 

    # renumber the wrong IDs
    bad_IDs_part_indexes = np.where(IDs_part>=Ntotal)[0]
    bad_IDs_part         = IDs_part[bad_IDs_part_indexes]
    if len(bad_IDs_part)>0:
        IDs_part[bad_IDs_part_indexes] = \
            rename_IDs(bad_IDs_part, bad_IDs, missing_IDs)
    del bad_IDs_part_indexes, bad_IDs_part

    # sort the positions of the particles
    pos_part_sorted = np.empty((Ntotal,3),dtype=np.float32)
    pos_part_sorted[IDs_part] = pos_part;  del pos_part

    # sort the masses of the particles
    mass_part_sorted = np.empty(Ntotal,dtype=np.float32)
    mass_part_sorted[IDs_part] = mass_part;  del mass_part, IDs_part
    #################################################

    #################################################
    # compute the HI masses of the gas particles
    HI_frac = Rahmati(snapshot_fname, TREECOOL_file, T_block=True,  #HI/H
                           Gamma_UVB=None, SF_temperature=1e4,
                           self_shielding_correction=True, correct_H2='KMT')
    mass_g = readsnap.read_block(snapshot_fname,"MASS",parttype=0)*1e10  #Msun/h
    pos_g  = readsnap.read_block(snapshot_fname,"POS ",parttype=0)/1e3  #Mpc/h
    M_HI   = 0.76*mass_g*HI_frac;  del HI_frac

    # read and renumber the IDs of the gas particles
    IDs_g = readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1  #normalized
    bad_IDs_g_indexes = np.where(IDs_g>=Ntotal)[0]
    bad_IDs_g         = IDs_g[bad_IDs_g_indexes]
    if len(bad_IDs_g)>0:
        IDs_g[bad_IDs_g_indexes] = rename_IDs(bad_IDs_g,bad_IDs,missing_IDs)
    del bad_IDs_g_indexes, bad_IDs_g

    # create a sorted array with the HI masses
    M_HI_sorted = np.zeros(Ntotal,dtype=np.float64)
    M_HI_sorted[IDs_g] = M_HI

    gas_mass = np.zeros(Ntotal,dtype=np.float64)
    gas_mass[IDs_g] = mass_g;  del IDs_g, mass_g
    #################################################

    #################################################
    # read the stellar masses of the particles
    mass_s = readsnap.read_block(snapshot_fname,"MASS",parttype=4)*1e10  #Msun/h

    # read and renumber the IDs of the star particles
    IDs_s  = readsnap.read_block(snapshot_fname,"ID  ",parttype=4)-1 #normalized
    bad_IDs_s_indexes = np.where(IDs_s>=Ntotal)[0]
    bad_IDs_s         = IDs_s[bad_IDs_s_indexes]
    if len(bad_IDs_s)>0:
        IDs_s[bad_IDs_s_indexes] = rename_IDs(bad_IDs_s,bad_IDs,missing_IDs)
    del bad_IDs_s_indexes, bad_IDs_s

    # create a sorted array with the stellar masses
    M_stars_sorted = np.zeros(Ntotal,dtype=np.float64)
    M_stars_sorted[IDs_s] = mass_s;  del IDs_s, mass_s
    #################################################
    

    #################################################
    # do a loop over all halos
    M_HI_halos          = np.zeros(len(R200),dtype=np.float64)
    M_HI_galaxies_halos = np.zeros(len(R200),dtype=np.float64)
    galaxies            = np.zeros(len(R200),dtype=np.int32) 
    galaxies_with_HI    = np.zeros(len(R200),dtype=np.int32) 
    galaxies_with_stars = np.zeros(len(R200),dtype=np.int32) 
    
    for i in range(len(R200)):

        ######## compute the HI mass of the halo ########
        distance = np.sqrt((pos_g[:,0]-pos_h[i,0])**2 + \
                           (pos_g[:,1]-pos_h[i,1])**2 + \
                           (pos_g[:,2]-pos_h[i,2])**2)

        indexes  = np.where(distance<R200[i])[0]
        HI_mass  = np.sum(M_HI[indexes],dtype=np.float64)
        del indexes, distance;  print('HI mass = %.4e'%HI_mass)
        #################################################

        ######## find the galaxies within R200 ########
        distance = galaxy_pos - pos_h[i]
        distance = np.sqrt(distance[:,0]**2+distance[:,1]**2+distance[:,2]**2)
        indexes  = np.where(distance<R200[i])[0]        
        number_galaxies = len(indexes);  del distance
        print('galaxies found within the halo:',number_galaxies)
        ###############################################

        ###### find the IDs particles in the galaxies within R200 ######
        length = 0   #find the total number of particles
        for j in indexes:  length += galaxy_number[j]
        print('particles belonging to galaxies within R200:',length)
        
        IDs_part_gal = np.empty(length,dtype=np.uint32);  offset = 0
        for j in indexes:
            size_gal     = galaxy_number[j]
            offset_gal   = galaxy_offset[j]
            gal_IDs      = galaxy_IDs[offset_gal:offset_gal+size_gal] 
            HI_galaxy    = np.sum(M_HI_sorted[gal_IDs],dtype=np.float64)
            mass_galaxy  = np.sum(mass_part_sorted[gal_IDs],dtype=np.float64)
            stellar_mass = np.sum(M_stars_sorted[gal_IDs],dtype=np.float64)
            gas_galaxy   = np.sum(gas_mass[gal_IDs],dtype=np.float64)
            if HI_galaxy>0.0:   galaxies_with_HI[i] += 1
            if stellar_mass>0:  galaxies_with_stars[i] += 1
            #if galaxy_pos[j,0]>502.7 and galaxy_pos[j,0]<503.2 and \
            #        galaxy_pos[j,1]>497.1 and galaxy_pos[j,1]<497.3:
            print('%3d ---> %.4e %.4e : %.4e %.4e %.4e'\
                %(j,mass_galaxy,galaxy_mass[j],gas_galaxy,
                  HI_galaxy,stellar_mass))
                #print galaxy_pos[j]
            #if np.absolute(mass_galaxy-galaxy_mass[j])/mass_galaxy>1e-5:
            #    sys.exit()
            IDs_part_gal[offset:offset+size_gal] = gal_IDs
            offset += size_gal
        del length, indexes
        ################################################################

        # find the positions of the particles within galaxies in R200
        gal_part_pos = pos_part_sorted[IDs_part_gal]

        # select the particles belonging to galaxies within R200 of the halo
        distance = gal_part_pos - pos_h[i]
        distance = np.sqrt(distance[:,0]**2+distance[:,1]**2+distance[:,2]**2)
        indexes = np.where(distance<R200[i])[0]  
        gal_part_pos = gal_part_pos[indexes]
        IDs_part_gal = IDs_part_gal[indexes]
        HI_mass_galaxies = np.sum(M_HI_sorted[IDs_part_gal],dtype=np.float64)
        print('HI mass in galaxies within R200 = %.4e'\
            %HI_mass_galaxies)


        M_HI_halos[i]          = HI_mass
        M_HI_galaxies_halos[i] = HI_mass_galaxies
        galaxies[i]            = number_galaxies

    return M200, M_HI_halos, M_HI_galaxies_halos, galaxies, galaxies_with_HI,\
        galaxies_with_stars, pos_h
