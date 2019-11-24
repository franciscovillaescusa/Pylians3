import numpy as np
#import scipy.integrate as si
import integration_library as IL
import readfof, readsubf
import os, sys
import units_library as UL


############################ CONSTANTS ############################
pi       = np.pi
rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3
deltac   = 1.686
###################################################################


# This is the main driver routine to compute the halo mass function from several
# fitting formulae.
# k_in ---------> input k
# Pk_in --------> input Pk
# OmegaM -------> value of Omega_m
# Masses -------> array with the masses where to compute the HMF
# author -------> 'ST','Tinker','Tinker10','Crocce','Jenkins','Warren',
#                 'Watson','Watson_FoF','Angulo'
# bins ---------> In order to carry out the integral, we need a k-array sorted and
#                 equally spaced in log10. This sets the number of bins in that array
# z ------------> for 'Tinker', 'Tinker10' and Crocce
# delta --------> for 'Tinker' and 'Tinker10'
def MF_theory(k_in, Pk_in, OmegaM, Masses, author, bins=10000, z=0, delta=200.0):

    rhoM = rho_crit*OmegaM

    # sort the input k-array
    indexes = np.argsort(k_in)
    k_in = k_in[indexes];  Pk_in = Pk_in[indexes]
    
    # interpolate the input k-array into bins points
    k  = np.logspace(np.log10(k_in[0]), np.log10(k_in[-1]), bins)
    Pk = np.interp(np.log(k), np.log(k_in), Pk_in)
    
    if author=='ST':            MF = ST_mass_function(k, Pk, rhoM, Masses)
    elif author=="Tinker":      MF = Tinker_mass_function(k, Pk, rhoM, Masses,
                                                          z, delta)
    elif author=="Tinker10":    MF = Tinker_2010_mass_function(k, Pk, rhoM, Masses,
                                                               z, delta)
    elif author=="Crocce":      MF = Crocce_mass_function(k, Pk, rhoM, Masses, z)
    elif author=="Jenkins":     MF = Jenkins_mass_function(k, Pk, rhoM, Masses)
    elif author=="Warren":      MF = Warren_mass_function(k, Pk, rhoM, Masses)
    elif author=="Watson":      MF = Watson_mass_function(k, Pk, rhoM, Masses)
    elif author=="Watson_FOF":  MF = Watson_mass_function_FoF(k, Pk, rhoM, Masses)
    elif author=="Angulo":      MF = Angulo_subhalos_mass_function(k, Pk, rhoM, Masses)
    else:  raise Exception("%s model not implemented!!!"%author)

    return MF

    

"""
###### old method to compute integrals ######
#derivative function for integrating sigma(R)
def deriv_sigma(y,x,k,Pk,R):
    Pkp=np.interp(np.log10(x),np.log10(k),np.log10(Pk)); Pkp=10**Pkp
    kR=x*R
    W=3.0*(np.sin(kR)-kR*np.cos(kR))/kR**3 
    return [x**2*Pkp*W**2]

#this function computes sigma(R)
def sigma_old(k,Pk,R):
    k_limits=[k[0],k[-1]]; yinit=[0.0]

    I=si.odeint(deriv_sigma,yinit,k_limits,args=(k,Pk,R),
                rtol=1e-8,atol=1e-8,
                mxstep=10000000)[1][0]/(2.0*pi**2)

    return np.sqrt(I)
"""

def sigma(k,Pk,R):
    yinit = np.array([0.0], dtype=np.float64)
    eps   = 1e-13  #change this for higher/lower accuracy
    h1    = 1e-12
    hmin  = 0.0
    
    W   = 3.0*(np.sin(k*R) - k*R*np.cos(k*R))/(k*R)**3
    Pk1 = Pk*W**2*k**2/(2.0*pi**2)
    
    return np.sqrt(IL.odeint(yinit, k[0], k[-1], eps,
                             h1, hmin, np.log10(k), Pk1,
                             'sigma', verbose=False)[0])

#this function computes the derivate of sigma(M) wrt M
def dSdM(k, Pk, rhoM, M):

    R1=(3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
    s1=sigma(k,Pk,R1)

    M2=M*1.0001
    R2=(3.0*M2/(4.0*pi*rhoM))**(1.0/3.0)
    s2=sigma(k,Pk,R2)

    return (s2-s1)/(M2-M)

##############################################################################
#This function computes the Sheth-Tormen mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def ST_mass_function(k, Pk, rhoM, Masses):

    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        nu  = (1.686/sigma(k,Pk,R))**2
        nup = 0.707*nu

        dndM[i]=-2.0*(rhoM/M)*dSdM(k,Pk,rhoM,M)/sigma(k,Pk,R)
        dndM[i]*=0.3222*(1.0+1.0/nup**0.3)*np.sqrt(0.5*nup)
        dndM[i]*=np.exp(-0.5*nup)/np.sqrt(pi)

    return dndM

##############################################################################
#This function computes the Tinker mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Tinker_mass_function(k, Pk, rhoM, Masses, z, delta=200.0):

    alpha=10**(-(0.75/np.log10(delta/75.0))**1.2)

    D=[200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3600.0]
    A=[0.186, 0.200, 0.212, 0.218, 0.248, 0.255,  0.260,  0.260,  0.260]
    a=[1.47,  1.52,  1.56,  1.61,  1.87,  2.13,   2.30,   2.53,   2.66]
    b=[2.57,  2.25,  2.05,  1.87,  1.59,  1.51,   1.46,   1.44,   1.41]
    c=[1.19,  1.27,  1.34,  1.45,  1.58,  1.80,   1.97,   2.24,   2.44]
    D=np.array(D); A=np.array(A); a=np.array(a); b=np.array(b); c=np.array(c)

    A=np.interp(delta,D,A)
    a=np.interp(delta,D,a)
    b=np.interp(delta,D,b)
    c=np.interp(delta,D,c)

    # this is for R200_critical with OmegaM=0.2708
    A*=(1.0+z)**(-0.14)
    a*=(1.0+z)**(-0.06)
    b*=(1.0+z)**(-alpha)

    print('Delta=',delta)
    print('A=',A)
    print('a=',a)
    print('b=',b)
    print('c=',c)
    
    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   =(3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = A*((b/s)**(a)+1.0)*np.exp(-c/s**2)

        dndM[i] = -(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM

##############################################################################
def Tinker_2010_mass_function(k, Pk, rhoM, Masses, z, delta=200.0):

    if delta!=200.0:
        raise Exception("only implemented delta=200, please update library")

    alpha  = 0.368
    beta0  = 0.589;       beta  = beta0*(1.0+z)**0.20
    gamma0 = 0.864;       gamma = gamma0*(1.0+z)**(-0.01)
    phi0   = -0.729;      phi   = phi0*(1.0+z)**(-0.08)
    eta0   = -0.243;      eta   = eta0*(1.0+z)**0.27
    
    dndM = np.empty(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        nu = 1.686/sigma(k,Pk,R)
        fnu = alpha*(1.0+(beta*nu)**(-2.0*phi))*nu**(2.0*eta)*\
            np.exp(-gamma*nu**2/2.0)

        Mp  = M*1.01;  Rp = (3.0*Mp/(4.0*pi*rhoM))**(1.0/3.0)
        nup = 1.686/sigma(k,Pk,Rp)
        dlnnu_dlnM = np.log(nup-nu)/np.log(Mp-M)

        dndM[i] = -(rhoM/M**2)*(nu*fnu)*dlnnu_dlnM

    return dndM
    


##############################################################################
#This function computes the Crocce mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Crocce_mass_function(k, Pk, rhoM, Masses, z):

    A=0.58*(1.0+z)**(-0.13) 
    a=1.37*(1.0+z)**(-0.15)
    b=0.3*(1.0+z)**(-0.084)
    c=1.036*(1.0+z)**(-0.024)
    
    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = A*(s**(-a)+b)*np.exp(-c/s**2)

        dndM[i]=-(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM

##############################################################################
#This function computes the Jenkins mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Jenkins_mass_function(k, Pk, rhoM, Masses):

    A=0.315
    b=0.61
    c=3.8
    
    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = A*np.exp(-np.absolute(np.log(1.0/s)+b)**c)

        dndM[i] = -(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM

##############################################################################
#This function computes the Warren mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Warren_mass_function(k, Pk, rhoM, Masses):

    A=0.7234
    a=1.625
    b=0.2538
    c=1.1982
    
    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = A*(s**(-a)+b)*np.exp(-c/s**2)

        dndM[i] = -(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM

##############################################################################
#This function computes the Warren mass function for FoF halos. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Watson_mass_function_FoF(k, Pk, rhoM, Masses):

    A=0.282
    a=2.163
    b=1.406
    c=1.210
    
    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = A*((b/s)**a+1.0)*np.exp(-c/s**2)

        dndM[i] = -(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM

##############################################################################
#This function computes the Warren mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Watson_mass_function(k, Pk, rhoM, Masses):

    delta=200.0
    OmegaM =rhoM/rho_crit

    A=0.194
    a=1.805
    b=2.267
    c=1.287

    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = A*(s**(-a)+b)*np.exp(-c/s**2)

        factor=np.exp(0.023*(delta/178.0-1.0))
        factor*=(delta/178.0)**(-0.456*OmegaM-0.139)
        factor*=np.exp(0.072*(1-delta/178.0)/s**2.130)

        f_s*=factor

        dndM[i] = -(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM

#############################################################################
#This function computes the Warren mass function. Returns dn/dM
#If the mass function wants to be computed in a given mass bins, use Masses
def Angulo_subhalos_mass_function(k, Pk, rhoM, Masses):
    
    dndM = np.zeros(Masses.shape[0], dtype=np.float64)
        
    for i,M in enumerate(Masses):
        R   = (3.0*M/(4.0*pi*rhoM))**(1.0/3.0)
        s   = sigma(k,Pk,R)
        f_s = 0.265*(1.675/s+1.0)**1.9*np.exp(-1.4/s**2)

        dndM[i] = -(rhoM/M)*dSdM(k,Pk,rhoM,M)*f_s/s

    return dndM


#############################################################################
#This functions computes the halo mass function (dn/dM (M)) of a given object
#and write the results to a file. The arguments are:
#groups_fname ---> folder containing the halos/subhalos files
#groups_number --> number of the snapshot (e.g. 1, 2, 22...)
#f_out ----------> the name of the output file
#min_mass -------> the lower limit of the mass function (units of 1e10 Msun/h)
#min_mass -------> the upper limit of the mass function (units of 1e10 Msun/h)
#bins -----------> the number of bins used to compute the mass function
#BoxSize --------> Size of the simulation box (in Mpc/h)
#obj ------------> the object over which compute the mass function:
#   *FoF halos -----------> 'FoF'
#   *SO 200xmean halos ---> 'halos_m200'
#   *subhalos -------- ---> 'subhalos'
#long_IDs_flag -> flag for long IDs (True or False)
#SFR_flag ------> flag for SFR (True or False)
#When using FoF halos their masses are corrected to account for sampling effects
#if min_mass=None the code will take the minimum mass of the halo catalogue
#the same applies to max_mass
def mass_function(groups_fname,groups_number,obj,BoxSize,bins,f_out,
                  min_mass=None,max_mass=None,
                  long_ids_flag=True,SFR_flag=False):

    #bins_mass=np.logspace(np.log10(min_mass),np.log10(max_mass),bins+1)
    #mass_mean=10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
    #dM=bins_mass[1:]-bins_mass[:-1]

    if obj=='FoF':
        #read FoF halos information
        fof=readfof.FoF_catalog(groups_fname,groups_number,
                                long_ids=long_ids_flag,swap=False,SFR=SFR_flag)
        F_pos=fof.GroupPos/1e3        #positions in Mpc/h
        F_mass=fof.GroupMass*1e10     #masses in Msun/h
        F_part=fof.GroupLen           #number particles belonging to the group
        F_Mpart=F_mass[0]/F_part[0]   #mass of a single particle in Msun/h
        del fof

        #some verbose
        print('\nNumber of FoF halos=',len(F_pos))
        print('%f < X_fof < %f'%(np.min(F_pos[:,0]),np.max(F_pos[:,0])))
        print('%f < Y_fof < %f'%(np.min(F_pos[:,1]),np.max(F_pos[:,1])))
        print('%f < Z_fof < %f'%(np.min(F_pos[:,2]),np.max(F_pos[:,2])))
        print('%e < M_fof < %e\n'%(np.min(F_mass),np.max(F_mass)))

        #Correct the masses of the FoF halos
        F_mass=F_Mpart*(F_part*(1.0-F_part**(-0.6)))

        #compute the minimum and maximum mass
        if min_mass==None:
            min_mass=np.min(F_mass)
        if max_mass==None:
            max_mass=np.max(F_mass)

        print('M_min = %e'%(min_mass))
        print('M_max = %e\n'%(max_mass))

        #find the masses and mass intervals
        bins_mass=np.logspace(np.log10(min_mass),np.log10(max_mass),bins+1)
        mass_mean=10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
        dM=bins_mass[1:]-bins_mass[:-1]

        #compute the number of halos within each mass interval
        number=np.histogram(F_mass,bins=bins_mass)[0]
        print(number); print(np.sum(number,dtype=np.float64))

    elif obj=='halos_m200':
        #read CDM halos information
        halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                       group_veldisp=True,masstab=True,
                                       long_ids=True,swap=False)

        H_pos=halos.group_pos/1e3               #positions in Mpc/h
        H_mass=halos.group_m_mean200*1e10       #masses in Msun/h
        H_radius_m=halos.group_r_mean200/1e3    #radius in Mpc/h
        del halos

        #some verbose
        print('\nNumber of halos=',len(H_pos))
        print('%f < X < %f'%(np.min(H_pos[:,0]),np.max(H_pos[:,0])))
        print('%f < Y < %f'%(np.min(H_pos[:,1]),np.max(H_pos[:,1])))
        print('%f < Z < %f'%(np.min(H_pos[:,2]),np.max(H_pos[:,2])))
        print('%e < M < %e\n'%(np.min(H_mass),np.max(H_mass)))

        #compute the minimum and maximum mass
        if min_mass==None:
            min_mass=np.min(H_mass)
        if max_mass==None:
            max_mass=np.max(H_mass)

        #compute the number of halos within each mass interval
        bins_mass=np.logspace(np.log10(min_mass),np.log10(max_mass),bins+1)
        mass_mean=10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
        dM=bins_mass[1:]-bins_mass[:-1]

        number=np.histogram(H_mass,bins=bins_mass)[0]
        print(number); print(np.sum(number,dtype=np.float64))

    elif obj=='halos_c200':
        #read CDM halos information
        halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                       group_veldisp=True,masstab=True,
                                       long_ids=True,swap=False)

        H_pos=halos.group_pos/1e3               #positions in Mpc/h
        H_mass=halos.group_m_crit200*1e10       #masses in Msun/h
        H_radius_m=halos.group_r_crit200/1e3    #radius in Mpc/h
        del halos

        #some verbose
        print('\nNumber of halos=',len(H_pos))
        print('%f < X < %f'%(np.min(H_pos[:,0]),np.max(H_pos[:,0])))
        print('%f < Y < %f'%(np.min(H_pos[:,1]),np.max(H_pos[:,1])))
        print('%f < Z < %f'%(np.min(H_pos[:,2]),np.max(H_pos[:,2])))
        print('%e < M < %e\n'%(np.min(H_mass),np.max(H_mass)))

        #compute the minimum and maximum mass
        if min_mass==None:
            min_mass=np.min(H_mass)
        if max_mass==None:
            max_mass=np.max(H_mass)

        #compute the number of halos within each mass interval
        bins_mass=np.logspace(np.log10(min_mass),np.log10(max_mass),bins+1)
        mass_mean=10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
        dM=bins_mass[1:]-bins_mass[:-1]

        number=np.histogram(H_mass,bins=bins_mass)[0]
        print(number); print(np.sum(number,dtype=np.float64))

    elif obj=='subhalos':
        #read CDM halos information
        halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                       group_veldisp=True,masstab=True,
                                       long_ids=True,swap=False)

        S_pos=halos.sub_pos/1e3            #positions in Mpc/h
        S_mass=halos.sub_mass*1e10         #masses in Msun/h
        del halos

        #some verbose
        print('\nNumber of subhalos=',len(S_pos))
        print('%f < X < %f'%(np.min(S_pos[:,0]),np.max(S_pos[:,0])))
        print('%f < Y < %f'%(np.min(S_pos[:,1]),np.max(S_pos[:,1])))
        print('%f < Z < %f'%(np.min(S_pos[:,2]),np.max(S_pos[:,2])))
        print('%e < M < %e\n'%(np.min(S_mass),np.max(S_mass)))

        #compute the minimum and maximum mass
        if min_mass==None:
            min_mass=np.min(S_mass)
        if max_mass==None:
            max_mass=np.max(S_mass)

        #compute the number of halos within each mass interval
        bins_mass=np.logspace(np.log10(min_mass),np.log10(max_mass),bins+1)
        mass_mean=10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))
        dM=bins_mass[1:]-bins_mass[:-1]

        number=np.histogram(S_mass,bins=bins_mass)[0]
        print(number); print(np.sum(number,dtype=np.float64))

    else:
        print('bad object type selected')
        sys.exit()

    MF=number/(dM*BoxSize**3)
    delta_MF=np.sqrt(number)/(dM*BoxSize**3)

    f=open(f_out,'w')
    for i in range(bins):
        f.write(str(mass_mean[i])+' '+str(MF[i])+' '+str(delta_MF[i])+'\n')
    f.close()

#############################################################################

#This functions computes the function f(sigma) see eq. 1 of Crocce et al 2010
#for a given object, and write the results to a file. The arguments are:
#groups_fname ---> folder containing the halos/subhalos files
#groups_number --> number of the snapshot (e.g. 1, 2, 22...)
#f_out ----------> the name of the output file
#min_mass -------> the lower limit of the mass function (units of 1e10 Msun/h)
#min_mass -------> the upper limit of the mass function (units of 1e10 Msun/h)
#bins -----------> the number of bins used to compute the mass function
#BoxSize --------> Size of the simulation box (in Mpc/h)
#obj ------------> the object over which compute the mass function:
#   *FoF halos -----------> 'FoF'
#   *SO 200xmean halos ---> 'halos_m200'
#Omega_M -------> This is the value of Omega_M = Omega_CDM+Omega_B(+Omega_nu)
#k, Pk ---------> Those are the values of the DM P(k) or the CDM P(k) from CAMB
#When using FoF halos their masses are corrected to account for sampling effects

def mass_function_fsigma(groups_fname,groups_number,f_out,min_mass,max_mass,
                         bins,BoxSize,obj,Omega_M,k,Pk):

    rhoM=rho_crit*Omega_M
    bins_mass=np.logspace(np.log10(min_mass),np.log10(max_mass),bins+1)
    mass_mean=10**(0.5*(np.log10(bins_mass[1:])+np.log10(bins_mass[:-1])))

    if obj=='FoF':
        #read FoF halos information
        fof=readfof.FoF_catalog(groups_fname,groups_number,
                                long_ids=True,swap=False)
        F_pos=fof.GroupPos/1e3        #positions in Mpc/h
        F_mass=fof.GroupMass*1e10     #masses in Msun/h
        F_part=fof.GroupLen           #number particles belonging to the group
        F_Mpart=F_mass[0]/F_part[0]   #mass of a single particle in Msun/h
        del fof

        #Correct the masses of the FoF halos
        F_mass=F_Mpart*(F_part*(1.0-F_part**(-0.6)))

        #some verbose
        print('Number of FoF halos=',len(F_pos))
        print(np.min(F_pos[:,0]),'< X_fof <',np.max(F_pos[:,0]))
        print(np.min(F_pos[:,1]),'< Y_fof <',np.max(F_pos[:,1]))
        print(np.min(F_pos[:,2]),'< Z_fof <',np.max(F_pos[:,2]))
        print(np.min(F_mass),'< M_fof <',np.max(F_mass))

        number=np.histogram(F_mass,bins=bins_mass)[0]
        print(number); print(np.sum(number,dtype=np.float64))

    elif obj=='halos_m200':
        #read CDM halos information
        halos=readsubf.subfind_catalog(groups_fname,groups_number,
                                       group_veldisp=True,masstab=True,
                                       long_ids=True,swap=False)

        H_pos=halos.group_pos/1e3               #positions in Mpc/h
        H_mass=halos.group_m_mean200*1e10       #masses in Msun/h
        H_radius_m=halos.group_r_mean200/1e3    #radius in Mpc/h
        del halos

        #some verbose
        print('Number of halos=',len(H_pos))
        print(np.min(H_pos[:,0]),'< X_fof <',np.max(H_pos[:,0]))
        print(np.min(H_pos[:,1]),'< Y_fof <',np.max(H_pos[:,1]))
        print(np.min(H_pos[:,2]),'< Z_fof <',np.max(H_pos[:,2]))
        print(np.min(H_mass),'< M_fof <',np.max(H_mass))

        number=np.histogram(H_mass,bins=bins_mass)[0]
        print(number); print(np.sum(number,dtype=np.float64))

    else:
        print('bad object type selected')
        sys.exit()

    sigma_mean=np.empty(bins); f_sigma=np.empty(bins)
    delta_f_sigma=np.empty(bins)
    for i in range(bins):
        R1=(3.0*bins_mass[i]/(4.0*pi*rhoM))**(1.0/3.0)
        sigma1=sigma(k,Pk,R1)

        R2=(3.0*bins_mass[i+1]/(4.0*pi*rhoM))**(1.0/3.0)
        sigma2=sigma(k,Pk,R2)

        sigma_mean[i]=0.5*(sigma2+sigma1)
        f_sigma[i]=(number[i]/np.log(sigma2/sigma1))/BoxSize**3
        f_sigma[i]=-(mass_mean[i]/rhoM)*f_sigma[i]
        delta_f_sigma[i]=(np.sqrt(number[i])/np.log(sigma2/sigma1))/BoxSize**3
        delta_f_sigma[i]=-(mass_mean[i]/rhoM)*delta_f_sigma[i]

    f=open(f_out,'w')
    for i in range(bins):
        f.write(str(sigma_mean[i])+' '+str(f_sigma[i])+' '+str(delta_f_sigma[i])+' '+str(mass_mean[i])+'\n')
    f.close()








