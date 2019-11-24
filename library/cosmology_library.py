import numpy as np
import scipy.integrate as si
import sys,os,time

#################### FUNCTIONS USED TO COMPUTE INTEGRALS #####################

#we use this function to compute the comoving distance to a given redshift
def func(y,x,Omega_m,Omega_L):
    return [1.0/np.sqrt(Omega_m*(1.0+x)**3+Omega_L)]
##############################################################################



##############################################################################
#This functions computes the comoving distance to redshift z, in Mpc/h
#As input it needs z, Omega_m and Omega_L. It assumes a flat cosmology
def comoving_distance(z,Omega_m,Omega_L):
    H0=100.0 #km/s/(Mpc/h)
    c=3e5    #km/s

    #compute the comoving distance to redshift z
    yinit=[0.0]
    z_limits=[0.0,z]
    I=si.odeint(func,yinit,z_limits,args=(Omega_m,Omega_L),
                rtol=1e-8,atol=1e-8,mxstep=100000,h0=1e-6)[1][0]
    r=c/H0*I

    return r
##############################################################################

##############################################################################
def func_lgf(y,x,Omega_m,Omega_L):
    #print(x, 1.0/(x**3 * (np.sqrt(Omega_m/x**3 + Omega_L))**3))
    return 1.0/(x*np.sqrt(Omega_m/x**3 + Omega_L))**3

#This function computes the linear growth factor. See Eq. 1 of 0006089
#Notice that in that formula H(a) = (Omega_m/a^3+Omega_L)^1/2 and that the 
#growth is D(a), not g(a). We normalize it such as D(a=1)=1
def linear_growth_factor(z,Omega_m,Omega_L):
    
    # compute linear growth factor at z and z=0
    yinit = [0.0];  a_limits = [1e-30, 1.0/(1.0+z), 1.0/(1.0+0.0)]
    I = si.odeint(func_lgf,yinit,a_limits,args=(Omega_m,Omega_L),
                  rtol=1e-10,atol=1e-10,mxstep=100000,h0=1e-20)[1:]
    redshifts = np.array([ [z], [0.0] ])
    Ha = np.sqrt(Omega_m*(1.0+redshifts)**3 + Omega_L)
    D = (5.0*Omega_m/2.0)*Ha*I

    return D[0]/D[1]
##############################################################################


#This function computes the absoption distance:
#dX = H0*(1+z)^2/H(z)*dz
#Omega_m ----> value of the Omega_m cosmological parameter
#Omega_L ----> value of the Omega_L cosmological parameter
#z ----------> cosmological redshift
#BoxSize ----> size of the simulation box in Mpc/h
def absorption_distance(Omega_m,Omega_L,z,BoxSize):
    iterations=40; tol=1e-4; i=0; final=False
    dz_max=10.0; dz_min=0.0; dz=0.5*(dz_min+dz_max)
    r0=comoving_distance(z,Omega_m,Omega_L) #Mpc/h
    while not(final):
        dr=comoving_distance(z+dz,Omega_m,Omega_L)-r0
        if (np.absolute(dr-BoxSize)/BoxSize)<tol or i>iterations:
            final=True
        else:
            i+=1
            if dr>BoxSize:
                dz_max=dz
            else:
                dz_min=dz
            dz=0.5*(dz_min+dz_max)

    dX=(1.0+z)**2/np.sqrt(Omega_m*(1.0+z)**3+Omega_L)*dz
    return dX
##############################################################################

# This routine implements the Takahashi 2012 halofit formula (1208.2701)
# we have neglected the terms with dark energy variation (Eqs. A6 and A7)
# Omega_m ---------------> value of Omega_m at z=0
# Omega_l ---------------> value of Omega_l at z=0
# z ---------------------> redshift
# k_lin,Pk_lin ----------> linear power spectrum at z=0
# for redshifts different to 0 the code computes the growth factor and 
# rescale it. It returns the non-linear P(k) at redshift z. We have checked
# that for a Planck cosmology it reproduces CAMB within 0.6% up to z=5
def Halofit_12(Omega_m,Omega_l,z,k_lin,Pk_lin):

    # compute growth factor at redshift z and rescale P(k)
    Dz = linear_growth_factor(z,Omega_m,Omega_l)
    Pk_lin = Pk_lin*Dz**2
    
    ######### find the value of k_sigma #########
    Rmin,Rmax = 0.01, 10.0  #Mpc/h
    found = False;  precision = 1e-5
    while not(found):
        R = 0.5*(Rmin + Rmax)
        
        yinit = [0.0];  k_limits = [k_lin[0],k_lin[-1]]
        sigma2 = si.odeint(sigma_func, yinit, k_limits, args=(k_lin,Pk_lin,R), 
                           mxstep=1000000, rtol=1e-8, atol=1e-21, 
                           h0=1e-10)[1][0]
        sigma2 = sigma2/(2.0*np.pi**2)

        if abs(sigma2-1.0)<precision:  found = True
        elif sigma2>1.0:  Rmin = R
        else:             Rmax = R
    k_sigma = 1.0/R  #h/Mpc
    #############################################

    ####### compute value of neff and C #########
    # notice that we are doing dlog sigma^2/dlnR = 
    # log(sigma^2(log(R)+log(h))) - log(sigma^2(log(R))) / log(h) = 
    # log(sigma^2(log(R*h))) - log(sigma^2(log(R))) / log(h)
    h = 1.05
    Rp = R*h;  yinit = [0.0];  k_limits = [k_lin[0],k_lin[-1]]
    sigma2p = si.odeint(sigma_func, yinit, k_limits, args=(k_lin,Pk_lin,Rp), 
                        mxstep=1000000, rtol=1e-8, atol=1e-21, h0=1e-10)[1][0]
    sigma2p = sigma2p/(2.0*np.pi**2)

    Rm = R/h;  yinit = [0.0];  k_limits = [k_lin[0],k_lin[-1]]
    sigma2m = si.odeint(sigma_func, yinit, k_limits, args=(k_lin,Pk_lin,Rm), 
                        mxstep=1000000, rtol=1e-8, atol=1e-21, h0=1e-10)[1][0]
    sigma2m = sigma2m/(2.0*np.pi**2)

    # sanity check
    if abs(sigma2p-sigma2)<1e3*precision or abs(sigma2-sigma2m)<1e3*precision:
        print('value of h too small for given precision');  sys.exit()

    neff = -(np.log(sigma2p)-np.log(sigma2m))/(2.0*np.log(h)) - 3.0
    C = -(np.log(sigma2p) - 2.0*np.log(sigma2) + np.log(sigma2m))/np.log(h)**2
    #############################################

    ################ constants ##################
    Omegamz = Omega_m*(1.0+z)**3/(Omega_m*(1.0+z)**3 + Omega_l)

    f1 = Omegamz**(-0.0307);  f2 = Omegamz**(-0.0585);  f3 = Omegamz**(0.0743)

    an = 10**(1.5222 + 2.8553*neff + 2.3706*neff**2 + 0.9903*neff**3 + \
                  0.2250*neff**4 - 0.6038*C)
    bn = 10**(-0.5642 + 0.5864*neff + 0.5716*neff**2 - 1.5474*C)
    cn = 10**(0.3698 + 2.0404*neff + 0.8161*neff**2 + 0.5869*C)
    gamman = 0.1971 - 0.0843*neff + 0.8460*C
    alphan = abs(6.0835 + 1.3373*neff - 0.1959*neff**2 - 5.5274*C)
    betan  = 2.0379 - 0.7354*neff + 0.3157*neff**2 + 1.2490*neff**3 + \
        0.3980*neff**4 - 0.1682*C
    mun = 0.0
    nun = 10**(5.2105 + 3.6902*neff)
    ############################################


    Pk_hf  = np.zeros(len(k_lin),dtype=np.float64)
    for i,k in enumerate(k_lin):

        # dimensionless linear power spectrum
        delta2_lin = k**3*Pk_lin[i]/(2.0*np.pi**2)

        y = (k/k_sigma);  fy = y/4.0 + y**2/8.0

        # two-halo term
        delta2_Q = delta2_lin*(1.0 + delta2_lin)**betan/\
            (1.0 + alphan*delta2_lin)*np.exp(-fy)

        # one-halo term
        delta2_HH = an*y**(3.0*f1)/(1.0 + bn*y**f2 + (cn*f3*y)**(3.0-gamman))
        delta2_H  = delta2_HH/(1.0 + mun/y +  nun/y**2)

        # total non-linear dimensionless power spectrum
        delta2_hf = delta2_Q + delta2_H

        # non-linear power spectrum
        Pk_hf[i]  = (2*np.pi**2)*delta2_hf/k**3

    return Pk_hf
    #############################################

def sigma_func(y,x,k,Pk,R):
    return [np.interp(x,k,Pk)*x**2*np.exp(-x**2*R**2)]
##############################################################################







###############################################################################
################################### USAGE #####################################
###############################################################################

###### comoving distance ######
"""
z=3.0
Omega_m=0.3
Omega_L=0.7

r=comoving_distance(z,Omega_m,Omega_L)
print('comoving distance to z = %2.2f ---> %f Mpc/h'%(z,r))
"""

###### linear growth factor ######
"""
z       = 1.0
Omega_m = 0.308
Omega_l = 0.692
h       = 0.6781

Da = linear_growth_factor(z,Omega_m,Omega_l,h)
print('Linear growth factor at z = %.1f : %.3e'%(z,Da))
"""

###### absorption distance ######
"""
Omega_m=0.274247
Omega_L=0.725753
z=3.0
BoxSize=60.0 #Mpc/h

dX=absorption_distance(Omega_m,Omega_L,z,BoxSize)
print('dX =',dX)
"""

###### halofit P(k) ######
"""
Omega_m = 0.3175
Omega_l = 0.6825
z       = 1.0
k_lin,Pk_lin = np.loadtxt('ics_matterpow_0.dat',unpack=True)
Pk_hf = Halofit_12(Omega_m,Omega_l,z,k_lin,Pk_lin)
"""
