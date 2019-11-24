import numpy as np
import sys,os
import mass_function_library as MFL
import units_library as UL
import integration_library as IL

############################ CONSTANTS ############################
pi       = np.pi
rho_crit = (UL.units()).rho_crit #h^2 Msun/Mpc^3
deltac   = 1.686
###################################################################

##############################################################################
# This functions computes the halo bias, b(M), in a set of different halo masses
# k_in --------> input k
# Pk_in -------> input Pk
# OmegaM ------> value of Omega_m
# Masses ------> array with the halo masses where to evaluate the halo bias
# author ------> SMT01', 'Tinker'
def bias(k_in, Pk_in, OmegaM, Masses, author, bins=10000):

    rhoM = rho_crit*OmegaM

    # sort the input k-array
    indexes = np.argsort(k_in)
    k_in = k_in[indexes];  Pk_in = Pk_in[indexes]
    
    # interpolate the input k-array into bins points
    k  = np.logspace(np.log10(k_in[0]), np.log10(k_in[-1]), bins)
    Pk = np.interp(np.log(k), np.log(k_in), Pk_in)
    
    bias = np.zeros(Masses.shape[0], dtype=np.float64)

    if author=="SMT01":
        a = 0.707;  b = 0.5;  c = 0.6

        for i in range(Masses.shape[0]):
            R = (3.0*Masses[i]/(4.0*pi*rhoM))**(1.0/3.0)
            anu = a*(deltac/MFL.sigma(k,Pk,R))**2

            bias[i] = np.sqrt(a)*anu + np.sqrt(a)*b*anu**(1.0-c)
            bias[i] = bias[i] - anu**c/(anu**c + b*(1.0-c)*(1.0-0.5*c))
            bias[i] = 1.0+bias[i]/(np.sqrt(a)*deltac)        
    
    elif author=="Tinker":
        Delta=200.0;  y=np.log10(Delta)

        A=1.0+0.24*y*np.exp(-(4.0/y)**4);          a=0.44*y-0.88
        B=0.183;                                   b=1.5
        C=0.019+0.107*y+0.19*np.exp(-(4.0/y)**4);  c=2.4

        for i in range(Masses.shape[0]):
            R = (3.0*Masses[i]/(4.0*pi*rhoM))**(1.0/3.0)
            nu = deltac/MFL.sigma(k,Pk,R)
            bias[i] = 1.0-A*nu**a/(nu**a+deltac**a)+B*nu**b+C*nu**c

    return bias

##############################################################################
# This function returns the effective bias, see eq. 15 of Marulli 2011
def bias_eff(k, Pk, OmegaM, Masses, z, author):

    # compute halo mass function
    if author=='SMT01':
        dndM = MFL.MF_theory(k,Pk,OmegaM,Masses,'ST',bins=10000,z=z,delta=200.0)

    if author=='Tinker':
        dndM = MFL.MF_theory(k,Pk,OmegaM,Masses,'Tinker',bins=10000,z=z,delta=200.0)

    # compute halo bias
    b = bias(k, Pk, OmegaM, Masses, author, bins=10000)

    # integration parameters
    eps, h1, hmin = 1e-10, 1e0, 0.0

    yinit = np.array([0.0], dtype=np.float64)
    numerator   = IL.odeint_example2(yinit, np.min(Masses), np.max(Masses), eps,
                                     h1, hmin, Masses, dndM*b, verbose=False)[0]

    yinit = np.array([0.0], dtype=np.float64)
    denominator = IL.odeint_example2(yinit, np.min(Masses), np.max(Masses), eps,
                                     h1, hmin, Masses, dndM, verbose=False)[0]

    bias_eff = numerator/denominator

    return bias_eff








    



