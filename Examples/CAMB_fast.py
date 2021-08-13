import numpy as np
import CAMB_library as CL

################################## INPUT ######################################
# neutrino parameters
hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
Mnu       = 0.00         #eV
Nnu       = 3            #number of massive neutrinos
Neff      = 3.046

# cosmological parameters
h       = 0.6711
Omega_m = 0.3175
Omega_b = 0.049
Omega_k = 0.0
s8      = 0.834
tau     = None

# initial P(k) parameters
ns           = 0.9624
As           = 2.13e-9
pivot_scalar = 0.05
pivot_tensor = 0.05

# redshifts and k-range
redshifts    = [0.0, 0.5, 1, 2, 3, 127] 
kmax         = 10.0
k_per_logint = 10

# CAMB accuracy
AccuracyBoost  = 5.0 
lSampleBoost   = 5.0 
lAccuracyBoost = 5.0
###############################################################################

result = CL.PkL(Omega_m=Omega_m, Omega_b=Omega_b, h=h, ns=ns, s8=s8,
                Mnu=Mnu, As=As, Omega_k=Omega_k, 
                pivot_scalar=pivot_scalar, pivot_tensor=pivot_tensor, 
                Nnu=Nnu, hierarchy=hierarchy, Neff=Neff, tau=tau,
                redshifts=redshifts, kmax=kmax, k_per_logint=k_per_logint,
                AccuracyBoost=AccuracyBoost, lSampleBoost=lSampleBoost,
                lAccuracyBoost=lAccuracyBoost, verbose=False)

zs  = result.z
k   = result.k
Pkm = result.Pkmm
Tk  = result.Tk

for i,z in enumerate(zs):
    np.savetxt('Pk_m_z=%.1f.txt'%z, np.transpose([k, Pkm[i]]))
    np.savetxt('Tk_z=%.1f.txt'%z, np.transpose(Tk[:,:,i]))
