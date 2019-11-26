import numpy as np
import camb
import sys,os


################################## INPUT ######################################
# neutrino parameters
hierarchy = 'normal' #'degenerate', 'normal', 'inverted'
Mnu       = 0.06     #eV
Nnu       = 3        #number of massive neutrinos
Neff      = 3.046

# cosmological parameters
h       = 0.6711
Omega_c = 0.2685 - Mnu/(93.14*h**2)
Omega_b = 0.049
Omega_k = 0.0
tau     = None

# initial P(k) parameters
ns           = 0.9624
As           = 2.13e-9
pivot_scalar = 0.05
pivot_tensor = 0.05

# redshifts and k-range
redshifts    = [0.0, 0.5, 1, 2, 3, 99] 
kmax         = 10.0
k_per_logint = 10

# dz, relative difference dz/z to compute growths
dz = 0.01
###############################################################################

# create a new redshift list to compute growth rates
zs = []
for z in redshifts:
    dz_abs = (1.0+z)*dz
    if z==0.0:
        zs.append(z);  zs.append(z+dz_abs)
    else:
        zs.append(z-dz_abs);  zs.append(z);  zs.append(z+dz_abs)
z_list = redshifts;  redshifts = zs


Omega_cb = Omega_c + Omega_b

pars = camb.CAMBparams()

# set accuracy of the calculation
pars.set_accuracy(AccuracyBoost=5.0, lSampleBoost=5.0, 
                  lAccuracyBoost=5.0, HighAccuracyDefault=True, 
                  DoLateRadTruncation=True)

# set value of the cosmological parameters
pars.set_cosmology(H0=h*100.0, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, 
                   mnu=Mnu, omk=Omega_k, 
                   neutrino_hierarchy=hierarchy, 
                   num_massive_neutrinos=Nnu,
                   nnu=Neff,
                   tau=tau)
                   
# set the value of the primordial power spectrum parameters
pars.InitPower.set_params(As=As, ns=ns, 
                          pivot_scalar=pivot_scalar, 
                          pivot_tensor=pivot_tensor)

# set redshifts, k-range and k-sampling
pars.set_matter_power(redshifts=redshifts, kmax=kmax, 
                      k_per_logint=k_per_logint)

# compute results
results = camb.get_results(pars)

# get raw matter power spectrum and transfer functions with strange k-binning
#k, zs, Pk = results.get_linear_matter_power_spectrum()
#Tk        = (results.get_matter_transfer_data()).transfer_data

# interpolate to get Pmm, Pcc...etc
k, zs, Pkmm = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=7, var2=7, 
                                                have_power_spectra=True, 
                                                params=None)

k, zs, Pkcc = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=2, var2=2, 
                                                have_power_spectra=True, 
                                                params=None)

k, zs, Pkbb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=3, var2=3, 
                                                have_power_spectra=True, 
                                                params=None)

k, zs, Pkcb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=2, var2=3, 
                                                have_power_spectra=True, 
                                                params=None)

Pkcb = (Omega_c**2*Pkcc + Omega_b**2*Pkbb +\
        2.0*Omega_b*Omega_c*Pkcb)/Omega_cb**2

k, zs, Pknn = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                npoints=400, var1=6, var2=6, 
                                                have_power_spectra=True, 
                                                params=None)


print(pars)

# get sigma_8 and Hz in km/s/(kpc/h)
s8 = np.array(results.get_sigma8())
Hz = results.hubble_parameter(99.0)
print('H(z=99)      = %.4f km/s/(kpc/h)'%(Hz/1e3/h))
print('sigma_8(z=0) = %.4f'%s8[-1])


# do a loop over all redshifts
for i,z in enumerate(zs):

    fout1 = 'Pk_mm_z=%.3f.txt'%z
    fout2 = 'Pk_cc_z=%.3f.txt'%z
    fout3 = 'Pk_bb_z=%.3f.txt'%z
    fout4 = 'Pk_cb_z=%.3f.txt'%z
    fout5 = 'Pk_nn_z=%.3f.txt'%z

    np.savetxt(fout1,np.transpose([k,Pkmm[i,:]]))
    np.savetxt(fout2,np.transpose([k,Pkcc[i,:]]))
    np.savetxt(fout3,np.transpose([k,Pkbb[i,:]]))
    np.savetxt(fout4,np.transpose([k,Pkcb[i,:]]))
    np.savetxt(fout5,np.transpose([k,Pknn[i,:]]))


    #fout = 'Pk_trans_z=%.3f.txt'%z
    # notice that transfer functions have an inverted order:i=0 ==>z_max
    #np.savetxt(fout,np.transpose([Tk[0,:,i],Tk[1,:,i],Tk[2,:,i],Tk[3,:,i],
    #                               Tk[4,:,i],Tk[5,:,i],Tk[6,:,i]]))


# compute growth rates
for z in z_list:
    
    dz_abs = (1.0+z)*dz
    for suffix in ['mm','cb','nn']:

        fout = 'f%s_z=%.3f.txt'%(suffix,z)
        f2   = 'Pk_%s_z=%.3f.txt'%(suffix,z+dz_abs)

        if z==0.0:
            f1 = 'Pk_%s_z=%.3f.txt'%(suffix,z);  fac = 1.0
            
        else:
            f1 = 'Pk_%s_z=%.3f.txt'%(suffix,z-dz_abs);  fac = 2.0
            
        k1,Pk1 = np.loadtxt(f1,unpack=True)
        k2,Pk2 = np.loadtxt(f2,unpack=True)

        if np.any(k1!=k2):
            raise Exception('k values difer!!!')

        f = -0.5*(1.0+z)*np.log(Pk2/Pk1)/(fac*dz_abs)
        np.savetxt(fout,np.transpose([k1,f]))

        os.system('rm '+f2)
        if z!=0.0:  os.system('rm '+f1)
