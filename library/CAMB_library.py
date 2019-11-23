import numpy as np
import camb
import sys,os

# This routine computes the linear power spectra using CAMB given the input
# cosmological parameters. To do the rescaling with s8 we always need to generate
# the z=0 linear matter Pk, i.e. in redshifts there always should be 0
# PkL.z -------> redshifts [0, 0.5, 1, 2 ...]
# PkL.k -------> wavenumbers
# PkL.s8 ------> array with the values of sigma8
# PkL.Hz ------> array with the values of Hz
# PkL.Pkmm ----> matrix with matter Pk: Pkmm[1,:] = mm P(k) at z[1]
# PkL.Pkcc ----> matrix with matter Pk: Pkcc[1,:] = cc P(k) at z[1]
# PkL.Pkbb ----> matrix with matter Pk: Pkbb[1,:] = bb P(k) at z[1]
# PkL.Pkcb ----> matrix with matter Pk: Pkcb[1,:] = cb P(k) at z[1]
# PkL.Pknn ----> matrix with matter Pk: Pkcc[1,:] = nu P(k) at z[1]
class PkL:
    def __init__(self, Omega_m=0.3175, Omega_b=0.049, h=0.6711, ns=0.9624, s8=None,
                 Mnu=0.0, As=2.13e-9, Omega_k=0.0, 
                 pivot_scalar=0.05, pivot_tensor=0.05, 
                 Nnu=3, hierarchy='degenerate', Neff=3.046, tau=None,
                 redshifts=[0, 0.5, 1, 2, 3], kmax=10.0, k_per_logint=50,
                 verbose=False):

        Omega_c  = Omega_m - Omega_b - Mnu/(93.14*h**2)
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
                           num_massive_neutrinos=Nnu, nnu=Neff,
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

        # get raw matter P(k) and transfer functions with weird k-binning
        #k, zs, Pk = results.get_linear_matter_power_spectrum()
        #Tk        = (results.get_matter_transfer_data()).transfer_data

        # interpolate to get Pmm, Pcc...etc
        k,z,Pmm = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                    npoints=500, var1=7, var2=7,
                                                    have_power_spectra=True, 
                                                    params=None)

        k,z,Pcc = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                    npoints=500, var1=2, var2=2,
                                                    have_power_spectra=True, 
                                                    params=None)

        k,z,Pbb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                    npoints=500, var1=3, var2=3,
                                                    have_power_spectra=True, 
                                                    params=None)

        k,z,Pcb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                    npoints=500, var1=2, var2=3,
                                                    have_power_spectra=True, 
                                                    params=None)

        Pcb = (Omega_c**2*Pcc + Omega_b**2*Pbb +\
               2.0*Omega_b*Omega_c*Pcb)/Omega_cb**2

        k,z,Pnn = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
                                                    npoints=500, var1=6, var2=6,
                                                    have_power_spectra=True, 
                                                    params=None)

        # rescale by sigma_8
        s8_linear = results.get_sigma8()[-1]
        if s8!=None and z[0]!=0.0:
            raise Exception('To rescale by s8 we need to generate the linear Pk at z=0')
        factor = (s8/s8_linear)**2

        # get sigma_8 and Hz in km/s/(kpc/h)
        self.s8 = np.array(results.get_sigma8())[::-1]*np.sqrt(factor)
        self.Hz = np.array([results.hubble_parameter(red) for red in z])

        self.z    = z;    self.k   = k 
        self.Pkmm = Pmm*factor;  self.Pknn = Pnn*factor
        self.Pkcc = Pcc*factor;  self.Pkbb = Pbb*factor;  self.Pkcb = Pcb*factor
        
        if verbose:  print(pars)
        
        #fout = 'Pk_trans_z=%.3f.txt'%z
        # notice that transfer functions have an inverted order:i=0 ==>z_max
        #np.savetxt(fout,np.transpose([Tk[0,:,i],Tk[1,:,i],Tk[2,:,i],Tk[3,:,i],
        #                               Tk[4,:,i],Tk[5,:,i],Tk[6,:,i]]))
