# This script generates the density and velocity fields of a Gaussian density field
# given an input power spectrum. 
import numpy as np
import scipy.fftpack
import density_field_library as DFL
import Pk_library as PKL
import sys

############################### INPUT #########################################
BoxSize = 1000.0  #Mpc/h
dims    = 128     #Generate delta(k) using a grid with dims^3 cells

Rayleigh_sampling = False
precision         = 'double'  #'single' or 'double'

do_sirko   = True   #set delta_k[0,0,0] to 0 (False) or to Pkf[0] (True)
SphereMode = 'cube' #choose between 'sphere' and 'cube'

Pk_file = 'ics_matterpow_10.dat'

seed = 105  #seed for the random numbers

do_lognormal = False

# Pk/Xi parameters
axis    = 0
MAS     = 'None'
threads = 1

z_in    = 10.0    #only needed for the velocities
Omega_m = 0.3175  #only needed for the velocities
Omega_l = 0.6825  #only needed for the velocities

fout_df  = 'Cube_sirko_128^3_z=10.txt'         #file with the density field
f_Pk     = 'Pk_sirko_128^3_density_field.dat'  #file with density field Pk
f_CF     = 'CF_sirko_128^3_density_field.dat'  #file with density field Xi
Pk_theta = 'Pk_theta_field.dat'                #file with the P(k) of \grad V
###############################################################################

dims2, dims3, middle = dims**2, dims**3, dims//2
k_nyquist = dims//2  #Nyquist frequency in units of the fundamental frequency

# compute the value of Omega_m(z), the growth rate and the Hubble function
# Notice that it would be better to compute H(z_in) and the growth rate directly
# from CAMB, but in the past we checked that H_zin*growth_rate is quite 
# insesitive to Omega_radiation
Omega_mz    = Omega_m*(1.0+z_in)**3/(Omega_m*(1.0+z_in)**3+Omega_l)
growth_rate = Omega_mz**0.54545454
H_zin       = 100.0*np.sqrt(Omega_m*(1.0+z_in)**3+Omega_l) #km/s/(Mpc/h)
vel_prefact = -BoxSize/(2.0*np.pi)*H_zin*1.0/(1.0+z_in)*growth_rate #sign is -!!
print('vel_prefact = %.4e km/s'%vel_prefact)
print('a*H(z_i)*f  = %.4e km/s'%(H_zin*1.0/(1.0+z_in)*growth_rate))
# the 1/(BoxSize/(2*pi)) above comes from the ratio \vec{k}/k^2 of \vec{v}_k

# read the P(k) file and set correct units
kf, Pkf = np.loadtxt(Pk_file,unpack=True)
kf = kf*BoxSize/(2.0*np.pi);    Pkf = Pkf/BoxSize**3

# generate the field delta(k)
delta_k,Vx_k,Vy_k,Vz_k = \
    DFL.delta_k(dims, precision, kf, Pkf, Rayleigh_sampling, do_sirko, seed)


############### delta(r) ##################
# compute delta(r) by FFT delta(k)
print('Fourier transforming delta(k)...')
delta_r = scipy.fftpack.fftn(delta_k,overwrite_x=True); del delta_k
delta_r = np.ravel(delta_r)

# since delta(r) has to be real, check that the imaginary part is negligible
ratio = np.absolute(delta_r.imag/delta_r.real)
print('max value of |delta_r.imag/delta_r.real| =',np.max(ratio)); del ratio

# in principle delta(r) will have an imaginary part. Just keep the real one
delta_r = delta_r.real;    print(np.min(delta_r),'< delta(r) <',np.max(delta_r))
print('<delta(r)> =',np.mean(delta_r,dtype=np.float64))
###########################################

################# Vx(r) ###################
# compute Vx(r) by FFT Vx(k)
print('Fourier transforming Vx(k)...')
Vx_r = scipy.fftpack.fftn(Vx_k,overwrite_x=True); del Vx_k
Vx_r = np.ravel(Vx_r); Vx_r*=vel_prefact

#since delta(r) has to be real, check that the imaginary part is negligible
ratio = np.absolute(Vx_r.imag/Vx_r.real)
print('max value of |Vx_r.imag/Vx_r.real| =',np.max(ratio)); del ratio

#in principle Vx(r) will have an imaginary part. Just keep the real one
Vx_r = Vx_r.real
print(np.min(Vx_r),'< Vx <',np.max(Vx_r))
###########################################

################# Vy(r) ###################
# compute Vy(r) by FFT Vy(k)
print('Fourier transforming Vy(k)...')
Vy_r = scipy.fftpack.fftn(Vy_k,overwrite_x=True); del Vy_k
Vy_r = np.ravel(Vy_r); Vy_r*=vel_prefact

# since delta(r) has to be real, check that the imaginary part is negligible
ratio = np.absolute(Vy_r.imag/Vy_r.real)
print('max value of |Vy_r.imag/Vy_r.real| =',np.max(ratio)); del ratio

# in principle Vy(r) will have an imaginary part. Just keep the real one
Vy_r = Vy_r.real
print(np.min(Vy_r),'< Vy <',np.max(Vy_r))
###########################################

################# Vz(r) ###################
# compute Vz(r) by FFT Vz(k)
print('Fourier transforming Vz(k)...')
Vz_r = scipy.fftpack.fftn(Vz_k,overwrite_x=True); del Vz_k
Vz_r = np.ravel(Vz_r); Vz_r*=vel_prefact

# since delta(r) has to be real, check that the imaginary part is negligible
ratio = np.absolute(Vz_r.imag/Vz_r.real)
print('max value of |Vz_r.imag/Vz_r.real| =',np.max(ratio)); del ratio

# in principle Vz(r) will have an imaginary part. Just keep the real one
Vz_r = Vz_r.real
print(np.min(Vz_r),'< Vz <',np.max(Vz_r))
###########################################

# do the lognormal transformation
if do_lognormal:
    sigma_G = np.sqrt(np.mean(delta_r**2,dtype=np.float64))
    print('sigma_G =',sigma_G)
    delta_r = np.exp(delta_r-sigma_G**2/2.0) - 1.0
    print('<delta_NL(r)> =',np.mean(delta_r,dtype=np.float64))
    print(np.min(delta_r),'< delta_NL(r) <',np.max(delta_r))


# save density and velocity field to file
np.save(fout_df, [delta_r, Vx_r, Vy_r, Vz_r])


############ Final checks ###########
# Compute power spectrum & correlation function of the density field 
Pk = PKL.Pk(delta_r, BoxSize, axis, MAS, threads)
CF = PKL.Xi(delta_r, BoxSize, MAS, axis, threads)
np.savetxt(f_Pk, np.transpose([Pk.k3D, Pk.Pk[:,0]]))
np.savetxt(f_CF, np.transpose([CF.r3D, CF.xi[:,0]]));  del delta_r

"""
#compute the power spectrum of the theta=i\grad V
Vx_r = np.reshape(Vx_r,(dims,dims,dims))
Vy_r = np.reshape(Vy_r,(dims,dims,dims))
Vz_r = np.reshape(Vz_r,(dims,dims,dims))

Vx_k = scipy.fftpack.ifftn(Vx_r,overwrite_x=True); del Vx_r
Vy_k = scipy.fftpack.ifftn(Vy_r,overwrite_x=True); del Vy_r
Vz_k = scipy.fftpack.ifftn(Vz_r,overwrite_x=True); del Vz_r

theta_k = DFL.theta(Vx_k,Vy_k,Vz_k,dims)*2.0*np.pi/BoxSize
del Vx_k, Vy_k, Vz_k
theta_r = scipy.fftpack.fftn(theta_k,overwrite_x=True); del theta_k
theta_r = np.ravel(theta_r)
[k,Pk]=PSL.power_spectrum_given_delta(theta_r,dims,BoxSize,
                                      aliasing_method='None')
np.savetxt(Pk_theta,np.transpose([k,Pk])); del theta_r
"""
#####################################



