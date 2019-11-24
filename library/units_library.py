# This is the library where we store the value of the different constants
# In order to use this library do:
# import units_library as UL
# U = UL.units()
# kpc = U.kpc_cm

class units:
    def __init__(self):
    
        self.rho_crit = 2.77536627e11 #h^2 Msun/Mpc^3
        self.c_kms    = 3e5           #km/s
        self.Mpc_cm   = 3.0856e24     #cm
        self.kpc_cm   = 3.0856e21     #cm
        self.Msun_g   = 1.989e33      #g
        self.Ymass    = 0.24          #helium mass fraction
        self.mH_g     = 1.6726e-24    #proton mass in grams
        self.yr_s     = 3.15576e7     #seconds
        self.km_cm    = 1e5           #cm
        self.kB       = 1.3806e-26    #gr (km/s)^2 K^{-1}
        self.nu0_MHz  = 1420.0        #21-cm frequency in MHz
