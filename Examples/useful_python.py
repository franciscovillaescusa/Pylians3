# Here there are some useful things

# save a short array to a file (can be slow for large arrays):
a = np.arange(10); b = 2.0*a; c = 3.0*a
np.savetxt('borrar.dat',np.transpose([a,b,c]))

# read from a ASCII file
a = np.loadtxt('borrar.dat')
a,b = np.loadtxt('borrar.dat',unpack=True)
