import numpy as np
import plotting_library as PL
import sys,os

from pylab import *
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import LogNorm

################################# INPUT #######################################
# snapshot
snapshot = '../snapdir_003/snap_003'

# density field parameters
x_min, x_max = 0.0, 1000.0
y_min, y_max = 0.0, 1000.0
z_min, z_max = 0.0, 10.0
grid         = 512
ptypes       = [1]   # 0-Gas, 1-CDM, 2-NU, 4-Stars; can deal with several species
plane        = 'XY'  #'XY','YZ' or 'XZ'
MAS          = 'PCS' #'NGP', 'CIC', 'TSC', 'PCS' 
save_df      = True  #whether save the density field into a file

# image parameters
fout            = 'Image.png'
min_overdensity = 0.1      #minimum overdensity to plot
max_overdensity = 100.0    #maximum overdensity to plot
scale           = 'linear' #'linear' or 'log'
cmap            = 'nipy_spectral'
###############################################################################

# compute 2D overdensity field
dx, x, dy, y, overdensity = density_field_2D(snapshot, x_min, x_max, y_min, y_max,
                                             z_min, z_max, grid, ptypes, plane, MAS, save_df)


############### IMAGE ###############
print('\nCreating the figure...')
fig = figure()    #create the figure
ax1 = fig.add_subplot(111) 

ax1.set_xlim([x, x+dx])  #set the range for the x-axis
ax1.set_ylim([y, y+dy])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Mpc}$',fontsize=18)  #y-axis label

if min_overdensity==None:  min_overdensity = np.min(overdensity)
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

if scale=='linear':
    cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                     extent=[x, x+dx, y, y+dy], interpolation='sinc',
                     vmin=min_overdensity,vmax=max_overdensity)
else:
    cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                     extent=[x, x+dx, y, y+dy], interpolation='sinc',
                     norm = LogNorm(vmin=min_overdensity,vmax=max_overdensity))

cbar = fig.colorbar(cax)
cbar.set_label(r"$\rho/\bar{\rho}$",fontsize=20)
savefig(fout, bbox_inches='tight')
close(fig)
#####################################

