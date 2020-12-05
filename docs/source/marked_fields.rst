*************
Marked fields
*************

Pylians can also compute density fields where each particle (or halo, galaxy, void...etc) has a different weight. This can be useful in the case where the observed field is constructed in that way, e.g. the 21cm field is built from gas particles weighting each of them by its HI mass. This is an example of how to do this with Pylians:


.. code-block:: python
		
   import numpy as np
   import MAS_library as MASL

   # input parameters
   grid    = 512    #grid size
   BoxSize = 1000   #Mpc/h
   MAS     = 'CIC'  #Cloud-in-Cell

   # define the array hosting the density field
   delta = np.zeros((grid,grid,grid), dtype=np.float32)

   # read the particle positions
   pos     = np.loadtxt('myfile.txt')   #Mpc/h 
   pos     = pos.astype(np.float32)     #pos should be a numpy float array
   weights = np.loadtxt('weights.txt')  #weights of the particles
   weights = weights.astype(np.float32) #weights should be a numpy float array

   # compute density field taking into account the particle weights
   MASL.MA(pos,delta,BoxSize,MAS,W=weights)

   # compute density contrast
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0 


