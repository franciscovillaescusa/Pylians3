******************
ICs power spectrum
******************

In some cases we may want to compute the power spectrum of the initial density field of a simulation. The particle positions from the initial conditions can be used to estimate that. However, this will only be an approximation.

We have modified the N-GenIC code by Volker Springel to output the modes amplitudes, phases, and wavenumbers of the initial density field; you can find this N-GenIC version `here <https://github.com/franciscovillaescusa/N-GenIC_growth>`_.

When generating initial conditions with that code (switch the ``DOUTPUT_DF`` flag in the Makefile), several files will be written starting with ``Amplitudes_``, ``Coordinates_``, and ``Phases_``. Pylians provides the routine ``Pk_NGenIC_IC_field`` that can be used to compute the power spectrum of the initial field from those files. The arguments of that function are these:

- ``f_coordinates``. The prefix of the files containing the modes coordinates. Note that only the prefix is needed, not the whole file name (or file names when multiple files).
- ``f_amplitudes``. The prefix of the files containing the modes amplitudes. Note that only the prefix is needed, not the whole file name (or file names when multiple files).
- ``BoxSize``. Size of the simulation box in Mpc/h.

An example of how to use the routine is this

.. code-block:: python

   import numpy as np
   import Pk_library as PKL

   # parameters
   f_coordinates = 'Coordinates_ptype_1' #modes coordinates of the dark matter (ptype 1)
   f_amplitudes  = 'Amplitudes_ptype_1'  #modes amplitudes of the dark matter (ptype 1)
   BoxSize       = 512.0                 #Mpc/h

   # compute Pk of the initial field
   # k will be in h/Mpc, while P(k) will have (Mpc/h)^3 units
   k, Pk, Nmodes = PKL.Pk_NGenIC_IC_field(f_coordinates, f_amplitudes, BoxSize)
