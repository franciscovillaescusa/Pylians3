*********
Minkowski functionals
*********

Pylians provides routines to measure the Minkowski functionals (MFs). They characterize the morphology of the cosmic web uniquely and completely in the sense of Hadwidger's theorem. See https://arxiv.org/abs/astro-ph/9312028 and https://arxiv.org/abs/astro-ph/9702130 for a systematic introduction to the MFs. They have recently been proposed as a promising probe of modified gravity (https://arxiv.org/abs/1704.02325,https://arxiv.org/abs/2305.04520,https://arxiv.org/abs/2412.05662) and massive neutrinos \cite{https://arxiv.org/abs/2204.02945,https://arxiv.org/abs/2302.08162}. A recent application of the MFs to the BOSS DR12 CMASS galaxy sample has shown that the MFs can extract valuable cosmological information and place tighter constraints on cosmological parameters than 2PCF (https://arxiv.org/abs/2501.01698)

The ingredients needed to compute the MFs are:

- ``delta''. The density contrast field.
- ``CellSize''. Cell size of the density field
- ``thres_mask''. Regions with density lower than thres_mask will be excluded in the measurement
- ``thresholds''. Density threshold above which the excursion set is defined. The MFs will be ouput as a function of the threshold.

An example is this

.. code-block:: python

   import smoothing_library as SL
   import MFs_library as MFL

   BoxSize = 1000.0 #Mpc/h
   R       = 5.0  #Mpc.h
   grid    = 250
   Filter  = 'Gaussian'
   threads = 8
   CellSize = int(BoxSize/grid)

   # compute FFT of the filter
   W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

   # smooth the field
   delta = SL.field_smoothing(field, W_k, threads)
   delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

   thres_mask = -2 #smaller than -1.0, so the measurement of MFs will be performed for the whole box.
   thresholds = np.linspace(-1,7,num=81)
   mfs = MFs(delta,CellSize,thres_mask,thresholds)
   data = mfs.MFs3D