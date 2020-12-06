.. _smoothing: 

*********
Smoothing
*********

Pylians provides routines to smooth fields with several filters. The ingredients needed are:

- ``field``. This is a 3D float numpy array that contains the input field to be smoothed.
- ``BoxSize``. This is the size of the box with the input density field. 
- ``R``. This is the smoothing scale.
- ``grid``. This is the grid size of the input field, i.e. ``field.shape[0]``.
-  ``threads``. Number of openmp threads to be used.
-  ``Filter``. Filter to use. ``'Top-Hat'`` or ``'Gaussian'``.
-  ``W_k``. This is a 3D complex64 numpy array containing the Fourier-transform of the filter. Notice that when smoothing a discrete field, like the one stored on a regular grid, the Fourier-transform of the filter need to be computed in the same way as the for the field, i.e. through DFT instead of FT.

An example is this

.. code-block:: python

   import smoothing_library as SL

   BoxSize = 75.0 #Mpc/h
   R       = 5.0  #Mpc.h
   grid    = field.shape[0]
   Filter  = 'Top-Hat'
   threads = 28

   # compute FFT of the filter
   W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

   # smooth the field
   field_smoothed = SL.field_smoothing(field, W_k, threads)
