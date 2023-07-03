Pylians
=======

Pylians stands for **Py**\thon **li**\braries for the **a**\nalysis of **n**\umerical **s**\imulations. They are a set of python libraries, written in python, cython and C, whose purposes is to facilitate the analysis of numerical simulations (both N-body and hydrodynamic). Pylians runs on both python2 and python3. Among other things, they can be used to:

- Compute density fields
- Compute power spectra
- Compute bispectra
- Compute correlation functions
- Identify voids
- Populate halos with galaxies using an HOD
- Apply HI+H2 corrections to the output of hydrodynamic simulations
- Make 21cm maps
- Compute DLAs column density distribution functions
- Plot density fields and make movies

`Pylians <https://en.wikipedia.org/wiki/Nestor_(mythology)>`__ were the native or inhabitant of the Homeric town of Pylos. 


.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation

.. _density_fields:
.. toctree:: 
   :maxdepth: 2
   :caption: Density fields

   construction
   smoothing
   interpolation
   gadget_df
   hydro_sims
	     
.. toctree::
   :maxdepth: 2
   :caption: Functionalities

   Pk
   xi
   Bk
   voids
   RSD

.. toctree::
   :maxdepth: 2
   :caption: Neutral hydrogen
      
.. toctree:: 
   :maxdepth: 2
   :caption: Miscellaneous

   plots
   cosmology
   mass_function
   gaussian_fields
   integrals
   gadget
   Pk_ics
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Other

   license
   citation
   contact
