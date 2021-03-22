******************
Halo mass function
******************

Pylians provides the routine ``MF_theory`` to compute the halo mass function of a given model. The arguments of this function are:

k_in, Pk_in, OmegaM, Masses, author, bins=10000, z=0, delta=200.0

- ``k``. 1D numpy array with the value of the linear matter power spectrum wavenumbers.
- ``Pk``. 1D numpy array with the amplitude of the linear matter power spectrum on the wavenumbers ``k``.
- ``OmegaM``. Value of :math:`\Omega_{\rm m}`.
- ``Masses``. 1D numpy array with the value of the halo masses over which compute the halo mass function.
- ``author``. The `model` for the halo mass function. Options are: ``ST``, ``Tinker``, ``Tinker10``, ``Crocce``, ``Jenkins``, ``Warren``, ``Watson``, ``Watson_FoF``, ``Angulo``.
- ``bins``. In order to carry out the integrals, the k bins need to be sorted and equally spaced in log10. This parameter determines the number of bins to use. The more the better, but a very large number will have very little impact. Default 10000.
- ``z``. Redshift at which to estimate the halo mass function. Only needed for the ``Tinker``, ``Tinker10``, and ``Crocce`` mass functions.
- ``delta``. The overdensity value. Default is 200. Only needed for ``Tinker`` and ``Tinker10``.

.. Note::

   For cosmologies with massive neutrinos, :math:`\Omega_{\rm m}` should be set to :math:`\Omega_{\rm c}+\Omega_{\rm b}` and the linear power spectrum should be the CDM+baryons linear power spectrum; see e.g. `1311.1212 <https://arxiv.org/abs/1311.1212>`_ and `1311.1514 <https://arxiv.org/abs/1311.1514>`_.
  
An example of how to use this routine is this:

.. code-block:: python

   import numpy as np
   import mass_function_library as MFL

   # halo mass function parameters
   f_Pk   = 'Pk_linear_z=0.txt'  #file with linear Pk
   OmegaM = 0.3175
   Masses = np.logspace(11, 15, 100) #array with halo masses
   author = 'ST'   #Sheth-Tormen halo mass function
   bins   = 10000  #number of bins to use for Pk
   z      = 0.0    #redshift; only used for Tinker, Tinker10 and Crocce
   delta  = 200.0  #overdensity; only for Tinker and Tinker10

   # read linear matter Pk
   k, Pk = np.loadtxt(f_Pk, unpack=True)
   
   # compute halo mass function
   HMF = MFL.MF_theory(k, Pk, OmegaM, Masses, author, bins, z, delta)

   
variance
~~~~~~~~

Pylians provides the routine ``sigma`` that can be used to compute :math:`\sigma_R`, defined as

.. math::

   \sigma_R = \int_0^\infty P(k)W(k,R)^2k^2/(2\pi^2)

where :math:`W(k,R)` is the Fourier transform of a top-hat function with radius :math:`R`:

.. math::

   W(k,R) = \frac{3[\sin(kR) - kR\cos(kR)]}{(kR)^3}

The most standard applicaiton of this routine is to compute the value of :math:`\sigma_8` given a linear power spectrum:  

.. code-block:: python

   import numpy as np
   import mass_function_library as MFL

   # read linear power spectrum
   k, Pk = np.loadtxt('My_linear_Pk.txt', unpack=True)

   # compute the value of sigma_8
   sigma_8 = MFL.sigma(k, Pk, 8.0)
