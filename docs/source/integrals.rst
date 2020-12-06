*********
Integrals
*********

Pylians provide routines to carry out numerical integrals in a more efficient way than scipy integrate. The philosophy is that to compute the integral :math:`\int_a^b f(x)dx` the user passes the integrator two arrays, one with some values of :math:`x` between a and b and another with the values of :math:`f(x)` at those positions. The integrator will interpolate internally the input data to evaluate the function at an arbitrary position :math:`x`. Pylians implements in c the fortran odeint function and wrap in python through cython.

For instance, to compute :math:`\int_0^5 (3x^3+2x+5) dx` one would do

.. code-block:: python
		
   import numpy as np
   import integration_library as IL

   # integral value, its limits and precision parameters
   yinit = np.zeros(1, dtype=np.float64) 
   x1    = 0.0
   x2    = 5.0
   eps   = 1e-8 
   h1    = 1e-10 
   hmin  = 0.0   

   # integral method and integrand function
   function = 'linear'
   bins     = 1000
   x        = np.linspace(x1, x2, bins)
   y        = 3*x**2 + 2*x + 5

   I = IL.odeint(yinit, x1, x2, eps, h1, hmin, x, y, function, verbose=True)[0]

The value of integral is stored in ``I``. The ``odeint`` routine needs the following ingredients:

- ``yinit``. The value of the integral is stored in this variable. Should be a 1D double numpy array with one single element equal to 0. If several integrals are being computed sequentially this variable need to be declared for each integral.
- ``x1``. Lower limit of the integral.
- ``x2``. Upper limit of the integral.
- ``eps``. Maximum local relative error tolerated. Typically set its value to be 1e8-1e10 lower than the value of the integral. Verify the convergence of the results by studying the dependence of the integral on this number.
- ``h1``. Initial guess for the first time step (cannot be 0).
- ``hmin``. Minimum allowerd time step (can be 0).
- ``verbose``. Set it to ``True`` to print some information on the integral computation.

There are two main methods to carry out the integral, depending on how the interpolation is performed. 

- ``function = 'linear'``. The function is evaluated by interpolating linearly the input values of :math:`x` and :math:`y=f(x)`.
    - ``x``. 1D double numpy array containing the input, equally spaced, values of :math:`x`. 
    - ``y``. 1D double numpy array containing the values of :math:`y=f(x)` at the ``x`` array positions.
 
- ``function = 'log'``. The function is evaluated by interpolating logaritmically the input values of :math:`x` and :math:`y=f(x)`.
    - ``x``. 1D double numpy array containing the input, equally spaced in log, values of :math:`\log_{10}(x)`. 
    - ``y``. 1D double numpy array containing the values of :math:`y=f(x)` at the ``x`` array positions.

An example of using the log-interpolation to compute the integral :math:`\int_1^2 e^x dx` is this

.. code-block:: python
		
   import numpy as np
   import integration_library as IL

   # integral value, its limits and precision parameters
   yinit = np.zeros(1, dtype=np.float64) 
   x1    = 1.0
   x2    = 2.0
   eps   = 1e-10 
   h1    = 1e-12 
   hmin  = 0.0   
   
   # integral method and integrand function
   function = 'log'
   bins     = 1000
   x        = np.logspace(np.log10(x1), np.log10(x2), bins)
   y        = np.exp(x)

   I = IL.odeint(yinit, x1, x2, eps, h1, hmin, np.log10(x), y,
		 function, verbose=True)[0]

.. warning::
   Be careful when using the log-interpolation, since the code will crash if a zero or negative value is encounter. 

The user can create its own function to avoid evaluating the integrand via interpolations. This function has to be placed in the file library/integration_library/integration_library.pyx (see linear and sigma functions as examples). After that, a new function call has to be created in the function odeint of that file (see linear and log as examples).
