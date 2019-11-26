# This scripts shows how to compute numerically integrals with scipy.integrate
import numpy as np
import scipy.integrate as si


def func_1D_odeint(y,x,a,b):
    return a + b*x**2

def func_1D_quad(x,a,b):
    return a + b*x**2

def func_2D(y,x,a,b):    
    return a*b*x*y**2

############## INPUT ##############
a = 1.0;  b = 1.0
###################################

# Example to compute a 1D integral with odeint:  I = \int_0^3 (a+b*x^2) dx
yinit    = [0.0];     x_limits = [0.0, 3.0] 
I = si.odeint(func_1D_odeint, yinit, x_limits, args=(a,b), mxstep=1000000,
              rtol=1e-8, atol=1e-10,  h0=1e-10)[1][0]
print('\n############### 1D integral with odeint ###############')
print('Numerical Integral = %.6f'%I)
print('Analytic  Integral = %.6f'%(3*a+9*b))
print('#######################################################')

# Example to compute a 1D integral with quad:  I = \int_0^3 (a+b*x^2) dx
I,dI = si.quad(func_1D_quad, 0, 3, args=(a,b), epsabs=1e-8, epsrel=1e-8)
print('\n################ 1D integral with quad ################')
print('Numerical Integral = %.6f +- %.3e'%(I,dI))
print('Analytic  Integral = %.6f'%(3*a+9*b))
print('#######################################################')


# Example to compute a 2D integral:  I = \int_0^1 dx \int_0^(x^2) dy a*b*x*y^2
I,dI = si.dblquad(func_2D, 0, 1, lambda x:0, lambda x:x**2, 
                  args=(a,b), epsabs=1e-8, epsrel=1e-8)
print('\n############### 2D integral with dblquad ##############')
print('Numerical Integral = %.6f +- %.3e'%(I,dI))
print('Analytic  Integral = %.6f'%(a*b/24.0))
print('#######################################################\n')
