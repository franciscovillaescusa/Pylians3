# This simple script can be used as
# import numpy as np
# import cython_code as CC
# a = np.arange(123456789, dtype=np.int32)
# CC.pyprint()
# CC.pysum(a)
import numpy as np 
cimport numpy as np
import sys,os,time
cimport cython

cdef extern void cprint()
cdef extern double cc_sum(double *a, long elements)

# python function that calls a C function that just prints 'Hello'
def pyprint():
    cprint()

# python function that takes an array of doubles and computes the sum
# of their cubes by calling a C function
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def csum(double[::1] array):

    cdef double suma

    start = time.time()
    suma = cc_sum(&array[0], array.shape[0])
    print('\nTime taken C func = %.3f ms'%(1e3*(time.time()-start)))
    print('Sum = %.8e'%suma)


# python function that takes an array of doubles and computes the sum
# of their cubes by calling using cython
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def cysum(double[::1] array):

    cdef double suma=0.0
    cdef long i, elements=array.shape[0]

    start = time.time()
    for i in range(elements):
        suma += array[i]*array[i]*array[i]
    print('\nTime taken cython = %.3f ms'%(1e3*(time.time()-start)))
    print('Sum = %.8e'%suma)


# python function that takes an array of doubles and computes the sum
# of their cubes by calling using cython
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def pysum(array):

    start = time.time()
    suma = np.sum(array**3, dtype=np.float64)
    print('\nTime taken numpy = %.3f ms'%(1e3*(time.time()-start)))
    print('Sum = %.8e'%suma)
