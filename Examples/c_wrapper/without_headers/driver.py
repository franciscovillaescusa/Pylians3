# This script creates a numpy array and the it compute the sum of the
# cube of its elements by calling three different functions:
# a pure c function, a cython function and using numpy

import cython_code as CC 
import numpy as np 

array = np.arange(100000000, dtype=np.float64)

CC.csum(array)
CC.cysum(array)
CC.pysum(array)
