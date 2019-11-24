import numpy as np
import sys,os,time
import integration_library as IL


elements = 100000
x = np.linspace(0, 10, elements)
y = x**4
steps = elements*100

start = time.time()
I = IL.trapezoidal(x,y)
print('I = %.8f'%I)
print('Time taken = %.2f ms'%(1e3*(time.time()-start)))

start = time.time()
I = IL.simpson(x,y,steps)
print('I = %.8f'%I)
print('Time taken = %.2f ms'%(1e3*(time.time()-start)))
