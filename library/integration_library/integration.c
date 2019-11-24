#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "integration.h"

// This function assumes data linearly spaced in a
inline double func(double x, double *a, double *b, int elements,
		   double a_min, double a_max)
{
  int index;
  
  index = (int)((x-a_min)/(a_max-a_min)*(elements-1));
  if (index>=elements-1)
    return b[elements-1];

  return (b[index+1] - b[index])/(a[index+1]-a[index])*(x-a[index]) + b[index];

}


// For this integration method, there is no need to use more steps than
// elements in the input array
double rectangular(double x_min, double x_max, double *x, double *y,
		   long elements)
{
  double x_value, I=0.0;
  long step;

  for (step=0; step<elements; step++)
    {
      x_value = (x_max-x_min)*(step+0.5)/elements;
      I += func(x_value, x, y, elements, x_min, x_max);
      //printf("step %ld ----> y_value = %.5e\n",step,I);
    }

  return I*(x_max-x_min)/elements;
  
}

double simpson(double x_min, double x_max, double *x, double *y,
	       long elements, long steps)
{
  double x_value, dx_full, dx_half, I=0.0;
  long i;

  dx_full = (x_max - x_min)/steps;
  dx_half = dx_full/2.0;
  
  I += func(x_min, x, y, elements, x_min, x_max);
  I += func(x_max, x, y, elements, x_min, x_max);
  
  for (i=1; i<steps-1; i++)
    {
      x_value = x_min + i*dx_full;
      I += 2*func(x_value, x, y, elements, x_min, x_max);
    }

  for (i=0; i<steps; i++)
    {
      x_value = x_min + dx_half + i*dx_full;
      I += 4*func(x_value, x, y, elements, x_min, x_max);
    }

  return I*(x_max-x_min)/(6*steps);

  
}








