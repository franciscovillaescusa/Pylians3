#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "ccode.h"

void cprint()
{
	printf("Hello\n");
}

extern double cfunc(double a);

double csum(double *a, long elements)
{
	long i;
	double sum=0.0;

	for (i=0; i<elements; i++)
	{
		sum += cfunc(a[i]);
	}	

	return sum;
}