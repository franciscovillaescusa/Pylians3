#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void cprint()
{
	printf("Hello\n");
}

inline double cfunc(double a) { return a*a*a; }

double cc_sum(double *a, long elements)
{
	long i;
	double sum=0.0;

	for (i=0; i<elements; i++)
	{
		sum += cfunc(a[i]);
	}	

	return sum;
}
