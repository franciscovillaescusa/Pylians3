cdef extern from "integration.h":
       double rectangular(double x_min, double x_max, double *x, double *y,
		          long elements)
       double simpson(double x_min, double x_max, double *x, double *y,
                      long elements, long steps)
       
       double *rkdumb(double vstart[], int nvar, double x1, double x2,
                      long nstep, double a[], double b[], long elements,
	              void (*derivs)(double, double [], double [],
			             double [], double [], long))

       void odeint(double ystart[], int nvar, double x1, double x2, double eps,
	           double h1, double hmin, long *nok, long *nbad,
	           double a[], double b[], long elements,
	           void (*derivs)(double, double [], double [], double [],
			          double [], long))
