static double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))

static double minarg1,minarg2;
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ? (minarg1) : (minarg2))

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

double rectangular(double x_min, double x_max, double *x, double *y,
		   long elements);
double simpson(double x_min, double x_max, double *x, double *y,
	       long elements, long steps);

void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	 double a[], double b[], long elements,
	 //void (*derivs)(double, double [], double []));
	 void (*derivs)(double, double [], double [],
			double [], double [], long));


double *rkdumb(double vstart[], int nvar, double x1, double x2, long nstep,
	       double a[], double b[], long elements,
	       //void (*derivs)(double, double [], double []));
	       void (*derivs)(double, double [], double [],
			      double [], double [], long));

void derivs(double, double [], double [], double [], double [], long);

void rkqs(double y[], double dydx[], int n, double *x, double htry, double eps,
	  double yscal[], double *hdid, double *hnext,
	  double a[], double b[], long elements,
	  void (*derivs)(double, double [], double [], double [],
			 double [], long));

void rkck(double y[], double dydx[], int n, double x, double h,
	  double yout[], double yerr[], double a[], double b[], long elements,
	  void (*derivs)(double, double [], double [], double [],
			 double [], long));

void odeint(double ystart[], int nvar, double x1, double x2, double eps,
	    double h1, double hmin, long *nok, long *nbad,
	    double a[], double b[], long elements,
	    void (*derivs)(double, double [], double [], double [],
			   double [], long));
