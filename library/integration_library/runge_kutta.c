#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "integration.h"

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
#define MAXSTP 1000000
#define TINY 1.0e-30

// $$$$$$$$$$$$$$$$ RUNGE-KUTTA 4th order constant step $$$$$$$$$$$$$$$
//Given values for the variables y[0..n-1] and their derivatives dydx[0..n-1] known at x, use the fourth-order Runge-Kutta method to advance the solution over an interval h and return the incremented variables as yout[0..n-1], which need not be a distinct array from y. The user supplies the routine derivs(x,y,dydx), which returns derivatives dydx at x.
void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	 double a[], double b[], long elements,
	 //void (*derivs)(double, double [], double []))
	 void (*derivs)(double, double [], double [], double [], double [], long))
{
  int i;
  double xh, hh, h6, *dym, *dyt, *yt;

  dym = (double *)malloc(n*sizeof(double));
  dyt = (double *)malloc(n*sizeof(double));
  yt  = (double *)malloc(n*sizeof(double));

  hh = h*0.5;
  h6 = h/6.0;
  xh = x+hh;

  for (i=0;i<n;i++) yt[i]=y[i]+hh*dydx[i];
  //(*derivs)(xh,yt,dyt);
  (*derivs)(xh,yt,dyt,a,b,elements);
  for (i=0;i<n;i++) yt[i]=y[i]+hh*dyt[i];
  //(*derivs)(xh,yt,dym);
  (*derivs)(xh,yt,dym,a,b,elements);
  for (i=0;i<n;i++) {
    yt[i]=y[i]+h*dym[i];
    dym[i] += dyt[i];
  }
  //(*derivs)(x+h,yt,dyt);
  (*derivs)(x+h,yt,dyt,a,b,elements);
  for (i=0;i<n;i++)
    yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);

  free(yt);  free(dyt);  free(dym);
}

//Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx) evaluates derivatives. Results are stored in the global variables y[0..nvar-1][1..nstep+1] and xx[1..nstep+1].
double *rkdumb(double vstart[], int nvar, double x1, double x2, long nstep,
	       double a[], double b[], long elements,
	       //void (*derivs)(double, double [], double []))
				 void (*derivs)(double, double [], double [], double [],
			      double [], long))
{

  int i;
  long k;
  double x, h, *v, *vout, *dv;

  v    = (double *)malloc(nvar*sizeof(double));
  vout = (double *)malloc(nvar*sizeof(double));
  dv   = (double *)malloc(nvar*sizeof(double));

  for (i=0;i<nvar;i++) {
    v[i]=vstart[i];
  }

  x=x1;
  h=(x2-x1)/nstep;

  for (k=0;k<nstep;k++) {
    //(*derivs)(x,v,dv);
		(*derivs)(x,v,dv,a,b,elements);
    rk4(v,dv,nvar,x,h,vout,a,b,elements,derivs);
    if ((double)(x+h) == x){
      printf("Step size too small in routine rkdumb\n");
      exit(1);
    }
    x += h;

    for (i=0;i<nvar;i++)
      v[i]=vout[i];
  }

  free(dv);  free(vout);
  return v;

}
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



// $$$$$$$$$$$$$$$$$$ RUNGE-KUTTA 5th order adaptive step $$$$$$$$$$$$$$$$$$
// Fifth-order Runge-Kutta step with monitoring of local truncation error to
// ensure accuracy and adjust step size. Input are the dependent variable vector
// y[0..n-1] and its derivative dydx[0..n-1] at the starting value of the
// independent variable x. Also input are the stepsize to be attempted htry,
// the required accuracy eps, and the vector yscal[0..n-1] against which the
// error is scaled. On output, y and x are replaced by their new values, hdid
// is the stepsize that was actually accomplished, and hnext is the estimated
// next stepsize. derivs is the user-supplied routine that computes the
// right-hand side derivatives.
void rkqs(double y[], double dydx[], int n, double *x, double htry, double eps,
	    		double yscal[], double *hdid, double *hnext,
					double a[], double b[], long elements,
					void (*derivs)(double, double [], double [], double [],
					double [], long))
{
	int i;
	double errmax, h, htemp, xnew, *yerr, *ytemp;

	yerr  = (double *)malloc(n*sizeof(double));
	ytemp = (double *)malloc(n*sizeof(double));
	h     = htry; //Set stepsize to the initial trial value.

	for (;;) {
		rkck(y,dydx,n,*x,h,ytemp,yerr,a,b,elements,derivs); //Take a step.
		errmax = 0.0; //Evaluate accuracy
		for (i=0;i<n;i++)
			errmax=FMAX(errmax,fabs(yerr[i]/yscal[i]));
		errmax /= eps; //Scale relative to required tolerance.
		if (errmax <= 1.0) break; //Step succeeded. Compute size of next step.
		htemp = SAFETY*h*pow(errmax,PSHRNK);
		//Truncation error too large, reduce stepsize.
		h=(h >= 0.0 ? FMAX(htemp,0.1*h) : FMIN(htemp,0.1*h));
		//No more than a factor of 10.
		xnew = (*x)+h;
		if (xnew == *x){
			printf("stepsize underflow in rkqs\n");
			exit(1);
		}
	}

	if (errmax > ERRCON)
		*hnext = SAFETY*h*pow(errmax,PGROW);
	else
		*hnext = 5.0*h;

	*x += (*hdid=h);
	for (i=0;i<n;i++)
		y[i]=ytemp[i];

	free(ytemp);  free(yerr);
}

//Given values for n variables y[0..n-1] and their derivatives dydx[0..n-1] known
//at x, use the fifth-order Cash-Karp Runge-Kutta method to advance the
//solution over an interval h and return the incremented variables as
//yout[0..n-1]. Also return an estimate of the local truncation error in yout
//using the embedded fourth-order method. The user supplies the routine
//derivs(x,y,dydx), which returns derivatives dydx at x.
void rkck(double y[], double dydx[], int n, double x, double h,
					double yout[], double yerr[], double a[], double b[], long elements,
					//void (*derivs)(float, float [], float []))
					void (*derivs)(double, double [], double [], double [],
					double [], long))
{

	int i;

	static double a2=0.2, a3=0.3, a4=0.6, a5=1.0, a6=0.875,
								b21=0.2, b31=3.0/40.0, b32=9.0/40.0, b41=0.3, b42 = -0.9,
								b43=1.2, b51 = -11.0/54.0, b52=2.5, b53 = -70.0/27.0,
								b54=35.0/27.0, b61=1631.0/55296.0, b62=175.0/512.0,
								b63=575.0/13824.0, b64=44275.0/110592.0, b65=253.0/4096.0,
								c1=37.0/378.0, c3=250.0/621.0, c4=125.0/594.0, c6=512.0/1771.0,
							 	dc5 = -277.00/14336.0;
	double dc1=c1-2825.0/27648.0,  dc3=c3-18575.0/48384.0,
				 dc4=c4-13525.0/55296.0, dc6=c6-0.25;
	double *ak2, *ak3, *ak4, *ak5, *ak6, *ytemp;

	ak2   = (double *)malloc(n*sizeof(double));
	ak3   = (double *)malloc(n*sizeof(double));
	ak4   = (double *)malloc(n*sizeof(double));
	ak5   = (double *)malloc(n*sizeof(double));
	ak6   = (double *)malloc(n*sizeof(double));
	ytemp = (double *)malloc(n*sizeof(double));

	for (i=0;i<n;i++)                          //first step
		ytemp[i]=y[i]+b21*h*dydx[i];
	//(*derivs)(x+a2*h,ytemp,ak2);
	(*derivs)(x+a2*h,ytemp,ak2,a,b,elements);  //second step

	for (i=0;i<n;i++)
		ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
	//(*derivs)(x+a3*h,ytemp,ak3);
	(*derivs)(x+a3*h,ytemp,ak3,a,b,elements);  //third step

	for (i=0;i<n;i++)
		ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
	//(*derivs)(x+a4*h,ytemp,ak4);
	(*derivs)(x+a4*h,ytemp,ak4,a,b,elements);  //fourth step

	for (i=0;i<n;i++)
		ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
	//(*derivs)(x+a5*h,ytemp,ak5);
	(*derivs)(x+a5*h,ytemp,ak5,a,b,elements);  //second step

	for (i=0;i<n;i++)
		ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
	//(*derivs)(x+a6*h,ytemp,ak6);
	(*derivs)(x+a6*h,ytemp,ak6,a,b,elements);  //sixth step

	for (i=0;i<n;i++)           //Accumulate increments with proper weights.
		yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);

	//Estimate error as difference between fourth and fifth order methods.
	for (i=0;i<n;i++)
		yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);

	free(ytemp);  free(ak6);  free(ak5);  free(ak4);  free(ak3);  free(ak2);
}

//User storage for intermediate results. Preset kmax and dxsav in the calling
//program. If kmax ̸= 0 results are stored at approximate intervals dxsav in the
//arrays xp[1..kount], yp[1..nvar] [1..kount], where kount is output by odeint.
//Defining declarations for these variables, with memory allocations xp[1..kmax]
//and yp[1..nvar][1..kmax] for the arrays, should be in the calling program.
//extern int kmax, kount;
//extern float *xp,**yp,dxsav;

//Runge-Kutta driver with adaptive stepsize control. Integrate starting values
//ystart[1..nvar] from x1 to x2 with accuracy eps, storing intermediate results
//in global variables. h1 should be set as a guessed first stepsize, hmin as
//the minimum allowed stepsize (can be zero). On output nok and nbad are the
//number of good and bad (but retried and fixed) steps taken, and ystart is
//replaced by values at the end of the integration interval. derivs is the
//user-supplied routine for calculating the right-hand side derivative, while
//rkqs is the name of the stepper routine to be used.
void odeint(double ystart[], int nvar, double x1, double x2, double eps,
						double h1, double hmin, long *nok, long *nbad,
						double a[], double b[], long elements,
						void (*derivs)(double, double [], double [], double [],
													 double [], long))
{

	 int i;
	 long nstp;
	 double x, hnext, hdid, h, *yscal, *y, *dydx;
	 //double xsav;

	 yscal = (double *)malloc(nvar*sizeof(double));
	 y     = (double *)malloc(nvar*sizeof(double));
	 dydx  = (double *)malloc(nvar*sizeof(double));
	 //yscal=vector(1,nvar); y=vector(1,nvar); dydx=vector(1,nvar);

	 x = x1;
	 h = SIGN(h1,x2-x1);
   //*nok = (*nbad) = kount = 0;

	 for (i=0;i<nvar;i++)
	 		y[i]=ystart[i];

	 //if (kmax > 0) xsav=x-dxsav*2.0;    //Assures storage of first step

	 for (nstp=0;nstp<MAXSTP;nstp++)   //Take at most MAXSTP steps
	 {
		 (*derivs)(x,y,dydx,a,b,elements);

		 //Scaling used to monitor accuracy.
		 //This general-purpose choice can be modified if need be.
		 for (i=0;i<nvar;i++)
		 		yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+TINY;
		/*
		 if (kmax > 0 && kount < kmax-1 && fabs(x-xsav) > fabs(dxsav))
		 	{
				xp[++kount]=x;     //Store intermediate results.
				for (i=1;i<=nvar;i++) yp[i][kount]=y[i];
				xsav=x;
			}*/

			if ((x+h-x2)*(x+h-x1) > 0.0)  //If stepsize can overshoot, decrease
				h=x2-x;

			(*rkqs)(y,dydx,nvar,&x,h,eps,yscal,&hdid,&hnext, a, b, elements, derivs);
			if (hdid == h) ++(*nok);
			else ++(*nbad);
			if ((x-x2)*(x2-x1) >= 0.0)      //Are we done?
				{
					for (i=0;i<nvar;i++)
						ystart[i]=y[i];   //put final values into ystart

					/*if (kmax)
						{
							xp[++kount]=x;  //Save final step.
							for (i=1;i<=nvar;i++) yp[i][kount]=y[i];
						}*/

					free(dydx);  free(y);  free(yscal);
 					return;   //Normal exit.
				}
			if (fabs(hnext) <= hmin)
			{
				printf("Step size too small in odeint\n");
				exit(1);
			}
			h=hnext;
		}
		printf("Too many steps in routine odeint\n");
		exit(1);
}
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
