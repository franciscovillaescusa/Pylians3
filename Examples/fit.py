# This scripts is an example of a MCMC fit. We generate points from a linear law
# y = a*x+b and we add noise to them. We then try to obtain the values of a and b
# from the data

import numpy as np
import emcee,corner
from scipy.optimize import minimize
import sys,os

# This function returns the predicted y-values for the particular model considered
# x --------> array with the x-values of the data
# theta ----> array with the value of the model parameters [a,b]
def model_theory(x,theta):
    a,b = theta
    return a*x+b


# This functions returns the log-likelihood for the theory model
# theta -------> array with parameters to fit [a,b]
# x,y,dy ------> data to fit
def lnlike_theory(theta,x,y,dy):
    # put priors here
    if (theta[0]>1000.0) or (theta[0]<-1000.0) or (theta[1]>1000.0) \
            or (theta[1]<-1000.0):
        return -np.inf
    else:
        y_model = model_theory(x,theta)
        chi2 = -np.sum(((y-y_model)/dy)**2,dtype=np.float64)
        return chi2


#################################### INPUT ####################################
x_min, x_max = 0.0, 10.0 #minimum and maximum value for the x-axis
points = 100  #number of points to use in the fit
sigma  = 1.0  #standard deviation for the errors
theta_model = [3.0, 5.0]  #real value of the parameters
theta0      = [0.0, 0.0]  #intial guess of the value of the parameters

# MCMC parameters
nwalkers   = 100
chain_pts  = 10000
f_contours = 'Ellipses.pdf' 
###############################################################################

# find number of degrees of freedom and parameter space dimension
ndim = len(theta_model);  ndof = points - ndim


####################### GENERATE DATA #############################
# generate noise data from underlying model y=x
x  = np.linspace(x_min, x_max, points)
dy = np.random.randn(points)*sigma
y  = theta_model[0]*x + theta_model[1] + dy  # y = a*x + b + epsilon

# save data to file
np.savetxt('data.txt',np.transpose([x,y])) 
####################################################################


######################## FIT DATA WITH NUMPY #######################
chi2_func = lambda *args: -lnlike_theory(*args)
best_fit  = minimize(chi2_func,theta0,
                     args=(x,y,dy),method='Powell')
theta_best_fit = best_fit["x"]
chi2 = chi2_func(theta_best_fit,x,y,dy)*1.0/ndof
    
a, b  = theta_best_fit
print('\n##### NUMPY fit results #####')
print('a    = %7.3e'%a)
print('b    = %7.3e'%b)
print('chi2 = %7.3f\n'%chi2)
#####################################################################


########################## FIT WITH MCMC ############################
# starting point of each MCMC chain: small ball around best fit or estimate
alpha = 0.1 #you can play with this number
pos = [theta_best_fit*(1.0+alpha*np.random.randn(ndim)) for i in range(nwalkers)]

# run MCMC
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlike_theory,args=(x,y,dy))
sampler.run_mcmc(pos,chain_pts);  del pos

# get the points in the MCMC chains
samples = sampler.chain[:,300:,:].reshape((-1,ndim))
sampler.reset()

# make the plot and save it to file
fig = corner.corner(samples,truths=theta_model,labels=[r"$a$",r"$b$"])
fig.savefig(f_contours)

# compute best fit and associated error
a, b, =  map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
             zip(*np.percentile(samples, [16, 50, 84],axis=0)))
del samples, sampler
                                      
# compute chi^2 of best fit
theta_mcmc = [a[0],b[0]]
chi2 = chi2_func(theta_mcmc,x,y,dy)/ndof

# show results of MCMC
print('\n##### MCMC fit results #####')
print('a    = %7.3e + %.2e - %.2e'%(a[0],a[1],a[2]))
print('b    = %7.3e + %.2e - %.2e'%(b[0],b[1],b[2]))
print('chi2 = %7.3f'%chi2)

# save best-fit model to file
y_best_fit = model_theory(x,theta_mcmc)
np.savetxt('Best-fit.txt',np.transpose([x,y_best_fit]))
#####################################################################


