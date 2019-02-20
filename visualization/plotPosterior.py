import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def gaussian_kernel_sampling(x,y,xgrid,sigma):
    kernel = lambda x,mu,sigma : np.exp(-0.5*(x-mu)**2/sigma**2)
    ygrid  = np.zeros_like(xgrid)
    for i in range(len(x)):
        ygrid += kernel(xgrid,x[i],sigma)
    ygrid *= y.max() / ygrid.max()
    return ygrid

def gaussian_kernel_regression(x,y,xgrid,sigma):
    kernel = lambda x,mu,sigma : np.exp(-0.5*(x-mu)**2/sigma**2)
    ygrid  = np.zeros_like(xgrid)
    

    
prior_mu    = 0.02
prior_sigma = 0.01
mcmcoutfile = '/home/adegennaro/Projects/AEOLUS/mcmc/output/mcmc.out'
burnin      = 0

mcmc        = np.genfromtxt(mcmcoutfile , delimiter=',')
params      = np.ravel(mcmc[burnin:,0:-1])
posterior   = mcmc[burnin:,-1]

prior_x     = np.linspace(-2,2,100)*prior_sigma + prior_mu
prior       = np.exp(-0.5*(prior_x-prior_mu)**2/prior_sigma**2)

# Interpolate posterior
posterior_x      = np.linspace(params.min(),params.max(),1000)
posterior_interp = gaussian_kernel_sampling(params,posterior,posterior_x,sigma=prior_sigma/20.)

plt.plot(prior_x,prior,'b')
plt.scatter(params,posterior,c='r')
plt.plot(posterior_x,posterior_interp,'r')
plt.legend(['Prior','Posterior Distribution','Posterior Values'],fontsize=15)
plt.show()
