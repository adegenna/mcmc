import numpy as np
import matplotlib.pyplot as plt

prior_mu    = 0.0
prior_sigma = 0.01
mcmcoutfile = '/home/adegennaro/Projects/AEOLUS/mcmc/output/mcmc.out'

mcmc        = np.genfromtxt(mcmcoutfile , delimiter=',')
params      = np.ravel(mcmc[:,0:-1])
posterior   = mcmc[:,-1]

prior_x     = np.linspace(-0.02,0.02,100)
prior       = np.exp(-0.5*(prior_x-prior_mu)**2/prior_sigma**2)

plt.plot(prior_x,prior,'b')
plt.scatter(params,posterior,c='r')
plt.show()
