import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import sys

class MCMC():
    """
    Class for MCMC sampling.
    """
    def __init__(self,inputs):
        self.inputs     = inputs
        self.P          = self.inputs.num_params
        self.param_iter = 0
        self.params     = []
        self.posterior  = []
        
    def set_forward_model(self,physics):
        """
        Method to set the forward model used by the MCMC sampler.
        """
        self.forward_model = physics

    def initialize_parameters(self,p0):
        """
        Method to initialize parameters.
        """        
        self.params = p0.reshape((-1,1))

    def append_parameters(self,pnew):
        """
        Method to append to parameters list.
        """
        self.params      = np.vstack([self.params,pnew])
        self.param_iter += 1
        
    def random_parameter_step(self):
        """
        Method to take a random step in parameter space.
        """
        p0   = self.params[self.param_iter]
        pnew = p0 + self.sigma_step * np.random.normal(0,1,(p0.shape))
        self.append_parameters(pnew)

    def compute_model_data_discrepancy_Tfinal(self,Umodel):
        """
        Method to compute discrepancy function between data and model.
        """
        Umodel_tf = Umodel[-1]
        return np.linalg.norm( self.U_truth - Umodel_tf )

    def compute_likelihood_gaussian(self,discrepancy):
        """
        Method to compute likelihood function with Gaussian assumption.
        """
        return np.exp( -0.5*(discrepancy)/(self.sigma_likelihood**2) )

    def compute_posterior_sample(self,param,likelihood):
        """
        Method to compute a sample of the posterior.
        """
        p0  = self.evaluate_prior(param)
        return likelihood * p0

    def append_posterior_sample(self,posterior):
        """
        Method to append a sample to the posterior record.
        """
        self.posterior = np.hstack( [self.posterior,posterior] )
        
    def calculate_posterior(self):
        """
        Main method for sampling from the posterior.
        """
        self.initialize_parameters(self.initial_parameters)
        for i in range(self.inputs.posterior_samples):
            if (i != 0):
                self.random_parameter_step()
            self.physics.solve()
            U_model     = self.physics.get_state_history()
            discrepancy = self.compute_model_data_discrepancy_Tfinal(Umodel)
            likelihood  = self.compute_likelihood_gaussian(discrepancy)
            posterior   = self.compute_posterior_sample(likelihood)
            self.append_posterior_sample(posterior)
            
