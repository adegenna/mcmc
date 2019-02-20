import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import sys

class MCMC():
    """
    Class for MCMC sampling.
    """
    def __init__(self,inputs):
        self.initial_parameters = inputs.initial_parameters
        self.P                  = len(self.initial_parameters)
        self.sigma_likelihood   = inputs.sigma_likelihood
        self.sigma_step         = inputs.sigma_step
        self.posterior_samples  = inputs.posterior_samples
        self.prior_mu           = inputs.prior_mu
        self.prior_sigma        = inputs.prior_sigma
        self.param_iter         = 0
        self.params             = []
        self.posterior          = []
        self.outdir             = inputs.outdir
        
    def set_true_state(self,U_truth):
        """
        Method to set the true state.
        """
        self.U_truth = U_truth
        
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

    def compute_model_data_discrepancy_Tfinal(self,Umodel_tf):
        """
        Method to compute discrepancy function between data and model.
        """
        return np.linalg.norm( self.U_truth - Umodel_tf )

    def compute_likelihood_gaussian(self,discrepancy):
        """
        Method to compute likelihood function with Gaussian assumption.
        """
        return np.exp( -0.5*(discrepancy)**2/(self.sigma_likelihood**2) )

    def evaluate_gaussian_prior(self,x):
        """
        Method to evaluate Gaussian prior at a point x.
        """
        eval_x = 1.0
        for i in range(self.P):
            eval_x *= np.exp( -0.5*(x-self.prior_mu[i])**2/(self.prior_sigma[i]**2) )
        eval_x /= np.sqrt( (2*np.pi)**self.P * np.prod(self.prior_sigma) )
        return eval_x
        
    def compute_posterior_sample(self,param,param_m1,likelihood,likelihood_m1):
        """
        Method to compute a sample of the posterior:
        alpha = L*P / L_m1*P_m1
        """
        tol_zero     = 1e-2
        p0           = self.evaluate_gaussian_prior(param)
        p0_m1        = self.evaluate_gaussian_prior(param_m1)
        posterior    = likelihood * p0
        posterior_m1 = likelihood_m1 * p0_m1
        alpha        = posterior/(posterior_m1 + tol_zero)
        return posterior,alpha

    def append_posterior_sample(self,posterior):
        """
        Method to append a sample to the posterior record.
        """
        self.posterior = np.hstack( [self.posterior,posterior] )

    def decide_acceptance(self,alpha):
        dice = np.random.uniform(0,1)
        return (alpha > dice)

    def delete_current_params(self):
        self.params       = self.params[0:-1]
        self.param_iter -= 1
    
    def calculate_posterior(self):
        """
        Main method for sampling from the posterior.
        """
        self.initialize_parameters(self.initial_parameters)
        for i in range(self.posterior_samples):
            if (i != 0):
                self.random_parameter_step()
                self.forward_model.reset_state()
            params_i    = self.params[self.param_iter]
            print("Computing sample %d " %(i+1) + ": " + str(params_i))
            self.forward_model.set_parameters(params_i)
            self.forward_model.reset()
            self.forward_model.solve()
            U_model     = self.forward_model.get_current_state()
            U_model     = self.forward_model.state.state2D_to_1D( U_model )
            discrepancy = self.compute_model_data_discrepancy_Tfinal(U_model)
            likelihood  = self.compute_likelihood_gaussian(discrepancy)
            if (i == 0):
                params_im1     = params_i
                likelihood_im1 = likelihood
            posterior,alpha = self.compute_posterior_sample(params_i,params_im1,likelihood,likelihood_im1)
            bool_accept     = self.decide_acceptance(alpha)
            if bool_accept:
                print("Sample %d discrepancy/likelihood/posterior/alpha: %.2f/%.2f/%.2f/%.2f" %((i+1),discrepancy,likelihood,posterior,alpha) + " ACCEPT")
                self.append_posterior_sample(posterior)
                params_im1     = params_i.copy()
                likelihood_im1 = likelihood.copy()
            else:
                print("Sample %d discrepancy/likelihood/posterior/alpha: %.2f/%.2f/%.2f/%.2f" %((i+1),discrepancy,likelihood,posterior,alpha) + " REJECT")
                self.delete_current_params()

    def write(self,outfile):
        """
        Writes to outdir/ the results of the sampling, ie. (parameter_i , posterior_i).
        """
        f = open(outfile,'w+')
        for i in range(self.params.shape[0]):
            line = str(self.params[i])[1:-1] + " , " + str(self.posterior[i])
            f.write(line)
            f.write('\n')            
        f.close()
