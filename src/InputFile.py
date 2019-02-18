import numpy as np
from pprint import pprint
import sys

class InputFile():
    """
    Class for packaging all input/config file options together.

    **Inputs**

    ----------
    args : command line arguments 
        (passed to constructor at runtime) command line arguments used in shell call for main driver script.

    **Options**

    ----------
    projdir : string
        Absolute path to project directory
    datadir : string
        Absolute path to data directory
    loaddir : string
        Absolute path to load directory
    outdir : string
        Absolute path to output directory
    initialstatepath : string
        Absolute path to initial state file
    epsilon : float
        Diffusion coefficient
    dt : float
        Timestep
    t_steps : int
        Number of timesteps
    saveperiod : int
        Sampling period for saving to output
    eyre_a : int
        Eyre time-scheme parameter
    """
    
    def __init__(self,args=[]):
        try:
            inputfilename         = args.inputfilename_mcmc
            inputfilestream       = open(inputfilename)
            self.projdir          = inputfilestream.readline().strip().split('= ')[1];
            self.datadir          = inputfilestream.readline().strip().split('= ')[1];
            self.loaddir          = inputfilestream.readline().strip().split('= ')[1];
            self.outdir           = inputfilestream.readline().strip().split('= ')[1];
            self.initialstatepath = inputfilestream.readline().strip().split('= ')[1];
            self.epsilon          = float(inputfilestream.readline().strip().split('= ')[1])
            self.dt               = float(inputfilestream.readline().strip().split('= ')[1])
            self.t_steps          = int(inputfilestream.readline().strip().split('= ')[1])
            self.saveperiod       = int(inputfilestream.readline().strip().split('= ')[1])
            self.eyre_a           = int(inputfilestream.readline().strip().split('= ')[1]);
            inputfilestream.close();
        except:
            print("Using no input file (blank initialization).")
    def printInputs(self):
        """
        Method to print all config options.
        """
        attrs = vars(self);
        print('\n');
        print("********************* INPUTS *********************")
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        print("**************************************************")
        print('\n');