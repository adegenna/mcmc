import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from InputFile import *
from MCMC import *
sys.path.append('../../')
import cahnhilliard_2d as ch

def main():
    """
    Main driver script for integrating the 2D Cahn Hilliard equations.

    **Inputs**

    ----------
    args : command line arguments
        Command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    **Outputs**

    -------
    inputs.outdir/results.txt : text file 
        time-integrated state
    """

    # Read inputs
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename_mcmc',\
                        metavar='inputfilename_mcmc',type=str,\
                        help='Filename of the input file for the mcmc sampler')
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file for the forward solver')
    args          = parser.parse_args()
    inputs_mcmc   = InputFile(args)
    inputs_solver = ch.InputFile.InputFile(args)
    inputs_mcmc.printInputs()
    inputs_solver.printInputs()

    # Physics setup
    C0         = np.genfromtxt(inputs_solver.initialstatepath , delimiter=',')
    state      = ch.CahnHilliardState.CahnHilliardState(C0)
    physics    = ch.CahnHilliardPhysics.CahnHilliardPhysics(inputs_solver, state)

    # MCMC sampler setup
    C_truth    = np.genfromtxt(inputs_mcmc.truestatepath)[-1]
    mcmc       = MCMC(inputs_mcmc)
    mcmc.set_true_state(C_truth)
    mcmc.set_forward_model(physics)
    mcmc.calculate_posterior()
    
    # Output
    mcmc.write(inputs_mcmc.outdir + "mcmc.out")
    
if __name__ == '__main__':
    main()
