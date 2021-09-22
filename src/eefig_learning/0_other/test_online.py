# Tools
import os, sys
import numpy as np

abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path (Floder "EEFIG_Learing")

# NOTE: Making python files in forlder scripts available to this file
sys.path.insert(1, os.path.join(abspath, "scripts"))

# Imports
from initial_mat_data import *
from lpv_mpc_eefig.visualization import PlotGranules
from LPV_MPC_EEFIG import LPV_MPC_EEFIG


# MAIN
def main ():

    # Load Data From .mat Files (MATLAB type file)
    data_input_EEFIG = data_conv_EEFIG(os.path.join(abspath, "test", "data", "data_learn_EEFIG.mat"))

    # SETTINGS EEFIG
    settings = {}

    settings["nx"] = 3
    settings["nu"] = 2
    settings["lmbd"] = 0.9 # USELESS?
    settings["effective_N"] = 200 # USELESS?
    settings["thr_continuous_anomalies"] = 5
    settings["separation"] = 0.1
    settings["ff"] = 0.975
    settings["nphi"] = 20

    lpv_mpc_eefig = LPV_MPC_EEFIG(settings)

    # Load Configuration
    #lpv_mpc_eefig.eefig.load(os.path.join(abspath, "saves", "EEFIG_run_0"))

    # PLOT
    plot = PlotGranules()

    # Execute
    for i in range(len(data_input_EEFIG['x'])):

        x   = data_input_EEFIG.x[i].tolist() # states
        u   = data_input_EEFIG.u[i].tolist() # commands

        xk = np.hstack([x, u])
       
        lpv_mpc_eefig.update(xk)
        A, B = lpv_mpc_eefig.get_LPV_matrices(xk)

        # Plot Granules
        plot.save_xk(xk)
        plot.save_granules(lpv_mpc_eefig)
        plot.save_tracker(lpv_mpc_eefig)
        
        print(A)
        print(B)

        print("=============")

    # Save Configuration
    # lpv_mpc_eefig.eefig.save(os.path.join(abspath, "saves", "EEFIG_run_0"))

    # Plot
    anim = plot.plot(lpv_mpc_eefig)
    plot.show()

    print(lpv_mpc_eefig.nEEFIG)


# MAIN
if __name__ == "__main__":
    main()
