# Tools
import os, sys
import numpy as np
import matplotlib.pyplot as plt

abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path (Floder "EEFIG_Learing")

# NOTE: Making python files in forlder scripts available to this file
sys.path.insert(1, os.path.join(abspath, "scripts"))

# Imports
from LPV_MPC_EEFIG import LPV_MPC_EEFIG
from lpv_mpc_eefig.common.offline_learning import find_clusters, train

# Others
from utils import load_data


# Main
def main ():

    ############
    # SETTINGS #
    ############
    
    debug           = True
    remove          = True
    silhouette      = False
    expand_packs    = True

    moving_avg      = False
    mv              = 10

    ## the number of clusters??
    n_clusters      = 1

    #############
    # LOAD DATA #
    #############

    path_csv_file = os.path.join(abspath, "data", "csv", "2021-09-22-08-41-01.csv")

    # path_csv_file = os.path.join(abspath, "data", "csv", "0.8_vel_2021-06-03-16-23-21.csv")
    topics = ["__time","/est_state_info/vx","/est_state_info/vy","/est_state_info/yaw_rate","/control/accel/data","/control/steering/data"]

    desired_dt = 1./65 # the controller frequency
    data, dt, time = load_data(path_csv_file, topics, desired_dt, mode="interp")
    print("dt: ", dt)

    data = [data]

    # #########
    # # LEARN #
    # #########

    labels, n_clusters = find_clusters (data, n_clusters, silhouette=silhouette, debug=debug)

    lpv_mpc_eefig = get_lpv_mpc_eefig()
    lpv_mpc_eefig = train (lpv_mpc_eefig, data, labels, expand_packs=expand_packs, remove=remove)

    ########
    # SAVE #
    ########

    lpv_mpc_eefig.save(os.path.join(abspath, "saves", "rc_car_model1"))
    print("Saved successfully!")


def get_lpv_mpc_eefig ():

    ###############
    # SETUP EEFIG #
    ###############

    settings = {}

    settings["nx"] = 3
    settings["nu"] = 2
    settings["lmbd"] = 0.9 # USELESS?
    settings["effective_N"] = 200 # USELESS?
    settings["thr_continuous_anomalies"] = 5
    settings["separation"] = 2
    settings["ff"] = 0.975
    settings["nphi"] = 20

    lpv_mpc_eefig = LPV_MPC_EEFIG(settings)
    
    return lpv_mpc_eefig


# Main
if __name__ == "__main__":
    main()

