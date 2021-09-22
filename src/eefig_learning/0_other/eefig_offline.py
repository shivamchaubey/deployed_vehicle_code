# Tools
import numpy as np

import os, sys
abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path

# NOTE: Making python files in forlder scripts available to this file
sys.path.insert(1, os.path.join(abspath, "common"))
sys.path.insert(1, os.path.join(abspath, "visualization"))

# Imports
from granule import Granule
from eefig import EEFIG
from plot_granules import PlotGranules


# Evolving Ellipsoidal Fuzzy Information Granules
def EEFIG_offline (data):

    # PLOT
    plot = PlotGranules()

    # SETTINGS EEFIG
    settings = {}

    settings["nx"] = 3
    settings["nu"] = 2
    settings["lmbd"] = 0.9
    settings["effective_N"] = 200
    settings["thr_continuous_anomalies"] = 5
    settings["separation"] = 2
    settings["ff"] = 0

    eefig = EEFIG(settings)

    # Modify Initialization EEFIG
    buffer = 80
    nphi = 20

    aux_gran = Granule(eefig.p, data[:buffer, :])
    eefig.EEFIG = [aux_gran]
    eefig.nEEFIG = 1

    eefig.tracker_m = np.mean(data[:buffer, :], axis = 0) # Mean by columns

    for i in range(buffer, data.shape[0]):

        xk = data[i, :]

        # Detect & Update Counter
        eefig.update_anomalies(xk)

        # If New Granule Create
        if eefig.check_new_granule_creation(xk, i):
            eefig.create_new_granule()

        # Update Granule Distribution
        gran_idx = eefig.data_evaluation(xk)

        # Estimation of the A's matrices - Consequent Estimation via WLS
        # NOTE: Does not pass through in the first iterration
        if i > buffer:

            xr = data[i - nphi + 1 : i + 1, :eefig.nx] # xr containts the states x of the buffer (eq. 24)
            psik = data[i - nphi : i, :]

            eefig.create_using_WLS(gran_idx, xr, psik)

            # Set A and B values of the newly created Granule
            if eefig.EEFIG[-1].A.size == 0:
                eefig.EEFIG[-1].A = eefig.EEFIG[gran_idx].A
                eefig.EEFIG[-1].B = eefig.EEFIG[gran_idx].B

        # Plot Granules
        plot.save_xk(xk)
        plot.save_granules(eefig)

    plot.plot(eefig)

    return eefig
