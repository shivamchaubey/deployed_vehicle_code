# Tools
import os, sys
import numpy as np

abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path (Floder "EEFIG_Learing")

# NOTE: Making python files in forlder scripts available to this file
sys.path.insert(1, os.path.join(abspath, "scripts"))

# Imports
from lpv_mpc_eefig.visualization import PlotGranules
from LPV_MPC_EEFIG import LPV_MPC_EEFIG
from lpv_mpc_eefig.common import Granule


def get_lpv_mpc_eefig ():

    ###############
    # SETUP EEFIG #
    ###############

    settings = {}

    settings["nx"] = 2
    settings["nu"] = 1
    settings["lmbd"] = 0.9 # USELESS?
    settings["effective_N"] = 200 # USELESS?
    settings["thr_continuous_anomalies"] = 5
    settings["separation"] = 2
    settings["ff"] = 0.975
    settings["nphi"] = 20

    lpv_mpc_eefig = LPV_MPC_EEFIG(settings)
    
    return lpv_mpc_eefig



def main_3D():
    mean = [0, 0, 0]
    cov = [[100, 0, 0], [0, 1, 0], [0, 0, 1]]  # diagonal covariance

    theta_x = np.deg2rad(360*np.random.rand())
    theta_y = np.deg2rad(360*np.random.rand())
    theta_z = np.deg2rad(360*np.random.rand())
    R_x = np.array([[1.0, 0, 0],
                    [0.0, np.cos(theta_x), -np.sin(theta_x)],
                    [0.0, np.sin(theta_x), np.cos(theta_x)]])

    R_y = np.array([[np.cos(theta_y), 0.0, np.sin(theta_y)],
                    [0.0, 1.0, 0],
                    [-np.sin(theta_y), 0.0, np.cos(theta_y)]])

    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0.0],
                    [np.sin(theta_z), np.cos(theta_z), 0.0],
                    [0.0, 0.0, 1.0]])
    R = R_x @ R_y @ R_z

    print(R)
    cov = np.linalg.inv(R) @ cov @ R

    x = np.random.multivariate_normal(mean, cov, 5000)
    x = np.array(x)
    print(x)
    lpv_mpc_eefig = get_lpv_mpc_eefig()
    plot = PlotGranules()
    granule = Granule(lpv_mpc_eefig.p)
    
    granule.C = np.linalg.inv(cov)
    granule.m = np.array(mean)
    lpv_mpc_eefig.set_granules([granule])

    for i in range(x.shape[0]):
        plot.save_xk(x[i,:])
        plot.save_granules(lpv_mpc_eefig)

    anim = plot.plot(lpv_mpc_eefig)
    plot.show()

def main_2D():
    mean = [0, 0]
    cov = [[100, 0], [0, 1]]  # diagonal covariance

    theta = np.deg2rad(10)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

    cov = np.linalg.inv(R) @ cov @ R

    x = np.random.multivariate_normal(mean, cov, 5000)
    x = np.array(x)
    print(x)
    lpv_mpc_eefig = get_lpv_mpc_eefig()
    plot = PlotGranules()
    granule = Granule(lpv_mpc_eefig.p)
    
    granule.C = np.linalg.inv(cov)
    granule.m = np.array([mean])
    lpv_mpc_eefig.set_granules([granule])

    for i in range(x.shape[0]):
        plot.save_xk(x[i,:])
        plot.save_granules(lpv_mpc_eefig)

    anim = plot.plot(lpv_mpc_eefig)
    plot.show()

if __name__ == "__main__":
    # main_2D()
    main_3D()
