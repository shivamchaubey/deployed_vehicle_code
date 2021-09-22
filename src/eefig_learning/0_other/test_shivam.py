# Tools
import os, sys
import numpy as np
from math import pi
import scipy.io as sio

abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path

# NOTE: Making python files in forlder scripts available to this file
sys.path.insert(1, os.path.join(abspath, "scripts", "EEFIG_offline"))

# Imports
from eefig_offline import EEFIG_offline


def import_data_npy ():

    # IMPORT TEST DATA
    # ----------------------------------------------------
    files_dir = os.path.join(abspath, "test", "data", "d12_m04_y2021_hr19_min20_sec12")
    files = os.listdir(files_dir)

    for filename in files:
        data = filename.split('.')[0]
        globals()[data] = np.load(os.path.join(files_dir, filename), allow_pickle = True, encoding = 'latin1').item()
    # ----------------------------------------------------

    start_index = 175
    end_index = 900

    valid_control = control_his_resistive_estimation['duty_cycle'][start_index:end_index]
    valid_steer = control_his_resistive_estimation['steering'][start_index:end_index]

    valid_vx = np.array(enc_his_resistive_estimation['vx'][start_index:end_index])
    valid_X = np.array(fcam_his_resistive_estimation['X'][start_index:end_index])
    valid_Y = np.array(fcam_his_resistive_estimation['Y'][start_index:end_index])
    valid_w = np.array(imu_his_resistive_estimation['yaw_rate'][start_index:end_index])
    valid_w -= np.mean(valid_w[:10])
    valid_yaw = np.array(imu_his_resistive_estimation['yaw'][start_index:end_index])
    valid_yaw = -valid_yaw + valid_yaw[0] + pi/2

    valid_vx_MA = np.array(enc_MA_his_resistive_estimation['vx'][start_index:end_index])
    valid_X_MA = np.array(fcam_MA_his_resistive_estimation['X'][start_index:end_index])
    valid_Y_MA = np.array(fcam_MA_his_resistive_estimation['Y'][start_index:end_index])
    valid_w_MA = np.array(imu_MA_his_resistive_estimation['yaw_rate'][start_index:end_index])
    valid_w_MA -= np.mean(valid_w_MA[:10])
    valid_yaw_MA = np.array(imu_MA_his_resistive_estimation['yaw'][start_index:end_index])
    valid_yaw_MA = -valid_yaw_MA + valid_yaw_MA[0] + pi/2

    valid_time = np.array(enc_his_resistive_estimation['timestamp_ms'][start_index:end_index])
    valid_time = valid_time - valid_time[0]

    # PACKAGE TEST DATA
    # ----------------------------------------------------
    nx = 3 # number of states (vx, vy, w)
    nu = 2 # number of inputs (delta, acc)
    data = np.zeros([end_index - start_index, nx + nu])

    # States
    data[:, 0] = valid_vx_MA
    data[:, 1] = np.linspace(0.01, 0.2, end_index - start_index)
    data[:, 2] = valid_w_MA

    # Inputs
    data[:, 3] = valid_steer
    data[:, 4] = valid_control

    return data


def import_data_mat ():

    # IMPORT TEST DATA
    # ----------------------------------------------------
    data = sio.loadmat(os.path.join(abspath, "test", "data", "data_learn_EEFIG.mat"))['data_learn_EEFIG']

    data = data[0]
    for i in range(1, len(data)):
        if i == 1:
            data_ = data[i]
        else:
            data_ = np.vstack([data_, data[i]])

    # PACKAGE TEST DATA
    # ----------------------------------------------------
    data = data_[:, :5]

    return data


if __name__ == "__main__":

    data = import_data_mat()
    eefig = EEFIG_offline(data)
    print(eefig.nEEFIG)
