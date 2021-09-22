# Tools
import pandas as pd
import numpy as np


# Load CSV/ROS Data
def load_data(path_csv_file, topics, fix_dt, mode="interp"):

    # Load
    data = pd.read_csv(path_csv_file)
    print "data", data.head(n=2)
    data = data[topics]  # select topics

    # fill nan values by interpolating row values
    data.interpolate(method='linear', inplace=True)
    data = data.dropna()  # delete first nan rows

    # Convert to Numpy
    data = data.to_numpy()
    time = data[:, 0] - data[0, 0]  # substract for timestamp
    data = data[:, 1:]  # remove time


    # INTERPOLATE MODE
    if mode == "interp":
        print "fix_dt",fix_dt
        fix_time = np.linspace(0, time[-1], num=int(time[-1] / fix_dt))

        fix_data = np.zeros([len(fix_time), data.shape[1]])
        for i in range(data.shape[1]):
            fix_data[:, i] = np.interp(fix_time, time, data[:, i])

        return fix_data, fix_dt, fix_time


    # ONLY REAL TIME
    elif mode == "exact":

        mean_dt = 0
        fix_time = np.array([time[0]])
        fix_data = data[0]

        for i in range(1, data.shape[0]):

            step = time[i] - fix_time[-1]  # step

            if step > fix_dt:  # if delta time (dt) has passed, save value
                mean_dt += step
                fix_time = np.hstack([fix_time, time[i]])
                fix_data = np.vstack([fix_data, data[i]])

        mean_dt /= fix_data.shape[0]

        return fix_data, mean_dt, fix_time


    # DEFAULT
    else:
        return data, -1, time


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_torque_BCNeMotorsport(torque):

    r = 0.254
    mass = 240

    return torque/(mass*r) * 3.67


def normalize_brake_BCNeMotorsport(brake):

    mass = 240
    ApisF = 4.9087 / 10000
    ApisR = 4.9087 / 10000
    Rreff = 0.0779
    Rdyn = 0.2286
    mu_brakes = 0.45
    Nr = 4
    Nf = 8

    return 100 * brake * (100000 * mu_brakes * Rreff * (ApisF * Nf + ApisR * Nr)) / (Rdyn * mass)
