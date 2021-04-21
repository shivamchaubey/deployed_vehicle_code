#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that (1) you retain this notice
# and (2) you provide clear attribution to UC Berkeley, including a link
# to http://barc-project.com
#
# Author: J. Noonan
# Email: jpnoonan@berkeley.edu
#
# This code provides a way to see the car's trajectory, orientation, and velocity profile in
# real time with referenced to the track defined a priori.
#
# ---------------------------------------------------------------------------
import sys

import rospy
import numpy as np
from plotter.msg import simulatorStates, hedge_imu_fusion
import matplotlib.pyplot as plt
import pdb
import matplotlib.patches as patches
from tf import transformations
import scipy.io as sio


np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

# For printing with less decimals
# print "Vx = {0:0.2f}".format(estimatedStates[0])
class vehicle_real():

    def __init__(self):
        rospy.Subscriber("hedge_fusion_filter",hedge_imu_fusion,self.hedge_callback,queue_size=1)
        self.real_x   = 0.0
        self.real_y   = 0.0
        self.real_psi = 0.0

    def hedge_callback(self, msg):

        self.real_x = msg.x_m
        self.real_y = msg.y_m 
        quaternion = (msg.qx,msg.qy,msg.qz,msg.qw)
        euler = transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        self.real_psi = yaw
        
class vehicle_sim():
    """Object collecting closed loop data points
    Attributes:
        updateInitialConditions: function which updates initial conditions and clear the memory
    """
    def __init__(self):

        rospy.Subscriber("simulatorStates", simulatorStates, self.simState_callback,queue_size=1)
        self.sim_x   = 0.0
        self.sim_y   = 0.0
        self.sim_psi = 0.0

    def simState_callback(self, msg):

        self.sim_x = msg.x
        self.sim_y = msg.y 
        self.sim_psi = msg.psi


def main():

    rospy.init_node("realTimePlotting")

    sim = vehicle_sim()
    real = vehicle_real()
    loop_rate   = 20
    rate        = rospy.Rate(loop_rate)
    lim = 5
    plim = 0.2
    ( plt, fig, axtr, line_planning, point_simc, point_rlc, line_SS, line_sim, line_rl, rec_rl,
         rec_sim, rec_planning ) = _initializeFigure_xy(lim,lim)

    sim_x_his  = []
    sim_y_his  = []
    real_x_his = []
    real_y_his = []


    while not rospy.is_shutdown():

        #simulated vehicle
        x_sim   = sim.sim_x
        y_sim   = sim.sim_y
        psi_sim = sim.sim_psi
        sim_x_his.append(x_sim)
        sim_y_his.append(y_sim)

        x_real = real.real_x
        y_real = real.real_y
        psi_real = real.real_psi
        real_x_his.append(x_real)
        real_y_his.append(y_real)

        # plot dimension update
        x_p = max([abs(x_sim),abs(x_real)])
        y_p = max([abs(y_sim),abs(y_real)])
        # print ('x_p',x_p,'lim',lim)
        if (lim < abs(x_p)):
            lim = abs(x_p)
            lim = int((1.0+plim)*lim)
            plt.xlim([-lim,lim])
            plt.ylim([-lim,lim])
        if (lim < abs(y_p)):
            lim = abs(y_p)
            plt.xlim([-lim,lim])
            plt.ylim([-lim,lim])
        
        l = 0.42; w = 0.19

        # print ('lim',lim,'x',x)

        #plotting real vehicle
        car_real_x, car_real_y = getCarPosition(x_real, y_real, psi_real, w, l)
        rec_rl.set_xy(np.array([car_real_x, car_real_y]).T)
        point_rlc.set_data(x_real, y_real)
        line_rl.set_data(real_x_his, real_y_his)

        # plotting simulated data
        car_sim_x, car_sim_y = getCarPosition(x_sim, y_sim, psi_sim, w, l)
        rec_sim.set_xy(np.array([car_sim_x, car_sim_y]).T)
        point_simc.set_data(x_sim, y_sim)
        line_sim.set_data(sim_x_his, sim_y_his)

        fig.canvas.draw()

        rate.sleep()



def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y







# ===================================================================================================================================== #
# ============================================================= Internal Functions ==================================================== #
# ===================================================================================================================================== #



# def _initializeFigure_xy(map, mode, X_Planner_Pts, Y_Planner_Pts):
def _initializeFigure_xy(x_lim,y_lim):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()
    # Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    # # Points1 = np.zeros((Points, 2))
    # # Points2 = np.zeros((Points, 2))
    # # Points0 = np.zeros((Points, 2))
    # Points1 = np.zeros((Points, 3))
    # Points2 = np.zeros((Points, 3))
    # Points0 = np.zeros((Points, 3))

    # for i in range(0, int(Points)):
    #     Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
    #     Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
    #     Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    # plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    # plt.plot(Points0[:, 0], Points0[:, 1], '--')
    # plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    # plt.plot(Points2[:, 0], Points2[:, 1], '-b')

    # These lines plot the planned offline trajectory in the main figure:
    # plt.plot(X_Planner_Pts[0, 0:290], Y_Planner_Pts[0, 0:290], '--r')
    # plt.plot(X_Planner_Pts[0, 290:460], Y_Planner_Pts[0, 290:460], '--r')
    # plt.plot(X_Planner_Pts[0, :], Y_Planner_Pts[0, :], '--r')


    line_sim,       = axtr.plot(xdata, ydata, '-k')
    line_rl,        = axtr.plot(xdata, ydata, '-b')  # Plots the traveled positions
    point_simc,     = axtr.plot(xdata, ydata, '-or')       # Plots the current positions
    line_SS,        = axtr.plot(xdata, ydata, 'og')
    point_rlc,      = axtr.plot(xdata, ydata, '-or')
    line_planning,  = axtr.plot(xdata, ydata, '-ok')

    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])

    rec_rl = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10,label='Real vehicle')
    axtr.add_patch(rec_rl)
    # # Vehicle:
    rec_sim = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10,label='simulated vehicle')
    axtr.add_patch(rec_sim)
    plt.legend()
    # # Planner vehicle:
    rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)
    # axtr.add_patch(rec_planning)



    plt.show()

    return plt, fig, axtr, line_planning, point_simc, point_rlc, line_SS, line_sim, line_rl, rec_rl, rec_sim, rec_planning






def _initializeFigure(map):
    xdata = []; ydata = []
    plt.ion()
    fig = plt.figure(figsize=(40,20))

    axvx = fig.add_subplot(3, 2, 1)
    linevx, = axvx.plot(xdata, ydata, 'or-')
    axvx.set_ylim([0, 1.5])
    plt.ylabel("vx")
    plt.xlabel("t")

    axvy = fig.add_subplot(3, 2, 2)
    linevy, = axvy.plot(xdata, ydata, 'or-')
    plt.ylabel("vy")
    plt.xlabel("s")

    axwz = fig.add_subplot(3, 2, 3)
    linewz, = axwz.plot(xdata, ydata, 'or-')
    plt.ylabel("wz")
    plt.xlabel("s")

    axepsi = fig.add_subplot(3, 2, 4)
    lineepsi, = axepsi.plot(xdata, ydata, 'or-')
    axepsi.set_ylim([-np.pi/2,np.pi/2])
    plt.ylabel("epsi")
    plt.xlabel("s")

    axey = fig.add_subplot(3, 2, 5)
    lineey, = axey.plot(xdata, ydata, 'or-')
    axey.set_ylim([-map.width,map.width])
    plt.ylabel("ey")
    plt.xlabel("s")

    Points = np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    axtr = fig.add_subplot(3, 2, 6)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    line_tr, = axtr.plot(xdata, ydata, '-or')
    line_pred, = axtr.plot(xdata, ydata, '-or')

    plt.show()

    return fig, linevx, linevy, linewz, lineepsi, lineey, line_tr, line_pred



# ===================================================================================================================================== #
# ========================================================= End of Internal Functions ================================================= #
# ===================================================================================================================================== #

if __name__ == '__main__':
    try:
        main()

    except rospy.ROSInterruptException:
        pass
