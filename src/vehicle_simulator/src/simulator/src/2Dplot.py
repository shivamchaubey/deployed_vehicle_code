#!/usr/bin/env python


import os
import sys
sys.path.append(sys.path[0]+'/ControllerObject')
sys.path.append(sys.path[0]+'/Utilities')
sys.path.append(sys.path[0]+'/data')


import numpy as np
import pdb
import numpy.linalg as la
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simulator.msg import simulatorStates
from sensor_fusion.msg import sensorReading, control, hedge_imu_fusion, hedge_imu_raw
from math import sin, cos, atan, pi
from trackInitialization import Map




# ======================================================================================================================
# ======================================================================================================================
# ====================================== Internal utilities functions ==================================================
# ======================================================================================================================
# ======================================================================================================================

def computeAngle(point1, origin, point2):
    # The orientation of this angle matches that of the coordinate system. Tha is why a minus sign is needed
    v1 = np.array(point1) - np.array(origin)
    v2 = np.array(point2) - np.array(origin)
    #
    # cosang = np.dot(v1, v2)
    # sinang = la.norm(np.cross(v1, v2))
    #
    # dp = np.dot(v1, v2)
    # laa = la.norm(v1)
    # lba = la.norm(v2)
    # costheta = dp / (laa * lba)

    dot = v1[0] * v2[0] + v1[1] * v2[1]  # dot product between [x1, y1] and [x2, y2]
    det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    return angle # np.arctan2(sinang, cosang)


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res


# def unityTestChangeOfCoordinates(map, ClosedLoopData):
#     """For each point in ClosedLoopData change (X, Y) into (s, ey) and back to (X, Y) to check accurancy
#     """
#     TestResult = 1
#     for i in range(0, ClosedLoopData.x.shape[0]):
#         xdat = ClosedLoopData.x
#         xglobdat = ClosedLoopData.x_glob

#         s, ey, _, _ = map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5], xglobdat[i, 3])
#         v1 = np.array([s, ey])
#         v2 = np.array(xdat[i, 4:6])
#         v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
#         v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
#         # print v1, v2, np.dot(v1 - v2, v1 - v2), np.dot(v3 - v4, v3 - v4)

#         if np.dot(v3 - v4, v3 - v4) > 0.00000001:
#             TestResult = 0
#             print "ERROR", v1, v2, v3, v4
#             pdb.set_trace()
#             v1 = np.array(map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5]))
#             v2 = np.array(xdat[i, 4:6])
#             v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
#             v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
#             print np.dot(v3 - v4, v3 - v4)
#             pdb.set_trace()

#     if TestResult == 1:
#         print "Change of coordinates test passed!"



def unityTestChangeOfCoordinates(map, ClosedLoopData):
    """For each point in ClosedLoopData change (X, Y) into (s, ey) and back to (X, Y) to check accurancy
    """
    TestResult = 1
    for i in range(0, ClosedLoopData.x.shape[0]):
        xdat = ClosedLoopData.x
        xglobdat = ClosedLoopData.x_glob

        s, ey, _, _ = map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5], xglobdat[i, 3])
        v1 = np.array([s, ey])
        v2 = np.array(xdat[i, 4:6])
        v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
        v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
        # print v1, v2, np.dot(v1 - v2, v1 - v2), np.dot(v3 - v4, v3 - v4)

        if np.dot(v3 - v4, v3 - v4) > 0.00000001:
            TestResult = 0
            print ("ERROR", v1, v2, v3, v4)
            pdb.set_trace()
            v1 = np.array(map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5]))
            v2 = np.array(xdat[i, 4:6])
            v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
            v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
            print (np.dot(v3 - v4, v3 - v4))
            pdb.set_trace()

    if TestResult == 1:
        print ("Change of coordinates test passed!")


def _initializeFigure_xy(map):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    axtr = plt.axes()

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    # Points1 = np.zeros((Points, 2))
    # Points2 = np.zeros((Points, 2))
    # Points0 = np.zeros((Points, 2))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))    

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o') #points on center track
    # np.save('inner_track',np.array([Points0[:, 0], Points0[:, 1]]))
    plt.plot(Points1[:, 0], Points1[:, 1], '-b') # inner track
    plt.plot(Points2[:, 0], Points2[:, 1], '-b') #outer track



    line_cl,        = axtr.plot(xdata, ydata, '-k', label = 'LPVerr MPC error model', linewidth = 2, alpha = 0.5)
    line_est,    = axtr.plot(xdata, ydata, '--or')  # Plots the traveled positions
    line_gps_cl,    = axtr.plot(xdata, ydata, '--ob')  # Plots the traveled positions
    line_tr,        = axtr.plot(xdata, ydata, '-r', label = 'LPV Observer model', linewidth = 6, alpha = 0.5)       # Plots the current positions
    line_SS,        = axtr.plot(xdata, ydata, '-g', label = 'Non-linear full state model', linewidth = 10, alpha = 0.5)
    line_pred,      = axtr.plot(xdata, ydata, '-or')
    line_planning,  = axtr.plot(xdata, ydata, '-ok')
    
    l = 0.4; w = 0.2 #legth and width of the car

    v = np.array([[ 0.4,  0.2],
                  [ 0.4, -0.2],
                  [-0.4, -0.2],
                  [-0.4,  0.2]])

    rec_est = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10)
    axtr.add_patch(rec_est)

    # Vehicle:
    # rec_sim = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10)

#     if mode == "simulations":
    # axtr.add_patch(rec_sim)    

    # Planner vehicle:
    # rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)


    # plt.show()
    # plt.pause(2)
    plt.legend()
    return fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_est, line_gps_cl, rec_est


def errorFigure2(map):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    axtr = plt.axes()

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    # Points1 = np.zeros((Points, 2))
    # Points2 = np.zeros((Points, 2))
    # Points0 = np.zeros((Points, 2))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))    

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o') #points on center track
    # np.save('inner_track',np.array([Points0[:, 0], Points0[:, 1]]))
    plt.plot(Points1[:, 0], Points1[:, 1], '-b') # inner track
    plt.plot(Points2[:, 0], Points2[:, 1], '-b') #outer track



    line_cl,        = axtr.plot(xdata, ydata, '-k', legend = 'LPV error model', linewidth = 5, alpha = 0.5)
    line_est,    = axtr.plot(xdata, ydata, '--or')  # Plots the traveled positions
    line_gps_cl,    = axtr.plot(xdata, ydata, '--ob')  # Plots the traveled positions
    line_tr,        = axtr.plot(xdata, ydata, '-r',"LPV estimation model", linewidth = 3, alpha = 0.3)       # Plots the current positions
    line_SS,        = axtr.plot(xdata, ydata, '-g', label = 'Non-linear model', linewidth = 10, alpha = 0.5)
    line_pred,      = axtr.plot(xdata, ydata, '-or')
    line_planning,  = axtr.plot(xdata, ydata, '-ok')
    
    l = 0.4; w = 0.2 #legth and width of the car

    v = np.array([[ 0.4,  0.2],
                  [ 0.4, -0.2],
                  [-0.4, -0.2],
                  [-0.4,  0.2]])

    rec_est = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10)
    axtr.add_patch(rec_est)

    # Vehicle:
    # rec_sim = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10)

#     if mode == "simulations":
    # axtr.add_patch(rec_sim)    

    # Planner vehicle:
    # rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)


    # plt.show()
    # plt.pause(2)
    plt.legend()
    return fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_est, line_gps_cl, rec_est



def errorFigure():
    xdata = []; ydata = []
    fig, axs = plt.subplots(3, 3, figsize=(10,10))

    plt.ion()

    ### Estimator track error
    line_est_s,      = axs[0,0].plot(xdata, ydata, '-k', label = 's est error model')
    line_est_ey,     = axs[0,1].plot(xdata, ydata, '-r', label = 'ey est error model')  # Plots the traveled positions
    line_est_epsi,   = axs[0,2].plot(xdata, ydata, '-b', label = 'epsi est error model')  # Plots the traveled positions

    axs[0,0].legend(loc = 'lower left')
    axs[0,0].grid()
    axs[0,1].legend(loc = 'lower left')
    axs[0,1].grid()
    axs[0,2].legend(loc = 'lower left')
    axs[0,2].grid()

    # axs[0,:].set_title("Vehicle error w.r.t MAP for Observer")
    ### Vehicle nonlinear track error
    line_nl_s,      = axs[1,0].plot(xdata, ydata, '-k', label = 's NL model')
    line_nl_ey,     = axs[1,1].plot(xdata, ydata, '-r', label = 'ey NL model')  # Plots the traveled positions
    line_nl_epsi,   = axs[1,2].plot(xdata, ydata, '-b', label = 'epsi NL model')  # Plots the traveled positions

    axs[1,0].legend(loc = 'lower left')
    axs[1,0].grid()
    axs[1,1].legend(loc = 'lower left')
    axs[1,1].grid()
    axs[1,2].legend(loc = 'lower left')
    axs[1,2].grid()

    # axs[1,:].set_title("Vehicle error w.r.t MAP non-linear model")
    
    ### Vehicle LPV error model error 
    line_LPVerr_s,      = axs[2,0].plot(xdata, ydata, '-k', label = 's LPVerr model')
    line_LPVerr_ey,     = axs[2,1].plot(xdata, ydata, '-r', label = 'ey LPVerr model')  # Plots the traveled positions
    line_LPVerr_epsi,   = axs[2,2].plot(xdata, ydata, '-b', label = 'epsi LPVerr model')  # Plots the traveled positions
    
    axs[2,0].legend(loc = 'lower left')
    axs[2,0].grid()
    axs[2,1].legend(loc = 'lower left')
    axs[2,1].grid()
    axs[2,2].legend(loc = 'lower left')
    axs[2,2].grid()


    # axs[2,:].set_title("Vehicle error w.r.t MAP from LPVerr model for MPC")


    return plt, fig, line_est_s, line_est_ey, line_est_epsi, line_nl_s, line_nl_ey, line_nl_epsi, line_LPVerr_s, line_LPVerr_ey, line_LPVerr_epsi


def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y



class vehicle_control(object):
    """ Object collecting CMD command data
    Attributes:
        Input command:
            1.a 2.df
        Time stamp
            1.t0  2.curr_time
    """
    def __init__(self,t0):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """
        rospy.Subscriber('control/accel', Float32, self.accel_callback, queue_size=1)
        rospy.Subscriber('control/steering', Float32, self.steering_callback, queue_size=1)

        # ECU measurement
        self.duty_cycle  = 0.0 #dutycyle
        self.steer = 0.0

        # time stamp
        self.t0         = t0
        self.curr_time_dc  = rospy.get_rostime().to_sec() - self.t0
        self.curr_time_steer  = rospy.get_rostime().to_sec() - self.t0

    def accel_callback(self,data):
        """Unpack message from sensor, ECU"""
        self.curr_time_dc = rospy.get_rostime().to_sec() - self.t0
        self.duty_cycle  = data.data

    def steering_callback(self,data):
        self.curr_time_steer = rospy.get_rostime().to_sec() - self.t0
        self.steer = data.data

    def data_retrive(self, msg):

        msg.timestamp_ms_DC = self.curr_time_dc
        msg.timestamp_ms_steer = self.curr_time_steer
        msg.duty_cycle  = self.duty_cycle
        msg.steer = self.steer
        return msg


# class EstimatorData(object): ## for real vehicle 
#     """Data from estimator"""
#     def __init__(self):
#         """Subscriber to estimator"""
#         rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback)
#         self.offset_x = 0.46822099
#         self.offset_y = -1.09683919
#         self.offset_yaw = -pi/2
#         self.R = np.array([[cos(self.offset_yaw),-sin(self.offset_yaw)],[sin(self.offset_yaw), cos(self.offset_yaw)]])
#         self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
#     def estimator_callback(self, msg):
#         """
#         Unpack the messages from the estimator
#         """
#         self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw + self.offset_yaw]).T
#         self.CurrentState[3:5] = np.dot(self.R,self.CurrentState[3:5])
#         self.CurrentState[3:5] = self.CurrentState[3:5] - np.array([self.offset_x, self.offset_y]).T



class EstimatorData(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback)
        self.offset_x = 0.
        self.offset_y = 0.
        self.offset_yaw = 0
        self.R = np.array([[cos(self.offset_yaw),-sin(self.offset_yaw)],[sin(self.offset_yaw), cos(self.offset_yaw)]])
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def estimator_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw + self.offset_yaw]).T
        self.CurrentState[3:5] = np.dot(self.R,self.CurrentState[3:5])
        self.CurrentState[3:5] = self.CurrentState[3:5] - np.array([self.offset_x, self.offset_y]).T



class Vehicle_states(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        rospy.Subscriber('vehicle_simulatorStates', simulatorStates, self.vehicle_state_callback)
        self.offset_x = 0.
        self.offset_y = 0.
        self.offset_yaw = 0.

        self.x      = 0. 
        self.y      = 0.
        self.vx     = 0.
        self.vy     = 0.
        self.yaw    = 0. 
        self.omega  = 0.

        self.R = np.array([[cos(self.offset_yaw),-sin(self.offset_yaw)],[sin(self.offset_yaw), cos(self.offset_yaw)]])
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def vehicle_state_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.omega, msg.x, msg.y, msg.yaw + self.offset_yaw]).T
        # self.CurrentState[3:5] = np.dot(self.R,self.CurrentState[3:5])
        # self.CurrentState[3:5] = self.CurrentState[3:5] - np.array([self.offset_x, self.offset_y]).T


class Vehicle_states_LPVerr(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        rospy.Subscriber('vehicleLPVerr_simulatorStates', simulatorStates, self.vehicle_state_callback)
        self.offset_x = 0.
        self.offset_y = 0.
        self.offset_yaw = 0.

        self.x      = 0. 
        self.y      = 0.
        self.vx     = 0.
        self.vy     = 0.
        self.yaw    = 0. 
        self.omega  = 0.

        self.R = np.array([[cos(self.offset_yaw),-sin(self.offset_yaw)],[sin(self.offset_yaw), cos(self.offset_yaw)]])
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def vehicle_state_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.omega, msg.x, msg.y, msg.yaw + self.offset_yaw]).T
        # self.CurrentState[3:5] = np.dot(self.R,self.CurrentState[3:5])
        # self.CurrentState[3:5] = self.CurrentState[3:5] - np.array([self.offset_x, self.offset_y]).T
        

class Vehicle_states_LPV(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        rospy.Subscriber('vehicleLPV_simulatorStates', simulatorStates, self.vehicle_state_callback)
        self.offset_x = 0.
        self.offset_y = 0.
        self.offset_yaw = 0.

        self.x      = 0. 
        self.y      = 0.
        self.vx     = 0.
        self.vy     = 0.
        self.yaw    = 0. 
        self.omega  = 0.

        self.R = np.array([[cos(self.offset_yaw),-sin(self.offset_yaw)],[sin(self.offset_yaw), cos(self.offset_yaw)]])
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def vehicle_state_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.omega, msg.x, msg.y, msg.yaw + self.offset_yaw]).T
        # self.CurrentState[3:5] = np.dot(self.R,self.CurrentState[3:5])
        # self.CurrentState[3:5] = self.CurrentState[3:5] - np.array([self.offset_x, self.offset_y]).T
        




def main():


    rospy.init_node('realtime_response', anonymous=True)
    loop_rate       = 100
    rate            = rospy.Rate(loop_rate)

    track_map = Map()


    est = EstimatorData()
    vehicle_state = Vehicle_states()
    vehicle_state_LPVerr = Vehicle_states_LPVerr()
    vehicle_state_LPV = Vehicle_states_LPV()

    (fig, axtr, line_planning, line_tr, line_pred, line_SS, line_cl, line_est, line_gps_cl, rec_est ) = _initializeFigure_xy(track_map)


    map_error_plot = False

    if map_error_plot == True:
        (plt_err, fig_err, line_est_s, line_est_ey, line_est_epsi, line_nl_s, line_nl_ey, line_nl_epsi, line_LPVerr_s, line_LPVerr_ey, line_LPVerr_epsi) = errorFigure()



    line_est_s_his = []
    line_est_ey_his = []
    line_est_epsi_his = []

    line_nl_s_his = []
    line_nl_ey_his = []
    line_nl_epsi_his = []

    line_LPVerr_s_his = []
    line_LPVerr_ey_his = []
    line_LPVerr_epsi_his = []


    est_x_his = []
    est_y_his = []

    veh_x_his = []
    veh_y_his = []

    lpverr_x_his = []
    lpverr_y_his = []

    lpv_x_his = []
    lpv_y_his = []

    lim = 5.0
    plt.xlim([-2.5,lim])
    plt.ylim([-2.5,lim])


    

    counter = 0
    while not (rospy.is_shutdown()):

        x_est   = est.CurrentState[3]
        y_est   = est.CurrentState[4]
        yaw_est = wrap(est.CurrentState[5])
        print "est.CurrentState",est.CurrentState
        est_x_his.append(x_est)
        est_y_his.append(y_est)


        x_veh   = vehicle_state.CurrentState[3]
        y_veh   = vehicle_state.CurrentState[4]
        yaw_veh = wrap(vehicle_state.CurrentState[5])
        print "vehicle_state.CurrentState",vehicle_state.CurrentState
        veh_x_his.append(x_veh)
        veh_y_his.append(y_veh)


        x_lpverr   = vehicle_state_LPVerr.CurrentState[3]
        y_lpverr   = vehicle_state_LPVerr.CurrentState[4]
        yaw_lpverr = wrap(vehicle_state_LPVerr.CurrentState[5])
        print "vehicle_state_LPVerr.CurrentState",vehicle_state_LPVerr.CurrentState
        lpverr_x_his.append(x_lpverr)
        lpverr_y_his.append(y_lpverr)
        

        x_lpv   = vehicle_state_LPV.CurrentState[3]
        y_lpv   = vehicle_state_LPV.CurrentState[4]
        yaw_lpv = wrap(vehicle_state_LPV.CurrentState[5])
        print "vehicle_state_LPV.CurrentState",vehicle_state_LPV.CurrentState
        lpv_x_his.append(x_lpv)
        lpv_y_his.append(y_lpv)

        est_s, est_ey, est_epsi, est_insideMap = track_map.getLocalPosition(est.CurrentState[3], est.CurrentState[4], wrap(est.CurrentState[5]))

        veh_s, veh_ey, veh_epsi, veh_insideMap = track_map.getLocalPosition(vehicle_state.CurrentState[3], vehicle_state.CurrentState[4], wrap(vehicle_state.CurrentState[5]))

        lpverr_s, lpverr_ey, lpverr_epsi, lpverr_insideMap = track_map.getLocalPosition(vehicle_state_LPVerr.CurrentState[3], vehicle_state_LPVerr.CurrentState[4], wrap(vehicle_state_LPVerr.CurrentState[5]))

        lpv_s, lpv_ey, lpv_epsi, lpv_insideMap = track_map.getLocalPosition(vehicle_state_LPV.CurrentState[3], vehicle_state_LPV.CurrentState[4], wrap(vehicle_state_LPV.CurrentState[5]))

        if map_error_plot == True:
            line_est_s_his.append(est_s)
            line_est_ey_his.append(est_ey)
            line_est_epsi_his.append(est_epsi)

            line_nl_s_his.append(veh_s)
            line_nl_ey_his.append(veh_ey)
            line_nl_epsi_his.append(veh_epsi)
            
            line_LPVerr_s_his.append(lpverr_s)
            line_LPVerr_ey_his.append(lpverr_ey)
            line_LPVerr_epsi_his.append(lpverr_epsi)

            line_est_s.set_data(np.arange(len(line_est_s_his)), line_est_s_his)
            # limx = len(line_est_s_his)
            # limy = max(line_est_s_his)
            # plt_err.xlim([0,limx])
            # plt_err.ylim([-limy,limy])

            line_est_ey.set_data(np.arange(len(line_est_ey_his)), line_est_ey_his)
            line_est_epsi.set_data(np.arange(len(line_est_epsi_his)), line_est_epsi_his)

            
            line_nl_s.set_data(np.arange(len(line_nl_s_his)), line_nl_s_his)
            line_nl_ey.set_data(np.arange(len(line_nl_ey_his)), line_nl_ey_his)
            line_nl_epsi.set_data(np.arange(len(line_nl_epsi_his)), line_nl_epsi_his)

            line_LPVerr_s.set_data(np.arange(len(line_LPVerr_s_his)), line_LPVerr_s_his)
            line_LPVerr_ey.set_data(np.arange(len(line_LPVerr_ey_his)), line_LPVerr_ey_his)
            line_LPVerr_epsi.set_data(np.arange(len(line_LPVerr_epsi_his)), line_LPVerr_epsi_his)

            fig_err.canvas.draw()
            plt_err.show()
            plt_err.pause(1.0/30)


        print "Vehicle Estimated states",'s', est_s,  'ey', est_ey,  'epsi', est_epsi,  'insideMap', est_insideMap
        print "Vehicle non-linear states",'s', veh_s,  'ey', veh_ey,  'epsi', veh_epsi,  'insideMap', veh_insideMap
        print "Vehicle LPVerr states REVERSE CAL",'s', lpverr_s,  'ey', lpverr_ey,  'epsi', lpverr_epsi,  'insideMap', lpverr_insideMap
        print "Vehicle LPV states",'s', lpv_s,  'ey', lpv_ey,  'epsi', lpv_epsi,  'insideMap', lpv_insideMap

        l = 0.42; w = 0.19

        # print ('lim',lim,'x',x)

        #plotting real vehicle
        car_est_x, car_est_y = getCarPosition(x_est, y_est, yaw_est, w, l)
        rec_est.set_xy(np.array([car_est_x, car_est_y]).T)
        # point_rlc.set_data(x_est, y_est)
        # line_est.set_data(est_x_his, est_y_his)


        line_SS.set_data(veh_x_his, veh_y_his)
        line_cl.set_data(lpverr_x_his, lpverr_y_his)
        line_tr.set_data(lpv_x_his, lpv_y_his)

        fig.canvas.draw()
        plt.show()
        plt.pause(1.0/300)
        counter +=1
        rate.sleep()




def Body_Frame_Errors (x, y, psi, xd, yd, psid, s0, vx, vy, curv, dt):

    ex = (x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)

    ey = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)

    epsi = wrap(psi - psid)

    s = s0 + ( (vx*np.cos(epsi) - vy*np.sin(epsi)) / (1-ey*curv) ) * dt

    return s, ex, ey, epsi



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass