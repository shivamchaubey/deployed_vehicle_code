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


def plot_vehicle_kinematics(x_lim,y_lim):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_ol,        = axtr.plot(xdata, ydata, '-k', label = 'Open loop simulation')
    line_est,       = axtr.plot(xdata, ydata, '-r', label = 'Estimated states')  # Plots the traveled positions
    line_meas,      = axtr.plot(xdata, ydata, '-b', label = 'Measured position camera')  # Plots the traveled positions
    # line_tr,        = axtr.plot(xdata, ydata, '-r', linewidth = 6, alpha = 0.5)       # Plots the current positions
    # line_SS,        = axtr.plot(xdata, ydata, '-g', , linewidth = 10, alpha = 0.5)
    # line_pred,      = axtr.plot(xdata, ydata, '-or')
    # line_planning,  = axtr.plot(xdata, ydata, '-ok')
    
    l = 0.4; w = 0.2 #legth and width of the car

    v = np.array([[ 1,  1],
                  [ 1, -1],
                  [-1, -1],
                  [-1,  1]])

    # Estimated states:
    rec_est = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10)
    axtr.add_patch(rec_est)

    # Open loop simulation:
    rec_ol = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)
    axtr.add_patch(rec_ol)

    # Open loop simulation:
    rec_meas = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10)
    axtr.add_patch(rec_meas)


    plt.legend()
    return fig, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas





def plot_vehicle_states(x_lim,y_lim):

    xdata = []; ydata = []
    
    fig = plt.figure(figsize=(10,8))

    fig, axs = plt.subplots(2, 3, figsize=(10,10))

    plt.ion()

    # plt.xlim([-1*x_lim,x_lim])
    # plt.ylim([-1*y_lim,y_lim])

    # axtr = plt.axes()


    ### longitudinal Velocity
    line_vx_est,        = axs[0,0].plot(xdata, ydata, '-k', label = r'$v_x$: Estimated longitudinal velocity')
    line_vx_ol,         = axs[0,0].plot(xdata, ydata, '-r', label = r'$v_x$: Open loop longitudinal velocity')  # Plots the traveled positions
    line_vx_meas,       = axs[0,0].plot(xdata, ydata, '-b', label = r'$v_x$: Measured longitudinal velocity')  # Plots the traveled positions


    ### lateral Velocity
    line_vy_est,        = axs[0,1].plot(xdata, ydata, '-k', label = r'$v_y$: Estimated lateral velocity')
    line_vy_ol,         = axs[0,1].plot(xdata, ydata, '-r', label = r'$v_y$: Open loop lateral velocity')  # Plots the traveled positions
    line_vy_meas,       = axs[0,1].plot(xdata, ydata, '-b', label = r'$v_y$: Measured lateral velocity')  # Plots the traveled positions


    ### Angular rate
    line_omega_est,     = axs[0,2].plot(xdata, ydata, '-k', label = r'$\omega$: Estimated angular velocity')
    line_omega_ol,      = axs[0,2].plot(xdata, ydata, '-r', label = r'$\omega$: Open loop angular velocity')  # Plots the traveled positions
    line_omega_meas,    = axs[0,2].plot(xdata, ydata, '-b', label = r'$\omega$: Measured angular velocity')  # Plots the traveled positions


    ### Global X -position
    line_X_est,     = axs[1,0].plot(xdata, ydata, '-k', label = r'$X$: Estimated X - position')
    line_X_ol,      = axs[1,0].plot(xdata, ydata, '-r', label = r'$X$: Open loop X - position')  # Plots the traveled positions
    line_X_meas,    = axs[1,0].plot(xdata, ydata, '-b', label = r'$X$: Measured X - position')  # Plots the traveled positions


    ### Global Y -position
    line_Y_est,     = axs[1,1].plot(xdata, ydata, '-k', label = r'$\omega$: Estimated Y - position')
    line_Y_ol,      = axs[1,1].plot(xdata, ydata, '-r', label = r'$\omega$: Open loop Y - position')  # Plots the traveled positions
    line_Y_meas,    = axs[1,1].plot(xdata, ydata, '-b', label = r'$\omega$: Measured Y - position')  # Plots the traveled positions

    ### Yaw
    line_yaw_est,     = axs[1,2].plot(xdata, ydata, '-k', label = r'$\omega$: Estimated yaw')
    line_yaw_ol,      = axs[1,2].plot(xdata, ydata, '-r', label = r'$\omega$: Open loop yaw')  # Plots the traveled positions
    line_yaw_meas,    = axs[1,2].plot(xdata, ydata, '-b', label = r'$\omega$: Measured yaw')  # Plots the traveled positions


    plt.legend()

    return fig, axtr, line_vx_ol, line_vx_est, line_vx_meas, line_vy_ol, line_vy_est, line_vy_meas, line_omega_ol, line_omega_est, line_omega_meas,\
    line_X_ol, line_X_est, line_X_meas, line_Y_ol, line_Y_est, line_Y_meas, line_yaw_ol, line_yaw_est, line_yaw_meas



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



class EstimatorData(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        print "subscribed to vehicle estimated states"

        rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback, queue_size=1)
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def estimator_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T
       


class Vehicle_measurement(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        
        print "subscribed to vehicle measurement"
        rospy.Subscriber('meas_state_info', sensorReading, self.meas_state_callback, queue_size=1)
        
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def meas_state_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        print "self.CurrentState",self.CurrentState 

        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T
        

class Vehicle_ol(object): ## For simulation
    """Data from estimator"""
    def __init__(self):
        """Subscriber to estimator"""
        print "subscribed to vehicle open loop states"

        rospy.Subscriber('ol_state_info', sensorReading, self.vehicle_ol_state_callback, queue_size=1)

        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def vehicle_ol_state_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T
        print "self.CurrentState", self.CurrentState



def main():


    rospy.init_node('observer_performance_tester', anonymous=True)
    loop_rate       = 200
    rate            = rospy.Rate(loop_rate)

    track_map = Map()


    vehicle_state_est = EstimatorData()
    vehicle_state_meas = Vehicle_measurement()
    vehicle_state_ol = Vehicle_ol()

    x_lim = 10
    y_lim = 10


    ### Vehicle kinematics
    (fig, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas) = plot_vehicle_kinematics(x_lim,y_lim)

    ol_x_his     = []
    est_x_his    = []
    meas_x_his   = []
    ol_y_his     = []
    est_y_his    = []
    meas_y_his   = []




    ### vehicle states
    (fig_dy, axtr_dy, line_vx_ol, line_vx_est, line_vx_meas, line_vy_ol, line_vy_est, line_vy_meas, line_omega_ol, line_omega_est, line_omega_meas,\
    line_X_ol, line_X_est, line_X_meas, line_Y_ol, line_Y_est, line_Y_meas, line_yaw_ol, line_yaw_est, line_yaw_meas) = plot_vehicle_states()


    line_vx_ol       =  []
    line_vx_est      =  []
    line_vx_meas     =  []
    line_vy_ol       =  []
    line_vy_est      =  []
    line_vy_meas     =  []
    line_omega_ol    =  []
    line_omega_est   =  []
    line_omega_meas  =  []
    line_X_ol        =  []
    line_X_est       =  []
    line_X_meas      =  []
    line_Y_ol        =  []
    line_Y_est       =  []
    line_Y_meas      =  []
    line_yaw_ol      =  []
    line_yaw_est     =  []
    line_yaw_meas    =  []


    counter = 0
    while not (rospy.is_shutdown()):


        ####################### unpack messages ##################################

        ( vx_est  , vy_est  , omega_est  , x_est  , y_est  , yaw_est  )  = vehicle_state_est.CurrentState
        ( vx_ol   , vy_ol   , omega_ol   , x_ol   , y_ol   , yaw_ol   )  = vehicle_state_ol.CurrentState
        ( vx_meas , vy_meas , omega_meas , x_meas , y_meas , yaw_meas )  = vehicle_state_meas.CurrentState

        ##########################################################################



        ############################# vehicle motion plot #########################

        l = 0.42; w = 0.19

        est_x_his.append(x_est)
        est_y_his.append(y_est)

        ol_x_his.append(x_ol)
        ol_y_his.append(y_ol)
                    
        meas_x_his.append(x_meas)
        meas_y_his.append(y_meas)

        car_est_x, car_est_y = getCarPosition(x_est, y_est, yaw_est, w, l)
        rec_est.set_xy(np.array([car_est_x, car_est_y]).T)

        car_ol_x, car_ol_y = getCarPosition(x_ol, y_ol, yaw_ol, w, l)
        rec_ol.set_xy(np.array([car_ol_x, car_ol_y]).T)

        car_meas_x, car_meas_y = getCarPosition(x_meas, y_meas, yaw_meas, w, l)
        rec_meas.set_xy(np.array([car_meas_x, car_meas_y]).T)

        line_est.set_data(est_x_his, est_y_his)
        line_ol.set_data(ol_x_his, ol_y_his)
        line_meas.set_data(meas_x_his, meas_y_his)

        fig.canvas.draw()
        plt.show()
        plt.pause(1.0/300)


        #############################################################################

        ##############################  vehicle states plot ##############################


        line_vx_ol.append()       
        line_vx_est.append()      
        line_vx_meas.append()     
        line_vy_ol.append()       
        line_vy_est.append()      
        line_vy_meas.append()     
        line_omega_ol.append()    
        line_omega_est.append()   
        line_omega_meas.append()  
        line_X_ol.append(x_ol)        
        line_X_est.append(x_est)       
        line_X_meas.append(x_meas)      
        line_Y_ol.append(y_ol)        
        line_Y_est.append(y_est)       
        line_Y_meas.append(y_meas)      
        line_yaw_ol.append(yaw_ol)      
        line_yaw_est.append(yaw_est)     
        line_yaw_meas.append(yaw_meas)    




        
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