#!/usr/bin/env python


import os
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
from trackInitialization import Map
import numpy as np
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simulator.msg import simulatorStates
from planner.msg import My_Planning
from sensor_fusion.msg import sensorReading
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Path
import tf
from controller.msg import mpcPrediction


#### plotter for control action ####
def plot_control(x_lim,y_lim):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_dutycycle,   = axtr.plot(xdata, ydata, '-g', label = 'Duty cycle')
    line_steer,       = axtr.plot(xdata, ydata, '-b', label = 'Steering angle')  

    plt.legend()
    plt.grid()
    

    return fig, plt, line_dutycycle, line_steer


#### plotter for vehicle estimation on the track ####
def plot_vehicle_global_position(x_lim_min, y_lim_min, x_lim_max, y_lim_max, map):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([x_lim_min, x_lim_max])
    plt.ylim([y_lim_min, y_lim_max])
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
    plt.plot(Points1[:, 0], Points1[:, 1], '-r' , linewidth  = 4) # inner track
    plt.plot(Points2[:, 0], Points2[:, 1], '-r' , linewidth  = 4) #outer track
    plt.plot(Points0[:, 0], Points0[:, 1], '--y' , linewidth  = 2) #outer track


    line_ol,        = axtr.plot(xdata, ydata, '-ob', label = 'Planner trajectory')
    line_est,       = axtr.plot(xdata, ydata, '-b', label = 'Estimated states')  
    line_meas,      = axtr.plot(xdata, ydata, '-m', label = 'Measured position camera') 
    line_lpv_pred,  = axtr.plot(xdata, ydata, '-o', color='orange' , label = 'Model prediction')
    line_mpc_pred,  = axtr.plot(xdata, ydata, '-oc', label = 'MPC prediction', linewidth  = 5) 

    l = 0.42/2; w = 0.19/2

    v = 0.1*np.array([[ 1,  1],
                  [ 1, -1],
                  [-1, -1],
                  [-1,  1]])

    rec_est = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10)
    axtr.add_patch(rec_est)

    rec_ol = patches.Polygon(v, alpha=0.7, closed=True, fc='g', ec='k', zorder=10)
    axtr.add_patch(rec_ol)

    rec_meas = patches.Polygon(v, alpha=0.7, closed=True, fc='m', ec='k', zorder=10)
    axtr.add_patch(rec_meas)

    plt.legend()
    plt.grid()

    return fig, plt, axtr, line_est, line_ol, line_meas,  line_lpv_pred, line_mpc_pred , rec_est, rec_ol, rec_meas


#### plotter for vehicle 6 states ####
def plot_vehicle_states(window):

    xdata = []; ydata = [];
    fig, axs = plt.subplots(3, 2, figsize=(50,50))
    plt.ion()


    ### longitudinal Velocity
    line_vx_est,        = axs[0,0].plot(xdata, ydata, '-b', label = r'$v_x$: Estimated longitudinal velocity')
    line_vx_ol,         = axs[0,0].plot(xdata, ydata, '-g', label = r'$v_x$: Vehicle model longitudinal velocity')  
    line_vx_meas,       = axs[0,0].plot(xdata, ydata, '-m', label = r'$v_x$: Measured longitudinal velocity')  
    axs[0,0].set_xlim(0,window)
    axs[0,0].set_ylim(-0.2,1.5)
    axs[0,0].legend(prop={'size': 12}, framealpha = 0.2)
    axs[0,0].grid()



    ### lateral Velocity
    line_vy_est,        = axs[0,1].plot(xdata, ydata, '-b', label = r'$v_y$: Estimated lateral velocity')
    line_vy_ol,         = axs[0,1].plot(xdata, ydata, '-g', label = r'$v_y$: Vehicle model lateral velocity')  
    line_vy_meas,       = axs[0,1].plot(xdata, ydata, '-m', label = r'$v_y$: Measured lateral velocity')  
    axs[0,1].set_xlim(0,window)
    axs[0,1].set_ylim(-0.2,1.5)
    axs[0,1].legend(prop={'size': 12}, framealpha = 0.2)
    axs[0,1].grid()

    ### Angular rate
    line_omega_est,     = axs[1,0].plot(xdata, ydata, '-b', label = r'$\omega$: Estimated angular velocity')
    line_omega_ol,      = axs[1,0].plot(xdata, ydata, '-g', label = r'$\omega$: Vehicle model angular velocity')  
    line_omega_meas,    = axs[1,0].plot(xdata, ydata, '-m', label = r'$\omega$: Measured angular velocity')  
    axs[1,0].set_xlim(0,window)
    axs[1,0].set_ylim(-2,2)
    axs[1,0].legend(prop={'size': 12} , framealpha = 0.2)
    axs[1,0].grid()
    
    ### Global X -position
    line_X_est,     = axs[1,1].plot(xdata, ydata, '-b', label = r'$X$: Estimated X - position')
    line_X_ol,      = axs[1,1].plot(xdata, ydata, '-g', label = r'$X$: Vehicle model X - position')  
    line_X_meas,    = axs[1,1].plot(xdata, ydata, '-m', label = r'$X$: Measured X - position')  
    axs[1,1].set_xlim(0,window)
    axs[1,1].set_ylim(-10,10)
    axs[1,1].legend(prop={'size': 12} , framealpha = 0.2)
    axs[1,1].grid()
    

    ### Global Y -position
    line_Y_est,     = axs[2,0].plot(xdata, ydata, '-b', label = r'$Y$: Estimated Y - position')
    line_Y_ol,      = axs[2,0].plot(xdata, ydata, '-g', label = r'$Y$: Vehicle model Y - position')  
    line_Y_meas,    = axs[2,0].plot(xdata, ydata, '-m', label = r'$Y$: Measured Y - position')  
    axs[2,0].set_xlim(0,window)
    axs[2,0].set_ylim(-10,10)
    axs[2,0].legend(prop={'size': 12} , framealpha = 0.2)
    axs[2,0].grid()


    ### Yaw
    line_yaw_est,     = axs[2,1].plot(xdata, ydata, '-b', label = r'$\theta$: Estimated yaw')
    line_yaw_ol,      = axs[2,1].plot(xdata, ydata, '-g', label = r'$\theta$: Vehicle model yaw')  
    line_yaw_meas,    = axs[2,1].plot(xdata, ydata, '-m', label = r'$\theta$: Measured yaw')  
    axs[2,1].set_xlim(0,window)
    axs[2,1].set_ylim(-4,4)
    axs[2,1].legend(prop={'size': 12} , framealpha = 0.2)
    axs[2,1].grid()


    return fig, axs, plt, line_vx_ol, line_vx_est, line_vx_meas, line_vy_ol, line_vy_est, line_vy_meas, line_omega_ol, line_omega_est, line_omega_meas,\
    line_X_ol, line_X_est, line_X_meas, line_Y_ol, line_Y_est, line_Y_meas, line_yaw_ol, line_yaw_est, line_yaw_meas



#### plot vehicle state error for the track ####
def plot_estimated_states(x_lim,y_lim):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_ey,        = axtr.plot(xdata, ydata, '-g', label = 'Lateral error from center track')
    line_eyaw,       = axtr.plot(xdata, ydata, '-b', label = 'Angle deviation from center track')  

    return fig, plt, axtr, line_ey, line_eyaw


#### gives the coordinate of the patches for plotting the rectangular (vehicle) orientation and position.
def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y


'''All the three subscription is made from the estimator as it produces all this information'''
class EstimatorData(object):
    def __init__(self):
        print "subscribed to vehicle estimated states"
        rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback, queue_size=1)
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def estimator_callback(self, msg):
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T

class Vehicle_measurement(object):
    def __init__(self):
        
        print "subscribed to vehicle measurement"
        rospy.Subscriber('meas_state_info', sensorReading, self.meas_state_callback, queue_size=1)
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def meas_state_callback(self, msg):
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T
        
class Vehicle_ol(object):
    def __init__(self):
        print "subscribed to vehicle Vehicle model states"
        rospy.Subscriber('ol_state_info', sensorReading, self.vehicle_ol_state_callback, queue_size=1)
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def vehicle_ol_state_callback(self, msg):
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T


# class LPV_preditction(object):
#     def __init__(self):
#         print "subscribed to vehicle Vehicle model states"
#         rospy.Subscriber('control/LPV_prediction', Path, self.vehicle_ol_state_callback, queue_size=1)
#         self.x_list = []
#         self.y_list = []
#         self.yaw_list = []
    
#     def vehicle_ol_state_callback(self, msg):
#         self.x_list = []
#         self.y_list = []
#         self.yaw_list = []
        
#         for i in range(len(msg.poses)):
#             quaternion = msg.poses[i].pose.orientation
#             # print "quaternion", quaternion
#             (roll, pitch, yaw)  = tf.transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
#             # print "yaw", yaw
#             # print 'msg.poses[i].pose.position.x', msg.poses[i].pose.position.x
#             # print 'msg.poses[i].pose.position.y', msg.poses[i].pose.position.y

#             self.yaw_list.append(yaw)
#             self.x_list.append(msg.poses[i].pose.position.x)
#             self.y_list.append(msg.poses[i].pose.position.y)

class MPC_prediction(object):
    def __init__(self):
        print "subscribed to vehicle Vehicle model states"
        rospy.Subscriber('control/MPC_prediction', mpcPrediction, self.vehicle_ol_state_callback, queue_size=1)
        
        N           = 2
        self.x_list = np.zeros(N)
        self.y_list = np.zeros(N)
        self.yaw_list = np.zeros(N)
        self.map      = Map()

    def vehicle_ol_state_callback(self, msg):
        self.x_list = np.zeros(len(msg.s))
        self.y_list = np.zeros(len(msg.s))
        self.yaw_list = np.zeros(len(msg.s))
        
        for i in range(len(msg.s)):
            x, y, yaw = self.map.getGlobalPosition(msg.s[i], msg.ey[i])
            
            # print "x, y, yaw", x, y, yaw
            self.yaw_list[i] = yaw
            self.x_list[i]   = x
            self.y_list[i]   = y

        self.yaw_list = np.array(self.yaw_list, dtype = np.float)
        self.x_list   = np.array(self.x_list, dtype = np.float)
        self.y_list   = np.array(self.y_list, dtype = np.float)

    def update(self):


        # self.yaw_list = np.array(self.yaw_list)
        # self.x_list = np.array(self.x_list)
        # self.y_list = np.array(self.y_list)

        return self.yaw_list, self.x_list, self.y_list

########## Planner reference from MPC planner #############
class planner_ref(object):
    """Data from estimator"""
    def __init__(self):

        rospy.Subscriber("My_Planning", My_Planning, self.planner_callback)
        print "Subscribed to planner"
    
        self.x_d    = []
        self.y_d    = []
        self.psi_d  = []
        self.vx_d   = []
        self.curv_d = []

    def planner_callback(self, msg):
        """
        Unpack the messages from the planner
        """
        self.x_d    = msg.x_d
        self.y_d    = msg.y_d
        self.psi_d  = msg.psi_d
        self.vx_d   = msg.vx_d
        self.curv_d = msg.curv_d


class LPV_prediction(object):
    def __init__(self):
        print "subscribed to vehicle Vehicle model states"
        rospy.Subscriber('control/LPV_prediction', mpcPrediction, self.vehicle_ol_state_callback, queue_size=1)
        
        N           = 2
        self.x_list = np.zeros(N)
        self.y_list = np.zeros(N)
        self.yaw_list = np.zeros(N)
        self.map      = Map()

    def vehicle_ol_state_callback(self, msg):
        self.x_list = np.zeros(len(msg.s))
        self.y_list = np.zeros(len(msg.s))
        self.yaw_list = np.zeros(len(msg.s))
        
        for i in range(len(msg.s)):
            x, y, yaw = self.map.getGlobalPosition(msg.s[i], msg.ey[i])
            
            # print "x, y, yaw", x, y, yaw
            self.yaw_list[i] = yaw
            self.x_list[i]   = x
            self.y_list[i]   = y

        self.yaw_list = np.array(self.yaw_list, dtype = np.float)
        self.x_list   = np.array(self.x_list, dtype = np.float)
        self.y_list   = np.array(self.y_list, dtype = np.float)


    def update(self):


        # self.yaw_list = np.array(self.yaw_list)
        # self.x_list = np.array(self.x_list)
        # self.y_list = np.array(self.y_list)

        return self.yaw_list, self.x_list, self.y_list


class vehicle_control_action(object):
    """ Object collecting CMD command data
    Attributes:
        Input command:
            1.duty_cycle 2.steer
    """
    def __init__(self):
        """ Initialization"""
        rospy.Subscriber('control/accel', Float32, self.accel_callback, queue_size=1)
        rospy.Subscriber('control/steering', Float32, self.steering_callback, queue_size=1)

        # ECU measurement
        self.duty_cycle  = 0.0 #dutycyle
        self.steer = 0.0

    def accel_callback(self,data):
        """Unpack message from sensor, ECU"""
        self.duty_cycle  = data.data

    def steering_callback(self,data):
        self.steer = data.data


def main():


    rospy.init_node('MPC_controller_plot', anonymous=True)
    loop_rate       = rospy.get_param("MPC_plotter/loop_rate")
    rate            = rospy.Rate(loop_rate)

    track_map = Map()


    vehicle_state_est  = EstimatorData()
    vehicle_state_meas = Vehicle_measurement()
    vehicle_state_ol   = Vehicle_ol()
    vehicle_control    = vehicle_control_action()
    lpv_pred_points    = LPV_prediction()
    mpc_pred_points    = MPC_prediction()

    planner_info        = planner_ref()

    image_dy_his = []
    image_veh_his = []


    vehicle_visualization = rospy.get_param("MPC_plotter/vehicle_visualization")
    states_visualization  = rospy.get_param("MPC_plotter/states_visualization")
    error_visualization   = rospy.get_param("MPC_plotter/error_visualization")
    control_visualization = rospy.get_param("MPC_plotter/control_visualization")
    plot_end_result       = rospy.get_param("MPC_plotter/plot_end_result")
    

    window_size = 100
    margin = 0.5 ## margin percentage fox axes: make dynamic window size
        
    if vehicle_visualization == True:

        x_lim_init_max = 3.0
        x_lim_init_min = -1.75
        
        y_lim_init_max = 4.0
        y_lim_init_min = -0.5

        dynamic_graph_size = False
        ### Vehicle kinematics

        (fig_veh, plt_veh, axtr, line_est, line_ol, line_meas, line_lpv_pred, line_mpc_pred, rec_est, rec_ol, rec_meas) = plot_vehicle_global_position(x_lim_init_min, y_lim_init_min, x_lim_init_max, y_lim_init_max, track_map)

    ol_x_his     = []
    est_x_his    = []
    meas_x_his   = []
    ol_y_his     = []
    est_y_his    = []
    meas_y_his   = []



    if states_visualization == True:


        ### vehicle states
        (fig_dy, axs_dy, plt_dy, line_vx_ol, line_vx_est, line_vx_meas, line_vy_ol, line_vy_est, line_vy_meas, line_omega_ol, line_omega_est, line_omega_meas,\
        line_X_ol, line_X_est, line_X_meas, line_Y_ol, line_Y_est, line_Y_meas, line_yaw_ol, line_yaw_est, line_yaw_meas) = plot_vehicle_states(window_size)


    line_vx_ol_his       =  []
    line_vx_est_his      =  []
    line_vx_meas_his     =  []
    line_vy_ol_his       =  []
    line_vy_est_his      =  []
    line_vy_meas_his     =  []
    line_omega_ol_his    =  []
    line_omega_est_his   =  []
    line_omega_meas_his  =  []
    line_X_ol_his        =  []
    line_X_est_his       =  []
    line_X_meas_his      =  []
    line_Y_ol_his        =  []
    line_Y_est_his       =  []
    line_Y_meas_his      =  []
    line_yaw_ol_his      =  []
    line_yaw_est_his     =  []
    line_yaw_meas_his    =  []

    if control_visualization  == True:    
        x_lim = 100
        y_lim = 1.2
        fig_cont, plt_cont, line_dutycycle, line_steer = plot_control(x_lim,y_lim)

    line_dutycycle_his = []
    line_steer_his     = []         


    counter = 0
    while not (rospy.is_shutdown()):


        ########################################### unpack messages ############################################

        ( vx_est  , vy_est  , omega_est  , X_est  , Y_est  , yaw_est  )  = vehicle_state_est.CurrentState
        ( vx_ol   , vy_ol   , omega_ol   , X_ol   , Y_ol   , yaw_ol   )  = vehicle_state_ol.CurrentState 

        # ( vx_ol   , vy_ol   , omega_ol   , X_ol   , Y_ol   , yaw_ol   )  = planner_ref.x_d, planner_ref.y_d, planner_ref.psi_d 
        ( vx_meas , vy_meas , omega_meas , X_meas , Y_meas , yaw_meas )  = vehicle_state_meas.CurrentState

        (lpv_pred_x, lpv_pred_y)  =  lpv_pred_points.x_list, lpv_pred_points.y_list
        (mpc_pred_x, mpc_pred_y)  =  mpc_pred_points.x_list, mpc_pred_points.y_list

        # (lpv_pred_x, lpv_pred_y), _  =  np.array(np.squeeze(lpv_pred_points.update())), np.array(np.squeeze(lpv_pred_points.y_list))
        # (mpc_pred_x, mpc_pred_y), _  =  np.array(np.squeeze(mpc_pred_points.x_list)), np.array(np.squeeze(mpc_pred_points.y_list))


        # lpv_pred_x, lpv_pred_y, _  =  lpv_pred_points.update()
        # mpc_pred_x, mpc_pred_y, _  =  mpc_pred_points.update()

        ########################################################################################################



        ############################################## vehicle motion plot ######################################

        est_x_his.append(X_est)
        est_y_his.append(Y_est)

        ol_x_his.append(planner_info.x_d)
        ol_y_his.append(planner_info.y_d)
                    
        meas_x_his.append(X_meas)
        meas_y_his.append(Y_meas)

        if vehicle_visualization == True: 

            l = 0.42/2; w = 0.19/2

            # print "X_est  , Y_est  , yaw_est", X_est  , Y_est  , yaw_est
            # print "X_ol  , Y_ol  , yaw_ol", X_ol  , Y_ol  , yaw_ol
            # print "X_meas  , Y_meas  , yaw_meas", X_meas  , Y_meas  , yaw_meas




            car_est_x, car_est_y = getCarPosition(X_est, Y_est, yaw_est, w, l)
            rec_est.set_xy(np.array([car_est_x, car_est_y]).T)

            car_ol_x, car_ol_y = getCarPosition(X_ol, Y_ol, yaw_ol, w, l)
            rec_ol.set_xy(np.array([car_ol_x, car_ol_y]).T)

            car_meas_x, car_meas_y = getCarPosition(X_meas, Y_meas, yaw_meas, w, l)
            rec_meas.set_xy(np.array([car_meas_x, car_meas_y]).T)

            line_est.set_data(est_x_his, est_y_his)
            # line_ol.set_data(ol_x_his, ol_y_his)
            print "planner_info.x_d", planner_info.x_d
            line_ol.set_data(planner_info.x_d, planner_info.y_d)

            line_meas.set_data(meas_x_his, meas_y_his)

            # print "lpv_pred_x, lpv_pred_y", lpv_pred_x, lpv_pred_y

            if (len(lpv_pred_x) == len(lpv_pred_y)) and (False in np.isnan(lpv_pred_x)) and (False in np.isnan(lpv_pred_y)):
                line_lpv_pred.set_data(lpv_pred_x, lpv_pred_y)

            # print "mpc_pred_x, mpc_pred_y", mpc_pred_x, mpc_pred_y    
            if (len(mpc_pred_x) == len(mpc_pred_y)) and (False in np.isnan(mpc_pred_x)) and (False in np.isnan(mpc_pred_y)):
                line_mpc_pred.set_data(mpc_pred_x, mpc_pred_y)

            ############# Dynamic window size ##############

            if dynamic_graph_size == True:
                min_x_lim = min(min(ol_x_his) - margin*min(ol_x_his), min(meas_x_his) - margin*min(meas_x_his)) 
                max_x_lim = max(max(ol_x_his) + margin*max(ol_x_his), max(meas_x_his) + margin*max(meas_x_his))
                min_y_lim = min(min(ol_y_his) - margin*min(ol_y_his), min(meas_y_his) - margin*min(meas_y_his))
                max_y_lim = max(max(ol_y_his) + margin*max(ol_y_his), max(meas_y_his) + margin*max(meas_y_his))

                if (x_lim_init_max < max_x_lim):
                    x_lim_init_max = max_x_lim
                    axtr.set_xlim( x_lim_init_min, x_lim_init_max )

                
                if (x_lim_init_min > min_x_lim):
                    x_lim_init_min = min_x_lim
                    axtr.set_xlim( x_lim_init_min, x_lim_init_max )


                if (y_lim_init_max < max_y_lim):
                    y_lim_init_max = max_y_lim
                    axtr.set_ylim( y_lim_init_min, y_lim_init_max )

                if (y_lim_init_min > min_y_lim):
                    y_lim_init_min = min_y_lim
                    axtr.set_ylim( y_lim_init_min, y_lim_init_max )

            StringValue = "vx = "+str(vx_est) 
            axtr.set_title(StringValue)

            fig_veh.canvas.draw()
            plt_veh.show()
       

        ##########################################################################################################


        #############################################  vehicle states plot #######################################

        line_vx_ol_his.append(vx_ol)
        line_vx_est_his.append(vx_est)
        line_vx_meas_his.append(vx_meas)
        line_vy_ol_his.append(vy_ol)
        line_vy_est_his.append(vy_est)
        line_vy_meas_his.append(vy_meas)
        line_omega_ol_his.append(omega_ol)
        line_omega_est_his.append(omega_est)
        line_omega_meas_his.append(omega_meas)
        line_X_ol_his.append(X_ol)
        line_X_est_his.append(X_est)
        line_X_meas_his.append(X_meas)
        line_Y_ol_his.append(Y_ol)
        line_Y_est_his.append(Y_est)
        line_Y_meas_his.append(Y_meas)
        line_yaw_ol_his.append(yaw_ol)
        line_yaw_est_his.append(yaw_est)
        line_yaw_meas_his.append(yaw_meas)


        if states_visualization == True and counter > 100:

                # axs_dy[0,0].set_ylim(min(line_vx_ol_his) + , max(line_vx_ol_his) + ) # FOR SETTING THE DYNAMIC AXES
                # axs_dy[0,0].set_ylim(min(line_vx_ol_his) + , max(line_vx_ol_his) + ) # FOR SETTING THE DYNAMIC AXES
            line_vx_ol.set_data( range(counter, counter + window_size ) ,line_vx_ol_his[ -window_size : ])
            line_vx_est.set_data( range(counter, counter + window_size ) ,line_vx_est_his[ -window_size : ])
            line_vx_meas.set_data( range(counter, counter + window_size ) ,line_vx_meas_his[ -window_size : ])
            axs_dy[0,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
            line_vy_ol.set_data( range(counter, counter + window_size ) ,line_vy_ol_his[ -window_size : ])
            line_vy_est.set_data( range(counter, counter + window_size ) ,line_vy_est_his[ -window_size : ])
            line_vy_meas.set_data( range(counter, counter + window_size ) ,line_vy_meas_his[ -window_size : ])
            axs_dy[0,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            line_omega_ol.set_data( range(counter, counter + window_size ) ,line_omega_ol_his[ -window_size : ])
            line_omega_est.set_data( range(counter, counter + window_size ) ,line_omega_est_his[ -window_size : ])
            line_omega_meas.set_data( range(counter, counter + window_size ) ,line_omega_meas_his[ -window_size : ])
            axs_dy[1,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
            line_X_ol.set_data( range(counter, counter + window_size ) ,line_X_ol_his[ -window_size : ])
            line_X_est.set_data( range(counter, counter + window_size ) ,line_X_est_his[ -window_size : ])
            line_X_meas.set_data( range(counter, counter + window_size ) ,line_X_meas_his[ -window_size : ])
            axs_dy[1,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            line_Y_ol.set_data( range(counter, counter + window_size ) ,line_Y_ol_his[ -window_size : ])
            line_Y_est.set_data( range(counter, counter + window_size ) ,line_Y_est_his[ -window_size : ])
            line_Y_meas.set_data( range(counter, counter + window_size ) ,line_Y_meas_his[ -window_size : ])
            axs_dy[2,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            line_yaw_ol.set_data( range(counter, counter + window_size ) ,line_yaw_ol_his[ -window_size : ])
            line_yaw_est.set_data( range(counter, counter + window_size ) ,line_yaw_est_his[ -window_size : ])
            line_yaw_meas.set_data( range(counter, counter + window_size ) ,line_yaw_meas_his[ -window_size : ])
            axs_dy[2,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            axs_dy[2,1].set_ylim(min(min(line_yaw_meas_his[ -window_size : ]) + margin*min(line_yaw_meas_his[ -window_size : ])\
                , min(line_yaw_ol_his[ -window_size : ]) + margin*min(line_yaw_ol_his[ -window_size : ])), \
            max(max(line_yaw_meas_his[ -window_size : ]) + margin*max(line_yaw_meas_his[ -window_size : ]), max(line_yaw_ol_his[ -window_size : ]) + margin*max(line_yaw_ol_his[ -window_size : ]))) # FOR SETTING THE DYNAMIC AXES

            fig_dy.canvas.draw()
            plt_dy.show()
            
        
        line_dutycycle_his.append(vehicle_control.duty_cycle)
        line_steer_his.append(vehicle_control.steer)

        if control_visualization == True and counter > 100: 

            line_dutycycle.set_data(range(counter, counter + window_size ) ,line_dutycycle_his[ -window_size : ])
            line_steer.set_data(range(counter, counter + window_size ) ,line_steer_his[ -window_size : ])
            plt_cont.xlim(counter, counter + window_size )

            fig_cont.canvas.draw()
            plt_cont.show()

            


        plt.pause(1.0/3000)


        counter +=1
        rate.sleep()


    if plot_end_result == True:

        ######## Vehicle position ##########
        x_lim_init_max = 3.0
        x_lim_init_min = -1.75
        y_lim_init_max = 4.0
        y_lim_init_min = -0.5

        dynamic_graph_size = False
        ### Vehicle kinematics

        (fig_veh, plt_veh, axtr, line_est, line_ol, line_meas, line_lpv_pred, line_mpc_pred, rec_est, rec_ol, rec_meas) = plot_vehicle_global_position(x_lim_init_min, y_lim_init_min, x_lim_init_max, y_lim_init_max, track_map)

        line_est.set_data(est_x_his, est_y_his)
        line_ol.set_data(ol_x_his, ol_y_his)
        line_meas.set_data(meas_x_his, meas_y_his)






        # ############# Vehicle states ############
        window_size = len(line_vx_est_his)
        margin = 0.1
        (fig_dy, axs_dy, plt_dy, line_vx_ol, line_vx_est, line_vx_meas, line_vy_ol, line_vy_est, line_vy_meas, line_omega_ol, line_omega_est, line_omega_meas,\
        line_X_ol, line_X_est, line_X_meas, line_Y_ol, line_Y_est, line_Y_meas, line_yaw_ol, line_yaw_est, line_yaw_meas) = plot_vehicle_states(window_size)

        line_vx_ol.set_data( range(counter, counter + window_size ) ,line_vx_ol_his[ -window_size : ])
        line_vx_est.set_data( range(counter, counter + window_size ) ,line_vx_est_his[ -window_size : ])
        line_vx_meas.set_data( range(counter, counter + window_size ) ,line_vx_meas_his[ -window_size : ])
        axs_dy[0,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
        
        
        line_vy_ol.set_data( range(counter, counter + window_size ) ,line_vy_ol_his[ -window_size : ])
        line_vy_est.set_data( range(counter, counter + window_size ) ,line_vy_est_his[ -window_size : ])
        line_vy_meas.set_data( range(counter, counter + window_size ) ,line_vy_meas_his[ -window_size : ])
        axs_dy[0,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
        

        line_omega_ol.set_data( range(counter, counter + window_size ) ,line_omega_ol_his[ -window_size : ])
        line_omega_est.set_data( range(counter, counter + window_size ) ,line_omega_est_his[ -window_size : ])
        line_omega_meas.set_data( range(counter, counter + window_size ) ,line_omega_meas_his[ -window_size : ])
        axs_dy[1,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
        
        
        line_X_ol.set_data( range(counter, counter + window_size ) ,line_X_ol_his[ -window_size : ])
        line_X_est.set_data( range(counter, counter + window_size ) ,line_X_est_his[ -window_size : ])
        line_X_meas.set_data( range(counter, counter + window_size ) ,line_X_meas_his[ -window_size : ])
        axs_dy[1,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
        

        line_Y_ol.set_data( range(counter, counter + window_size ) ,line_Y_ol_his[ -window_size : ])
        line_Y_est.set_data( range(counter, counter + window_size ) ,line_Y_est_his[ -window_size : ])
        line_Y_meas.set_data( range(counter, counter + window_size ) ,line_Y_meas_his[ -window_size : ])
        axs_dy[2,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
        

        line_yaw_ol.set_data( range(counter, counter + window_size ) ,line_yaw_ol_his[ -window_size : ])
        line_yaw_est.set_data( range(counter, counter + window_size ) ,line_yaw_est_his[ -window_size : ])
        line_yaw_meas.set_data( range(counter, counter + window_size ) ,line_yaw_meas_his[ -window_size : ])
        axs_dy[2,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
        

        axs_dy[2,1].set_ylim(min(min(line_yaw_meas_his[ -window_size : ]) + margin*min(line_yaw_meas_his[ -window_size : ])\
            , min(line_yaw_ol_his[ -window_size : ]) + margin*min(line_yaw_ol_his[ -window_size : ])), \
        max(max(line_yaw_meas_his[ -window_size : ]) + margin*max(line_yaw_meas_his[ -window_size : ]), max(line_yaw_ol_his[ -window_size : ]) + margin*max(line_yaw_ol_his[ -window_size : ]))) # FOR SETTING THE DYNAMIC AXES






        # ########### vehicle control ###############
        x_lim = len(line_dutycycle_his)
        y_lim = 1.2
        fig_cont, plt_cont, line_dutycycle, line_steer = plot_control(x_lim,y_lim)
        line_dutycycle.set_data(range(counter, counter + window_size ) ,line_dutycycle_his[ -window_size : ])
        line_steer.set_data(range(counter, counter + window_size ) ,line_steer_his[ -window_size : ])
        plt_cont.xlim(counter, counter + window_size )

        import cv2

        while True:

            fig_veh.canvas.draw()
            plt_veh.show()
            plt.pause(1)

            fig_dy.canvas.draw()
            plt_dy.show()
            plt_dy.pause(1)

            fig_cont.canvas.draw()
            plt_cont.show()
            plt_dy.pause(1)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


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

