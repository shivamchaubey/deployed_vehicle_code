#!/usr/bin/env python


import os
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
import numpy as np
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simulator.msg import simulatorStates
from sensor_fusion.msg import sensorReading
from trackInitialization import Map
from std_msgs.msg import Bool, Float32


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
def plot_vehicle_global_position(map):
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
    plt.plot(Points1[:, 0], Points1[:, 1], '-b') # inner track
    plt.plot(Points2[:, 0], Points2[:, 1], '-b') #outer track
    plt.plot(Points0[:, 0], Points0[:, 1], '-y') #outer track


    line_ol,        = axtr.plot(xdata, ydata, '-g', label = 'Vehicle model simulation')
    line_est,       = axtr.plot(xdata, ydata, '-b', label = 'Estimated states')  
    line_meas,      = axtr.plot(xdata, ydata, '-m', label = 'Measured position camera') 

    l = 0.4; w = 0.2

    v = np.array([[ 1,  1],
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

    return fig, plt, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas


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
    loop_rate       = 1000
    rate            = rospy.Rate(loop_rate)

    track_map = Map()


    vehicle_state_est  = EstimatorData()
    vehicle_state_meas = Vehicle_measurement()
    vehicle_state_ol   = Vehicle_ol()
    vehicle_control    = vehicle_control_action()

    image_dy_his = []
    image_veh_his = []


    vehicle_visualization = True
    states_visualization  = False
    error_visualization   = True
    control_visualization  = True
    window_size = 100
    
    if vehicle_visualization == True:

        margin = 0.5 ## margin percentage fox axes: make dynamic window size
        x_lim_init_max = 5
        x_lim_init_min = -x_lim_init_max
        
        y_lim_init_max = 5
        y_lim_init_min = -y_lim_init_max


        ### Vehicle kinematics
        (fig_veh, plt_veh, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas) = plot_vehicle_global_position(track_map)

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


        line_vx_ol_his       =  [0.0]*window_size
        line_vx_est_his      =  [0.0]*window_size
        line_vx_meas_his     =  [0.0]*window_size
        line_vy_ol_his       =  [0.0]*window_size
        line_vy_est_his      =  [0.0]*window_size
        line_vy_meas_his     =  [0.0]*window_size
        line_omega_ol_his    =  [0.0]*window_size
        line_omega_est_his   =  [0.0]*window_size
        line_omega_meas_his  =  [0.0]*window_size
        line_X_ol_his        =  [0.0]*window_size
        line_X_est_his       =  [0.0]*window_size
        line_X_meas_his      =  [0.0]*window_size
        line_Y_ol_his        =  [0.0]*window_size
        line_Y_est_his       =  [0.0]*window_size
        line_Y_meas_his      =  [0.0]*window_size
        line_yaw_ol_his      =  [0.0]*window_size
        line_yaw_est_his     =  [0.0]*window_size
        line_yaw_meas_his    =  [0.0]*window_size

    if control_visualization  == True:    
        x_lim = 100
        y_lim = 1.2
        fig_cont, plt_cont, line_dutycycle, line_steer = plot_control(x_lim,y_lim)

        line_dutycycle_his = [0.0]*window_size
        line_steer_his     = [0.0]*window_size         


    counter = 0
    while not (rospy.is_shutdown()):


        ########################################### unpack messages ############################################

        ( vx_est  , vy_est  , omega_est  , X_est  , Y_est  , yaw_est  )  = vehicle_state_est.CurrentState
        ( vx_ol   , vy_ol   , omega_ol   , X_ol   , Y_ol   , yaw_ol   )  = vehicle_state_ol.CurrentState
        ( vx_meas , vy_meas , omega_meas , X_meas , Y_meas , yaw_meas )  = vehicle_state_meas.CurrentState

        ########################################################################################################



        ############################################## vehicle motion plot ######################################

        if vehicle_visualization == True: 

            l = 0.42; w = 0.19

            est_x_his.append(X_est)
            est_y_his.append(Y_est)

            ol_x_his.append(X_ol)
            ol_y_his.append(Y_ol)
                        
            meas_x_his.append(X_meas)
            meas_y_his.append(Y_meas)

            car_est_x, car_est_y = getCarPosition(X_est, Y_est, yaw_est, w, l)
            rec_est.set_xy(np.array([car_est_x, car_est_y]).T)

            car_ol_x, car_ol_y = getCarPosition(X_ol, Y_ol, yaw_ol, w, l)
            rec_ol.set_xy(np.array([car_ol_x, car_ol_y]).T)

            car_meas_x, car_meas_y = getCarPosition(X_meas, Y_meas, yaw_meas, w, l)
            rec_meas.set_xy(np.array([car_meas_x, car_meas_y]).T)

            line_est.set_data(est_x_his, est_y_his)
            line_ol.set_data(ol_x_his, ol_y_his)
            line_meas.set_data(meas_x_his, meas_y_his)

            ############# Dynamic window size ##############
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

            fig_veh.canvas.draw()
            plt_veh.show()
       

        ##########################################################################################################


        #############################################  vehicle states plot #######################################

        if states_visualization == True:

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

            line_vx_ol_his.pop(0)
            line_vx_est_his.pop(0)
            line_vx_meas_his.pop(0)
            line_vy_ol_his.pop(0)
            line_vy_est_his.pop(0)
            line_vy_meas_his.pop(0)
            line_omega_ol_his.pop(0)
            line_omega_est_his.pop(0)
            line_omega_meas_his.pop(0)
            line_X_ol_his.pop(0)
            line_X_est_his.pop(0)
            line_X_meas_his.pop(0)
            line_Y_ol_his.pop(0)
            line_Y_est_his.pop(0)
            line_Y_meas_his.pop(0)
            line_yaw_ol_his.pop(0)
            line_yaw_est_his.pop(0)
            line_yaw_meas_his.pop(0)



                # axs_dy[0,0].set_ylim(min(line_vx_ol_his) + , max(line_vx_ol_his) + ) # FOR SETTING THE DYNAMIC AXES
            line_vx_ol.set_data( range(counter, counter + window_size ) ,line_vx_ol_his)
            line_vx_est.set_data( range(counter, counter + window_size ) ,line_vx_est_his)
            line_vx_meas.set_data( range(counter, counter + window_size ) ,line_vx_meas_his)
            axs_dy[0,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
            line_vy_ol.set_data( range(counter, counter + window_size ) ,line_vy_ol_his)
            line_vy_est.set_data( range(counter, counter + window_size ) ,line_vy_est_his)
            line_vy_meas.set_data( range(counter, counter + window_size ) ,line_vy_meas_his)
            axs_dy[0,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            line_omega_ol.set_data( range(counter, counter + window_size ) ,line_omega_ol_his)
            line_omega_est.set_data( range(counter, counter + window_size ) ,line_omega_est_his)
            line_omega_meas.set_data( range(counter, counter + window_size ) ,line_omega_meas_his)
            axs_dy[1,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
            line_X_ol.set_data( range(counter, counter + window_size ) ,line_X_ol_his)
            line_X_est.set_data( range(counter, counter + window_size ) ,line_X_est_his)
            line_X_meas.set_data( range(counter, counter + window_size ) ,line_X_meas_his)
            axs_dy[1,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            line_Y_ol.set_data( range(counter, counter + window_size ) ,line_Y_ol_his)
            line_Y_est.set_data( range(counter, counter + window_size ) ,line_Y_est_his)
            line_Y_meas.set_data( range(counter, counter + window_size ) ,line_Y_meas_his)
            axs_dy[2,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

            line_yaw_ol.set_data( range(counter, counter + window_size ) ,line_yaw_ol_his)
            line_yaw_est.set_data( range(counter, counter + window_size ) ,line_yaw_est_his)
            line_yaw_meas.set_data( range(counter, counter + window_size ) ,line_yaw_meas_his)
            axs_dy[2,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            axs_dy[2,1].set_ylim(min(min(line_yaw_meas_his) - margin*min(line_yaw_meas_his)\
                , min(line_yaw_ol_his) - margin*min(line_yaw_ol_his)), \
            max(max(line_yaw_meas_his) + margin*max(line_yaw_meas_his), max(line_yaw_ol_his) + margin*max(line_yaw_ol_his))) # FOR SETTING THE DYNAMIC AXES
            
            fig_dy.canvas.draw()    
            plt_dy.show()


        if control_visualization == True:
            
            line_dutycycle_his.append(vehicle_control.duty_cycle)
            line_steer_his.append(vehicle_control.steer)

            line_dutycycle_his.pop(0)
            line_steer_his.pop(0)

            line_dutycycle.set_data(range(counter, counter + window_size ) ,line_dutycycle_his)
            line_steer.set_data(range(counter, counter + window_size ) ,line_steer_his)
            plt_cont.xlim(counter, counter + window_size)

            fig_cont.canvas.draw()
            plt_cont.show()

            


        plt.pause(1.0/3000)


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

