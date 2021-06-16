#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' This code is to compare the performance of the estimation technique. The
estimated states, measured states and the simulator states are
plotted to compare the performance. This code is bit slow due to online
plotting of matplotlib. Use other plotter for real time debugging such as
plotjuggler. This code an be used to record the data for documentation
 '''

import numpy as np
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sensor_fusion.msg import sensorReading, control
from simulator.msg import simulatorStates
from PIL import Image
from std_msgs.msg import Bool, Float32
import datetime
import os
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
from trackInitialization import Map


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

#### plotter for vehicle motion ####
def plot_vehicle_kinematics(x_lim_min,y_lim_min, x_lim_max, y_lim_max, map, track_plot_on, sim_on):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([x_lim_min,x_lim_max])
    plt.ylim([y_lim_min,y_lim_max])

    axtr = plt.axes()


    if track_plot_on == True:
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


    line_ol,        = axtr.plot(xdata, ydata, '-g', label = 'Vehicle model simulation')
    line_est,       = axtr.plot(xdata, ydata, '-b', label = 'Estimated states')  
    line_meas,      = axtr.plot(xdata, ydata, '-m', label = 'Measured position camera') 
    
    if sim_on:
        line_real,      = axtr.plot(xdata, ydata, '-m', label = 'Simulated vehicle') 


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

    if sim_on:
        rec_real = patches.Polygon(v, alpha=0.7, closed=True, fc='m', ec='k', zorder=10)
        axtr.add_patch(rec_real)

    plt.legend()
    plt.grid()
    
    if sim_on:
    
        return fig, plt, axtr, line_est, line_ol, line_real, line_meas, rec_est, rec_ol, rec_meas, rec_real
    else:
        return fig, plt, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas




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
    axs[0,0].set_ylim(-1,5)
    axs[0,0].legend(prop={'size': 12}, framealpha = 0.2)
    axs[0,0].grid()



    ### lateral Velocity
    line_vy_est,        = axs[0,1].plot(xdata, ydata, '-b', label = r'$v_y$: Estimated lateral velocity')
    line_vy_ol,         = axs[0,1].plot(xdata, ydata, '-g', label = r'$v_y$: Vehicle model lateral velocity')  
    line_vy_meas,       = axs[0,1].plot(xdata, ydata, '-m', label = r'$v_y$: Measured lateral velocity')  
    axs[0,1].set_xlim(0,window)
    axs[0,1].set_ylim(-1,5)
    axs[0,1].legend(prop={'size': 12}, framealpha = 0.2)
    axs[0,1].grid()

    ### Angular rate
    line_omega_est,     = axs[1,0].plot(xdata, ydata, '-b', label = r'$\omega$: Estimated angular velocity')
    line_omega_ol,      = axs[1,0].plot(xdata, ydata, '-g', label = r'$\omega$: Vehicle model angular velocity')  
    line_omega_meas,    = axs[1,0].plot(xdata, ydata, '-m', label = r'$\omega$: Measured angular velocity')  
    axs[1,0].set_xlim(0,window)
    axs[1,0].set_ylim(-5,5)
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


#### gives the coordinate of the patches for plotting the rectangular (vehicle) orientation and position.
def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y


def main():


    rospy.init_node('observer_performance_tester', anonymous=True)
    
    loop_rate       = 1000.0
    rate            = rospy.Rate(loop_rate)


    path = '/home/auto/Desktop/autonomus_vehicle_project/project/development/proto/plotter/log'
    dir_name = '/estimator'
    file_name = '/observer_data.npy'

    data = np.load(path + dir_name +file_name, allow_pickle = True).item()


    vx_est = data['est_state'][:,0]
    vy_est = data['est_state'][:,1]
    omega_est = data['est_state'][:,2]

    X_est = data['est_state'][:,3]
    Y_est = data['est_state'][:,4]
    yaw_est = data['est_state'][:,5]


    vx_meas = data['full_meas'][:,0]
    omega_meas = data['full_meas'][:,1]

    X_meas = data['full_meas'][:,2]
    Y_meas = data['full_meas'][:,3]
    yaw_meas = data['full_meas'][:,4]

    X_lidar = data['lidar_pose'][:,0]
    Y_lidar = data['lidar_pose'][:,1]

    d_cont = data['control'][:,0]
    steer_cont = data['control'][:,1]


    track_visualization = True
    track_map = Map()

    
    image_dy_his = []
    image_veh_his = []

    vehicle_visualization = True
    states_visualization  = False
    control_visualization = False
    record_data = False
    sim_on = False

    window_size = 100

    if vehicle_visualization == True:

        margin = 0.5 ## margin percentage fox axes: make dynamic window size
        x_lim_init_min = -1.75
        y_lim_init_min = -0.75
        x_lim_init_max = 3
        y_lim_init_max = 3.5


        ### Vehicle kinematics
        if sim_on:
            (fig_veh, plt_veh, axtr, line_est, line_ol, line_real, line_meas, rec_est, rec_ol, rec_meas, rec_real) = plot_vehicle_kinematics(x_lim_init_min, y_lim_init_min, x_lim_init_max,y_lim_init_max, track_map, track_visualization, sim_on)

        else:
            (fig_veh, plt_veh, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas) = plot_vehicle_kinematics(x_lim_init_min, y_lim_init_min, x_lim_init_max,y_lim_init_max, track_map, track_visualization, sim_on)


    ol_x_his     = []
    est_x_his    = []
    meas_x_his   = []
    real_x_his   = []

    ol_y_his     = []
    est_y_his    = []
    meas_y_his   = []
    real_y_his   = []



    if states_visualization == True:


        ### vehicle states
        (fig_dy, axs_dy, plt_dy, line_vx_ol, line_vx_est, line_vx_meas, line_vy_ol, line_vy_est, line_vy_meas, line_omega_ol, line_omega_est, line_omega_meas,\
        line_X_ol, line_X_est, line_X_meas, line_Y_ol, line_Y_est, line_Y_meas, line_yaw_ol, line_yaw_est, line_yaw_meas) = plot_vehicle_states(window_size)


    line_vx_ol_his       =  []
    line_vx_est_his      =  []
    line_vx_meas_his     =  []
    line_vx_real_his     =  []
    
    line_vy_ol_his       =  []
    line_vy_est_his      =  []
    line_vy_meas_his     =  []
    line_vy_real_his     =  []
    
    line_omega_ol_his    =  []
    line_omega_est_his   =  []
    line_omega_meas_his  =  []
    line_omega_real_his  =  []
    
    line_X_ol_his        =  []
    line_X_est_his       =  []
    line_X_meas_his      =  []
    line_X_real_his      =  []
    
    line_Y_ol_his        =  []
    line_Y_est_his       =  []
    line_Y_meas_his      =  []
    line_Y_real_his      =  []
    
    line_yaw_ol_his      =  []
    line_yaw_est_his     =  []
    line_yaw_meas_his    =  []
    line_yaw_real_his    =  []

    if control_visualization == True:    
        x_lim = 100
        y_lim = 1.2
        fig_cont, plt_cont, line_dutycycle, line_steer = plot_control(x_lim,y_lim)

    line_dutycycle_his = []
    line_steer_his     = []         


    counter = 6000
    plt.figure()

    while not (rospy.is_shutdown()):


        ########################################### unpack messages ############################################

        print "counter", counter
        ( vx_est  , vy_est  , omega_est  , X_est  , Y_est  , yaw_est  )  = data['est_state'][counter,:]
        ( vx_ol   , vy_ol   , omega_ol   , X_ol   , Y_ol   , yaw_ol   )  = [0, 0, 0, 0, 0, 0]
        ( vx_meas , omega_meas , X_meas , Y_meas , yaw_meas )  = data['full_meas'][counter,:]



        plt.scatter(X_est, Y_est)
        plt.show()
        plt.pause(0.000001)
    
        # vy_meas = 0.0
        # if sim_on:
        #     ( vx_real , vy_real , omega_real , X_real , Y_real , yaw_real )  = vehicle_state_real.CurrentState

            
        #     real_x_his.append(X_real)
        #     real_y_his.append(Y_real)

        # ########################################################################################################



        # ############################################## vehicle motion plot ######################################

        # est_x_his.append(X_est)
        # est_y_his.append(Y_est)
        # ol_x_his.append(X_ol)
        # ol_y_his.append(Y_ol)
                    
        # meas_x_his.append(X_meas)
        # meas_y_his.append(Y_meas)
        

        # if vehicle_visualization == True: 

        #     l = 0.42/2; w = 0.19/2

        #     car_est_x, car_est_y = getCarPosition(X_est, Y_est, yaw_est, w, l)
        #     rec_est.set_xy(np.array([car_est_x, car_est_y]).T)

        #     car_ol_x, car_ol_y = getCarPosition(X_ol, Y_ol, yaw_ol, w, l)
        #     rec_ol.set_xy(np.array([car_ol_x, car_ol_y]).T)

        #     car_meas_x, car_meas_y = getCarPosition(X_meas, Y_meas, yaw_meas, w, l)
        #     rec_meas.set_xy(np.array([car_meas_x, car_meas_y]).T)

            

        #     line_est.set_data(est_x_his, est_y_his)
        #     line_ol.set_data(ol_x_his, ol_y_his)
        #     line_meas.set_data(meas_x_his, meas_y_his)
                
        #     fig_veh.canvas.draw()
        #     plt_veh.show()
          

        # ##########################################################################################################

        # #############################################  vehicle states plot #######################################

        # line_vx_ol_his.append(vx_ol)
        # line_vx_est_his.append(vx_est)
        # line_vx_meas_his.append(vx_meas)

        # line_vy_ol_his.append(vy_ol)
        # line_vy_est_his.append(vy_est)
        # line_vy_meas_his.append(vy_meas)
        
        # line_omega_ol_his.append(omega_ol)
        # line_omega_est_his.append(omega_est)
        # line_omega_meas_his.append(omega_meas)
        
        # line_X_ol_his.append(X_ol)
        # line_X_est_his.append(X_est)
        # line_X_meas_his.append(X_meas)
        
        # line_Y_ol_his.append(Y_ol)
        # line_Y_est_his.append(Y_est)
        # line_Y_meas_his.append(Y_meas)
        
        # line_yaw_ol_his.append(yaw_ol)
        # line_yaw_est_his.append(yaw_est)
        # line_yaw_meas_his.append(yaw_meas)

        # if sim_on:
        #     line_vx_real_his.append(vx_real)
        #     line_vy_real_his.append(vy_real)
        #     line_omega_real_his.append(omega_real)
        #     line_X_real_his.append(X_real)
        #     line_Y_real_his.append(Y_real)
        #     line_yaw_real_his.append(yaw_real)

        # if states_visualization == True and counter >= 100:

        #     ### Keep size of window to 100 points 

        #     # axs_dy[0,0].set_ylim(min(line_vx_ol_his) + , max(line_vx_ol_his) + ) # FOR SETTING THE DYNAMIC AXES
        #     line_vx_ol.set_data( range(counter, counter + window_size ) ,line_vx_ol_his[ -window_size : ])
        #     line_vx_est.set_data( range(counter, counter + window_size ) ,line_vx_est_his[ -window_size : ])
        #     line_vx_meas.set_data( range(counter, counter + window_size ) ,line_vx_meas_his[ -window_size : ])
        #     axs_dy[0,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
        #     line_vy_ol.set_data( range(counter, counter + window_size ) ,line_vy_ol_his[ -window_size : ])
        #     line_vy_est.set_data( range(counter, counter + window_size ) ,line_vy_est_his[ -window_size : ])
        #     line_vy_meas.set_data( range(counter, counter + window_size ) ,line_vy_meas_his[ -window_size : ])
        #     axs_dy[0,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     line_omega_ol.set_data( range(counter, counter + window_size ) ,line_omega_ol_his[ -window_size : ])
        #     line_omega_est.set_data( range(counter, counter + window_size ) ,line_omega_est_his[ -window_size : ])
        #     line_omega_meas.set_data( range(counter, counter + window_size ) ,line_omega_meas_his[ -window_size : ])
        #     axs_dy[1,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
        #     line_X_ol.set_data( range(counter, counter + window_size ) ,line_X_ol_his[ -window_size : ])
        #     line_X_est.set_data( range(counter, counter + window_size ) ,line_X_est_his[ -window_size : ])
        #     line_X_meas.set_data( range(counter, counter + window_size ) ,line_X_meas_his[ -window_size : ])
        #     axs_dy[1,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     line_Y_ol.set_data( range(counter, counter + window_size ) ,line_Y_ol_his[ -window_size : ])
        #     line_Y_est.set_data( range(counter, counter + window_size ) ,line_Y_est_his[ -window_size : ])
        #     line_Y_meas.set_data( range(counter, counter + window_size ) ,line_Y_meas_his[ -window_size : ])
        #     axs_dy[2,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     line_yaw_ol.set_data( range(counter, counter + window_size ) ,line_yaw_ol_his[ -window_size : ])
        #     line_yaw_est.set_data( range(counter, counter + window_size ) ,line_yaw_est_his[ -window_size : ])
        #     line_yaw_meas.set_data( range(counter, counter + window_size ) ,line_yaw_meas_his[ -window_size : ])
        #     axs_dy[2,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     axs_dy[2,1].set_ylim(min(min(line_yaw_meas_his[ -window_size : ]) + margin*min(line_yaw_meas_his[ -window_size : ])\
        #         , min(line_yaw_ol_his[ -window_size : ]) + margin*min(line_yaw_ol_his[ -window_size : ])), \
        #     max(max(line_yaw_meas_his[ -window_size : ]) + margin*max(line_yaw_meas_his[ -window_size : ]), max(line_yaw_ol_his[ -window_size : ]) + margin*max(line_yaw_ol_his[ -window_size : ]))) # FOR SETTING THE DYNAMIC AXES

        #     fig_dy.canvas.draw()
        #     plt_dy.show()
        

        # ############################## Control input to the vehicle #################################

        # # line_dutycycle_his.append(vehicle_control.duty_cycle)
        # # line_steer_his.append(vehicle_control.steer)

        # if control_visualization == True and counter >= 100:

        #     line_dutycycle.set_data(range(counter, counter + window_size ) ,line_dutycycle_his[ -window_size : ])
        #     line_steer.set_data(range(counter, counter + window_size ) ,line_steer_his[ -window_size : ])
        #     plt_cont.xlim(counter, counter + window_size )
            
        #     fig_cont.canvas.draw()
        #     plt_cont.show()

        # # image_dy = Image.frombytes('RGB', fig_dy.canvas.get_width_height(),fig_dy.canvas.tostring_rgb())
        # # image_dy = np.fromstring(fig_dy.canvas.tostring_rgb(), dtype='uint8')
        # # print "image_dy.shape",image_dy.shape
        # # image_veh = np.fromstring(fig_veh.canvas.tostring_rgb(), dtype='uint8')

        # # image_dy_his.append(image_dy)
        # # image_veh_his.append(image_veh)

        # ##########################################################################################################

        
        
        

        # # create file name and append it to a list
        # # filename_veh = '/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehicle_motion/incorrect_yaw'+str(counter)+'.png'
        # # filename_dy = '/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehicle_states/incorrect_yaw'+str(counter)+'.png'
        
        # # filenames.append(filename)
        
        # # # repeat last frame
        # # if (index == len(coordinates_lists)-1):
        # #     for i in range(15):
        # #         filenames.append(filename)
                
        # # save frame



        # plt_veh.savefig(filename_veh)
        # plt_dy.savefig(filename_dy)

        # plt.pause(.1)
        # plt_veh.close()



        counter +=1
        rate.sleep()


    if record_data == True:

        data_est     = {'vx': line_vx_est_his , 'vy': line_vy_est_his , 'omega': line_omega_est_his , 'X': line_X_est_his , 'Y': line_Y_est_his , 'yaw': line_yaw_est_his } 
        data_meas    = {'vx': line_vx_meas_his , 'vy': line_vy_meas_his , 'omega': line_omega_meas_his , 'X': line_X_meas_his , 'Y': line_Y_meas_his , 'yaw': line_yaw_meas_his } 
        data_ol      = {'vx': line_vx_ol_his , 'vy': line_vy_ol_his , 'omega': line_omega_ol_his , 'X': line_X_ol_his , 'Y': line_Y_ol_his , 'yaw': line_yaw_ol_his } 
        if sim_on:
            data_real    = {'vx': line_vx_real_his , 'vy': line_vy_real_his , 'omega': line_omega_real_his , 'X': line_X_real_his , 'Y': line_Y_real_his , 'yaw': line_yaw_real_his } 
        data_control = {'duty': line_dutycycle_his , 'steer': line_steer_his }

        path = ('/').join(__file__.split('/')[:-2]) + '/data/observer/' 
            
        now = datetime.datetime.now()
        # path = path + now.strftime("d%d_m%m_y%Y/")
        path = path + now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")

        if not os.path.exists(path):
            os.makedirs(path)

        est_path  = path + '/est_his'
        meas_path = path + '/meas_his'
        real_path = path + '/real_his'
        ol_path   = path + '/ol_his'
        control_path = path + '/control_his'
        
        np.save(est_path,data_est)
        np.save(meas_path,data_meas)
        np.save(ol_path,data_ol)
        if sim_on:
            np.save(real_path,data_real)
        np.save(control_path,data_control)




    # Save into a GIF file that loops forever
    # image_dy_his[0].save("/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehilce_motion.gif", format='GIF', append_images=image_dy_his[1:], save_all=True, duration=300, loop=0)
    # from moviepy.editor import ImageSequenceClip
    # clip = ImageSequenceClip(list(np.array(image_veh_his)), fps=20)
    # clip.write_gif('test.gif', fps=20)

    print "Saved"
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
