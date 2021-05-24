#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' This code is to compare the performance of the estimation technique. The
estimated states, measured states and the open loop simulation states are
plotted to compare the performance. This code is bit slow due to online
plotting of matplotlib. Use other plotter for real time debugging such as
plotjuggler. This plot is created for documentation which has nice appearance.
 '''

import numpy as np
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sensor_fusion.msg import sensorReading, control
from simulator.msg import simulatorStates
from PIL import Image

#### plotter for vehicle motion ####
def plot_vehicle_kinematics(x_lim,y_lim):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_state,       = axtr.plot(xdata, ydata, '-b', label = 'Estimated states')  

    l = 0.4; w = 0.2

    v = np.array([[ 1,  1],
                  [ 1, -1],
                  [-1, -1],
                  [-1,  1]])

    rec_state = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10)
    axtr.add_patch(rec_state)

    plt.legend()
    plt.grid()
    

    return fig, plt, axtr, line_state, rec_state



#### gives the coordinate of the patches for plotting the rectangular (vehicle) orientation and position.
def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y


'''All the three subscription is made from the estimator as it produces all this information'''
class vehicle_simstates(object):
    def __init__(self):
        print "subscribed to vehicle_simulatorStates"
        rospy.Subscriber("vehicle_simulatorStates", simulatorStates, self.sim_callback, queue_size=1)
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def sim_callback(self, msg):
        self.CurrentState = np.array([msg.vx, msg.vy, msg.omega, msg.x, msg.y, msg.yaw]).T

class sensor_measurement(object):
    def __init__(self):
        
        print "subscribed to sensor measurement"
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




def main():


    rospy.init_node('simulator_plotter', anonymous=True)
    vehicle_visualization = rospy.get_param("simulator_plot/visualization")

    loop_rate       = 200
    rate            = rospy.Rate(loop_rate)

    vehicle_state = vehicle_simstates()

    state_x_his = []
    state_y_his = []

    


    if vehicle_visualization == True:

        margin = 0.5 ## margin percentage fox axes: make dynamic window size
        x_lim_init_max = 5
        x_lim_init_min = -x_lim_init_max
        
        y_lim_init_max = 5
        y_lim_init_min = -y_lim_init_max


        ### Vehicle kinematics
        (fig, plt, axtr, line_state, rec_state) = plot_vehicle_kinematics(x_lim_init_max,y_lim_init_max)


    counter = 0
    while not (rospy.is_shutdown()):


        ########################################### unpack messages ############################################

        ( vx_state  , vy_state  , omega_state  , X_state  , Y_state  , yaw_state  )  = vehicle_state.CurrentState

        print "vx_state  , vy_state  , omega_state  , X_state  , Y_state  , yaw_state  ", vx_state  , vy_state  , omega_state  , X_state  , Y_state  , yaw_state  
        ########################################################################################################



        ############################################## vehicle motion plot ######################################

        if vehicle_visualization == True: 

            l = 0.42; w = 0.19

            state_x_his.append(X_state)
            state_y_his.append(Y_state)

            
            car_state_x, car_state_y = getCarPosition(X_state, Y_state, yaw_state, w, l)
            rec_state.set_xy(np.array([car_state_x, car_state_y]).T)

            
            line_state.set_data(state_x_his, state_y_his)
            
            ############# Dynamic window size ##############
            min_x_lim =  min(state_x_his) - margin*min(state_x_his) 
            max_x_lim =  max(state_x_his) + margin*max(state_x_his)
            min_y_lim =  min(state_y_his) - margin*min(state_y_his)
            max_y_lim =  max(state_y_his) + margin*max(state_y_his)

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
                

        fig.canvas.draw()

        plt.show()
        
        # create file name and append it to a list
        # filename_veh = '/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehicle_motion/incorrect_yaw'+str(counter)+'.png'
        # filename_dy = '/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehicle_states/incorrect_yaw'+str(counter)+'.png'
        
        # filenames.append(filename)
        
        # # repeat last frame
        # if (index == len(coordinates_lists)-1):
        #     for i in range(15):
        #         filenames.append(filename)
                
        # save frame



        # plt_veh.savefig(filename_veh)
        # plt_dy.savefig(filename_dy)

        plt.pause(1.0/3000)
        # plt_veh.close()



        counter +=1
        rate.sleep()

    print "Saving GIF images"

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
