#!/usr/bin/env python


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

#### plotter for vehicle motion ####
def plot_vehicle_kinematics(x_lim,y_lim):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_ol,        = axtr.plot(xdata, ydata, '-g', label = 'Open loop simulation')
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




#### plotter for vehicle 6 states ####
def plot_vehicle_states(window):

    xdata = []; ydata = [];
    fig, axs = plt.subplots(3, 2, figsize=(50,50))
    plt.ion()


    ### longitudinal Velocity
    line_vx_est,        = axs[0,0].plot(xdata, ydata, '-b', label = r'$v_x$: Estimated longitudinal velocity')
    line_vx_ol,         = axs[0,0].plot(xdata, ydata, '-g', label = r'$v_x$: Open loop longitudinal velocity')  
    line_vx_meas,       = axs[0,0].plot(xdata, ydata, '-m', label = r'$v_x$: Measured longitudinal velocity')  
    axs[0,0].set_xlim(0,window)
    axs[0,0].set_ylim(-1,5)
    axs[0,0].legend(prop={'size': 12}, framealpha = 0.2)
    axs[0,0].grid()



    ### lateral Velocity
    line_vy_est,        = axs[0,1].plot(xdata, ydata, '-b', label = r'$v_y$: Estimated lateral velocity')
    line_vy_ol,         = axs[0,1].plot(xdata, ydata, '-g', label = r'$v_y$: Open loop lateral velocity')  
    line_vy_meas,       = axs[0,1].plot(xdata, ydata, '-m', label = r'$v_y$: Measured lateral velocity')  
    axs[0,1].set_xlim(0,window)
    axs[0,1].set_ylim(-1,5)
    axs[0,1].legend(prop={'size': 12}, framealpha = 0.2)
    axs[0,1].grid()

    ### Angular rate
    line_omega_est,     = axs[1,0].plot(xdata, ydata, '-b', label = r'$\omega$: Estimated angular velocity')
    line_omega_ol,      = axs[1,0].plot(xdata, ydata, '-g', label = r'$\omega$: Open loop angular velocity')  
    line_omega_meas,    = axs[1,0].plot(xdata, ydata, '-m', label = r'$\omega$: Measured angular velocity')  
    axs[1,0].set_xlim(0,window)
    axs[1,0].set_ylim(-5,5)
    axs[1,0].legend(prop={'size': 12} , framealpha = 0.2)
    axs[1,0].grid()
    
    ### Global X -position
    line_X_est,     = axs[1,1].plot(xdata, ydata, '-b', label = r'$X$: Estimated X - position')
    line_X_ol,      = axs[1,1].plot(xdata, ydata, '-g', label = r'$X$: Open loop X - position')  
    line_X_meas,    = axs[1,1].plot(xdata, ydata, '-m', label = r'$X$: Measured X - position')  
    axs[1,1].set_xlim(0,window)
    axs[1,1].set_ylim(-10,10)
    axs[1,1].legend(prop={'size': 12} , framealpha = 0.2)
    axs[1,1].grid()
    

    ### Global Y -position
    line_Y_est,     = axs[2,0].plot(xdata, ydata, '-b', label = r'$Y$: Estimated Y - position')
    line_Y_ol,      = axs[2,0].plot(xdata, ydata, '-g', label = r'$Y$: Open loop Y - position')  
    line_Y_meas,    = axs[2,0].plot(xdata, ydata, '-m', label = r'$Y$: Measured Y - position')  
    axs[2,0].set_xlim(0,window)
    axs[2,0].set_ylim(-10,10)
    axs[2,0].legend(prop={'size': 12} , framealpha = 0.2)
    axs[2,0].grid()


    ### Yaw
    line_yaw_est,     = axs[2,1].plot(xdata, ydata, '-b', label = r'$\theta$: Estimated yaw')
    line_yaw_ol,      = axs[2,1].plot(xdata, ydata, '-g', label = r'$\theta$: Open loop yaw')  
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
        print "subscribed to vehicle open loop states"
        rospy.Subscriber('ol_state_info', sensorReading, self.vehicle_ol_state_callback, queue_size=1)
        self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
    def vehicle_ol_state_callback(self, msg):
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T




def main():


    rospy.init_node('observer_performance_tester', anonymous=True)
    loop_rate       = 200
    rate            = rospy.Rate(loop_rate)

    vehicle_state_est = EstimatorData()
    vehicle_state_meas = Vehicle_measurement()
    vehicle_state_ol = Vehicle_ol()



    vehicle_visualization = True
    states_visualization = True


    if vehicle_visualization == True:

        margin = 0.5 ## margin percentage fox axes: make dynamic window size
        x_lim_init_max = 2
        x_lim_init_min = -x_lim_init_max
        
        y_lim_init_max = 2
        y_lim_init_min = -y_lim_init_max


        ### Vehicle kinematics
        (fig, plt_veh, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas) = plot_vehicle_kinematics(x_lim_init_max,y_lim_init_max)

        ol_x_his     = []
        est_x_his    = []
        meas_x_his   = []
        ol_y_his     = []
        est_y_his    = []
        meas_y_his   = []



    if states_visualization == True:

        window_size = 100

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
            min_x_lim = min(ol_x_his) - margin*min(ol_x_his) 
            max_x_lim = max(ol_x_his) + margin*max(ol_x_his)
            min_y_lim = min(ol_y_his) - margin*min(ol_y_his)
            max_y_lim = max(ol_y_his) + margin*max(ol_y_his)

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


            # print "range(len(line_vx_ol_his))", range(len(line_vx_ol_his))
            # print "line_vx_ol_his", line_vx_ol_his

            ### Keep size of window to 100 points 

            if counter >window_size :

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
                

                line_vx_ol.set_data( range(counter, counter + window_size + 1) ,line_vx_ol_his)
                line_vx_est.set_data( range(counter, counter + window_size + 1) ,line_vx_est_his)
                line_vx_meas.set_data( range(counter, counter + window_size + 1) ,line_vx_meas_his)
                axs_dy[0,0].set_xlim(counter, counter + window_size + 1) # FOR SETTING THE DYNAMIC AXES
                
                
                line_vy_ol.set_data( range(counter, counter + window_size + 1) ,line_vy_ol_his)
                line_vy_est.set_data( range(counter, counter + window_size + 1) ,line_vy_est_his)
                line_vy_meas.set_data( range(counter, counter + window_size + 1) ,line_vy_meas_his)
                axs_dy[0,1].set_xlim(counter, counter + window_size + 1) # FOR SETTING THE DYNAMIC AXES
                

                line_omega_ol.set_data( range(counter, counter + window_size + 1) ,line_omega_ol_his)
                line_omega_est.set_data( range(counter, counter + window_size + 1) ,line_omega_est_his)
                line_omega_meas.set_data( range(counter, counter + window_size + 1) ,line_omega_meas_his)
                axs_dy[1,0].set_xlim(counter, counter + window_size + 1) # FOR SETTING THE DYNAMIC AXES
                
                
                line_X_ol.set_data( range(counter, counter + window_size + 1) ,line_X_ol_his)
                line_X_est.set_data( range(counter, counter + window_size + 1) ,line_X_est_his)
                line_X_meas.set_data( range(counter, counter + window_size + 1) ,line_X_meas_his)
                axs_dy[1,1].set_xlim(counter, counter + window_size + 1) # FOR SETTING THE DYNAMIC AXES
                

                line_Y_ol.set_data( range(counter, counter + window_size + 1) ,line_Y_ol_his)
                line_Y_est.set_data( range(counter, counter + window_size + 1) ,line_Y_est_his)
                line_Y_meas.set_data( range(counter, counter + window_size + 1) ,line_Y_meas_his)
                axs_dy[2,0].set_xlim(counter, counter + window_size + 1) # FOR SETTING THE DYNAMIC AXES
                

                line_yaw_ol.set_data( range(counter, counter + window_size + 1) ,line_yaw_ol_his)
                line_yaw_est.set_data( range(counter, counter + window_size + 1) ,line_yaw_est_his)
                line_yaw_meas.set_data( range(counter, counter + window_size + 1) ,line_yaw_meas_his)
                axs_dy[2,1].set_xlim(counter, counter + window_size + 1) # FOR SETTING THE DYNAMIC AXES
                

                fig_dy.canvas.draw()



        ##########################################################################################################

        plt_dy.show()
        plt_veh.show()
        plt.pause(1.0/3000)

        counter +=1
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass