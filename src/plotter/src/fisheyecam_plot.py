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
from PIL import Image
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Pose
from math import pi, sin, cos, tan, atan
import datetime
import os
import matplotlib.animation as animation
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
from trackInitialization import Map

#### plotter for visual sensor vs visual-inertial sensor ##########
def plot_cam_pose(x_lim_min,y_lim_min, x_lim_max, y_lim_max, map, track_plot_on):

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


    line_pure,        = axtr.plot(xdata, ydata, '-g', label = 'Localization using visual only', linewidth =2 )
    line_fused,       = axtr.plot(xdata, ydata, '-b', label = 'Localization using visual-inertial', linewidth =2)  

    v = np.array([[ 1,  1],
                  [ 1, -1],
                  [-1, -1],
                  [-1,  1]])

    rec_pure = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10)
    axtr.add_patch(rec_pure)

    rec_fused = patches.Polygon(v, alpha=0.7, closed=True, fc='g', ec='k', zorder=10)
    axtr.add_patch(rec_fused)

    plt.legend()
    plt.grid()
    

    return fig, plt, line_pure, line_fused, rec_pure, rec_fused

### wrap the angle between [-pi,pi] ###
def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


#################################### SUbsribe to fisheye camera ##################################### 
class fiseye_cam():


    def __init__(self):
        '''
        1. /pure_cam_pose topic estimate the camera position in world frame using only camera information 
        2. /fused_cam_pose topic estimate the camera position in world frame using IMU and camera information 
        '''

        rospy.Subscriber('pure_cam_pose', Pose, self.pure_cam_pose_callback, queue_size=1)
        rospy.Subscriber('fused_cam_pose', Pose, self.fused_cam_pose_callback, queue_size=1)

        #### Homogeneous transformation for reference change####
        # self.x_tf     = rospy.get_param("switching_lqr_observer/x_tf")
        # self.y_tf     = rospy.get_param("switching_lqr_observer/y_tf")
        # theta_tf = rospy.get_param("switching_lqr_observer/theta_tf")*pi/180
        # self.R_tf = np.array([[cos(theta_tf), -sin(theta_tf)],
        #                  [sin(theta_tf),  cos(theta_tf)]])
        # self.yaw_tf   = rospy.get_param("switching_lqr_observer/yaw_tf")*pi/180

        self.x_tf     = 0.970
        self.y_tf     = -0.812
        theta_tf = -90.*pi/180
        self.R_tf = np.array([[cos(theta_tf), -sin(theta_tf)],
                         [sin(theta_tf),  cos(theta_tf)]])
        self.yaw_tf   = 90.*pi/180


        self.pure_x   = 0.0
        self.pure_y   = 0.0
        self.pure_yaw = 0.0
        

        self.fused_x             = 0.0
        self.fused_y             = 0.0
        self.fused_yaw           = 0.0


    def pure_cam_pose_callback(self, data):

        [self.pure_x, self.pure_y] = np.dot(self.R_tf, np.array([data.position.x,data.position.y]).T)
        self.pure_x = self.pure_x - self.x_tf
        self.pure_y = self.pure_y - self.y_tf
        self.pure_yaw = wrap(data.orientation.z + self.yaw_tf)


    def fused_cam_pose_callback(self, data):

        [self.fused_x, self.fused_y] = np.dot(self.R_tf, np.array([data.position.x,data.position.y]).T)
        self.fused_x = self.fused_x - self.x_tf
        self.fused_y = self.fused_y - self.y_tf
        self.fused_yaw = wrap(data.orientation.z + self.yaw_tf)

#### gives the coordinate of the patches for plotting the rectangular (vehicle) orientation and position.
def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y




def main():


    rospy.init_node('fisheyecam_plotter', anonymous=True)

    loop_rate       = rospy.get_param("cam_pose_plotter/loop_rate")
    rate            = rospy.Rate(loop_rate)

    track_visualization = rospy.get_param("cam_pose_plotter/track_visualization")
    record_on = rospy.get_param("cam_pose_plotter/record_on")
    pose_visualization = rospy.get_param("cam_pose_plotter/pose_visualization")


    cam_pose  = fiseye_cam()

    track_map = Map()

    cam_pose_pure_x         = 0
    cam_pose_pure_y         = 0

    cam_pose_fused_x        = 0
    cam_pose_fused_y        = 0

    cam_pose_pure_x_hist    = []
    cam_pose_pure_y_hist    = []

    cam_pose_fused_x_hist   = []
    cam_pose_fused_y_hist   = []

    record_on = True
    pose_visualization  = True

    if pose_visualization == True:
        x_lim_min = -1.75
        y_lim_min = -0.75
        x_lim_max = 3
        y_lim_max = 3.5
        
        fig, plt, line_pure, line_fused, rec_pure, rec_fused = plot_cam_pose(x_lim_min,y_lim_min, x_lim_max,y_lim_max, track_map, track_visualization)

    counter = 0
    while not (rospy.is_shutdown()):


        ########################################### unpack messages ############################################

        cam_pure_x, cam_pure_y, cam_pure_yaw  = cam_pose.pure_x, cam_pose.pure_y, cam_pose.pure_yaw
        cam_fused_x, cam_fused_y, cam_fused_yaw = cam_pose.fused_x, cam_pose.fused_y, cam_pose.fused_yaw

        ########################################################################################################


        cam_pose_pure_x_hist.append(cam_pure_x)
        cam_pose_pure_y_hist.append(cam_pure_y)
        cam_pose_fused_x_hist.append(cam_fused_x)
        cam_pose_fused_y_hist.append(cam_fused_y)

        if pose_visualization == True:
            line_pure.set_data(cam_pose_pure_x_hist ,cam_pose_pure_y_hist)
            line_fused.set_data(cam_pose_fused_x_hist ,cam_pose_fused_y_hist)

            l = 0.42/2; w = 0.19/2

            car_pure_x, car_pure_y = getCarPosition(cam_pure_x, cam_pure_y, cam_pure_yaw, w, l)
            rec_pure.set_xy(np.array([car_pure_x, car_pure_y]).T)
            
            car_fused_x, car_fused_y = getCarPosition(cam_fused_x, cam_fused_y, cam_fused_yaw, w, l)
            rec_fused.set_xy(np.array([car_fused_x, car_fused_y]).T)
                        

            fig.canvas.draw()

        ##########################################################################################################

            plt.show()

            plt.pause(1.0/3000)

        counter +=1
        rate.sleep()

    print "Saving GIF images"

    if record_on == True:
        path = ('/').join(__file__.split('/')[:-2]) + '/data/camera/' 
            
        now = datetime.datetime.now()
        # path = path + now.strftime("d%d_m%m_y%Y/")
        path = path + now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")

        if not os.path.exists(path):
            os.makedirs(path)

        pure_path = path + '/pure_cam_pose'
        fused_path = path + '/fused_cam_pose'  

        pure_data = {'X':cam_pose_pure_x_hist, 'Y': cam_pose_pure_y_hist}
        fused_data = {'X':cam_pose_fused_x_hist, 'Y': cam_pose_fused_y_hist}
        

        np.save(pure_path, pure_data)
        np.save(fused_path, fused_data)

    # from collections import deque
    # import itertools
    # fig, plt, line_pure, line_fused = plot_cam_pose(x_lim,y_lim)
    # # history_len = 50  # how many trajectory points to display
    # # history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

    # ani_x = []
    # ani_y = []

    # def data_gen():
    #     for cnt in itertools.count():
    #         yield cnt

    # def animate(i):
    #     ani_x.append(cam_pose_pure_x_hist[:i]) 
    #     ani_y.append(cam_pose_pure_y_hist[:i])
    #     line_pure.set_data(ani_x , ani_y)
    #     # line_fused.set_data(cam_pose_fused_x_hist[i] ,cam_pose_fused_y_hist[i])

    #     return line_pure, #, line_fused

    # ani = animation.FuncAnimation(fig, animate, len(cam_pose_pure_y_hist), interval=10, blit=True)
    # plt.show()



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
