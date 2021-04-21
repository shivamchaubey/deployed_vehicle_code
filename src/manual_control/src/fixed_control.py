#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import math

import rospy
# from geometry_msgs.msg import Twist
from geometry_msgs.msg import Twist, Pose
# from datetime import datetime
import datetime
import os

import sys
import datetime
import rospy
import numpy as np
import scipy.io as sio
import pdb
import pickle
import matplotlib.pyplot as plt
from plotter.msg import simulatorStates, hedge_imu_fusion
from std_msgs.msg import Bool, Float32



### under development ###
# class plot_data():

#     def __init__ 
#     # Create figure for plotting
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# xs = []
# ys = []

# # Initialize communication with TMP102
# tmp102.init()

# # This function is called periodically from FuncAnimation
# def animate(i, xs, ys):

#     # Read temperature (Celsius) from TMP102
#     temp_c = round(tmp102.read_temp(), 2)

#     # Add x and y to lists
#     xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
#     ys.append(temp_c)

#     # Limit x and y lists to 20 items
#     xs = xs[-20:]
#     ys = ys[-20:]

#     # Draw x and y lists
#     ax.clear()
#     ax.plot(xs, ys)

#     # Format plot
#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.30)
#     plt.title('TMP102 Temperature over Time')
#     plt.ylabel('Temperature (deg C)')

# # Set up plot to call animate() function periodically
# ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
# plt.show()
class motor_feedback(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('/vehicle_velocity_desired', Float32, self.vel_des_callback, queue_size=1)

        rospy.Subscriber('/vehicle_velocity_feed', Float32, self.vel_feed_callback, queue_size=1)
        

        self.vel_x_des    = 0.0
        self.vel_x_feed   = 0.0
        
        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

                #history
        # self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[]}
        # self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}

    def vel_des_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vel_x_des     = data.data  # from DC motor encoder
        
    def vel_feed_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vel_x_feed     = data.data  # from DC motor encoder

class ImuEncClass(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('twist', Twist, self.Twist_callback, queue_size=1)

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)

        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.psiDot  = 0.0

        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

                #history
        self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[]}
        self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}

    def Twist_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vx     = data.linear.x  # from DC motor encoder
        self.vy     = data.linear.y
        self.psiDot = data.angular.z # from IMU

        # self.twist_hist["timestamp_ms"].append(self.curr_time)
        # self.twist_hist["vx"].append(self.vx)
        # self.twist_hist["vy"].append(self.vy)
        # self.twist_hist["psiDot"].append(self.psiDot)


    def Pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = wrap(data.orientation.x)
        self.pitch  = wrap(data.orientation.y)
        self.yaw    = wrap(data.orientation.z) # from IMU

        # self.pose_hist["timestamp_ms"].append(self.curr_time)
        # self.pose_hist["roll"].append(self.roll)
        # self.pose_hist["pitch"].append(self.pitch)
        # self.pose_hist["yaw"].append(self.yaw)

class GpsClass(object):
    """ Object collecting GPS measurement data
    Attributes:
        Measurement:
            1.x 2.y
        Measurement history:
            1.x_his 2.y_his
        Time stamp
            1.t0  2.curr_time
    """
    def __init__(self,t0):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """
        rospy.Subscriber('hedge_imu_fusion', hedge_imu_fusion, self.gps_callback, queue_size=1)
        # rospy.Subscriber('hedge_imu_fusion', hedge_imu_fusion, self.gps_callback, queue_size=1)

        # GPS measurement
        self.x_m = 0
        self.y_m = 0
        self.z_m = 0
        self.qw = 0
        self.qx = 0
        self.qy = 0
        self.qz = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.ax = 0
        self.ay = 0
        self.az = 0

        # self.x_his  = 0.0
        # self.y_his  = 0.0

        # time stamp
        self.t0             = t0
        self.curr_time      = rospy.get_rostime().to_sec() - self.t0

    def gps_callback(self, data):
        """Unpack message from sensor, GPS"""
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0
#        dist = np.sqrt((data.x_m - self.x_his) ** 2 + (data.y_m - self.y_his) ** 2)
        self.x_m = data.x_m
        self.y_m = data.y_m
        self.z_m = data.z_m
        self.qw = data.qw
        self.qx = data.qx
        self.qy = data.qy
        self.qz = data.qz
        self.vx = data.vx
        self.vy = data.vy
        self.vz = data.vz
        self.ax = data.ax
        self.ay = data.ay
        self.az = data.az

        # self.x = data.x_m
        # self.y = data.y_m
        # self.saveHistory()

    def saveHistory(self):
        self.x_his  = self.x
        self.y_his  = self.y


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def main():
    rospy.init_node('fixed_control')

    control_hist = []

    time0 = rospy.get_time()

    
    ##################### control command publisher ######################
    accel_commands     = rospy.Publisher('control/accel', Float32, queue_size=1)
    steering_commands  = rospy.Publisher('control/steering', Float32, queue_size=1)
    controller_Flag    = rospy.Publisher('controller_flag', Bool, queue_size=1)


    ##################### sensor subscriber ################################
    # imu_enc = ImuEncClass(time0)
    # motor_feed = motor_feedback(time0)
    # gps_msg = GpsClass(time0)

    # record_data = rospy.get_param('fixed_control/record_data')

    record_data = False
    
    control_commands_his  = {"real_timestamp_ms":[],"timestamp_ms":[],"duty_cycle":[],"steering":[],"desired_velocity":[],"feedback_velocity":[]}
    # twist_hist = {"real_timestamp_ms":[],"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[]}
    # pose_hist = {"real_timestamp_ms":[],"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}
    # gps_data = {'real_timestamp_ms':[],'timestamp_ms':[],'x_m':[],'y_m':[],'z_m':[],'qw':[],'qx':[],'qy':[],'qz':[],'vx':[],'vy':[],'vz':[],'ax':[],'ay':[],'az':[]}

    
    _hz = rospy.get_param('~hz', 1/0.033)
    rate = rospy.Rate(_hz)

    _linear = 0
    _angular = 0

    c = 0
    start_val = 2 #sec ### after how much second the signal will be send.
    on_time = 1.0 #sec ### duration for which the signal will be send.
    duty_cycle_val = -0.4 #-1 to +1
    while not rospy.is_shutdown():

        t00 = rospy.get_time()        # print ("time at start publishing",time1-time0)
        accel_commands.publish(_linear)
        steering_commands.publish(_angular)
        # print ("time for finishing publishing",rospy.get_time()-time1)

        if t00-time0>start_val:
            if (t00-time0) < start_val + on_time:
                _linear = duty_cycle_val
                print ("time left =",start_val + on_time - (t00-time0))
		c += 1
            else:
                _linear = 0
            if (t00-time0) > start_val + on_time + 10:
                break 

        control_commands_his["real_timestamp_ms"]. append(t00)
        control_commands_his["timestamp_ms"]. append(t00-time0)
        control_commands_his["duty_cycle"]. append(_linear)
        control_commands_his["steering"]. append(_angular)
        
        # control_commands_his["desired_velocity"].append(motor_feed.vel_x_des)
        # control_commands_his["feedback_velocity"].append(motor_feed.vel_x_feed)

        # twist_hist["real_timestamp_ms"].append(t00)        
        # twist_hist["timestamp_ms"].append(t00-time0)
        # twist_hist["vx"].append(imu_enc.vx)
        # twist_hist["vy"].append(imu_enc.vy)
        # twist_hist["psiDot"].append(imu_enc.psiDot)

        # pose_hist["real_timestamp_ms"].append(t00)
        # pose_hist["timestamp_ms"].append(t00-time0)
        # pose_hist["roll"].append(imu_enc.roll)
        # pose_hist["pitch"].append(imu_enc.pitch)
        # pose_hist["yaw"].append(imu_enc.yaw)
        
        # gps_data["real_timestamp_ms"].append(t00)
        # gps_data["timestamp_ms"].append(t00-time0)
        # gps_data['x_m'].append(gps_msg.x_m)
        # gps_data['y_m'].append(gps_msg.y_m)
        # gps_data['z_m'].append(gps_msg.z_m)
        # gps_data['qw'].append(gps_msg.qw)
        # gps_data['qx'].append(gps_msg.qx)
        # gps_data['qy'].append(gps_msg.qy)
        # gps_data['qz'].append(gps_msg.qz)
        # gps_data['vx'].append(gps_msg.vx)
        # gps_data['vy'].append(gps_msg.vy)
        # gps_data['vz'].append(gps_msg.vz)
        # gps_data['ax'].append(gps_msg.ax)
        # gps_data['ay'].append(gps_msg.ay)
        # gps_data['az'].append(gps_msg.az)

        print ("time = {}, duty_cycle = {} , steering = {}, counter = {}".format(t00,_linear,_angular,c))


        rate.sleep()
    
    if record_data == True:
        path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
            
        now = datetime.datetime.now()
        path = path + now.strftime("d%d_m%m_y%Y/")
        dt_string = now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")
        
        control_path = path + 'control_his_'+ dt_string
        print('control save path and file:    ', control_path)

        imu_enc_pose_path = path + 'imu_enc_pose_his_'+ dt_string
        imu_enc_twist_path = path + 'imu_enc_twist_his_'+ dt_string
        gps_path = path + 'marvelmind_gps_his_'+ dt_string
        
        if not os.path.exists(path):
            os.makedirs(path)
        # print ("self.pose_hist", self.imu_enc.pose_hist)
        # print ("control_his",self.control_commands_his)
        np.save(control_path,control_commands_his)
        np.save(imu_enc_pose_path,pose_hist)
        np.save(imu_enc_twist_path,twist_hist)
        np.save(gps_path,gps_data)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

