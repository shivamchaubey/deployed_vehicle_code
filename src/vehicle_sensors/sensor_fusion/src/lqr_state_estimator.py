#!/usr/bin/env python

from math import tan, atan, cos, sin, pi, atan2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rospy
from numpy.random import randn,rand
import rosbag
import pandas as pd
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool, Float32
from sensor_fusion.msg import sensorReading, control, hedge_imu_fusion, hedge_imu_raw
import tf
import time
from numpy import linalg as LA
import datetime
import os
import sys
import scipy.io as sio
    

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

class motor_encoder():

    def __init__(self,t0,N = 10):

        rospy.Subscriber('/wheel_rpm_feedback', Float32, self.RPM_callback, queue_size=1)       

        self.curr_time    = 0.0
        self.wheel_rpm    = 0.0
        self.vx           = 0.0
        self.s            = 0.0

        self.wheel_rpm_MA_window = [0]*N #moving average
        self.wheel_rpm_MA = 0.0
        self.vx_MA        = 0.0
        self.s_MA         = 0.0
        self.wheel_radius     = 0.03*1.12178 #radius of wheel

        # time stamp
        self.t0     = t0
        self.N      = N
        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

    def RPM_callback(self, data):

        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.wheel_rpm = data.data

        self.wheel_rpm_MA_window.pop(0)
        self.wheel_rpm_MA_window.append(self.wheel_rpm)
        
        self.vx        = (self.wheel_rpm*2*pi/60.0)*self.wheel_radius
        self.s        += abs(self.vx)*(self.curr_time - self.prev_time)  

        ### Moving average
        self.wheel_rpm_MA    =  np.squeeze(np.convolve(self.wheel_rpm_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.vx_MA     = (self.wheel_rpm_MA*2*pi/60.0)*self.wheel_radius
        self.s_MA     += self.vx_MA*(self.curr_time - self.prev_time)  
        
        self.prev_time = self.curr_time

    def data_retrive(self, msg):

        msg.timestamp_ms = self.curr_time
        msg.X  = 0
        msg.Y  = 0
        msg.roll  = 0
        msg.yaw  = 0
        msg.pitch  = 0
        msg.vx  = self.vx
        msg.vy  = 0
        msg.yaw_rate  = 0
        msg.ax  = 0
        msg.ay  = 0
        msg.s  = self.s
        msg.x  = 0
        msg.y  = 0

        return msg

    def data_retrive_MA(self, msg):

        msg.timestamp_ms = self.curr_time
        msg.X  = 0
        msg.Y  = 0
        msg.roll  = 0
        msg.yaw  = 0
        msg.pitch  = 0
        msg.vx  = self.vx_MA
        msg.vy  = 0
        msg.yaw_rate  = 0
        msg.ax  = 0
        msg.ay  = 0
        msg.s  = self.s_MA
        msg.x  = 0
        msg.y  = 0

        return msg


class marvelmind():


    def __init__(self,t0,N):

        rospy.Subscriber('hedge_imu_fusion', hedge_imu_fusion, self.fusion_callback, queue_size=1)
        rospy.Subscriber('hedge_imu_raw', hedge_imu_raw, self.raw_callback, queue_size=1)
        

        ## Define Origin ##
        self.x_m_offset = 0
        self.y_m_offset = 0 

        # GPS measurement
        self.x_m    = 0.0
        self.y_m    = 0.0
        self.z_m    = 0.0
        self.qw     = 0.0
        self.qx     = 0.0
        self.qy     = 0.0
        self.qz     = 0.0
        self.vx     = 0.0
        self.vy     = 0.0
        self.vz     = 0.0
        self.ax     = 0.0
        self.ay     = 0.0
        self.az     = 0.0
        self.roll   = 0.0
        self.yaw    = 0.0
        self.pitch  = 0.0 

        self.yaw_rate =0.0

        ####### MOVING AVERAGE #######

        self.x_m_MA             = 0.0           
        self.x_m_MA_window      = [0.0]*N
        self.y_m_MA             = 0.0           
        self.y_m_MA_window      = [0.0]*N
        self.z_m_MA             = 0.0           
        self.z_m_MA_window      = [0.0]*N
        self.qw_MA              = 0.0           
        self.qw_MA_window       = [0.0]*N
        self.qx_MA              = 0.0           
        self.qx_MA_window       = [0.0]*N
        self.qy_MA              = 0.0           
        self.qy_MA_window       = [0.0]*N
        self.qz_MA              = 0.0           
        self.qz_MA_window       = [0.0]*N
        self.vx_MA              = 0.0           
        self.vx_MA_window       = [0.0]*N
        self.vy_MA              = 0.0           
        self.vy_MA_window       = [0.0]*N
        self.vz_MA              = 0.0           
        self.vz_MA_window       = [0.0]*N
        self.ax_MA              = 0.0           
        self.ax_MA_window       = [0.0]*N
        self.ay_MA              = 0.0           
        self.ay_MA_window       = [0.0]*N
        self.az_MA              = 0.0           
        self.az_MA_window       = [0.0]*N
        self.roll_MA            = 0.0           
        self.roll_MA_window = [0.0]*N
        self.yaw_MA             = 0.0           
        self.yaw_MA_window      = [0.0]*N
        self.pitch_MA           = 0.0           
        self.pitch_MA_window    = [0.0]*N

        self.yaw_rate_MA =0.0
        self.yaw_rate_MA_window = [0.0]*N

        self.imu = IMU(t0,10)
        self.encoder = motor_encoder(t0)
        self.s       = 0.0

        self.N      = N
        # self.x_his  = 0.0
        # self.y_his  = 0.0

        # time stamp
        self.t0             = t0
        self.curr_time      = rospy.get_rostime().to_sec() - self.t0

    def raw_callback(self,data):
        
        self.yaw_rate = data.gyro_z

        self.yaw_rate_MA_window.pop(0)
        self.yaw_rate_MA_window.append(self.yaw_rate)
        self.yaw_rate_MA = np.squeeze(np.convolve(self.yaw_rate_MA_window, np.ones(self.N)/self.N, mode='valid'))


    def fusion_callback(self, data):
        """Unpack message from sensor, GPS"""
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0
#        dist = np.sqrt((data.x_m - self.x_his) ** 2 + (data.y_m - self.y_his) ** 2)


        self.x_m = data.x_m - self.x_m_offset
        self.y_m = data.y_m - self.y_m_offset
        self.z_m = data.z_m
        
        ## Global frame to vehicle body frame velocity transformation
        Vx = -data.vx
        Vy = -data.vy

        self.vx = Vx*cos(self.imu.yaw) + Vy*sin(self.imu.yaw)
        self.vy = -Vx*sin(self.imu.yaw) + Vy*cos(self.imu.yaw)

        self.vz = data.vz
        self.ax = data.ax
        self.ay = data.ay
        self.az = data.az

        self.qw = data.qw
        self.qx = data.qx
        self.qy = data.qy
        self.qz = data.qz
        euler = tf.transformations.euler_from_quaternion([self.qx, self.qy, self.qz, self.qw]) 
        # euler = tf.transformations.euler_from_quaternion([0, 0, self.qz, self.qw])   
        
        self.roll  = euler[0] 
        self.pitch = euler[1]
        self.yaw   = euler[2]
        


        ###### MOVING AVERAGE #####

        self.x_m_MA_window.pop(0)
        self.x_m_MA_window.append(self.x_m)
        self.x_m_MA = np.squeeze(np.convolve(self.x_m_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.y_m_MA_window.pop(0)
        self.y_m_MA_window.append(self.y_m)
        self.y_m_MA = np.squeeze(np.convolve(self.y_m_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.z_m_MA_window.pop(0)
        self.z_m_MA_window.append(self.z_m)
        self.z_m_MA = np.squeeze(np.convolve(self.z_m_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.vx_MA_window.pop(0)
        self.vx_MA_window.append(self.vx)
        self.vx_MA = np.squeeze(np.convolve(self.vx_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.vy_MA_window.pop(0)
        self.vy_MA_window.append(self.vy)
        self.vy_MA = np.squeeze(np.convolve(self.vy_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.vz_MA_window.pop(0)
        self.vz_MA_window.append(self.vz)
        self.vz_MA = np.squeeze(np.convolve(self.vz_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.ax_MA_window.pop(0)
        self.ax_MA_window.append(self.ax)
        self.ax_MA = np.squeeze(np.convolve(self.ax_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.ay_MA_window.pop(0)
        self.ay_MA_window.append(self.ay)
        self.ay_MA = np.squeeze(np.convolve(self.ay_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.az_MA_window.pop(0)
        self.az_MA_window.append(self.az)
        self.az_MA = np.squeeze(np.convolve(self.az_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.qw_MA_window.pop(0)
        self.qw_MA_window.append(self.qw)
        self.qw_MA = np.squeeze(np.convolve(self.qw_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.qx_MA_window.pop(0)
        self.qx_MA_window.append(self.qx)
        self.qx_MA = np.squeeze(np.convolve(self.qx_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.qy_MA_window.pop(0)
        self.qy_MA_window.append(self.qy)
        self.qy_MA = np.squeeze(np.convolve(self.qy_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.qz_MA_window.pop(0)
        self.qz_MA_window.append(self.qz)
        self.qz_MA = np.squeeze(np.convolve(self.qz_MA_window, np.ones(self.N)/self.N, mode='valid'))
        
        quaternion = np.squeeze(np.array([self.qx_MA, self.qy_MA, self.qz_MA, self.qw_MA]))
        # print ("self.qx_MA, self.qy_MA, self.qz_MA, self.qw_MA", quaternion)
        euler = tf.transformations.euler_from_quaternion(quaternion)   

        self.roll_MA  = euler[0] 
        self.yaw_MA   = euler[2]
        self.pitch_MA = euler[1]


        dist =  LA.norm(np.array([self.x_m_MA_window[-1],self.y_m_MA_window[-1]])-np.array([self.x_m_MA_window[-2], self.y_m_MA_window[-2]]))

        if self.encoder.vx == 0.0:
            dist = 0.0

        self.s += dist


        self.prev_time = self.curr_time

    def calibrate_marvel(self,delay,offset):

        x_m_info = []
        y_m_info = []
        yaw_rate_info = []
        ax_info = []
        ay_info = []
        az_info = []
        vx_info = []
        vy_info = []
        yaw_info = []

        # t1 = rospy.get_rostime().to_sec()
        # while   t1 - self.t0 < delay: ### time for 5sec
        for i in range(50):
            # t1 = rospy.get_rostime().to_sec()
            yaw_rate_info.append(self.yaw_rate)
            ax_info.append(self.ax)
            ay_info.append(self.ay)
            az_info.append(self.az)
            
            vx_info.append(self.vx)
            vy_info.append(self.vy)
            yaw_info.append(self.yaw)
            x_m_info.append(self.x_m)
            y_m_info.append(self.y_m)
            # print ('time', t1 - self.t0)        


        self.x_m_offset = np.mean(x_m_info)
        self.y_m_offset = np.mean(y_m_info)

        # self.yaw_rate_offset  = np.mean(yaw_rate_info)
        # self.ax_offset      = np.mean(ax_info)
        # self.ay_offset      = np.mean(ay_info)
        # self.az_offset      = np.mean(az_info)
        # self.vx_offset      = np.mean(vx_info)
        # self.vy_offset      = np.mean(vy_info)
        # self.yaw_offset     = np.mean(yaw_info)  - offset

        # self.co_yaw     = np.var(yaw_info)
        # self.co_psiDot  = np.var(yaw_rate_info)
        # self.co_ax      = np.var(ax_info)
        # self.co_ay      = np.var(ay_info)
        # self.co_vx      = np.var(vx_info)
        # self.co_vy      = np.var(vy_info)

    def data_retrive(self, msg):

        msg.timestamp_ms = self.curr_time
        msg.X  = self.x_m
        msg.Y  = self.y_m
        msg.roll  = self.roll
        msg.yaw  = self.yaw
        msg.pitch  = self.pitch
        msg.vx  = self.vx
        msg.vy  = self.vy
        msg.yaw_rate  = self.yaw_rate
        msg.ax  = self.ax
        msg.ay  = self.ay
        msg.s  = self.s
        msg.x  = 0
        msg.y  = 0

        return msg

    def data_retrive_MA(self, msg):

        msg.timestamp_ms = self.curr_time
        msg.X  = self.x_m_MA
        msg.Y  = self.y_m_MA
        msg.roll  = self.roll_MA
        msg.yaw  = self.yaw_MA
        msg.pitch  = self.pitch_MA
        msg.vx  = self.vx_MA
        msg.vy  = self.vy_MA
        msg.yaw_rate  = self.yaw_rate_MA
        msg.ax  = self.ax_MA
        msg.ay  = self.ay_MA
        msg.s  = self.s
        msg.x  = 0
        msg.y  = 0
        
        return msg


class IMU():

    def __init__(self,t0, N):

        rospy.Subscriber('twist', Twist, self.Twist_callback, queue_size=1)

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)

        self.roll               = 0.0
        self.pitch              = 0.0
        self.yaw                = 0.0

        self.roll_MA            = 0.0
        self.roll_MA_window     = [0.0]*N
        self.pitch_MA           = 0.0
        self.pitch_MA_window    = [0.0]*N
        self.yaw_MA             = 0.0
        self.yaw_MA_window      = [0.0]*N

        self.roll_rate              = 0.0
        self.roll_rate_MA           = 0.0
        self.roll_rate_MA_window    = [0.0]*N

        self.pitch_rate             = 0.0
        self.pitch_rate_MA          = 0.0
        self.pitch_rate_MA_window   = [0.0]*N

        self.yaw_rate               = 0.0
        self.yaw_rate_MA            = 0.0
        self.yaw_rate_MA_window     = [0.0]*N
        self.yaw_rate_offset        = -0.218736468054
        
        self.vx      = 0.0
        self.vy      = 0.0

        self.vx_MA              = 0.0
        self.vy_MA              = 0.0
        self.psiDot_MA          = 0.0
        
        self.ax             = 0.0
        self.ax_MA          = 0.0
        self.ax_MA_window   = [0.0]*N
        self.ay             = 0.0
        self.ay_MA          = 0.0
        self.ay_MA_window   = [0.0]*N
        self.az             = 0.0
        self.az_MA          = 0.0
        self.az_MA_window   = [0.0]*N
        
        self.X              = 0.0
        self.X_MA           = 0.0

        self.Y              = 0.0
        self.Y_MA           = 0.0
        
        self.N      = N
        
        
        ### For dead reckoning correction                        
        self.encoder = motor_encoder(t0)

        self.yaw_offset     = 0.0
        self.psiDot_offset  = 0.0
        self.ax_offset      = 0.0
        self.ay_offset      = 0.0
        self.az_offset      = 0.0
        self.vx_offset      = 0.0
        self.vy_offset      = 0.0

        self.co_yaw     = 0.0
        self.co_psiDot  = 0.0
        self.co_ax      = 0.0
        self.co_ay      = 0.0
        self.co_az      = 0.0
        self.s          = 0.0 
        self.co_vx      = 0.0
        self.co_vy      = 0.0

        # time stamp
        # self.t0     = time.time()
        self.t0     = t0



        # Time for yawDot integration
        # self.curr_time_pose = self.t0
        # self.curr_time_twist = self.t0
        
        self.curr_time_pose = rospy.get_rostime().to_sec() - self.t0
        self.curr_time_twist = rospy.get_rostime().to_sec() - self.t0
        
        self.prev_time_pose = self.curr_time_pose
        self.prev_time_twist = self.curr_time_twist


    def gravity_compensate(self):
        g = [0.0, 0.0, 0.0]
        q = [self.qx , self.qy, self.qz, self.qz]
        acc = [self.ax, self.ay, self.az]
        # get expected direction of gravity
        g[0] = 2 * (q[1] * q[3] - q[0] * q[2])
        g[1] = 2 * (q[0] * q[1] + q[2] * q[3])
        g[2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
        self.ax, self.ay, self.az =  [acc[0] + g[0], acc[1] + g[1], acc[2] + g[2]]

        

        ############ Body frame to world frame transformation ##########
        # http://planning.cs.uiuc.edu/node102.html
        self.R_ypr = np.array([[cos(self.yaw)*cos(self.pitch) , cos(self.yaw)*sin(self.pitch)*sin(self.roll) - \
                                sin(self.yaw)*cos(self.roll) , cos(self.yaw)*sin(self.pitch)*cos(self.roll) + \
                                sin(self.yaw)*sin(self.roll)], \
                                [sin(self.yaw)*cos(self.pitch) , sin(self.yaw)*sin(self.pitch)*sin(self.roll) + \
                                cos(self.yaw)*cos(self.roll) , sin(self.yaw)*sin(self.pitch)*cos(self.roll) - \
                                cos(self.yaw)*sin(self.roll)], \
                                [-sin(self.pitch) , cos(self.pitch)*sin(self.roll) , cos(self.pitch)*cos(self.roll)]])
        
        ### gravity compensation
        # http://www.varesano.net/blog/fabio/simple-gravity-compensation-9-dom-imus
        gravity = np.dot(self.R_ypr.T,np.array([0, 0, self.az]).T)
        self.ax = self.ax + gravity[0]
        self.ay = self.ay + gravity[1]
        self.az = self.az #- gravity[2]




    def Twist_callback(self, data):
        
        self.curr_time_twist = rospy.get_rostime().to_sec() - self.t0
        # self.curr_time_twist = time.time()

        # self.ax     = -1*data.linear.y - self.ax_offset
        # self.ay     = data.linear.x - self.ay_offset
        # self.az     = data.linear.z #- self.az_offset

        self.ax     = data.linear.x*0.001 #- self.ax_offset
        self.ay     = -data.linear.y*0.001 #- self.ay_offset
        self.az     = -data.linear.z*0.001 #- self.az_offset

        self.ax_MA_window.pop(0)
        self.ax_MA_window.append(self.ax)
        self.ax_MA = np.squeeze(np.convolve(self.ax_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.ay_MA_window.pop(0)
        self.ay_MA_window.append(self.ay)
        self.ay_MA = np.squeeze(np.convolve(self.ay_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.az_MA_window.pop(0)
        self.az_MA_window.append(self.az)
        self.az_MA = np.squeeze(np.convolve(self.az_MA_window, np.ones(self.N)/self.N, mode='valid'))


        self.transform = False
        if self.transform == True:
            self.coordinate_transform()


        #### Dead RECOKNING ####        
        if self.encoder.vx == 0.0:
            self.ax = 0.0
            self.ay = 0.0
            self.ax_MA = 0.0
            self.ay_sMA = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self.vx_MA = 0.0
            self.vy_MA = 0.0

        # print (self.curr_time_twist,self.prev_time_twist, self.curr_time_twist-self.prev_time_twist)
        self.vx     = self.vx+self.ax*(self.curr_time_twist-self.prev_time_twist)  # from IMU
        self.vy     = self.vy+self.ay*(self.curr_time_twist-self.prev_time_twist)

        self.X = self.X +  self.vx*(self.curr_time_twist-self.prev_time_twist)
        self.Y = self.Y +  self.vy*(self.curr_time_twist-self.prev_time_twist)
        
        self.vx_MA     = self.vx_MA+self.ax_MA*(self.curr_time_twist-self.prev_time_twist)  # from IMU
        self.vy_MA     = self.vy_MA+self.ay_MA*(self.curr_time_twist-self.prev_time_twist)

        self.X_MA = self.X_MA +  self.vx_MA*(self.curr_time_twist-self.prev_time_twist)
        self.Y_MA = self.Y_MA +  self.vy_MA*(self.curr_time_twist-self.prev_time_twist)


        # self.psiDot = data.angular.z 

        # p is called roll rate, q pitch rate and r yaw rate.
        self.roll_rate    = data.angular.x   
        self.pitch_rate   = data.angular.y   
        self.yaw_rate     = - (data.angular.z  + self.yaw_rate_offset)
        
        self.roll_rate_MA_window.pop(0)
        self.roll_rate_MA_window.append(self.roll_rate)
        self.roll_rate_MA = np.squeeze(np.convolve(self.roll_rate_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.pitch_rate_MA_window.pop(0)
        self.pitch_rate_MA_window.append(self.pitch_rate)
        self.pitch_rate_MA = np.squeeze(np.convolve(self.pitch_rate_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.yaw_rate_MA_window.pop(0)
        self.yaw_rate_MA_window.append(self.yaw_rate)
        self.yaw_rate_MA = np.squeeze(np.convolve(self.yaw_rate_MA_window, np.ones(self.N)/self.N, mode='valid'))



        self.prev_time_twist = self.curr_time_twist

    def Pose_callback(self, data):
        self.curr_time_pose = rospy.get_rostime().to_sec() - self.t0
        # self.curr_time_pose = time.time()


        self.roll   = data.orientation.x 
        self.pitch  = data.orientation.y 
        self.yaw    = data.orientation.z - (self.yaw_offset)

        self.roll_MA_window.pop(0)
        self.roll_MA_window.append(self.roll)
        self.roll_MA  = np.squeeze(np.convolve(self.roll_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.pitch_MA_window.pop(0)
        self.pitch_MA_window.append(self.pitch)
        self.pitch_MA = np.squeeze(np.convolve(self.pitch_MA_window, np.ones(self.N)/self.N, mode='valid'))

        self.yaw_MA_window.pop(0) 
        self.yaw_MA_window.append(self.yaw)
        self.yaw_MA   = np.squeeze(np.convolve(self.yaw_MA_window, np.ones(self.N)/self.N, mode='valid'))


        self.prev_time_pose = self.curr_time_pose
        

    def calibrate_imu(self,delay,offset):


        yaw_rate_info   = []
        ax_info = []
        ay_info = []
        az_info = []
        vx_info = []
        vy_info = []
        yaw_info    = []

        # t1 = rospy.get_rostime().to_sec()
        # while   t1 - self.t0 < delay: ### time for 5sec
        for i in range(50):
            # t1 = rospy.get_rostime().to_sec()
            yaw_rate_info.append(self.yaw_rate)
            ax_info.append(self.ax)
            ay_info.append(self.ay)
            az_info.append(self.az)
            
            vx_info.append(self.vx)
            vy_info.append(self.vy)
            yaw_info.append(self.yaw)
            # print ('time', t1 - self.t0)        

        self.yaw_rate_offset  = np.mean(yaw_rate_info)
        self.ax_offset      = np.mean(ax_info)
        self.ay_offset      = np.mean(ay_info)
        self.az_offset      = np.mean(az_info)
        self.vx_offset      = np.mean(vx_info)
        self.vy_offset      = np.mean(vy_info)
        self.yaw_offset     = np.mean(yaw_info)  - offset

        self.co_yaw     = np.var(yaw_info)
        self.co_psiDot  = np.var(yaw_rate_info)
        self.co_ax      = np.var(ax_info)
        self.co_ay      = np.var(ay_info)
        self.co_vx      = np.var(vx_info)
        self.co_vy      = np.var(vy_info)



    def data_retrive(self, msg):

        msg.timestamp_ms = self.curr_time_pose
        msg.X  = 0
        msg.Y  = 0
        msg.roll  = self.roll
        msg.yaw  = self.yaw
        msg.pitch  = self.pitch
        msg.vx  = self.vx
        msg.vy  = self.vy
        msg.yaw_rate  = self.yaw_rate
        msg.ax  = self.ax
        msg.ay  = self.ay
        msg.s  = self.s
        msg.x  = self.X
        msg.y  = self.Y

        return msg

    def data_retrive_MA(self, msg):

        msg.timestamp_ms = self.curr_time_pose
        msg.X  = 0
        msg.Y  = 0
        msg.roll  = self.roll_MA
        msg.yaw  = self.yaw_MA
        msg.pitch  = self.pitch_MA
        msg.vx  = self.vx_MA
        msg.vy  = self.vy_MA
        msg.yaw_rate  = self.yaw_rate_MA
        msg.ax  = self.ax_MA
        msg.ay  = self.ay_MA
        msg.s  = self.s
        msg.x  = self.X_MA
        msg.y  = self.Y_MA

        return msg

class fiseye_cam():


    def __init__(self,t0,N = 10):


        # rospy.Subscriber('pure_cam_pose', Pose, self.pure_cam_pose_callback, queue_size=1)
        rospy.Subscriber('fused_cam_pose', Pose, self.fused_cam_pose_callback, queue_size=1)

        # self.pure_x   = 0.0

        # self.pure_y   = 0.0
        # self.pure_yaw = 0.0
        

        # self.fused_x           = 0.0
        # self.fused_x_MA_window =[0.0]*N
        # self.fused_x_MA        = 0.0 
        # self.fused_y           = 0.0
        # self.fused_y_MA_window = [0.0]*N
        # self.fused_y_MA        = 0.0

        self.x_m_offset         = 0 
        self.y_m_offset         = 0 

        self.N_v                = 50

        self.vx                 = 0.0
        self.vx_prev            = 0.0
        self.vx_MA_window       = [0.0]*self.N_v
        self.vx_MA              = 0.0
        # self.vx_MA_window    = [0.0]*N
        # self.vx_MA               = 0.0

        self.vy                 = 0.0
        self.vy_prev            = 0.0
        self.vy_MA_window       = [0.0]*self.N_v
        self.vy_MA              = 0.0


        self.ax                 = 0.0
        self.ay                 = 0.0


        self.encoder = motor_encoder(t0)
        # self.vy_MA_window    = [0.0]*N
        # self.vy_MA               = 0.0


        self.X                 = 0.0
        self.X_MA_window       = [0.0]*N
        self.X_MA              = 0.0
        self.X_MA_past         = 0.0

        self.Y                 = 0.0
        self.Y_MA_window       = [0.0]*N
        self.Y_MA              = 0.0
        self.Y_MA_past         = 0.0

        self.s = 0.0
        
        self.yaw = 0.0
        self.yaw_MA = 0.0
        self.yaw_MA_window = [0.0]*N

        # time stamp
        self.t0     = t0
        self.N      = N
        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

        #         #history
        # self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[],"ax":[],"ay":[],"az":[]}
        # self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}
        # self.wheel_rpm_hist = {"timestamp_ms":[],"wheel_rpm":[]}

    def pure_cam_pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.pure_x   = data.position.x
        self.pure_y   = data.position.y
        self.pure_yaw = data.orientation.z


    def fused_cam_pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.X   = data.position.x - self.x_m_offset
        self.Y   = data.position.y - self.y_m_offset


        self.X_MA_window.pop(0)
        self.X_MA_window.append(self.X)
        self.X_MA = np.squeeze(np.convolve(self.X_MA_window, np.ones(self.N)/self.N, mode='valid'))
        

        self.Y_MA_window.pop(0)
        self.Y_MA_window.append(self.Y)
        self.Y_MA = np.squeeze(np.convolve(self.Y_MA_window, np.ones(self.N)/self.N, mode='valid'))
        
        self.yaw = data.orientation.z

        self.yaw_MA_window.pop(0)
        self.yaw_MA_window.append(self.yaw)
        self.yaw_MA = np.squeeze(np.convolve(self.yaw_MA_window, np.ones(self.N)/self.N, mode='valid'))


        #### Position and velocity calculation ####

        ### Velocity and orientation in inertial frame (dx/dt)
        Gvx = (self.X_MA_window[-1]- self.X_MA_window[-2])/(self.curr_time -self.prev_time)

        Gvy = (self.Y_MA_window[-1]- self.Y_MA_window[-2])/(self.curr_time -self.prev_time)

        Gvx_MA = (self.X_MA - self.X_MA_past)/(self.curr_time -self.prev_time)

        Gvy_MA = (self.Y_MA - self.Y_MA_past)/(self.curr_time -self.prev_time)

        dist =  LA.norm(np.array([self.X_MA_window[-1],self.Y_MA_window[-1]])-np.array([self.X_MA_window[-2], self.Y_MA_window[-2]]))

        if self.encoder.vx == 0.0:
            Gvx = 0.0
            Gvy = 0.0        
            Gvx_MA = 0.0
            Gvy_MA = 0.0        
            
            dist = 0.0


        ### velocity in vehicle body frame
        # Vx = (dist - self.s)/(self.curr_time -self.prev_time)
        # Vy = (Gvx*cos(self.yaw)+Gvy*sin(self.yaw))

        Vx_MA = (Gvx_MA*cos(self.yaw_MA)+Gvy_MA*sin(self.yaw_MA))
        Vy_MA = (-Gvx_MA*sin(self.yaw_MA)+Gvy_MA*cos(self.yaw_MA))

        Vx = (Gvx*cos(self.yaw)+Gvy*sin(self.yaw))
        Vy = (-Gvx*sin(self.yaw)+Gvy*cos(self.yaw))


        self.s += dist

        
        ### Filter large values which is not possible ###

        # if Vx > 0.1:
        #     Vx = self.vx
        #     Vy = self.vy

        self.vx = Vx
        self.vy = Vy

        self.vx_MA_window.pop(0)
        self.vx_MA_window.append(Vx_MA)
        self.vx_MA = np.squeeze(np.convolve(self.vx_MA_window, np.ones(self.N_v)/self.N_v, mode='valid'))

        self.vy_MA_window.pop(0)
        self.vy_MA_window.append(Vy_MA)
        self.vy_MA = np.squeeze(np.convolve(self.vy_MA_window, np.ones(self.N_v)/self.N_v, mode='valid'))

        ### acceleration in vehicle body frame
        self.ax = (self.vx-self.vx_prev)/(self.curr_time -self.prev_time)
        self.ay = (self.vy-self.vy_prev)/(self.curr_time -self.prev_time)

        self.vy_prev = self.vy

        # print (np.array([self.X_MA_window[-1],self.Y_MA_window[-1]])-np.array([self.X_MA_window[-2], self.Y_MA_window[-2]]))


        self.prev_time = self.curr_time
        self.X_MA_past = self.X_MA
        self.Y_MA_past = self.Y_MA

    def calibrate_fcam(self,delay,offset):

        x_m_info = []
        y_m_info = []
        # yaw_rate_info = []
        # ax_info = []
        # ay_info = []
        # az_info = []
        # vx_info = []
        # vy_info = []
        # yaw_info = []

        # t1 = rospy.get_rostime().to_sec()
        # while   t1 - self.t0 < delay: ### time for 5sec

        for i in range(100):
            # t1 = rospy.get_rostime().to_sec()
            # yaw_rate_info.append(self.yaw_rate)
            # ax_info.append(self.ax)
            # ay_info.append(self.ay)
            # az_info.append(self.az)
            
            # vx_info.append(self.vx)
            # vy_info.append(self.vy)
            # yaw_info.append(self.yaw)
            x_m_info.append(self.X)
            y_m_info.append(self.Y)
        

        self.x_m_offset = np.mean(x_m_info)
        self.y_m_offset = np.mean(y_m_info)
        print ("self.x_m_offset",self.x_m_offset, "self.y_m_offset",self.y_m_offset)
        # self.yaw_rate_offset  = np.mean(yaw_rate_info)
        # self.ax_offset      = np.mean(ax_info)
        # self.ay_offset      = np.mean(ay_info)
        # self.az_offset      = np.mean(az_info)
        # self.vx_offset      = np.mean(vx_info)
        # self.vy_offset      = np.mean(vy_info)
        # self.yaw_offset     = np.mean(yaw_info)  - offset

        # self.co_yaw     = np.var(yaw_info)
        # self.co_psiDot  = np.var(yaw_rate_info)
        # self.co_ax      = np.var(ax_info)
        # self.co_ay      = np.var(ay_info)
        # self.co_vx      = np.var(vx_info)
        # self.co_vy      = np.var(vy_info)

    def data_retrive(self, msg):

        msg.timestamp_ms = self.curr_time
        msg.X  = self.X
        msg.Y  = self.Y
        msg.roll  = 0
        msg.yaw  = self.yaw
        msg.pitch  = 0
        msg.vx  = self.vx
        msg.vy  = self.vy
        msg.yaw_rate  = 0.0
        msg.ax  = self.ax
        msg.ay  = self.ay
        msg.s  = self.s
        msg.x  = 0.0
        msg.y  = 0.0

        return msg

    def data_retrive_MA(self, msg):
        msg.timestamp_ms = self.curr_time
        msg.X  = self.X_MA
        msg.Y  = self.Y_MA
        msg.roll  = 0
        msg.yaw  = self.yaw_MA
        msg.pitch  = 0
        msg.vx  = self.vx_MA
        msg.vy  = self.vy_MA
        msg.yaw_rate  = 0.0
        msg.ax  = self.ax
        msg.ay  = self.ay
        msg.s  = self.s
        msg.x  = 0.0
        msg.y  = 0.0

        return msg


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
    line_0,        = axtr.plot(xdata, ydata, '-r')  # Plots the traveled positions
    line_2,        = axtr.plot(xdata, ydata, '-g')  # Plots the traveled positions
    line_3,        = axtr.plot(xdata, ydata, '-b')  # Plots the traveled positions
    line_4,        = axtr.plot(xdata, ydata, '-y')  # Plots the traveled positions
    line_fusion,        = axtr.plot(xdata, ydata, '-m')  # Plots the traveled positions
    


    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])

    marker_0 = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10,label='ID0')
    axtr.add_patch(marker_0)
    # # Vehicle:
    marker_2 = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10,label='ID2')
    axtr.add_patch(marker_2)

    marker_3 = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10,label='ID3')
    # axtr.add_patch(marker_3)
    # # Vehicle:
    marker_4 = patches.Polygon(v, alpha=0.7, closed=True, fc='y', ec='k', zorder=10,label='ID4')
    # axtr.add_patch(marker_4)


    fusion = patches.Polygon(v, alpha=0.7, closed=True, fc='m', ec='k', zorder=10,label='fusion')
    # axtr.add_patch(fusion)
    

    plt.legend()
    # # Planner vehicle:
    rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)
    # axtr.add_patch(rec_planning)



    plt.show()

    return plt, fig, axtr, line_planning, point_simc, point_rlc, line_SS, line_sim, line_rl, line_0, line_2, line_3, line_4, line_fusion,\
     marker_0, marker_2, marker_3, marker_4, fusion ,rec_planning


def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y

def append_sensor_data(data,msg):
    data['timestamp_ms'].append(msg.timestamp_ms)
    data['X'].append(msg.X)
    data['Y'].append(msg.Y)
    data['roll'].append(msg.roll)
    data['yaw'].append(msg.yaw)
    data['pitch'].append(msg.pitch)
    data['vx'].append(msg.vx)
    data['vy'].append(msg.vy)
    data['yaw_rate'].append(msg.yaw_rate)
    data['ax'].append(msg.ax)
    data['ay'].append(msg.ay)
    data['s'].append(msg.s)
    data['x'].append(msg.x)
    data['y'].append(msg.y)

def append_control_data(data,msg):
   
    data['timestamp_ms_dutycycle'].append(msg.timestamp_ms_DC)
    data['timestamp_ms_steer'].append(msg.timestamp_ms_steer)
    data['steering'].append(msg.steer)
    data['duty_cycle'].append(msg.duty_cycle)



def Continuous_AB_Comp(self, vx, vy, omega, theta, delta):

    # m = rospy.get_param("m")
    # rho = rospy.get_param("rho")
    # lr = rospy.get_param("lr")
    # lf = rospy.get_param("lf")
    # Cm0 = rospy.get_param("Cm0")
    # Cm1 = rospy.get_param("Cm1")
    # C0 = rospy.get_param("C0")
    # C1 = rospy.get_param("C1")
    # Cd_A = rospy.get_param("Cd_A")
    # Caf = rospy.get_param("Caf")
    # Car = rospy.get_param("Car")
    # Iz = rospy.get_param("Iz")

    m = 2.424;
    rho = 1.225;
    lr = 0.1203;
    lf = 0.1377;
    Cm0 = 10.1305;
    Cm1 = 1.05294;
    C0 = 3.68918;
    C1 = 0.0306803;
    Cd_A = -0.657645;
    Caf = 1.3958;
    Car = 1.6775;
    Iz = 0.02;


    F_flat = 0;
    Fry = 0;
    Frx = 0;
    
    A31 = 0;
    A11 = 0;
    
    eps = 0.00000001 ## avoiding reaching to infinity
    eps = 0
    if abs(vx)> 0:
        F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)));
    
    
    if abs(vx)> 0:
    
        Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) ;
        A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2);
        A31 = -Fry*lr/((vx+eps)*Iz);
        
    A12 = omega;
    A21 = -omega;
    A22 = 0;
    
    if abs(vy) > 0.0:
        A22 = Fry/(m*(vy+eps));

    A41 = cos(theta);
    A42 = -sin(theta);
    A51 = sin(theta);
    A52 = cos(theta);


    B12 = 0;
    B32 = 0;
    B22 = 0;
    
    if abs(delta) > 0:
        B12 = -F_flat*sin(delta)/(m*(delta+eps));
        B22 = F_flat*cos(delta)/(m*(delta+eps));    
        B32 = F_flat*cos(delta)*lf/(Iz*(delta+eps));



    B11 = (1/m)*(Cm0 - Cm1*vx);
    
    A_obs = np.array([[A11, A12, 0,  0,   0,  0],\
                  [A21, A22, 0,  0,   0,  0],\
                  [A31,  0 , 0,  0,   0,  0],\
                  [A41, A42, 0,  0,   0,  0],\
                  [A51, A52, 0,  0,   0,  0],\
                  [ 0 ,  0 , 1,  0,   0,  0]])
    
    B_obs = np.array([[B11, B12],\
                  [ 0,  B22],\
                  [ 0,  B32],\
                  [ 0 ,  0 ],\
                  [ 0 ,  0 ],\
                  [ 0 ,  0 ]])

    # print ('self.A_obs',self.A_obs,'self.B_obs',self.B_obs)
    # print ("observer ::")
    
    return A_obs, B_obs

def L_Computation(vx,vy,w,theta,delta,LQR_gain,sched_var,seq):    

    
    sched_vx    = sched_var[0]
    sched_vy    = sched_var[1]
    sched_w     = sched_var[2]
    sched_theta = sched_var[3]
    sched_delta = sched_var[4]
    
    M_vx_min      = (sched_vx[1] - vx) / (sched_vx[1] - sched_vx[0] );
    M_vy_min      = (sched_vy[1] - vy) / (sched_vy[1] - sched_vy[0] );
    M_w_min       = (sched_w[1] - w) / (sched_w[1] - sched_w[0]); 
    M_theta_min   = (sched_theta[1] - theta) / (sched_theta[1] - sched_theta[0]); 
    M_delta_min   = (sched_delta[1] - delta) / (sched_delta[1] - sched_delta[0]); 

    M_vx_max      = (1 - M_vx_min);      
    M_vy_max      = (1 - M_vy_min);      
    M_w_max       = (1 - M_w_min);       
    M_theta_max   = (1 - M_theta_min);   
    M_delta_max   = (1 - M_delta_min);   

    M_vx          = [M_vx_min, M_vx_max];   
    M_vy          = [M_vy_min, M_vy_max];   
    M_w           = [M_w_min, M_w_max];     
    M_theta       = [M_theta_min, M_theta_max];     
    M_delta       = [M_delta_min, M_delta_max];     


    if vx > sched_vx[1] or vx < sched_vx[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Vx is out of the polytope ...' )
    elif vy > sched_vy[1] or vy < sched_vy[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Vy is out of the polytope ...' )
    elif delta > sched_delta[1] or delta < sched_delta[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Steering is out of the polytope ... = ',delta)
#         elif theta > sched_theta[1] or theta < sched_theta[0]:
#             print( '[ESTIMATOR/L_Gain_Comp]: Theta is out of the polytope ...', theta )


    mu = np.zeros((seq.shape[0],1))
    L_gain  = np.zeros((LQR_gain[:,:,1].shape[0], 5))

    for i in range(seq.shape[0]):
        mu[i] = M_vx[seq[i,0]] * M_vy[seq[i,1]] * M_w[seq[i,2]] * M_theta[seq[i,3]] * M_delta[seq[i,4]];
        L_gain  = L_gain  + mu[i]*LQR_gain[:,:,i];

    return L_gain



def data_retrive(msg, est_msg):

    msg.timestamp_ms = 0
    msg.X  = est_msg[3]
    msg.Y  = est_msg[4]
    msg.roll  = 0
    msg.yaw  = est_msg[5]
    msg.pitch  = 0
    msg.vx  = est_msg[0]
    msg.vy  = est_msg[1]
    msg.yaw_rate  = est_msg[2]
    msg.ax  = 0
    msg.ay  = 0
    msg.s  = 0
    msg.x  = 0
    msg.y  = 0

    return msg


def load_LQRgain():

    gain_path = '/home/auto/Desktop/autonomus_vehicle_project/project/development/proto/estimator/observer_gain_saved/LQR_17_04_2021'

    LQR_gain = np.array(sio.loadmat(gain_path+'/LQR_gain.mat')['data']['Lmi'].item())

    seq = sio.loadmat(gain_path+'/LQR_gain.mat')['data']['sequence'].item()
    ##sched_vx, sched_vy, sched_w, sched_theta, sched_delta
    seq = seq - 1 ##matlab index to python index

    sched_var = sio.loadmat(gain_path+'/LQR_gain.mat',matlab_compatible = 'True')['data']['sched_var'].item()
    sched_vx    = sched_var[0,:]
    sched_vy    = sched_var[1,:]
    sched_w     = sched_var[2,:]
    sched_theta = sched_var[3,:]
    sched_delta = sched_var[4,:]

    return LQR_gain, seq, sched_var



def main():
    rospy.init_node('state_estimation', anonymous=True)

    loop_rate       = 30
    rate            = rospy.Rate(loop_rate)
    time0 = rospy.get_rostime().to_sec()
    # print ("time0",time.time())

    counter = 0

    record_data = False

    N = 10
    enc    = motor_encoder(time0, 10)
    fcam   = fiseye_cam(time0, 5)
    imu    = IMU(time0, 100)

    marvel = marvelmind(time0, 300)
    control_input = vehicle_control(time0)

    delay = 5
    offset = pi/2
    print ("<<<< Initializing IMU orientation >>>>")
    imu.calibrate_imu(delay,offset)    
    fcam.calibrate_fcam(delay,offset)
    marvel.calibrate_marvel(delay,offset)
    print ("<<<< ORIGIN SET AND CALIBRATION DONE >>>>")


    ###### Estimator Initialize ######


    C       =  np.array([[1, 0, 0, 0, 0, 0], 
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]) 

    lr = 0.1203;
    lf = 0.1377;
    beta = lr*tan(deltap)/(lr+lf);
    vy = enc.vx*beta;

    est_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, imu.yaw ]).T
    est_state_hist = [] 
    est_state_hist.append(est_state)
    
    est_state_msg = sensorReading()
    ### Assign the path inside this function
    LQR_gain, seq, sched_var = load_LQRgain()
    est_state_pub  = rospy.Publisher('est_state_info', sensorReading, queue_size=1)



    control_data = control()
    control_hist = {'timestamp_ms_dutycycle':[],'timestamp_ms_steer':[],'steering':[], 'duty_cycle':[]}

    marvel_pub  = rospy.Publisher('marvel_fused', sensorReading, queue_size=1)
    marvel_data = sensorReading()
    marvel_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}

    enc_pub  = rospy.Publisher('encoder_fused', sensorReading, queue_size=1)
    enc_data = sensorReading()
    enc_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    imu_pub  = rospy.Publisher('imu_fused', sensorReading, queue_size=1)
    imu_data = sensorReading()
    imu_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    fcam_pub  = rospy.Publisher('fcam_fused', sensorReading, queue_size=1)
    fcam_data = sensorReading()
    fcam_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        

    marvel_MA_pub  = rospy.Publisher('marvel_MA_fused', sensorReading, queue_size=1)
    marvel_MA_data = sensorReading()
    marvel_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    enc_MA_pub  = rospy.Publisher('encoder_MA_fused', sensorReading, queue_size=1)
    enc_MA_data = sensorReading()
    enc_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    imu_MA_pub  = rospy.Publisher('imu_MA_fused', sensorReading, queue_size=1)
    imu_MA_data = sensorReading()
    imu_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    fcam_MA_pub  = rospy.Publisher('fcam_MA_fused', sensorReading, queue_size=1)
    fcam_MA_data = sensorReading()
    fcam_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}

    
    
    curr_time = rospy.get_rostime().to_sec() -t0
     
    prev_time = curr_time 
    

    while not (rospy.is_shutdown()):
        
        curr_time = rospy.get_rostime().to_sec() -t0
    
        if control_input.dutycyle > 0.1:
            dt = curr_time - prev_time 
            u = np.array([control_input.dutycyle, control_input.steer]).T
            A_obs, B_obs = Continuous_AB_Comp(est_state[0], est_state[0], est_state[0], est_state[0], u[1])
            L_gain = L_Computation(est_state[0], est_state[0], est_state[0], est_state[0], u[1], LQR_gain, sched_var, seq)
            y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, imu.yaw ]).T 
            
            est_state  = est_state + ( dt * np.dot( ( A_obs - np.dot(L_gain, C) ), est_state )
                            +    dt * np.dot(B_obs, u)
                            +    dt * np.dot(L_gain, y_meas) )

        
        est_state_pub.publish(data_retrive(est_state_msg, est_msg))
        est_state_hist.append(est_state)


        control_msg = control_input.data_retrive(control_data)        
        append_control_data(control_hist, control_msg)

        marvel_msg = marvel.data_retrive(marvel_data)
        marvel_pub.publish(marvel_msg)
        append_sensor_data(marvel_hist, marvel_msg)

        marvel_MA_msg = marvel.data_retrive_MA(marvel_MA_data)
        marvel_MA_pub.publish(marvel_MA_msg)
        append_sensor_data(marvel_MA_hist, marvel_MA_msg)


        enc_msg = enc.data_retrive(enc_data)
        enc_pub.publish(enc_msg)
        append_sensor_data(enc_hist, enc_msg)


        enc_MA_msg = enc.data_retrive_MA(enc_MA_data)
        enc_MA_pub.publish(enc_MA_msg)
        append_sensor_data(enc_MA_hist, enc_MA_msg)


        imu_msg = imu.data_retrive(imu_data)
        imu_pub.publish(imu_msg)
        append_sensor_data(imu_hist, imu_msg)


        imu_MA_msg = imu.data_retrive_MA(imu_MA_data)
        imu_MA_pub.publish(imu_MA_msg)
        append_sensor_data(imu_MA_hist, imu_MA_msg)


        fcam_msg = fcam.data_retrive(fcam_data)
        fcam_pub.publish(fcam_msg)
        append_sensor_data(fcam_hist, fcam_msg)


        fcam_MA_msg = fcam.data_retrive_MA(fcam_MA_data)
        fcam_MA_pub.publish(fcam_MA_msg)
        append_sensor_data(fcam_MA_hist, fcam_MA_msg)

        prev_time = curr_time 
        rate.sleep()


    if record_data == True:
        path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
            
        now = datetime.datetime.now()
        # path = path + now.strftime("d%d_m%m_y%Y/")
        path = path + now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")

        if not os.path.exists(path):
            os.makedirs(path)


        control_path = path + '/control_his_resistive_estimation'
        
        marvel_path = path + '/marvel_his_resistive_estimation'
        marvel_MA_path = path + '/marvel_MA_his_resistive_estimation'

        imu_path = path + '/imu_his_resistive_estimation'
        imu_MA_path = path + '/imu_MA_his_resistive_estimation'
        
        enc_path = path + '/enc_his_resistive_estimation'
        enc_MA_path = path + '/enc_MA_his_resistive_estimation'
        
        fcam_path = path + '/fcam_his_resistive_estimation'
        fcam_MA_path = path + '/fcam_MA_his_resistive_estimation'

        np.save(control_path,control_hist)

        np.save(marvel_path,marvel_hist)
        np.save(marvel_MA_path,marvel_MA_hist)
        
        np.save(imu_path,imu_hist)
        np.save(imu_MA_path,imu_MA_hist)

        np.save(enc_path,enc_hist)
        np.save(enc_MA_path,enc_MA_hist)
        
        np.save(fcam_path,fcam_hist)
        np.save(fcam_MA_path,fcam_MA_hist)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


'''
sensorReading message >>>>>>>>>


int64 timestamp_ms
float64 X
float64 Y
float64 roll
float64 yaw
float64 pitch
float64 vx
float64 vy
float64 yaw_rate
float64 ax
float64 ay
float64 s
float64 x
float64 y

'''