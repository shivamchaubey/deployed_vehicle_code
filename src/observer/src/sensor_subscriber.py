#!/usr/bin/env python

from math import tan, atan, cos, sin, pi, atan2, fmod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rospy
from numpy.random import randn,rand
import rosbag
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
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/observer/src/')
from observer_functions import wrap


######## vehicle control subscriber ##########
class vehicle_control(object):
    """ Object collecting controller msg command data
    Attributes:
        Input command:
            1.dutycycle 2.steering 
        Time stamp
            1.t0 2.curr_time
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

################################## motor encoder subscriber ###############################
class motor_encoder():
    """ Object collecting motor encoder msg data
    Attributes:
        Input command:
            1.wheel rpm  
        Output:
            1. vx: longitudinal velocity, optional 

            Optional:  
            1. s: distance traveled by wheel (turn on : state_estimation_on = True) 
            2. wheel_rpm_MA: wheel rpm moving average , vx_MA: velocity moving average ,
               s_MA: traveled distance moving average (turn on: moving_average_on = True)
    """


    def __init__(self,t0,N = 10):

        rospy.Subscriber('/wheel_rpm_feedback', Float32, self.RPM_callback, queue_size=1)       

        self.curr_time    = 0.0
        self.wheel_rpm    = 0.0
        self.vx           = 0.0
        self.s            = 0.0

        self.moving_average_on = False ## turn on if moving average is needed
        self.state_estimation_on = False
        self.wheel_rpm_MA_window = [0]*N #moving average
        self.wheel_rpm_MA = 0.0
        self.vx_MA        = 0.0
        self.s_MA         = 0.0
        self.wheel_radius     = 0.03*1.12178 #radius of wheel

        # time stamp
        self.t0     = t0
        self.N      = N

        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

    def RPM_callback(self, data):

        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.wheel_rpm = data.data
        self.vx        = (self.wheel_rpm*2*pi/60.0)*self.wheel_radius
            
        if self.state_estimation_on == False:
            self.s        += abs(self.vx)*(self.curr_time - self.prev_time)  

        if self.moving_average_on == True:
            self.wheel_rpm_MA_window.pop(0)
            self.wheel_rpm_MA_window.append(self.wheel_rpm)
    
            if self.state_estimation_on == False:
                ### Moving average
                self.wheel_rpm_MA    =  np.squeeze(np.convolve(self.wheel_rpm_MA_window, np.ones(self.N)/self.N, mode='valid'))

                self.vx_MA     = (self.wheel_rpm_MA*2*pi/60.0)*self.wheel_radius
                self.s_MA     += self.vx_MA*(self.curr_time - self.prev_time)  
            

        
        self.prev_time = self.curr_time

    ##### function to retrieve the data for publishing #######
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

    ##### function to retrieve the MA data for publishing #######
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


######################################## IMU subscriber ##########################################
class IMU():
    """ Object collecting IMU msg data
    Attributes:
        Input command:
            1. Pose: accelerations, attitude (roll, yaw, pitch)
            2. Twist:  angular velocity in x,y and z axis. 
            Note: Only yaw and angular velocity i z-axis is needed for the vehicle assuming small 
            deflection of spring.
        Output:
            1. Accelerations (x,y,z)
            2. Attitude (roll, yaw, pitch)
            3. angular rate (x,y,z)
            
            Optional:  
            1. state_estimation_on = True : Calculates hidden state 
                1. vx, vy (longitudinal and lateral velocities )
                2. X, Y  (Position in X-Y)
            2. moving_average_on = True : calculate moving average of all the 
                output including optional if turned 'True'
    """
    def __init__(self,t0, N =50):

        rospy.Subscriber('twist', Twist, self.Twist_callback, queue_size=1)

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)


        self.state_estimation_on = False
        self.moving_average_on = False

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
        self.yaw_rate_offset        = 0.0
        
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
        
        
        ''' Dead reckoning is being utilized to estimate the velocity and
            position of the vehicle from IMU sensor. To minimize the integral error or
            drift encoder msg is subscribed to know if the integration is required
            when vehicle moves otherwise no integration when vehicle does not move.
        ''' 
        self.encoder = motor_encoder(t0)

        ##### offset if calibration is needed #####
        self.yaw_offset     = 0.0
        self.psiDot_offset  = 0.0
        self.ax_offset      = 0.0
        self.ay_offset      = 0.0
        self.az_offset      = 0.0
        self.vx_offset      = 0.0
        self.vy_offset      = 0.0

        self.t0     = t0
        self.curr_time_pose = rospy.get_rostime().to_sec() - self.t0
        self.curr_time_twist = rospy.get_rostime().to_sec() - self.t0
        
        self.prev_time_pose = self.curr_time_pose
        self.prev_time_twist = self.curr_time_twist

    def Twist_callback(self, data):
        
        self.curr_time_twist = rospy.get_rostime().to_sec() - self.t0
        # self.curr_time_twist = time.time()

        self.ax     = data.linear.x*0.001 #- self.ax_offset
        self.ay     = -data.linear.y*0.001 #- self.ay_offset
        self.az     = -data.linear.z*0.001 #- self.az_offset

        if self.moving_average_on == True:
            self.ax_MA_window.pop(0)
            self.ax_MA_window.append(self.ax)
            self.ax_MA = np.squeeze(np.convolve(self.ax_MA_window, np.ones(self.N)/self.N, mode='valid'))

            self.ay_MA_window.pop(0)
            self.ay_MA_window.append(self.ay)
            self.ay_MA = np.squeeze(np.convolve(self.ay_MA_window, np.ones(self.N)/self.N, mode='valid'))

            self.az_MA_window.pop(0)
            self.az_MA_window.append(self.az)
            self.az_MA = np.squeeze(np.convolve(self.az_MA_window, np.ones(self.N)/self.N, mode='valid'))

        if self.state_estimation_on == True:
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

            self.vx     = self.vx+self.ax*(self.curr_time_twist-self.prev_time_twist)  # from IMU
            self.vy     = self.vy+self.ay*(self.curr_time_twist-self.prev_time_twist)

            self.X = self.X +  self.vx*(self.curr_time_twist-self.prev_time_twist)
            self.Y = self.Y +  self.vy*(self.curr_time_twist-self.prev_time_twist)
            
            self.vx_MA     = self.vx_MA+self.ax_MA*(self.curr_time_twist-self.prev_time_twist)  # from IMU
            self.vy_MA     = self.vy_MA+self.ay_MA*(self.curr_time_twist-self.prev_time_twist)

            self.X_MA = self.X_MA +  self.vx_MA*(self.curr_time_twist-self.prev_time_twist)
            self.Y_MA = self.Y_MA +  self.vy_MA*(self.curr_time_twist-self.prev_time_twist)


        self.roll_rate    = data.angular.x   
        self.pitch_rate   = data.angular.y   
        self.yaw_rate     = -(data.angular.z  + self.yaw_rate_offset)

        if self.moving_average_on == True:        

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
        self.roll   = data.orientation.x 
        self.pitch  = data.orientation.y 
        self.yaw    = wrap(data.orientation.z - (self.yaw_offset))

        if self.moving_average_on == True:
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
        

    ######### Function to calibrate the IMU from biases #############
    def calibrate_imu(self,delay,offset):


        yaw_rate_info   = []
        ax_info = []
        ay_info = []
        az_info = []
        vx_info = []
        vy_info = []
        yaw_info    = []

        for i in range(50):
            yaw_rate_info.append(self.yaw_rate)
            ax_info.append(self.ax)
            ay_info.append(self.ay)
            az_info.append(self.az)
            
            vx_info.append(self.vx)
            vy_info.append(self.vy)
            yaw_info.append(self.yaw)

        self.yaw_rate_offset  = np.mean(yaw_rate_info)
        self.ax_offset      = np.mean(ax_info)
        self.ay_offset      = np.mean(ay_info)
        self.az_offset      = np.mean(az_info)
        self.vx_offset      = np.mean(vx_info)
        self.vy_offset      = np.mean(vy_info)
        self.yaw_offset     = np.mean(yaw_info)  - offset


    ####### function for easy publishing the subscribed and manipulated states #####
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

    ####### function for easy publishing the subscribed and manipulated MA states #####
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

########################### X-Y position subscriber from camera #################################
class fiseye_cam():
    """ Object collecting camera msg data
    Attributes:
        Input command:
            1. Pure Pose: X and Y position (Obtained from only visual camera)
            2. Fused Pose: X and Y position (Obtained from only visual-inertial camera)
        Output:
            1. Pose X and Y
            
            Optional:
            1. pure_cam_on = True: If information from pure cam is needed for debugging  
            2. state_estimation_on = True : Calculates hidden state 
                1. vx, vy (longitudinal and lateral velocities)
                2. ax, ay  (acceleration in vehicle body frame)
            3. moving_average_on = True : calculate moving average of all the 
                output including optional if turned 'True'
    """

    def __init__(self,t0,N = 10):


        self.pure_cam_on = True

        if self.pure_cam_on == True:
            rospy.Subscriber('pure_cam_pose', Pose, self.pure_cam_pose_callback, queue_size=1)
    
        rospy.Subscriber('fused_cam_pose', Pose, self.fused_cam_pose_callback, queue_size=1)

        self.state_estimation_on = True
        self.moving_average_on = True

        #### Homogeneous transformation for reference change####
        self.x_tf     = rospy.get_param("switching_lqr_observer/x_tf")
        self.y_tf     = rospy.get_param("switching_lqr_observer/y_tf")
        theta_tf = rospy.get_param("switching_lqr_observer/theta_tf")*pi/180
        self.R_tf = np.array([[cos(theta_tf), -sin(theta_tf)],
                         [sin(theta_tf),  cos(theta_tf)]])
        self.yaw_tf   = rospy.get_param("switching_lqr_observer/yaw_tf")*pi/180


        self.pure_x   = 0.0
        self.pure_y   = 0.0
        self.pure_yaw = 0.0        

        self.fused_x           = 0.0
        self.fused_y           = 0.0
        self.fused_yaw         = 0.0
        
        self.N_v                = 50 #moving average window

        self.vx                 = 0.0
        self.vx_prev            = 0.0
        self.vx_MA_window       = [0.0]*self.N_v
        self.vx_MA              = 0.0

        self.vy                 = 0.0
        self.vy_prev            = 0.0
        self.vy_MA_window       = [0.0]*self.N_v
        self.vy_MA              = 0.0


        self.ax                 = 0.0
        self.ay                 = 0.0

        '''
        states such as velocity and acceleration is obtained using differential. Due to noise in the sensor
        it is hard to know if the vehicle is static or moving to solve this problem information from motor
        encoder is exploited.
        '''
        self.encoder = motor_encoder(t0)
        
        self.x_m_offset         = 0 
        self.y_m_offset         = 0 

        self.X                 = 0.0
        self.X_MA_window       = [0.0]*N
        self.X_MA              = 0.0
        self.X_MA_past         = 0.0

        '''
        the yaw is information is published via pose_estimate.py by subscribing to IMU msg.
        The IMU msg might be transformed as needed in that file 
        '''

        self.yaw               = 0.0 
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

    def pure_cam_pose_callback(self, data):

        self.pure_x   = data.position.x
        self.pure_y   = data.position.y
        self.pure_yaw = data.orientation.z


    def fused_cam_pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.fused_x    = data.position.x
        self.fused_y    = data.position.y
        self.fused_yaw  = data.orientation.z
        
        [self.X, self.Y] = np.dot(self.R_tf, np.array([self.fused_x, self.fused_y]).T)
        self.X = self.X - self.x_tf
        self.Y = self.Y - self.y_tf

        self.yaw = wrap(self.fused_yaw + self.yaw_tf)

        if self.moving_average_on == True:
            self.X_MA_window.pop(0)
            self.X_MA_window.append(self.X)
            self.X_MA = np.squeeze(np.convolve(self.X_MA_window, np.ones(self.N)/self.N, mode='valid'))
            

            self.Y_MA_window.pop(0)
            self.Y_MA_window.append(self.Y)
            self.Y_MA = np.squeeze(np.convolve(self.Y_MA_window, np.ones(self.N)/self.N, mode='valid'))

            self.yaw_MA_window.pop(0)
            self.yaw_MA_window.append(self.yaw)
            self.yaw_MA = np.squeeze(np.convolve(self.yaw_MA_window, np.ones(self.N)/self.N, mode='valid'))


            '''
                velocity and acceleration calculation in inertial frame (world frame)
            '''

            if self.state_estimation_on == True:
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

                self.X_MA_past = self.X_MA
                self.Y_MA_past = self.Y_MA

        self.prev_time = self.curr_time

    ######## function to calculate bias and remove it if needed ########
    def calibrate_fcam(self,delay,offset):

        x_m_info = []
        y_m_info = []

        for i in range(100):
            x_m_info.append(self.X)
            y_m_info.append(self.Y)

        self.x_m_offset = np.mean(x_m_info)
        self.y_m_offset = np.mean(y_m_info)

    ######### function to handle msg for easy publishing #########
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

    ######### function to handle MA msg for easy publishing #########
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
