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
    

################## LQR GAIN PATH ####################
gain_path = rospy.get_param("switching_lqr_observer/lqr_gain_path")
gain_path = ('/').join(sys.path[0].split('/')[:-1]) + gain_path
print "\n LQR gain path ={} \n".format(gain_path)

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

        print  "yaw_offset", self.yaw_offset


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
        self.yaw_rate     = -(data.angular.z  + self.yaw_rate_offset)
        
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

        # print "data.orientation.z", data.orientation.z, "yaw_offset", self.yaw_offset

        self.yaw    = wrap(data.orientation.z - (self.yaw_offset))

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


        #### Homogeneous transformation for reference change####
        self.x_tf     = rospy.get_param("switching_lqr_observer/x_tf")
        self.y_tf     = rospy.get_param("switching_lqr_observer/y_tf")
        theta_tf = rospy.get_param("switching_lqr_observer/theta_tf")*pi/180
        self.R_tf = np.array([[cos(theta_tf), -sin(theta_tf)],
                         [sin(theta_tf),  cos(theta_tf)]])
        self.yaw_tf   = rospy.get_param("switching_lqr_observer/yaw_tf")*pi/180


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

        self.yaw = 0.0
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

        # self.X   = data.position.x - self.x_m_offset
        # self.Y   = data.position.y - self.y_m_offset

        [self.X, self.Y] = np.dot(self.R_tf, np.array([data.position.x,data.position.y]).T)
        self.X = self.X - self.x_tf
        self.Y = self.Y - self.y_tf
        # print "data.orientation.z",data.orientation.z,"self.yaw_tf", self.yaw_tf
        self.yaw = wrap(data.orientation.z + self.yaw_tf)

        self.X_MA_window.pop(0)
        self.X_MA_window.append(self.X)
        self.X_MA = np.squeeze(np.convolve(self.X_MA_window, np.ones(self.N)/self.N, mode='valid'))
        

        self.Y_MA_window.pop(0)
        self.Y_MA_window.append(self.Y)
        self.Y_MA = np.squeeze(np.convolve(self.Y_MA_window, np.ones(self.N)/self.N, mode='valid'))

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
        # print ("self.x_m_offset",self.x_m_offset, "self.y_m_offset",self.y_m_offset)
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

    line_ol,        = axtr.plot(xdata, ydata, '-k', label = 'Open loop simulation')
    line_est,    = axtr.plot(xdata, ydata, '-r', label = 'Estimated states')  # Plots the traveled positions
    line_meas,    = axtr.plot(xdata, ydata, '-b', label = 'Measured position camera')  # Plots the traveled positions
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


def Continuous_AB_Comp(vx, vy, omega, theta, delta):


#     %%% Parameters
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
    
    eps = 0.0000001
    # eps = 0
    # if abs(vx)> 0:
    F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)));
    Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) ;
    A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2);
    A31 = -Fry*lr/((vx+eps)*Iz);
        
    A12 = omega;
    A21 = -omega;
    A22 = 0;
    
    # if abs(vy) > 0.0:
    A22 = Fry/(m*(vy+eps));

    A41 = cos(theta);
    A42 = -sin(theta);
    A51 = sin(theta);
    A52 = cos(theta);


    B12 = 0;
    B32 = 0;
    B22 = 0;
    

    B12 = -F_flat*sin(delta)/(m*(delta+eps));
    B22 = F_flat*cos(delta)/(m*(delta+eps));    
    B32 = F_flat*cos(delta)*lf/(Iz*(delta+eps));



    B11 = (1/m)*(Cm0 - Cm1*vx);
    
    A = np.array([[A11, A12, 0,  0,   0,  0],\
                  [A21, A22, 0,  0,   0,  0],\
                  [A31,  0 , 0,  0,   0,  0],\
                  [A41, A42, 0,  0,   0,  0],\
                  [A51, A52, 0,  0,   0,  0],\
                  [ 0 ,  0 , 1,  0,   0,  0]])
    
    # print "A = {}".format(A), "Det A = {}".format(LA.det(A))

    B = np.array([[B11, B12],\
                  [ 0,  B22],\
                  [ 0,  B32],\
                  [ 0 ,  0 ],\
                  [ 0 ,  0 ],\
                  [ 0 ,  0 ]])
        
    # print "B = {}".format(B), "Det B = {}".format(LA.det(B))

    return A, B


def Continuous_AB_Comp_old(vx, vy, omega, theta, delta):

    m = rospy.get_param("m")
    rho = rospy.get_param("rho")
    lr = rospy.get_param("lr")
    lf = rospy.get_param("lf")
    Cm0 = rospy.get_param("Cm0")
    Cm1 = rospy.get_param("Cm1")
    C0 = rospy.get_param("C0")
    C1 = rospy.get_param("C1")
    Cd_A = rospy.get_param("Cd_A")
    Caf = rospy.get_param("Caf")
    Car = rospy.get_param("Car")
    Iz = rospy.get_param("Iz")

    
    F_flat = 0;
    Fry = 0.0;
    Frx = 0.0;
    A11 = 0.0;
    A31 = 0.0;
    
    eps = 0.000001
    
    # if abs(vx)>0.0:
    F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)));
    
    
    
    Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) ;
    A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2);
    A31 = -Fry*lr/((vx+eps)*Iz);
            
    A12 = omega;
    A21 = -omega;
    A22 = 0.0

    # if abs(vy)>0.0:

    A22 = Fry/(m*(vy+eps));

    A41 = cos(theta);
    A42 = -sin(theta);
    A51 = sin(theta);
    A52 = cos(theta);

    B12 = 0.0 
    B22 = 0.0 
    B32 = 0.0 

    # if abs(delta)>0.0:

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
    
    return A_obs, B_obs




def Continuous_AB_Comp_old2(vx, vy, omega, theta, delta):

    m = rospy.get_param("m")
    rho = rospy.get_param("rho")
    lr = rospy.get_param("lr")
    lf = rospy.get_param("lf")
    Cm0 = rospy.get_param("Cm0")
    Cm1 = rospy.get_param("Cm1")
    C0 = rospy.get_param("C0")
    C1 = rospy.get_param("C1")
    Cd_A = rospy.get_param("Cd_A")
    Caf = rospy.get_param("Caf")
    Car = rospy.get_param("Car")
    Iz = rospy.get_param("Iz")

    
    F_flat = 0;
    Fry = 0;
    Frx = 0;
    
    
    eps = 0.00000001
    F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)));
    
    
    
    Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) ;
    A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2);
    A31 = -Fry*lr/((vx+eps)*Iz);
        
    A12 = omega;
    A21 = -omega;
    
    A22 = Fry/(m*(vy+eps));

    A41 = cos(theta);
    A42 = -sin(theta);
    A51 = sin(theta);
    A52 = cos(theta);


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

def data_retrive_est(msg, est_msg, yaw_measured, AC_sig, CC_sig):

    msg.timestamp_ms = 0
    msg.X  = est_msg[3]
    msg.Y  = est_msg[4]
    msg.roll  = 0
    msg.yaw  = est_msg[5]
    msg.pitch  = 0
    msg.vx  = est_msg[0]
    msg.vy  = est_msg[1]
    msg.yaw_rate  = est_msg[2]
    msg.ax  = AC_sig
    msg.ay  = CC_sig
    msg.s  = yaw_measured
    msg.x  = 0
    msg.y  = 0

    return msg

def meas_retrive(msg, est_msg):

    msg.timestamp_ms = 0
    msg.X  = est_msg[2]
    msg.Y  = est_msg[3]
    msg.roll  = 0
    msg.yaw  = est_msg[4]
    msg.pitch  = 0
    msg.vx  = est_msg[0]
    msg.vy  = 0
    msg.yaw_rate  = est_msg[1]
    msg.ax  = 0
    msg.ay  = 0
    msg.s  = 0
    msg.x  = 0
    msg.y  = 0

    return msg

def load_LQRgain():

    LQR_gain = np.array(sio.loadmat(gain_path)['data']['Lmi'].item())
    seq = sio.loadmat(gain_path)['data']['sequence'].item()
    seq = seq - 1 ##matlab index to python index
    sched_var = sio.loadmat(gain_path,matlab_compatible = 'True')['data']['sched_var'].item()
    
    return LQR_gain, seq, sched_var

def load_switchingLQRgain():

    LQR_gain1 = np.array(sio.loadmat(gain_path)['data']['Lmi1'].item())
    LQR_gain2 = np.array(sio.loadmat(gain_path)['data']['Lmi2'].item())
    LQR_gain3 = np.array(sio.loadmat(gain_path)['data']['Lmi3'].item())
    LQR_gain4 = np.array(sio.loadmat(gain_path)['data']['Lmi4'].item())
    
    LQR_gain = np.array([LQR_gain1, LQR_gain2, LQR_gain3, LQR_gain4])

    seq_1 = sio.loadmat(gain_path)['data']['sequence_1'].item()
    seq_1 = seq_1 - 1 ##matlab index to python index

    seq_2 = sio.loadmat(gain_path)['data']['sequence_2'].item()
    seq_2 = seq_2 - 1 ##matlab index to python index
    
    seq_3 = sio.loadmat(gain_path)['data']['sequence_3'].item()
    seq_3 = seq_3 - 1 ##matlab index to python index
    
    seq_4 = sio.loadmat(gain_path)['data']['sequence_4'].item()
    seq_4 = seq_4 - 1 ##matlab index to python index
    
    seq = np.array([seq_1, seq_2, seq_3, seq_4])

    sched_var = sio.loadmat(gain_path,matlab_compatible = 'True')['data']['sched_var'].item()

    sched_var1 = [sched_var[0], sched_var[1], sched_var[2], sched_var[3], sched_var[7]]
    sched_var2 = [sched_var[0], sched_var[1], sched_var[2], sched_var[4], sched_var[7]]
    sched_var3 = [sched_var[0], sched_var[1], sched_var[2], sched_var[5], sched_var[7]]
    sched_var4 = [sched_var[0], sched_var[1], sched_var[2], sched_var[6], sched_var[7]]

    sched_var = np.array([sched_var1, sched_var2, sched_var3, sched_var4])
    return LQR_gain, seq, sched_var


def constrainAngle(x):
    
    x = fmod(x + pi,2*pi);
    if (x < 0):
        x += 2*pi;
    return x - pi;

# // convert to [-360,360]
def angleConv(angle):
    return fmod(constrainAngle(angle),2*pi);

def angleDiff(a,b):
    dif = fmod(b - a + pi,2*pi);
    if (dif < 0):
        dif += 2*pi;
    return dif - pi;

def unwrap(previousAngle,newAngle):
    return previousAngle - angleDiff(newAngle,angleConv(previousAngle))



def yaw_correction(angle):
    eps = 0.0
    if angle < 0:
        angle = 2*pi - abs(angle)
    elif angle > 2*pi - eps:
        angle = angle%(2.0*pi)
    return angle

def wrap(angle):
    eps = 0.00
    if angle < -np.pi + eps:
        w_angle = 2 * np.pi + angle -eps
    elif angle > np.pi - eps :
        w_angle = angle - 2 * np.pi + eps 
    
    elif angle > 2*np.pi - eps :
        w_angle = angle%(2.0*pi)
    
    elif angle < -2*np.pi + eps :
        w_angle =  -(angle%(2.0*pi))

    else:
        w_angle = angle

    return w_angle

def yaw_smooth(angle_cur, angle_past):
    
    eps = 0.02 ### Boundary near 0 and 2pi for considering the crossing of axis.
    
    CC = False
    AC = False
    # print round(angle_cur,2),round(angle_past,2)
    if (round(2*pi,2) - eps) <= round(angle_cur,2) <= round(2*pi,2) and (round(angle_past,2) <= eps) and (round(angle_past,2) >= 0.0):
        # print "clockwise changes"
        CC = True
        
    if ((round(2*pi,2) -eps) <= round(angle_past,2) <= round(2*pi,2)) and (eps >= round(angle_cur,2) >= 0.0):
#         print "anticlockwise changes"
        AC = True
    return CC, AC


def yaw_error_throw():
    try:
        raise Exception('general exceptions not caught by specific handling')
    except ValueError as e:
        print('Error in yaw transformation')





def main():
    rospy.init_node('switching_lqr_state_estimation', anonymous=True)

    loop_rate   = rospy.get_param("switching_lqr_observer/publish_frequency")
    rate        = rospy.Rate(loop_rate)
    time0       = rospy.get_rostime().to_sec()
    
    counter     = 0
    record_data =    rospy.get_param("switching_lqr_observer/record_data")
    visualization  = rospy.get_param("switching_lqr_observer/visualization")

    LQR_gain, seq, sched_var =load_switchingLQRgain()

    N_enc  = rospy.get_param("switching_lqr_observer/enc_MA_window")
    N_fcam = rospy.get_param("switching_lqr_observer/fcam_MA_window")
    N_imu  = rospy.get_param("switching_lqr_observer/imu_MA_window")

    enc    = motor_encoder(time0, N_enc)
    fcam   = fiseye_cam(time0, N_fcam)
    imu    = IMU(time0, N_imu)

    time.sleep(3)
    print  "yaw_offset", fcam.yaw
    imu.yaw_offset = imu.yaw - fcam.yaw
    control_input = vehicle_control(time0)
    time.sleep(3)

    print "fcam.yaw",fcam.yaw
    
    if visualization == True:
        x_lim = 10
        y_lim = 10
        (fig, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas) = _initializeFigure_xy(x_lim,y_lim)

        ol_x_his     = []
        est_x_his    = []
        meas_x_his   = []
        ol_y_his     = []
        est_y_his    = []
        meas_y_his   = []

# class EstimatorData(object):
#     """Data from estimator"""
#     def __init__(self):
#         """Subscriber to estimator"""
#         rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback)
#         self.offset_x = 0.46822099
#         self.offset_y = -1.09683919
#         self.offset_yaw = -pi/2
#         self.R = np.array([[cos(self.offset_yaw),-sin(self.offset_yaw)],[sin(self.offset_yaw), cos(self.offset_yaw)]])
#         self.CurrentState = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
    
#     def estimator_callback(self, msg):
#         """
#         Unpack the messages from the estimator
#         """
#         self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw + self.offset_yaw]).T
#         self.CurrentState[3:5] = np.dot(self.R,self.CurrentState[3:5])
#         self.CurrentState[3:5] = self.CurrentState[3:5] - np.array([self.offset_x, self.offset_y]).T

    # delay  = 5
    # offset = pi/2
    print ("<<<< Initializing IMU orientation >>>>")
    # imu.calibrate_imu(delay,offset)    
    # fcam.calibrate_fcam(delay,R_tf)
    print ("<<<< ORIGIN SET AND CALIBRATION DONE >>>>")


    ###### observation matrix ######
    C       =  np.array([[1, 0, 0, 0, 0, 0], 
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]]) 

    # time.sleep(5)
    
    lr = rospy.get_param("lr")
    lf = rospy.get_param("lf")
    beta = lr*tan(control_input.steer)/(lr+lf); ## slip angle
    vy = enc.vx*beta;
    vy = 0.0

    # yaw_curr = yaw_correction(imu.yaw)
    yaw_curr = (imu.yaw)

    # est_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, fcam.yaw ]).T
    # est_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, yaw_correction(fcam.yaw) ]).T
    # est_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, imu.yaw ]).T
    est_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, yaw_curr ]).T


    est_state_hist = [] 
    est_state_hist.append(est_state)
    
    est_state_msg = sensorReading()
    est_state_pub  = rospy.Publisher('est_state_info', sensorReading, queue_size=1)


    #### Open loop simulation ###
    # ol_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, fcam.yaw ]).T
    # ol_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, yaw_correction(fcam.yaw) ]).T
    # ol_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, imu.yaw ]).T
    ol_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, yaw_curr ]).T

    ol_state_hist = [] 
    ol_state_hist.append(ol_state)
    
    ol_state_msg = sensorReading()
    ol_state_pub  = rospy.Publisher('ol_state_info', sensorReading, queue_size=1)

    meas_state_pub  = rospy.Publisher('meas_state_info', sensorReading, queue_size=1)
    meas_state_msg = sensorReading()

    control_data = control()
    control_hist = {'timestamp_ms_dutycycle':[],'timestamp_ms_steer':[],'steering':[], 'duty_cycle':[]}

    # enc_pub  = rospy.Publisher('encoder_fused', sensorReading, queue_size=1)
    # enc_data = sensorReading()
    # enc_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # imu_pub  = rospy.Publisher('imu_fused', sensorReading, queue_size=1)
    # imu_data = sensorReading()
    # imu_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # fcam_pub  = rospy.Publisher('fcam_fused', sensorReading, queue_size=1)
    # fcam_data = sensorReading()
    # fcam_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        

        
    # enc_MA_pub  = rospy.Publisher('encoder_MA_fused', sensorReading, queue_size=1)
    # enc_MA_data = sensorReading()
    # enc_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # imu_MA_pub  = rospy.Publisher('imu_MA_fused', sensorReading, queue_size=1)
    # imu_MA_data = sensorReading()
    # imu_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # fcam_MA_pub  = rospy.Publisher('fcam_MA_fused', sensorReading, queue_size=1)
    # fcam_MA_data = sensorReading()
    # fcam_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}

    
    yaw_check = wrap(fcam.yaw)
    curr_time = rospy.get_rostime().to_sec() - time0
     
    prev_time = curr_time 
    
    u = [0,0]
    
    #### YAW CORRECTION ####
    angle_past = imu.yaw
    

    while not (rospy.is_shutdown()):
        
        u = np.array([control_input.duty_cycle, control_input.steer]).T


        angle_cur = imu.yaw
        
        angle_acc = unwrap(angle_past, angle_cur)  

        angle_past = angle_acc
        
        # print "fcam.yaw",fcam.yaw

        curr_time = rospy.get_rostime().to_sec() - time0
    
        ######### YAW CALCULATION ########
        # angle_cur = yaw_correction(imu.yaw)
        # yaw_curr = yaw_correction(imu.yaw)

        # CC, AC = yaw_smooth(angle_cur, angle_past)

        # dyaw = yaw_curr - yaw_past  


        # yaw_diff = np.diff([angle_past1, angle_past2, angle_curr], n = 2)

        # CC = False
        # AC = False

        # AC_sig = 0
        # CC_sig = 0
        # if yaw_diff == True:
        #     angle_temp += angle_past
        #     direction = 1.0
        #     AC_sig = 4
        #     # print ("anticlockwise crossed")

        # if CC == True:
        #     angle_temp += angle_cur
        #     direction = -1.0
        #     CC_sig = -4
            
        #     # print ("clockwise crossed")

        # angle_acc = angle_cur + direction*angle_temp  

        # angle_acc += dyaw  

        # yaw_past = yaw_curr
        # angle_past1 = angle_past2
        # angle_past2 = angle_cur


        y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, angle_acc]).T 
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, fcam.yaw]).T
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, yaw_correction(fcam.yaw)]).T
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X_MA, fcam.Y_MA, imu.yaw]).T 
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X_MA, fcam.Y_MA, yaw_correction(imu.yaw)]).T 




        dt = curr_time - prev_time 
        
        if u[0] > 0.05:


            # yaw_trans = wrap(yaw_correction(y_meas[-1]))
            yaw_trans = (est_state[5] + pi) % (2 * pi) - pi
            # yaw_trans = est_state[5]
            # %% quadrant case
            if 0 <= yaw_trans <= pi/2:
            # % 1st quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[0], sched_var[0], seq[0])
               
            elif pi/2 < yaw_trans <= pi:
            # % 2nd quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[1], sched_var[1], seq[1])
                    
            elif -pi <= yaw_trans <= -pi/2:
            # % 3rd quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[2], sched_var[2], seq[2])
                
            elif (-pi/2 < yaw_trans < 0):
            # % 4th quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[3], sched_var[3], seq[3])
                
            else:
                
                print "est theta", yaw_trans, yaw_trans*180.0/pi 

                display("ERROR Normalize the theta")



                yaw_error_throw()



            # # %% quadrant case
            # if 0 <= est_state[5] <= pi/2:
            # # % 1st quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[0], sched_var[0], seq[0])
               
            # elif pi/2 < est_state[5] <= pi:
            # # % 2nd quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[1], sched_var[1], seq[1])
                    
            # elif pi < est_state[5] <= 3*pi/2:
            # # % 3rd quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[2], sched_var[2], seq[2])
                
            # elif (3*pi/2 < est_state[5] < 2*pi):
            # # % 4th quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[3], sched_var[3], seq[3])
                
            # else:
                
            #     print "est theta", est_state[5], 'Measured theta', yaw_correction(fcam.yaw)

            #     display("ERROR Normalize the theta")



                # yaw_error_throw()





            # print ("u",u)
            
            ####### LQR ESTIMATION ########
            A_obs, B_obs = Continuous_AB_Comp(est_state[0], est_state[1], est_state[2], est_state[5], u[1])
            # L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain, sched_var, seq)
            
            est_state  = est_state + ( dt * np.dot( ( A_obs - np.dot(L_gain, C) ), est_state )
                            +    dt * np.dot(B_obs, u)
                            +    dt * np.dot(L_gain, y_meas) )
            
            # print ("time taken for estimation ={}".format(rospy.get_rostime().to_sec() - time0 - curr_time))
            
            ##### OPEN LOOP SIMULATION ####
            A_sim, B_sim = Continuous_AB_Comp(ol_state[0], ol_state[1], ol_state[2], ol_state[5], u[1])
            ol_state = ol_state + dt*(np.dot(A_sim,ol_state) + np.dot(B_sim,u)) 
            # yaw_check += wrap(fcam.yaw)

        if abs(u[0]) <= 0.05:
                #     # vehicle_sim.vehicle_model(u, simulator_dt)
                    # if vehicle_sim.vx <= 0.01 :
            est_state[:-3] = 0.000001 
            ol_state[:-3] = 0.000001

        # else:
        #     if enc.vx == 0.0:
        #         est_state[0] = 0.0
        #         ol_state[0]  = 0.0
        #         est_state[1] = 0.0
        #         ol_state[1]  = 0.0

        #     if imu.yaw_rate <= 0.018:    
        #         est_state[2] = 0.0
        #         ol_state[2]  = 0.0

        print "\n <<<<<<<<< PRE WRAP >>>>>>>>>>>>>"
        print "est_state",est_state
        print "ol_state", ol_state


        # est_state[5] = wrap(est_state[5])
        # ol_state[5] = wrap(ol_state[5])
        # est_state[5] = yaw_correction(est_state[5])
        # ol_state[5] = yaw_correction(ol_state[5])
        
        print "\n <<<<<<<<< STATS >>>>>>>>>>>>>"
        print "measured states", y_meas
        print "est_state",est_state
        print "ol_state", ol_state
        print "input u", u
        print "dt", dt

        AC_sig = 0
        CC_sig = 0

        est_state_pub.publish(data_retrive_est(est_state_msg, est_state, y_meas[-1], AC_sig, CC_sig)) ## remember we want to check the transformed yaw angle for debugging that's why 
                                                                                    ##publishing this information in the topic of "s" which is not used for any purpose. 
        est_state_hist.append(est_state)  

        ol_state_pub.publish(data_retrive(ol_state_msg, ol_state))
        # ol_state_hist.append(ol_state)

        meas_pub = meas_retrive(meas_state_msg, y_meas)
        meas_state_pub.publish(meas_pub)
        # angle_past = angle_cur

        # control_msg = control_input.data_retrive(control_data)        
        # append_control_data(control_hist, control_msg)


        # enc_msg = enc.data_retrive(enc_data)
        # enc_pub.publish(enc_msg)
        # append_sensor_data(enc_hist, enc_msg)


        # enc_MA_msg = enc.data_retrive_MA(enc_MA_data)
        # enc_MA_pub.publish(enc_MA_msg)
        # append_sensor_data(enc_MA_hist, enc_MA_msg)


        # imu_msg = imu.data_retrive(imu_data)
        # imu_pub.publish(imu_msg)
        # append_sensor_data(imu_hist, imu_msg)


        # imu_MA_msg = imu.data_retrive_MA(imu_MA_data)
        # imu_MA_pub.publish(imu_MA_msg)
        # append_sensor_data(imu_MA_hist, imu_MA_msg)


        # fcam_msg = fcam.data_retrive(fcam_data)
        # fcam_pub.publish(fcam_msg)
        # append_sensor_data(fcam_hist, fcam_msg)


        # fcam_MA_msg = fcam.data_retrive_MA(fcam_MA_data)
        # fcam_MA_pub.publish(fcam_MA_msg)
        # append_sensor_data(fcam_MA_hist, fcam_MA_msg)

        prev_time = curr_time 


        if visualization == True:

            l = 0.42; w = 0.19

            (x_est , y_est , yaw_est )  = est_state[-3:]
            (x_ol  , y_ol  , yaw_ol  )  = ol_state[-3:]
            (x_meas, y_meas, yaw_meas)  = y_meas[-3:]

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

            meas_x, car_meas_y = getCarPosition(x_meas, y_meas, yaw_meas, w, l)
            rec_meas.set_xy(np.array([meas_x, car_meas_y]).T)

            line_est.set_data(est_x_his, est_y_his)
            line_ol.set_data(ol_x_his, ol_y_his)
            line_meas.set_data(meas_x_his, meas_y_his)

            fig.canvas.draw()
            plt.show()
            plt.pause(1.0/300)


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
