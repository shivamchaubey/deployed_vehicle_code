#!/usr/bin/env python
"""
    File name: stateEstimator.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu
    Update: 20/02/2020
    Python Version: 2.7.12
"""
# ---------------------------------------------------------------------------
# Attibution Information: This project ROS code-based was developed at UPC-IRI
# in the CS2AC group by Eugenio Alcala Baselga (eugenio.alcala@upc.edu).
#
# This is basically a estimation algorithm based on a polytopic gain scheduling
# approach, similar to Extended Kalman Filter although better.
# ---------------------------------------------------------------------------

import rospy
import os
import sys
from datetime import datetime
sys.path.append(sys.path[0]+'/Utilities')


homedir = os.path.expanduser("~")
sys.path.append(os.path.join(homedir,"barc/workspace/src/barc/src/library"))
from utilities import wrap
from lpv_mpc.msg import control_actions, pos_info, Vel_est, simulatorStates, My_IMU
from geometry_msgs.msg import Twist, Pose
from marvelmind_nav.msg import hedge_imu_fusion, hedge_pos
from std_msgs.msg import Header, Bool, Float32
from numpy import eye, zeros, diag, tan, cos, sin, vstack, linalg, pi
from numpy import ones, polyval, size, dot, add
from scipy.linalg import inv, cholesky
from tf import transformations
import math

from sensor_fusion.msg import sensorReading, hedge_imu_fusion, hedge_imu_raw

from math import cos, sin, atan, tan, acos, asin
import numpy as np
import scipy.io as sio
import pdb


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def main():
    # node initialization
    rospy.init_node("state_estimation")
    a_delay     = 0.0
    df_delay    = 0.0
    loop_rate   = rospy.get_param('state_estimator/loopRate')
    slow_mode_th = rospy.get_param("slow_mode_th")

    t0 = rospy.get_rostime().to_sec()

    # gps = GpsClsas(t0)
    fcam = fcamClass(t0)
    ecu = EcuClass(t0)
    
    #imu_enc = ImuEncClass(t0) #Created for the miniUPC vehicle
    
    imu_enc = ImuEncFused(t0)

    est = Estimator(t0,30)

    estMsg = pos_info()

    read_flag        = flag()

    rospy.sleep(0.5)   # Soluciona los problemas de inicializacion
    #print('[ESTIMATOR] The estimator is running!')
    # curr_time = rospy.get_rostime().to_sec() -t0
    
     
    curr_time = imu_enc.timestamp_ms
    prev_time = curr_time 
    while not rospy.is_shutdown():

        # curr_time = rospy.get_rostime().to_sec() -t0
        curr_time = imu_enc.timestamp_ms


        u = np.array([ecu.a, ecu.df]).T
        dt = curr_time - prev_time
        if u[1]> 0.15:

            est.estimateState( imu_enc.vx, imu_enc.yaw_rate, fcam.x, fcam.y, imu_enc.yaw,u,dt)
        
        estMsg.x          = est.x_est
        estMsg.y          = est.y_est
        estMsg.v_x        = est.vx_est
        estMsg.v_y        = est.vy_est
        estMsg.yaw        = imu_enc.yaw #est.yaw_est
        estMsg.yaw_rate   = est.yaw_rate_est

        # est.vx_his.append(est.vx_est)
        # if imu_enc.vx >= slow_mode_th:
        #     est.estimateState(ecu, imu_enc, fcam)

        #     estMsg.x          = est.x_est
        #     estMsg.y          = est.y_est
        #     estMsg.v_x        = est.vx_est
        #     estMsg.v_y        = est.vy_est
        #     estMsg.psi        = imu_enc.yaw #est.yaw_est
        #     estMsg.yaw_rate   = est.yaw_rate_est

        #     est.vx_his.append(est.vx_est)


        # else:
        #     # Real Car
        #     estMsg.x        = fcam.x
        #     estMsg.y        = fcam.y
        #     estMsg.v_x      = imu_enc.vx
        #     estMsg.v_y      = 0
        #     estMsg.psi      = imu_enc.yaw
        #     estMsg.yaw_rate   = 0

        #     est.x_est       = fcam.x
        #     est.y_est       = fcam.y
        #     est.vx_est      = imu_enc.vx
        #     est.vy_est      = 0
        #     est.yaw_est     = imu_enc.yaw
        #     est.yaw_rate_est  = 0

        
        # est.states_est  = np.array([ est.vx_est, est.vy_est, est.yaw_rate_est, est.x_est, est.y_est, est.yaw_est ])

        estMsg.u_a      = ecu.a
        estMsg.u_df     = ecu.df

        print ("estMsg",estMsg)
        est.state_pub_pos.publish(estMsg)
        prev_time = curr_time
        est.rate.sleep()




class flag(object):

    def __init__(self):
        self.flag = False
        rospy.Subscriber('flag', Bool, self.flag_callback, queue_size=1)

    def flag_callback(self,msg):
        self.flag = msg.data

class Estimator(object):
    """ Object collecting estimated state data
    Attributes:
        Estimated states:
            1.vx_est     2.vy_est    3.yaw_rate_est
            4.x_est      5.y_est     6.yaw_est

        Time stamp
            1.t0  2.curr_time

    Methods:
        stateEstimate( imu, fcam, enc, ecu, sim, observer, AB_comp, L_comp ):
            Estimate current state from sensor data

        GS_LPV_Est( states_est, y_meas, u, Continuous_AB_Comp, L_Gain_Comp ):
            Gain Scheduling LPV Estimator

        Continuous_AB_Comp( vx, vy, theta, steer):
            LPV State Space computation

        L_Gain_Comp(self, vx, vy, theta, steer):
            Scheduling-based polytopic estimator gain
    """
    def __init__(self, t0,loop_rate):

        dt                  = 1.0 / loop_rate
        self.rate           = rospy.Rate(loop_rate)
        # self.dt             = 0


#         self.a_delay        = a_delay
#         self.df_delay       = df_delay
#         self.motor_his      = [0.0]*int(a_delay/dt)
#         self.servo_his      = [0.0]*int(df_delay/dt)

        self.n_states  = 6
        self.n_meas    = 5
        self.L_gain    = np.zeros((self.n_states, self.n_meas))

        
    
        gain_path = '/home/auto/Desktop/autonomus_vehicle_project/project/development/proto/estimator/observer_gain_saved/LQR_16_04_2021'

        self.LQR_gain = np.array(sio.loadmat(gain_path+'/LQR_gain.mat')['data']['Lmi'].item())


        self.L_gain = np.zeros((self.n_states ,self.n_meas))
        seq = sio.loadmat(gain_path+'/LQR_gain.mat')['data']['sequence'].item()
        ##sched_vx, sched_vy, sched_w, sched_theta, sched_delta
        self.seq = seq - 1 ##matlab index to python index

        sched_var = sio.loadmat(gain_path+'/LQR_gain.mat')['data']['sched_var'].item()
        self.sched_vx    = sched_var[0]
        self.sched_vy    = sched_var[1]
        self.sched_w     = sched_var[2]
        self.sched_theta = sched_var[3]
        self.sched_delta = sched_var[4]

        self.state_pub_pos  = rospy.Publisher('pos_info', pos_info, queue_size=1)


        self.index = 0
        self.C_obs = np.array([[  1., 0., 0., 0., 0., 0. ],  # vx
                               [  0., 0., 1., 0., 0., 0. ],  # omega
                               [  0., 0., 0., 1., 0., 0. ],  # x
                               [  0., 0., 0., 0., 1., 0. ],  # y
                               [  0., 0., 0., 0., 0., 1. ]]) # yaw

#         self.state_pub_pos  = rospy.Publisher('pos_info', pos_info, queue_size=1)
        self.t0             = t0

        self.x_est          = 0.0
        self.y_est          = 0.0 
        self.vx_est         = 0.0
        self.vy_est         = 0.0
        self.yaw_est        = 0.0
        self.yaw_rate_est     = 0.0
        
        

        self.states_est     = np.array([ self.vx_est, self.vy_est, self.yaw_rate_est, self.x_est, self.y_est, self.yaw_est ]).T

        self.L_Computation(self.vx_est,self.vy_est,self.yaw_rate_est,self.yaw_est,0)   

        self.Continuous_AB_Comp(self.vx_est, self.vy_est, self.yaw_rate_est, self.yaw_est, 0)

        self.curr_time      = rospy.get_rostime().to_sec() - self.t0

        self.prev_time      = self.curr_time

        self.x_his          = 0
        self.y_his          = 0
        self.x_var_his      = 0
        self.y_var_his      = 0

        self.sim_model_x    = 0
        self.sim_model_y    = 0
        self.eps            = 0.02 # (meters) This variable defines the safety region in the Dead Reckoning algorithm

        self.vx_his         = []


    def estimateState(self, imu_enc_vx, imu_enc_yaw_rate, fcam_x, fcam_y, imu_enc_yaw,u,dt):
        """ Estimates the vector [ vx vy omega x y yaw ] from the measured one [ vx, omega, x, y, yaw ]
        using DC motor encoder, IMU and fcam sensors.
        Besides, it has been implemented a Dead reckoning algorithm to save the pose estimation when fcam sensor
        fails. """

        self.curr_time  = rospy.get_rostime().to_sec() - self.t0
        self.dt  = self.curr_time - self.prev_time
        dt  = self.dt
        y_meas = np.array([imu_enc_vx, imu_enc_yaw_rate, fcam_x, fcam_y, imu_enc_yaw]).T

        self.L_Computation(self.states_est[0],self.states_est[1],self.states_est[2],self.states_est[5], u[1])  

        self.Continuous_AB_Comp(self.states_est[0],self.states_est[1],self.states_est[2],self.states_est[5], u[1])

        correction = dt*(np.dot(self.L_gain, (y_meas - np.dot(self.C_obs,self.states_est))));

        self.states_est  = self.states_est + dt*((np.dot(self.A_obs,self.states_est) +     np.dot(self.B_obs,u)) + np.dot(self.L_gain, (y_meas - np.dot(self.C_obs,self.states_est))))        
        

        print (self.states_est,"self.states_est")
        self.vx_est         = self.states_est[0]
        self.vy_est         = self.states_est[1]
        self.yaw_rate_est     = self.states_est[2]

        self.x_est          = self.states_est[3]
        self.y_est          = self.states_est[4]
        self.yaw_est        = self.states_est[5]
        
        # self.vx_his = self.states_est[0]
        # self.x_his  = self.states_est[3]
        # self.y_his  = self.states_est[4]

        self.prev_time = self.curr_time



    def Continuous_AB_Comp(self, vx, vy, omega, theta, delta):

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

        # m = 2.424;
        # rho = 1.225;
        # lr = 0.1203;
        # lf = 0.1377;
        # Cm0 = 10.1305;
        # Cm1 = 1.05294;
        # C0 = 3.68918;
        # C1 = 0.0306803;
        # Cd_A = -0.657645;
        # Caf = 1.3958;
        # Car = 1.6775;
        # Iz = 0.02;


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
        
        self.A_obs = np.array([[A11, A12, 0,  0,   0,  0],\
                      [A21, A22, 0,  0,   0,  0],\
                      [A31,  0 , 0,  0,   0,  0],\
                      [A41, A42, 0,  0,   0,  0],\
                      [A51, A52, 0,  0,   0,  0],\
                      [ 0 ,  0 , 1,  0,   0,  0]])
        
        self.B_obs = np.array([[B11, B12],\
                      [ 0,  B22],\
                      [ 0,  B32],\
                      [ 0 ,  0 ],\
                      [ 0 ,  0 ],\
                      [ 0 ,  0 ]])

        
    def L_Computation(self,vx,vy,w,theta,delta):    

        

        
        M_vx_min      = (self.sched_vx[1] - vx) / (self.sched_vx[1] - self.sched_vx[0] );
        M_vy_min      = (self.sched_vy[1] - vy) / (self.sched_vy[1] - self.sched_vy[0] );
        M_w_min       = (self.sched_w[1] - w) / (self.sched_w[1] - self.sched_w[0]); 
        M_theta_min   = (self.sched_theta[1] - theta) / (self.sched_theta[1] - self.sched_theta[0]); 
        M_delta_min   = (self.sched_delta[1] - delta) / (self.sched_delta[1] - self.sched_delta[0]); 

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


        if vx > self.sched_vx[1] or vx < self.sched_vx[0]:
            print( '[ESTIMATOR/L_Gain_Comp]: Vx is out of the polytope ...' )
        elif vy > self.sched_vy[1] or vy < self.sched_vy[0]:
            print( '[ESTIMATOR/L_Gain_Comp]: Vy is out of the polytope ...' )
        elif delta > self.sched_delta[1] or delta < self.sched_delta[0]:
            print( '[ESTIMATOR/L_Gain_Comp]: Steering is out of the polytope ... = ',delta)


        mu = np.zeros((self.seq.shape[0],1))
        self.L_gain  = np.zeros((self.LQR_gain[:,:,1].shape[0], 5))

        for i in range(self.seq.shape[0]):
            mu[i] = M_vx[self.seq[i,0]] * M_vy[self.seq[i,1]] * M_w[self.seq[i,2]] * M_theta[self.seq[i,3]] * M_delta[self.seq[i,4]];
            self.L_gain  = self.L_gain  + mu[i]*self.LQR_gain[:,:,i];


            








# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================


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
        self.yaw_rate  = 0.0

        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

    def Twist_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vx     = data.linear.x  # from DC motor encoder
        self.vy     = 0.0
        self.yaw_rate = data.angular.z # from IMU

    def Pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = 0.0
        self.pitch  = 0.0
        self.yaw    = self.yaw_correction(data.orientation.z) # from IMU

    def yaw_correction(self,angle):
        if angle < 0:
            angle = 2*pi - abs(angle)
        return angle



class ImuEncFused(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('imu_fused', sensorReading, self.imuf_callback, queue_size=1)

        rospy.Subscriber('imu_MA_fused', sensorReading, self.imufMA_callback, queue_size=1)

        rospy.Subscriber('encoder_fused', sensorReading, self.encf_callback, queue_size=1)

        rospy.Subscriber('encoder_MA_fused', sensorReading, self.encfMA_callback, queue_size=1)


        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.yaw_rate  = 0.0

        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

    def imuf_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.yaw     = data.yaw
        self.yaw_rate  = data.yaw_rate

    def imufMA_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.yaw     = data.yaw
        self.yaw_rate  = data.yaw_rate

    def encf_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vx     = data.vx  # from DC motor encoder
        self.vy     = 0.0

    def encfMA_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vx     = data.vx  # from DC motor encoder
        self.vy     = 0.0





class GPSClass(object):
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

        # GPS measurement
        self.angle  = 0.0
        self.x      = 0.0
        self.y      = 0.0
        self.x_his  = 0.0
        self.y_his  = 0.0

        # time stamp
        self.t0             = t0
        self.curr_time      = rospy.get_rostime().to_sec() - self.t0

    def gps_callback(self, data):
        """Unpack message from sensor, GPS"""
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0
        self.x = data.x_m
        self.y = data.y_m
        self.saveHistory()

    def saveHistory(self):
        self.x_his  = self.x
        self.y_his  = self.y



class fcamClass(object):
    """ Object collecting fish eyecamera measurement data
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
        # rospy.Subscriber('fcam_fused', sensorReading, self.fcam_callback, queue_size=1)
        rospy.Subscriber('fcam_MA_fused', sensorReading, self.fcam_MA_callback, queue_size=1)
        
        # GPS measurement
        self.angle  = 0.0
        self.x      = 0.0
        self.y      = 0.0
        self.x_his  = 0.0
        self.y_his  = 0.0

        self.x_MA      = 0.0
        self.y_MA      = 0.0
        self.x_MA_his  = 0.0
        self.y_MA_his  = 0.0


        # time stamp
        self.t0             = t0
        self.curr_time      = rospy.get_rostime().to_sec() - self.t0

    def fcam_callback(self, data):
        """Unpack message from fish eye camera sensor"""
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0
        self.x          = data.X
        self.y          = data.Y

    def fcam_MA_callback(self, data):
        """Unpack averaged message from fish eye camera sensor"""
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0
        self.x_MA          = data.X
        self.y_MA          = data.Y
        # self.saveHistory()

    def saveHistory(self):
        self.x_his     = self.x
        self.y_his     = self.y
        self.x_MA_his  = self.x_MA
        self.y_MA_his  = self.y_MA






class EcuClass(object):
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
        self.a  = 0.0
        self.df = 0.0

        # time stamp
        self.t0         = t0
        self.curr_time  = rospy.get_rostime().to_sec() - self.t0

    def accel_callback(self,data):
        """Unpack message from sensor, ECU"""
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.a  = data.data

    def steering_callback(self,data):

        self.df = data.data






if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
