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

################################################### Plot states of vehicle #############################################
def plot_states(x_lim,y_lim):

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_ol,        = axtr.plot(xdata, ydata, '-k', label = 'Open loop simulation')
    line_est,    = axtr.plot(xdata, ydata, '-r', label = 'Estimated states')  # Plots the traveled positions
    line_meas,    = axtr.plot(xdata, ydata, '-b', label = 'Measured position camera')  # Plots the traveled positions
    
    l = 0.4; w = 0.2 #length and width of the car

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

#################### get rectangular points to plot in matplotlib patches ###################
def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y

############## function to save states either from subscribed messages or estimated states #########
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

############## function to save control action from subscribed messages ############
def append_control_data(data,msg):
    data['timestamp_ms_dutycycle'].append(msg.timestamp_ms_DC)
    data['timestamp_ms_steer'].append(msg.timestamp_ms_steer)
    data['steering'].append(msg.steer)
    data['duty_cycle'].append(msg.duty_cycle)


def Continuous_AB_Comp(vx, vy, omega, theta, delta):
    '''
    Calculate the vehicle LPV model
    Input: 
    1. vx (longitudinal velocity)
    2. vy (lateral velocity)
    3. omega (angular rate)
    4. theta (yaw) and delta (steering angle)

    Output:
    A, B: system matrices
    '''

    # vehicle Parameters
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
    
    A31 = 0;
    A11 = 0;
    
    eps = 0.0000001

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
    
    # if abs(delta) > 0.0:
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
    
    B = np.array([[B11, B12],\
                  [ 0,  B22],\
                  [ 0,  B32],\
                  [ 0 ,  0 ],\
                  [ 0 ,  0 ],\
                  [ 0 ,  0 ]])

    return A, B

##################################### Gain Calculation for observer ################################
def L_Computation(vx,vy,w,theta,delta,LQR_gain,sched_var,seq):    
    '''
    Gain scheduling function which interpolates the gain to find the optimal gain for the existing 
    varying system. The set of gains calculated offline using using fuzzy model of the system and 
    designing LMI for desired criteria such as Lyapunov, LQR.
    
    Input:
    1. Current states: vx, vy, w
    2. Current input: delta (steering angle)
    3. Gain calculated offline: LQR_gain
    4. Scheduling variable which present the bound on states and input : sched_var
    5. sequence: The sequence is important to know which tells about which subsystem belongs to which gain. 
       This can be known during the LMI formulation.
    Output:
    1. Interpolated gain: The gain which can be applied to current system to estimate the states. 

    '''
    
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

    ######### Set membership function #########
    M_vx          = [M_vx_min, M_vx_max];   
    M_vy          = [M_vy_min, M_vy_max];   
    M_w           = [M_w_min, M_w_max];     
    M_theta       = [M_theta_min, M_theta_max];     
    M_delta       = [M_delta_min, M_delta_max];     


    if vx > sched_vx[1] or vx < sched_vx[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Vx is out of the polytope ... = ', vx )
    elif vy > sched_vy[1] or vy < sched_vy[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Vy is out of the polytope ... = ', vy )
    elif w > sched_w[1] or w < sched_w[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Omega (w) is out of the polytope ...', w )
    elif theta > sched_theta[1] or theta < sched_theta[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Yaw (theta) is out of the polytope ...', theta )
    
    elif delta > sched_delta[1] or delta < sched_delta[0]:
        print( '[ESTIMATOR/L_Gain_Comp]: Steering is out of the polytope ... = ',delta)


    mu = np.zeros((seq.shape[0],1))
    L_gain  = np.zeros((LQR_gain[:,:,1].shape[0], 5))
    for i in range(seq.shape[0]):
        mu[i] = M_vx[seq[i,0]] * M_vy[seq[i,1]] * M_w[seq[i,2]] * M_theta[seq[i,3]] * M_delta[seq[i,4]]; #grade membership
        L_gain  = L_gain  + mu[i]*LQR_gain[:,:,i]; #interpolation

    return L_gain



def data_retrive(msg, est_msg):
    '''
     function to retrieve data from sensorReading() msg which can be further
     published or saved as history
    '''

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
    '''
     function to retrieve data from sensorReading() msg which can be further
     published or saved as history.
     This has some extra input which can be used for debugging purpose
     by assigning value to some variable which needs to be known and later
     visualize through rqt_graph or plotjuggler
    '''

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
    '''
    Same as before only difference is the indexing. The previous function can not be
    used for assigning the measured value as it has length of 5 and indexing is 
    different
    '''

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
    '''
    Function to load gain from .mat file for observer
    '''
    LQR_gain = np.array(sio.loadmat(gain_path)['data']['Lmi'].item())
    seq = sio.loadmat(gain_path)['data']['sequence'].item()
    seq = seq - 1 ##matlab index to python index
    sched_var = sio.loadmat(gain_path,matlab_compatible = 'True')['data']['sched_var'].item()
    
    return LQR_gain, seq, sched_var

def load_switchingLQRgain():
    '''
    Function to load gain from .mat file for observer
    This gain has 4 different gain for each quadrant which is advanced case of simple
    gain. This is done to solve the problem of trigonometric function (sin, cos) during
    the LMI formulation.
    '''

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




def unwrap(previousAngle,newAngle):
    '''
    Function to change discontinuous yaw measurement [-pi, pi]
    to continuous (-inf, inf) to correct the convergence of the
    observer otherwise the estimator works weird. 
    '''

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

    return previousAngle - angleDiff(newAngle,angleConv(previousAngle))


def wrap(angle):
    '''
    wrapping angle between [-pi, pi]
    Transform continuous angle to discontinuous
    '''
    return (angle + pi) % (2 * pi) - pi

def yaw_error_throw():
    '''
    Throw error if yaw in not unwraped properly:
    Used for debugging
    '''
    try:
        raise Exception('general exceptions not caught by specific handling')
    except ValueError as e:
        print('Error in yaw transformation')
