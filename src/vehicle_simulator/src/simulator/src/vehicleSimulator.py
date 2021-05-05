#!/usr/bin/env python
"""
    File name: barc_simulator_dyn.py
    Author: Shuqi Xu
    Email: shuqixu@kth.se
    Python Version: 2.7.12
"""
# ---------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that (1) you retain this notice
# and (2) you provide clear attribution to UC Berkeley, including a link
# to http://barc-project.com
#
# Attibution Information: The barc project ROS code-base was developed
# at UC Berkeley in the Model Predictive Control (MPC) lab by Jon Gonzales
# (jon.gonzales@berkeley.edu). The cloud services integation with ROS was developed
# by Kiet Lam  (kiet.lam@berkeley.edu). The web-server app Dator was
# based on an open source project by Bruce Wootton
#----------------------------------------------------------------------------


import rospy
import geometry_msgs.msg
from simulator.msg import simulatorStates
from geometry_msgs.msg import Vector3, Quaternion, Pose, Twist
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Imu
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from l4vehicle_msgs.msg import VehicleState
from math import atan, tan, cos, sin, pi
from numpy.random import randn,rand
from random import randrange, uniform
from tf import transformations
import numpy as np
import pdb


def main():
    rospy.init_node("simulator")
    fcam_freq_update = rospy.get_param("simulator/fcam_freq_update")
    imu_freq_update = rospy.get_param("simulator/imu_freq_update")
    enc_freq_update = rospy.get_param("simulator/enc_freq_update")

    simulator_dt    = rospy.get_param("simulator/dt")
    lowLevelDyn     = rospy.get_param("simulator/lowLevelDyn")
    # pub_rate   = rospy.get_param("simulator/pub_rate")

    print "[SIMULATOR] Is Low Level Dynamics active?: ", lowLevelDyn


    """ Object collecting GPS measurement data
    and using vehicle mode for simulation
    Attributes:
        Model params:
            1.L_f 2.L_r 3.m(car mass) 3.I_z(car inertial) 4.c_f(equivalent drag coefficient)
        States:
            1.x 2.y 3.vx 4.vy 5.ax 6.ay 7.psiDot
        States history:
            1.x_his 2.y_his 3.vx_his 4.vy_his 5.ax_his 6.ay_his 7.psiDot_his
        Simulator sampling time:
            1. dt
        Time stamp:
            1. time_his
    Methods:
        f(u):
            System model used to update the states
        pacejka(ang):
            Pacejka lateral tire modeling
    """

    time0 = rospy.get_rostime().to_sec()

    sensor_sim = Sensor_Simulator(fcam_freq_update, imu_freq_update, enc_freq_update, simulator_dt)
    vehicle_sim = Vehicle_Simulator()
    
    # subscribe to the ecu for updating the control inputs.
    ecu = vehicle_control_input(time0)

    link_msg = ModelState()
    # publish the link message for gazebo simulation

    flag  = FlagClass()
    # bool message type

    hist_sens_noise = []

    link_msg.model_name = "seat_car"
    link_msg.reference_frame = "world"

    link_msg.pose.position.z = 0.031
    link_msg.twist.linear.x = 0.0
    link_msg.twist.linear.y = 0
    link_msg.twist.linear.z = 0

    link_msg.twist.angular.x = 0.0
    link_msg.twist.angular.y = 0
    link_msg.twist.angular.z = 0

    #  what are these variables??
    a_his   = [0.0]*int(rospy.get_param("simulator/delay_a")/rospy.get_param("simulator/dt"))
    df_his  = [0.0]*int(rospy.get_param("simulator/delay_df")/rospy.get_param("simulator/dt"))

    #  what are these variables??
    vehicle_sim_offset_x    = rospy.get_param("simulator/init_x")
    vehicle_sim_offset_y    = rospy.get_param("simulator/init_y")
    vehicle_sim_offset_yaw  = rospy.get_param("simulator/init_yaw")

    pub_vehicle_simulatorStates = rospy.Publisher('vehicle_simulatorStates', simulatorStates, queue_size=1)
    # pub_simulatorStatesIDIADA = rospy.Publisher('vehicle_state', VehicleState, queue_size=1)
    pub_linkStates = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
    # pub_sysOut = rospy.Publisher('sensorStates', simulatorStates, queue_size=1)



    # simStatesIDADA = VehicleState()
    simStates = simulatorStates()

    print('[SIMULATOR] The simulator is running!')
    print('\n')

    # servo_inp ? what are the input to servo ? only theta or pwm signals ?? 
    servo_inp = 0.0

    # what is the use of these variables Tf?? 
    T  = simulator_dt
    # What is tf and how it is set to 0.07??
    Tf = 0.07
    counter = 0

    prev_time = rospy.get_rostime().to_sec() - time0

    while not (rospy.is_shutdown()):
        curr_time = rospy.get_rostime().to_sec() - time0
        #callback_time = rospy.get_time()

        if flag.status:
            pass
        # Simulator delayc
        counter += 1

        # what are these varibales ?? go to EcuClass() 
        # a_his.append(ecu.u[0])
        # df_his.append(ecu.u[1])


        # if lowLevelDyn == True:
        #   servo_inp = (1 - T / Tf) * servo_inp + ( T / Tf )*df_his.pop(0)
        #   u = [a_his.pop(0), servo_inp]
        # else:
        #   u = [a_his.pop(0), df_his.pop(0)] # EA: remove the first element of the array and return it to you.
            #print "Applyied the first of this vector: ",df_his ,"\n "

        dt = curr_time - prev_time 
        if abs(ecu.duty_cycle) > 0.1:
            u = [ecu.duty_cycle, ecu.steer]
        
            vehicle_sim.vehicle_model(u, dt)


        else:
            if vehicle_sim.vx == 0.0:
                vehicle_sim.vx = 0.0
                vehicle_sim.vy = 0.0
                
            if vehicle_sim.omega <= 0.01:    
                vehicle_sim.omega = 0.0
                vehicle_sim.omega  = 0.0

        simStates.x      = vehicle_sim.x 
        simStates.y      = vehicle_sim.y 
        simStates.vx     = vehicle_sim.vx
        simStates.vy     = vehicle_sim.vy
        simStates.yaw    = vehicle_sim.yaw
        simStates.omega  = vehicle_sim.omega

        # vehicle_simStatesIDADA.x                         = vehicle_sim.x
        # vehicle_simStatesIDADA.y                         = vehicle_sim.y
        # vehicle_simStatesIDADA.longitudinal_velocity     = vehicle_sim.vx
        # vehicle_simStatesIDADA.lateral_velocity          = vehicle_sim.vy
        # vehicle_simStatesIDADA.heading                   = vehicle_sim.yaw
        # vehicle_simStatesIDADA.angular_velocity          = vehicle_sim.psiDot

        #if counter >= pub_rate/simulator_dt:
            # Publish input

        pub_vehicle_simulatorStates.publish(simStates)
        # pub_simulatorStatesIDIADA.publish(simStatesIDADA)
        # aux = SimulateSensors(sim, pub_sysOut)
        # hist_sens_noise.append(aux)
        
        counter = 0

        sensor_sim.fcam_update(vehicle_sim)
        sensor_sim.imu_update(vehicle_sim)
        sensor_sim.enc_update(vehicle_sim)
        # gps.update(sim)

        # sim.saveHistory()

        sensor_sim.fcam_pub()
        sensor_sim.imu_pub()
        sensor_sim.enc_pub()


        if pub_linkStates.get_num_connections() > 0:
            link_msg.pose.position.x = vehicle_sim.x + vehicle_sim_offset_x
            link_msg.pose.position.y = -(vehicle_sim.y + vehicle_sim_offset_y)
            link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -vehicle_sim.yaw))
            pub_linkStates.publish(link_msg)

            rospy.wait_for_service('/gazebo/set_model_state')
            #try:
            #   set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            #   resp = set_state( link_msg )

            #except rospy.ServiceException, e:
            #   print "Service call failed: %s" % e


        # # ESTO SIN COMENTAR ES COMO ESTABA ANTES...
        # counter += 1
        # if counter == 5:
        #   enc.enc_pub()
        #   counter = 0


        #time_since_last_callback = rospy.get_time() - callback_time

        #print time_since_last_callback

        prev_time = curr_time
        vehicle_sim.rate.sleep()

    # str         = 'test2'    
    # day         = 'TestPreFinal'    
    # num_test    = 'Noise'    
    # newpath     = '/home/auto/results_simu_test/'+day+'/'+num_test+'/'
    # print "Hi!"
    # print len(hist_sens_noise)
    # print newpath+'NoiseSens'
    # if not os.path.exists(newpath+'/'):
    #     os.makedirs(newpath+'/')  
    # np.save(newpath+'NoiseSens', hist_sens_noise)
    # np.save(newpath+'NoiseProces', vehicle_sim.noise_hist) 
    # np.save(newpath+'NoiseU', ecu.hist_noise)
    # quit()

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


def vehicle_model(vx,vy,omega,theta,delta,D):
    

    m = 2.424;
    rho = 1.225;
    lr = 0.1203;
    lf = 0.1377;
    cm0 = 10.1305;
    cm1 = 1.05294;
    C0 = 3.68918;
    C1 = 0.0306803;
    Cd_A = -0.657645;
    Caf = 1.3958
    Car = 1.6775
    Iz = 0.02
    
    
    dX = vx*cos(theta) - vy*sin(theta)
    dY = vx*sin(theta) + vy*cos(theta)
    dyaw = omega
    
    

    F_flat = 0
    Fry    = 0
    Frx    = (cm0 - cm1*vx)*D - C0*vx - C1 - (Cd_A*rho*vx**2)/2;

    eps = 0.00000001
    # if abs(vx)>0:
    Fry = -2.0*Car*atan((vy - lr*omega)/(vx+eps)) ;
    F_flat = 2.0*Caf*(delta - atan((vy+lf*omega)/(vx+eps)));

    dvx = (1/m)*(Frx - F_flat*sin(delta) + m*vy*omega);

    dvy = (1/m)*(F_flat*cos(delta) + Fry - m*vx*omega);

    domega = (1.0/Iz)*(lf*F_flat*cos(delta) - lr*Fry);
    

    return dvx, ddy, domega, dX, dY, dyaw





class Vehicle_Simulator(object):
    """ Object collecting GPS measurement data
    Attributes:
        Model params:
            1.L_f 2.L_r 3.m(car mass) 3.I_z(car inertial) 4.c_f(equivalent drag coefficient)
        States:
            1.x 2.y 3.vx 4.vy 5.ax 6.ay 7.psiDot
        States history:
            1.x_his 2.y_his 3.vx_his 4.vy_his 5.ax_his 6.ay_his 7.psiDot_his
        Simulator sampling time:
            1. dt
        Time stamp:
            1. time_his
    Methods:
        f(u):
            System model used to update the states
        pacejka(ang):
            Pacejka lateral tire modeling
    """
    def __init__(self):

        self.m = rospy.get_param("m")
        self.rho = rospy.get_param("rho")
        self.lr = rospy.get_param("lr")
        self.lf = rospy.get_param("lf")
        self.Cm0 = rospy.get_param("Cm0")
        self.Cm1 = rospy.get_param("Cm1")
        self.C0 = rospy.get_param("C0")
        self.C1 = rospy.get_param("C1")
        self.Cd_A = rospy.get_param("Cd_A")
        self.Caf = rospy.get_param("Caf")
        self.Car = rospy.get_param("Car")
        self.Iz = rospy.get_param("Iz")

        self.g = 9.81

        # with noise
        self.x      = rospy.get_param("simulator/init_x")
        self.y      = rospy.get_param("simulator/init_y")
        self.vx     = 0.0#rospy.get_param("init_vx")
        self.vy     = 0.0
        self.ax     = 0.0
        self.ay     = 0.0

        self.yaw        = rospy.get_param("simulator/init_yaw")
        # self.yaw        = 3.20797317234
        self.dist_mode  = rospy.get_param("simulator/dist_mode")
        self.mu_sf      = rospy.get_param("simulator/mu_sf")
        self.Cd         = rospy.get_param("simulator/Cd")
        self.A_car      = rospy.get_param("simulator/A_car")


        self.omega = 0.0

        self.x_his      = []
        self.y_his      = []
        self.vx_his     = []
        self.vy_his     = []
        self.ax_his     = []
        self.ay_his     = []
        self.omega_his  = []
        self.noise_hist = []

        self.dt         = rospy.get_param("simulator/dt")
        self.rate       = rospy.Rate(1.0/self.dt)
        # self.rate         = rospy.Rate(1.0)
        self.time_his   = []

        # Get process noise limits
        self.x_std           = rospy.get_param("simulator/x_std_pr")
        self.y_std           = rospy.get_param("simulator/y_std_pr")
        self.vx_std          = rospy.get_param("simulator/vx_std_pr")
        self.vy_std          = rospy.get_param("simulator/vy_std_pr")
        self.omega_std      = rospy.get_param("simulator/psiDot_std_pr")
        self.yaw_std         = rospy.get_param("simulator/psi_std_pr")
        self.n_bound         = rospy.get_param("simulator/n_bound_pr")


    # Go through each line of code ??   
    def vehicle_model(self,u, dt):

        x       = self.x
        y       = self.y
        ax      = self.ax
        ay      = self.ay
        vx      = self.vx
        vy      = self.vy
        yaw     = self.yaw
        omega   = self.omega
        self.dt = dt
    
    
        dX = vx*cos(yaw) - vy*sin(yaw)
        dY = vx*sin(yaw) + vy*cos(yaw)
        dyaw = omega
        
        

        F_flat = 0
        Fry    = 0
        Frx    = (self.Cm0 - self.Cm1*vx)*u[0] - self.C0*vx - self.C1 - (self.Cd_A*self.rho*vx**2)/2;
        
        eps = 0.0000001
        # if abs(vx)>0:
        Fry = -2.0*self.Car*atan((vy - self.lr*omega)/(vx+eps)) ;
        F_flat = 2.0*self.Caf*(u[1] - atan((vy+self.lf*omega)/(vx+eps)));

        dvx = (1/self.m)*(Frx - F_flat*sin(u[1]) + self.m*vy*omega);

        dvy = (1/self.m)*(F_flat*cos(u[1]) + Fry - self.m*vx*omega);

        domega = (1.0/self.Iz)*(self.lf*F_flat*cos(u[1]) - self.lr*Fry);



        n4 = max(-self.x_std*self.n_bound, min(self.x_std*0.66*(randn()), self.x_std*self.n_bound))
        n4 = 0
        self.x      += self.dt*(dX + n4)

        n5 = max(-self.y_std*self.n_bound, min(self.y_std*0.66*(randn()), self.y_std*self.n_bound))
        n5 = 0
        self.y      += self.dt*(dY + n5)

        n1 = max(-self.vx_std*self.n_bound, min(self.vx_std*0.66*(randn()), self.vx_std*self.n_bound))
        n1 = 0
        self.vx     += self.dt*(dvx + n1) 

        n2 = max(-self.vy_std*self.n_bound, min(self.vy_std*0.66*(randn()), self.vy_std*self.n_bound))
        n2 = 0
        self.vy     += self.dt*(dvy + n2)

        self.ax      = dvx 
        self.ay      = dvy

        n3 = max(-self.yaw_std*self.n_bound, min(self.yaw_std*0.1*(randn()), self.yaw_std*self.n_bound))
        n3 = 0
        self.yaw    += self.dt*(omega + 0)
        self.yaw     = wrap(self.yaw)   

        n6 = max(-self.omega_std*self.n_bound, min(self.omega_std*0.66*(randn()), self.omega_std*self.n_bound))
        n6 = 0
        self.omega     += self.dt*(domega + n6)
        self.omega      = - self.omega ## because the IMU installed in vehicle has z-axis downwards

        self.noise_hist.append([n1,n2,n6,n4,n5,n3])

    def saveHistory(self):
        self.x_his.append(self.x)
        self.y_his.append(self.y)
        self.vx_his.append(self.vx)
        self.vy_his.append(self.vy)
        self.ax_his.append(self.ax)
        self.ay_his.append(self.ay)
        self.omegae_his.append(self.omega)
        self.time_his.append(rospy.get_rostime().to_sec())


class vehicle_control_input(object):
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



class Sensor_Simulator():


    def __init__(self, fcam_freq_update, imu_freq_update, enc_freq_update, simulator_dt):

        """
        Only messages published which are required for the observer.
        """

        ### Fisheye camera publisher ###
        self.pub_fcam  = rospy.Publisher("/fused_cam_pose", Pose, queue_size=1)

        ### IMU twist publisher ###
        self.pub_twist = rospy.Publisher('/twist', Twist, queue_size=1)

        ### IMU Pose publisher ###
        self.pub_pose  = rospy.Publisher('/pose', Pose,  queue_size=1)

        ### Encoder wheel rpm publisher ###
        self.pub_enc       = rospy.Publisher('/wheel_rpm_feedback', Float32, queue_size=1)
        

        ### Parameter used for noise simulation in sensors ###
        self.x_std   = rospy.get_param("simulator/x_std")
        self.y_std   = rospy.get_param("simulator/y_std")
        self.vx_std      = rospy.get_param("simulator/vx_std")
        self.vy_std      = rospy.get_param("simulator/vy_std")
        self.yaw_std     = rospy.get_param("simulator/yaw_std")
        self.omega_std   = rospy.get_param("simulator/omega_std")
        self.n_bound = rospy.get_param("simulator/n_bound")

        ### FISHER EYE CAMERA ###
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        ### IMU ####
        self.roll        = 0.0
        self.pitch       = 0.0
        self.yaw_imu     = 0.0
        self.roll_rate   = 0.0
        self.pitch_rate  = 0.0
        self.yaw_rate    = 0.0
        self.ax          = 0.0
        self.ay          = 0.0
        self.az          = 0.0
        
        ### Encoder
        self.wheel_rpm   = 0.0 
        
        ### Message type used for publishing ###
        self.msg_fcam = Pose()
        self.msg_imu_twist = Twist()
        self.msg_imu_pose  = Pose()
        self.msg_enc       = Float32()

        ### Update rate of the simulated sensors ###
        self.counter_fcam  = 0
        self.thUpdate_fcam = (1.0 /  fcam_freq_update) / simulator_dt
        self.counter_imu  = 0
        self.thUpdate_imu = (1.0 /  imu_freq_update) / simulator_dt
        self.counter_enc  = 0
        self.thUpdate_enc = (1.0 /  enc_freq_update) / simulator_dt

    #### SIMULATE FISHEYE CAMERA ####
    def fcam_update(self,sim):
        n = max(-self.x_std*self.n_bound, min(self.x_std*randn(), self.x_std*self.n_bound))
        self.x = sim.x + n

        n = max(-self.y_std*self.n_bound, min(self.y_std*randn(), self.y_std*self.n_bound))
        self.y = sim.y + n

        self.yaw = wrap(sim.yaw + uniform(0, 0.1))

    def fcam_pub(self):
        if self.counter_fcam > self.thUpdate_fcam:
            self.counter_fcam = 0
            self.msg_fcam.position.x = self.x
            self.msg_fcam.position.y = self.y
            self.msg_fcam.orientation.z = self.yaw
            self.pub_fcam.publish(self.msg_fcam)
            # print "Update fcam"
        else:
            # print "Not update fcam"
            self.counter_fcam = self.counter_fcam + 1


    #### SIMULATE IMU ###
    def imu_update(self,sim):

        n = max(-self.yaw_std*self.n_bound, min(self.yaw_std*randn(), self.yaw_std*self.n_bound))
        self.yaw_imu = wrap(sim.yaw + n)

        n = max(-self.omega_std*self.n_bound, min(self.omega_std*randn(), self.omega_std*self.n_bound))
        self.omega = sim.omega + n

    def imu_pub(self):
        if self.counter_imu > self.thUpdate_imu:
            self.counter_imu = 0

            self.msg_imu_twist.linear.z      = self.omega 
            self.msg_imu_pose.orientation.z  = self.yaw_imu

            self.pub_twist.publish(self.msg_imu_twist)
            self.pub_pose.publish(self.msg_imu_pose)
            # print "Update fcam"
        else:
            # print "Not update fcam"
            self.counter_imu = self.counter_imu + 1


    #### SIMULATE MOTOR ENCODER ###
    def enc_update(self,sim):

        self.wheel_radius = 0.03*1.12178 #radius of wheel
        self.wheel_rpm    = sim.vx*60.0/(2*pi*self.wheel_radius)
        self.wheel_rpm    = uniform(-20,20)

    def enc_pub(self):
        if self.counter_enc > self.thUpdate_enc:
            self.counter_enc = 0

            self.msg_enc.data      = self.wheel_rpm 
            self.pub_enc.publish(self.msg_enc)
            # print "Update fcam"
        else:
            # print "Not update fcam"
            self.counter_enc = self.counter_enc + 1



"""
class GpsClass(object):
    def __init__(self, gps_freq_update, simulator_dt):
        self.pub  = rospy.Publisher("hedge_pos", hedge_pos, queue_size=1)
        self.x = 0.0
        self.y = 0.0
        self.x_std   = rospy.get_param("simulator/x_std")
        self.y_std   = rospy.get_param("simulator/y_std")
        self.n_bound = rospy.get_param("simulator/n_bound")

        self.msg = hedge_pos()
        self.counter  = 0
        self.thUpdate = (1.0 /  gps_freq_update) / simulator_dt

    def update(self,sim):
        n = max(-self.x_std*self.n_bound, min(self.x_std*randn(), self.x_std*self.n_bound))
        self.x = sim.x + n

        n = max(-self.y_std*self.n_bound, min(self.y_std*randn(), self.y_std*self.n_bound))
        self.y = sim.y + n

    def gps_pub(self):
        if self.counter > self.thUpdate:
            self.counter = 0
            self.msg.x_m = self.x
            self.msg.y_m = self.y
            self.pub.publish(self.msg)
            # print "Update GPS"
        else:
            # print "Not update GPS"
            self.counter = self.counter + 1












class fcamClass(object):
    def __init__(self, fcam_freq_update, simulator_dt):
        self.pub  = rospy.Publisher("fused_cam_pose", Pose, queue_size=1)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.x_std   = rospy.get_param("simulator/x_std")
        self.y_std   = rospy.get_param("simulator/y_std")
        self.n_bound = rospy.get_param("simulator/n_bound")

        self.msg = Pose()
        self.counter  = 0
        self.thUpdate = (1.0 /  gps_freq_update) / simulator_dt

    def fcam_update(self,sim):
        n = max(-self.x_std*self.n_bound, min(self.x_std*randn(), self.x_std*self.n_bound))
        self.x = sim.x + n

        n = max(-self.y_std*self.n_bound, min(self.y_std*randn(), self.y_std*self.n_bound))
        self.y = sim.y + n

        self.yaw = sim.yaw + uniform(0, 0.2)

    def fcam_pub(self):
        if self.counter > self.thUpdate:
            self.counter = 0
            self.msg.position.x = self.x
            self.msg.position.y = self.y
            self.msg.orientation.z = self.yaw
            self.pub.publish(self.msg)
            # print "Update fcam"
        else:
            # print "Not update fcam"
            self.counter = self.counter + 1




class EcuClass(object):
    def __init__(self):
        self.sub = rospy.Subscriber("ecu", ECU, self.ecu_callback, queue_size=1)
        self.u = [0.0, 0.0]
        self.du_0    = rospy.get_param("simulator/du_0")
        self.du_1    = rospy.get_param("simulator/du_1")
        self.u_bound = rospy.get_param("simulator/u_bound")
        self.hist_noise = []

    def ecu_callback(self,data):

        n1 = max(-self.du_0*self.u_bound, min(self.du_0*(2*rand()-1), self.du_0*self.u_bound))
        n2 = max(-self.du_1*self.u_bound, min(self.du_1*(2*rand()-1), self.du_1*self.u_bound))
        self.hist_noise.append([n1,n2])
        self.u = [data.motor + n1, data.servo + n2]


"""


class FlagClass(object):

    def __init__(self):
        self.flag_sub = rospy.Subscriber("flag", Bool, self.flag_callback, queue_size=1)
        self.status   = False
    def flag_callback(self,msg):
        self.status = msg.data

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
