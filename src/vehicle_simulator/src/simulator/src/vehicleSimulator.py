#!/usr/bin/env python

##########################################################
##           IRI 1:10 Autonomous Car                    ##
##           ```````````````````````                    ##
##      Supervisor: Puig Cayuela Vicenc                 ##
##      Author:     Shivam Chaubey                      ##
##      Email:      shivam.chaubey1006@gmail.com        ##
##      Date:       18/04/2021                          ##
##########################################################

'''  
The vehicle simulator is based on the obtained model of the vehicle, the
sensor is simulated by inducing noises in the vehicle states. There are 6
states of the vehicle and only 5 measurements can be recorded with sensors.   
The simulator can be further improved if URDF file can be created for Gazebo simulator. 

'''
#----------------------------------------------------------------------------


import rospy
import geometry_msgs.msg
from simulator.msg import simulatorStates
from geometry_msgs.msg import Vector3, Quaternion, Pose, Twist, PoseStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Imu
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import atan, tan, cos, sin, pi, atan2
from numpy.random import randn,rand
from random import randrange, uniform
from tf import transformations
import numpy as np
import pdb


def main():
    
    ## node name
    rospy.init_node("simulator")

    ## update rate of vehicle simulator
    rate       = rospy.get_param("simulator/publish_frequency")
    simulator_dt    = 1.0/rate
    lowLevelDyn     = rospy.get_param("simulator/lowLevelDyn")
    
    ## update rate of sensor simulator
    fcam_freq_update = rospy.get_param("simulator/fcam_freq_update")
    imu_freq_update = rospy.get_param("simulator/imu_freq_update")
    enc_freq_update = rospy.get_param("simulator/enc_freq_update")

    ## vehicle state initialization 
    vehicle_sim_offset_x    = rospy.get_param("simulator/init_x")
    vehicle_sim_offset_y    = rospy.get_param("simulator/init_y")
    vehicle_sim_offset_yaw  = rospy.get_param("simulator/init_yaw")

    ''' Vehicle deadzone. The vehicle doesn't move at certain duty cycle.
        In our case it's -0.05 to 0.05.'''
    duty_th                 = rospy.get_param("duty_th")

    time0 = rospy.get_rostime().to_sec()

    ## vehicle simulator class initialized
    vehicle_sim = Vehicle_Simulator()
    
    ## vehicle sensors class initialized
    sensor_sim = Sensor_Simulator(vehicle_sim, fcam_freq_update, imu_freq_update, enc_freq_update, simulator_dt)
    
    ## subscribe to the ecu class for updating the control inputs.
    ecu = vehicle_control_input(time0)

    ## publish the link message for gazebo simulation
    link_msg = ModelState()

    ## For gazebo simulation states
    link_msg.model_name = "seat_car"
    link_msg.reference_frame = "world"

    link_msg.pose.position.z = 0.031
    link_msg.twist.linear.x = 0.0
    link_msg.twist.linear.y = 0
    link_msg.twist.linear.z = 0

    link_msg.twist.angular.x = 0.0
    link_msg.twist.angular.y = 0
    link_msg.twist.angular.z = 0


    ## Publishing actual vehicle states
    pub_vehicle_simulatorStates = rospy.Publisher('vehicle_simulatorStates', simulatorStates, queue_size=1)

    ## message type for vehicle states
    simStates = simulatorStates()

    counter = 0

    prev_time = rospy.get_rostime().to_sec() - time0

    while not (rospy.is_shutdown()):
        curr_time = rospy.get_rostime().to_sec() - time0


        u = [ecu.duty_cycle, ecu.steer]

        # dt = curr_time - prev_time 

        if abs(u[0]) > duty_th:            
            vehicle_sim.vehicle_model(u, simulator_dt)
        
        (simStates.vx, simStates.vy, simStates.omega, simStates.x, simStates.y, simStates.yaw) = vehicle_sim.states
        
        # if abs(u[0]) <= duty_th:
        #         #     # vehicle_sim.vehicle_model(u, simulator_dt)
        #             # if vehicle_sim.vx <= 0.01 :
        #             vehicle_sim.vx = 0.000001 
        #             vehicle_sim.vy = 0.000001
        #             vehicle_sim.omega = 0.000001 
        #             simStates.vx = 0.000001
        #             simStates.vy = 0.000001
        #             simStates.omega = 0.000001

        # simStates.x      = vehicle_sim.x 
        # simStates.y      = vehicle_sim.y 
        # simStates.vx     = vehicle_sim.vx
        # simStates.vy     = vehicle_sim.vy
        # simStates.yaw    = vehicle_sim.yaw
        # simStates.omega  = vehicle_sim.omega

        print("\n <<<< STATS >>>>")
        print("u", u) 
        print("simStates", simStates)

        #if counter >= pub_rate/simulator_dt:
            # Publish input

        pub_vehicle_simulatorStates.publish(simStates)
        
        counter = 0

        sensor_sim.fcam_update(vehicle_sim)
        sensor_sim.imu_update(vehicle_sim)
        sensor_sim.enc_update(vehicle_sim)
        sensor_sim.lidar_update(vehicle_sim)
        
        sensor_sim.fcam_pub()
        sensor_sim.imu_pub()
        sensor_sim.enc_pub()
        sensor_sim.lidar_pub()



        # if pub_linkStates.get_num_connections() > 0:
        #     link_msg.pose.position.x = vehicle_sim.x + vehicle_sim_offset_x
        #     link_msg.pose.position.y = -(vehicle_sim.y + vehicle_sim_offset_y)
        #     link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -vehicle_sim.yaw))
        #     pub_linkStates.publish(link_msg)

        #     rospy.wait_for_service('/gazebo/set_model_state')
            #try:
            #   set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            #   resp = set_state( link_msg )

            #except rospy.ServiceException, e:
            #   print "Service call failed: %s" % e


        #time_since_last_callback = rospy.get_time() - callback_time

        #print time_since_last_callback

        prev_time = curr_time
        vehicle_sim.rate.sleep()
        counter += 1 


def wrap(angle):
    eps = 0.00
    if angle < -np.pi + eps:
        w_angle = 2 * np.pi + angle -eps
    elif angle > np.pi - eps :
        w_angle = angle - 2 * np.pi + eps 
    
    else:
        w_angle = angle

    return w_angle




def vehicle_model(vx,vy,omega,theta,delta,D):
    
    m = 2.424;
    rho = 1.225;
    lr = 0.1203;
    lf = 0.1377 ;
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
    """ Object for vehicle simulator, the derived model used from parameter identification is used. 
    Attributes:
        Model params:
            1.m : mass of the vehicle, change the mass in case battery is being changed
            2.rho : Air density
            3.lr : Distance from front wheel to center of mass.
            4.lf : Distance from rear wheel to center of mass. 
            5.Cm0 : Co-efficient of driving motor related to force
            6.Cm1 : Co-efficient of driving motor related to velocity
            7.C0 : Rolling parameter coefficient 
            8.C1 : Resistive Frictional Co-efficient
            9.Cd_A : Aerodynamic Drag coefficient with frontal cross section area.
            10.Caf : Front wheel stiffness.
            11.Car : Rear wheel stiffness.
            12.Iz : Moment of inertia in z-axis. 
        States:
            1.v_x : Longitudinal velocity 
            2.v_y : Lateral velocity 
            3.omega : angular rate
            4.X : Global position of the vehicle in X - axis.
            5.Y : Global position of the vehicle in Y - axis.
            6.theta (yaw) : Global orientation of the vehicle. 
        Simulator sampling time:
            1. dt
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


        self.omega = 0.0

        self.x_his      = []
        self.y_his      = []
        self.vx_his     = []
        self.vy_his     = []
        self.ax_his     = []
        self.ay_his     = []
        self.omega_his  = []
        self.noise_hist = []

        self.rate       = rospy.Rate(rospy.get_param("simulator/publish_frequency"))
        self.dt         = 1.0/rospy.get_param("simulator/publish_frequency")
        
        self.time_his   = []

        # Get process noise limits
        self.x_std           = rospy.get_param("simulator/x_std_pr")
        self.y_std           = rospy.get_param("simulator/y_std_pr")
        self.vx_std          = rospy.get_param("simulator/vx_std_pr")
        self.vy_std          = rospy.get_param("simulator/vy_std_pr")
        self.omega_std       = rospy.get_param("simulator/psiDot_std_pr")
        self.yaw_std         = rospy.get_param("simulator/psi_std_pr")
        self.n_bound         = rospy.get_param("simulator/n_bound_pr")
        self.disturbance_on  = rospy.get_param("simulator/disturbance_on")
        self.disturbance     = np.array([0, 0, 0, 0, 0, 0]).T
        self.states          = np.array([self.vx, self.vy, self.omega, self.x, self.y, self.yaw])


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
        
        # eps = 0.0000001
        eps = 0.0

        if abs(vx)>0.4:
            self.Caf = 40.62927783;
            self.Car = 69.55846999;
            self.Iz = 1.01950479;
        else:

            self.Caf = 1.3958;
            self.Car = 1.6775;
            self.Iz = 1.01950479;

        if abs(vx)> 0:
            Fry = -2.0*self.Car*np.arctan((vy - self.lr*omega)/abs((vx+eps))) ;
            F_flat = 2.0*self.Caf*(u[1] - np.arctan((vy+self.lf*omega)/abs((vx+eps))));

        # if abs(vx)> 0.00001:
        #     Fry = -2.0*self.Car*((vy - self.lr*omega)/vx+eps) ;
        #     F_flat = 2.0*self.Caf*(u[1] - ((vy+self.lf*omega)/vx+eps));

        # if u[0]>0:
        #     Fry = -2.0*self.Car*np.arctan((vy - self.lr*omega)/(vx+eps)) ;
        #     F_flat = 2.0*self.Caf*(u[1] - np.arctan((vy+self.lf*omega)/(vx+eps)));


        # if u[0]<0:
        #     Fry = -2.0*self.Car*np.arctan((vy - self.lr*omega)/(abs(vx)+eps)) ;
        #     F_flat = 2.0*self.Caf*(u[1] - np.arctan((vy+self.lf*omega)/(abs(vx)+eps)));


        # if abs(vx)>0:
        #     Fry = -2.0*self.Car*atan((vy - self.lr*omega)/(abs(vx)+eps)) ;
        #     F_flat = 2.0*self.Caf*(u[1] - atan((vy+self.lf*omega)/(abs(vx)+eps)));


        dvx = (1/self.m)*(Frx - F_flat*sin(u[1]) + self.m*vy*omega);

        print("\n <<< LATERAL FORCES >>>")
        print("input",u) 
        print("F_flat", F_flat, "Fry", Fry)
        dvy = (1/self.m)*(F_flat*cos(u[1]) + Fry - self.m*vx*omega);
        print("F_flat", F_flat, "Fry", Fry, "F_flat*cos(u[1])", F_flat*cos(u[1]), "self.m*vx*omega", self.m*vx*omega)  
        
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
        # self.yaw     = wrap(self.yaw)   

        n6 = max(-self.omega_std*self.n_bound, min(self.omega_std*0.66*(randn()), self.omega_std*self.n_bound))
        n6 = 0
        self.omega     += self.dt*(domega + n6)
        
        self.states          = np.array([self.vx, self.vy, self.omega, self.x, self.y, self.yaw])

        if self.disturbance_on == True:
            self.w_vx    = max(-self.vx_std*self.n_bound, min(self.vx_std*(randn()), self.vx_std*self.n_bound))  
            self.w_vy    = max(-self.vy_std*self.n_bound, min(self.vy_std*(randn()), self.vy_std*self.n_bound)) 
            self.w_omega = max(-self.omega_std*self.n_bound, min(self.omega_std*(randn()), self.omega_std*self.n_bound)) 
            self.w_X     = max(-self.x_std*self.n_bound, min(self.x_std*(randn()), self.x_std*self.n_bound))
            self.W_Y     = max(-self.y_std*self.n_bound, min(self.y_std*(randn()), self.y_std*self.n_bound))
            self.W_yaw   = max(-self.yaw_std*self.n_bound, min(self.yaw_std*0.1*(randn()), self.yaw_std*self.n_bound))

            self.disturbance    = np.array([self.w_vx,self.w_vy,self.w_omega,self.w_X,self.W_Y,self.W_yaw]).T

            self.states = self.states +  self.disturbance

        # self.noise_hist.append([n1,n2,n6,n4,n5,n3])


class vehicle_control_input(object):
    """ Object collecting CMD command data
    Attributes:
        Input command:
            1.duty_cycle: duty cycle 2.steer : steering angle
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
    """ Object for vehicle sensor simulation, the sensors are simulated using the obtained states from the model.
    Attributes:
        sensor initialize:
            1. vehicle_sim : States from the vehicle simulator
            2. fcam_freq_update : update frequency of the fisheye camera
            3. imu_freq_update : update frequency of the IMU 
            4. enc_freq_update : update frequency of the motor encoder
            5. simulator_dt : update frequency of the vehicle states from the simulator

        Output:
            1.v_x + noise : Longitudinal velocity 
            2.omega + noise : angular rate
            3.X + noise : Global position of the vehicle in X - axis.
            4.Y + noise : Global position of the vehicle in Y - axis.
            5.theta (yaw) + noise : Global orientation of the vehicle. 

        Simulator sampling time:
            1. dt
    """

    def __init__(self, vehicle_sim, fcam_freq_update, imu_freq_update, enc_freq_update, simulator_dt):

        """
        Only messages published which are required for the observer. 
        5 states can only be measured, lateral velocity (v_y) can not be measured from any sensor.
        """

        ### Fisheye camera publisher ###
        self.pub_fcam  = rospy.Publisher("/fused_cam_pose", Pose, queue_size=1)

        ### Lidar publisher ###
        self.pub_lidar  = rospy.Publisher('/slam_out_pose', PoseStamped, queue_size=1)

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
        self.x = vehicle_sim.x
        self.y = vehicle_sim.y
        self.yaw = vehicle_sim.yaw

        ### IMU ####
        self.roll        = 0.0
        self.pitch       = 0.0
        self.yaw_imu     = vehicle_sim.yaw
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
        self.msg_lidar = PoseStamped()
        self.msg_imu_twist = Twist()
        self.msg_imu_pose  = Pose()
        self.msg_enc       = Float32()

        ### Update rate of the simulated sensors ###
        self.counter_fcam  = 1
        self.thUpdate_fcam = (1.0 /  fcam_freq_update) / simulator_dt
        self.counter_imu  = 1
        self.thUpdate_imu = (1.0 /  imu_freq_update) / simulator_dt
        self.counter_enc  = 1
        self.thUpdate_enc = (1.0 /  enc_freq_update) / simulator_dt

    #### SIMULATE FISHEYE CAMERA ####
    def fcam_update(self,sim):
        '''
        fisheye camera sensor simulation: 
        Input:  Takes input states X, Y, theta from vehicle simulator class.  
        Output: Provides only X, Y, theta + noises
        '''
        n = max(-self.x_std*self.n_bound, min(self.x_std*randn(), self.x_std*self.n_bound))
        # n = 0
        self.x = sim.x + n

        n = max(-self.y_std*self.n_bound, min(self.y_std*randn(), self.y_std*self.n_bound))
        # n = 0
        self.y = sim.y + n

        n = max(-self.yaw_std*self.n_bound, min(self.yaw_std*randn(), self.yaw_std*self.n_bound))

        # self.yaw = sim.yaw + n
        self.yaw = wrap(sim.yaw) + n

    def fcam_pub(self):
            '''
            For publishing at desired frequency
            '''
        # if self.counter_fcam >= self.thUpdate_fcam:
            self.counter_fcam = 0
            self.msg_fcam.position.x = self.x
            self.msg_fcam.position.y = self.y
            self.msg_fcam.orientation.z = self.yaw
            self.pub_fcam.publish(self.msg_fcam)
            
        # else:
            
        #     self.counter_fcam = self.counter_fcam + 1


    #### SIMULATE LIDAR ####
    def lidar_update(self,sim):
        '''
        LIDAR sensor simulation: 
        Input:  Takes input states X, Y, theta from vehicle simulator class.  
        Output: Provides only X, Y, theta + noises
        '''

        n = max(-self.x_std*self.n_bound, min(self.x_std*randn(), self.x_std*self.n_bound))
        # n = 0
        self.x = sim.x + n

        n = max(-self.y_std*self.n_bound, min(self.y_std*randn(), self.y_std*self.n_bound))
        # n = 0
        self.y = sim.y + n

        n = max(-self.yaw_std*self.n_bound, min(self.yaw_std*randn(), self.yaw_std*self.n_bound))

        # self.yaw = sim.yaw + n
        self.yaw = wrap(sim.yaw) + n

    def lidar_pub(self):

            '''
            For publishing at desired frequency
            '''

        # if self.counter_fcam >= self.thUpdate_fcam:
            self.counter_lidar = 0
            ## -ve direction is assigned because the real measurement has opposite direction of the vehicle coordinate.
            self.msg_lidar.pose.position.x = -self.x
            self.msg_lidar.pose.position.y = -self.y

            quaternion = transformations.quaternion_from_euler(0, 0, self.yaw)
            self.msg_lidar.pose.orientation.x = quaternion[0]
            self.msg_lidar.pose.orientation.y = quaternion[1]
            self.msg_lidar.pose.orientation.z = quaternion[2]
            self.msg_lidar.pose.orientation.w = quaternion[3]        
            self.pub_lidar.publish(self.msg_lidar)


    #### SIMULATE IMU ###
    def imu_update(self,sim):

        '''
        IMU sensor simulation: 
        Input:  Takes input states yaw (theta), omega (angular rate) from vehicle simulator class.  
        Output: Provides only yaw, omega + noises
        '''

        n = max(-self.yaw_std*self.n_bound, min(self.yaw_std*randn(), self.yaw_std*self.n_bound))
        # n = 0
        self.yaw_imu = wrap(sim.yaw + n)

        n = max(-self.omega_std*self.n_bound, min(self.omega_std*randn(), self.omega_std*self.n_bound))
        # n = 0
        self.omega = sim.omega + n
        self.pub_pose.publish(self.msg_imu_pose)


    def imu_pub(self):
        
            '''
            For publishing at desired frequency
            '''

        # if self.counter_imu >= self.thUpdate_imu:
        #     self.counter_imu = 0

            self.msg_imu_twist.angular.z      = -self.omega ## because the IMU installed in the vehicle has z-axis downwards
            self.msg_imu_pose.orientation.z  = self.yaw_imu

            self.pub_twist.publish(self.msg_imu_twist)
            self.pub_pose.publish(self.msg_imu_pose)
            # 
        # else:
            
        #     self.counter_imu = self.counter_imu + 1


    #### SIMULATE MOTOR ENCODER ###
    def enc_update(self,sim):

        '''
        Motor encoder sensor simulation: 
        Input:  Takes input states v_x (longitudinal velocity).
        Output: Provides wheel rpm + noises.
        '''

        n = max(-self.vx_std*self.n_bound, min(self.vx_std*randn(), self.vx_std*self.n_bound))

        self.wheel_radius = 0.03*1.12178 #radius of wheel
        self.wheel_rpm    = (sim.vx + n)*60.0/(2*pi*self.wheel_radius)# + uniform(-20,20)
        
        

    def enc_pub(self):
        
            '''
            For publishing at desired frequency
            '''

        # if self.counter_enc >= self.thUpdate_enc:
            self.counter_enc = 0

            self.msg_enc.data      = self.wheel_rpm 
            self.pub_enc.publish(self.msg_enc)

        # else:
            
        #     self.counter_enc = self.counter_enc + 1

class FlagClass(object):

    '''
    This message can be used for switching some desired states, or for fault diagnosis.
    '''

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
