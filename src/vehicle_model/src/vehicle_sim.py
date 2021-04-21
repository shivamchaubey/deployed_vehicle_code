#!/usr/bin/env python

from numpy import tan, arctan, cos, sin, pi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rospy
from vehicle_model.msg import simulatorStates
from std_msgs.msg import Bool, Float32
from numpy.random import randn,rand
import rosbag
import datetime
import os
import time


def main():
    print "[SIMULATOR] Is Low Level Dynamics active?: "
    rospy.init_node("simulator")
    # gps_freq_update = rospy.get_param("simulator/gps_freq_update")
    # simulator_dt    = rospy.get_param("simulator/dt")
    # lowLevelDyn     = rospy.get_param("simulator/lowLevelDyn")
    
    print ("[SIMULATOR] Is Low Level Dynamics active?: ")
    time0 = rospy.get_rostime().to_sec() 
    sim = Simulator(time0)
    # imu = ImuClass()
    # gps = GpsClass(gps_freq_update, simulator_dt)
    # enc = EncClass()
    ecu = EcuClass(time0)
    counter = 0

    t0 = rospy.Time.from_sec(time.time())
    record_sim_data = rospy.get_param('simulator/record_sim_data')
    # a_his   = [0.0]*int(rospy.get_param("simulator/delay_a")/rospy.get_param("simulator/dt"))
    # df_his  = [0.0]*int(rospy.get_param("simulator/delay_df")/rospy.get_param("simulator/dt"))

    
    pub_simulatorStates = rospy.Publisher('simulatorStates', simulatorStates, queue_size=1)
    simStates = simulatorStates()

    print('[SIMULATOR] The simulator is running!')
    print('\n')
    
    # servo_inp = 0.0
    # T  = simulator_dt
    # Tf = 0.07
    # try:
    record_data = {} 
    while not rospy.is_shutdown():
        # Simulator delay
        print (ecu.a,"ecu.a")
        u = [ecu.a,ecu.df]
        # a_his.append(ecu.u[0])
        # df_his.append(ecu.u[1])

        # if lowLevelDyn == True:
        #     servo_inp = (1 - T / Tf) * servo_inp + ( T / Tf )*df_his.pop(0)
        #     u = [a_his.pop(0), servo_inp]
        # else:
        #     u = [a_his.pop(0), df_his.pop(0)] # EA: remove the first element of the array and return it to you.
        #     #print "Applyied the first of this vector: ",df_his ,"\n "

        sim.f(u)
        t = rospy.Time.from_sec(time.time()) -t0
        seconds = t.to_sec() #floating point
        nanoseconds = t.to_nsec()

        # simStates.header.stamp.secc = nanoseconds
        simStates.header.stamp = t
        simStates.header.seq = counter
        simStates.header.frame_id = 'world'
        simStates.x      = sim.x
        simStates.y      = sim.y
        simStates.vx     = sim.vx
        simStates.vy     = sim.vy
        simStates.psi    = sim.yaw
        simStates.psiDot = sim.psiDot
        
        print ("Sim States = {}".format(simStates))
        # Publish input
        pub_simulatorStates.publish(simStates)

        # imu.update(sim)
        # gps.update(sim)
        # enc.update(sim)
        # gps.gps_pub()
        # imu.imu_pub()
        # enc.enc_pub()
        sim.rate.sleep()
        counter +=1
    # record_data = {'timestamp_ms':sim.time_his,'x_his':sim.x_his,'y_his':sim.y_his,'psi_his':sim.psi_his,'vx_his':sim.vx_his,'vy_his':sim.vy_his,'psiDot_his':sim.psiDot_his,'noise_hist':sim.noise_hist}
    # print ("record_datas",record_data)
    if record_sim_data == (True or true):
        sim.saveHistory()
    # except KeyboardInterrupt:
    #     sim.saveHistory()
    #     pass










class Simulator(object):
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
    def __init__(self,t0):


        # (119.27969348659003, 138.72030651340998)
        self.L_r    = 0.119
        self.L_f    = 0.139
        # self.L_f    = 0.125
        # self.L_r    = 0.125
        self.m      = 2.410 #including all the sensors and wire.
        self.I_z    = 0.03
        self.Cf     = 2*68
        self.Cr     = 2*71
        self.mu     = 0.05

        
        
        self.g = 9.81

        # with noise
        self.x      = 0.0
        self.y      = 0.0
        self.vx     = 0.0#rospy.get_param("init_vx")
        self.vy     = 0.0
        self.ax     = 0.0
        self.ay     = 0.0

        self.yaw        = 0.0
        self.dist_mode  = 0
        self.mu_sf      = 0.1
        self.Cd         = 0.020
        self.A_car      = 0.03


        self.psiDot = 0.0
        self.noise_mode = False
        self.x_his      = []
        self.y_his      = []
        self.vx_his     = []
        self.vy_his     = []
        self.ax_his     = []
        self.ay_his     = []
        self.psi_his    = []
        self.psiDot_his = []
        self.noise_hist = []
        self.dynamic_mode = False
        
        self.dt         = 0.005
        self.rate       = rospy.Rate(1.0/self.dt)
        # self.rate         = rospy.Rate(1.0)
        self.t0         = t0
        self.curr_time  = rospy.get_rostime().to_sec() -self.t0
        self.time_his   = []
        self.real_time_his   = []

        # Get process noise limits
        self.x_std           = 0.0
        self.y_std           = 0.0
        self.vx_std          = 0.0
        self.vy_std          = 0.0
        self.psiDot_std      = 0.0
        self.psi_std         = 0.0
        self.n_bound         = 0.0

        #Get sensor noise limits

        self.x_std_s           = 0.0
        self.y_std_s           = 0.0
        self.vx_std_s          = 0.0
        self.vy_std_s          = 0.0
        self.psiDot_std_s      = 0.0
        self.psi_std_s         = 0.0
        self.n_bound_s         = 0.0


    def f(self,u):
        a_F = 0.0
        a_R = 0.0
        self.dynamic_mode = False
        if abs(self.vx) > 0.01:
            #u[1] steering
            a_F = u[1] - arctan((self.vy + self.L_f*self.psiDot)/abs(self.vx))
            a_R = arctan((- self.vy + self.L_r*self.psiDot)/abs(self.vx))
            self.dynamic_mode = True
#             print ("Dynamic model")
        # FyF = self.pacejka(a_F)
        # FyR = self.pacejka(a_R)

        FyF = self.Cf * a_F
        FyR = self.Cr * a_R


        if abs(a_F) > 30.0/180.0*pi or abs(a_R) > 30.0/180.0*pi:
            print ("WARNING: Large slip angles in simulation")

        x       = self.x
        y       = self.y
        ax      = self.ax
        ay      = self.ay
        vx      = self.vx
        vy      = self.vy
        yaw     = self.yaw
        psiDot  = self.psiDot

        if self.noise_mode:

            dist = ( 10*self.Cd*1.225*self.A_car*(vx**2) + self.mu_sf*9.81*self.m)/self.m

        else:
            dist = self.mu*vx

        #print("disturbance: " , dist)


        #despreciem forces longitudinals i no fem l'aproximacio rara dels angles (respecte al pdf)
        if self.dist_mode == True:
            n4 = max(-self.x_std*self.n_bound, min(self.x_std*0.66*(randn()), self.x_std*self.n_bound))
        
            n5 = max(-self.y_std*self.n_bound, min(self.y_std*0.66*(randn()), self.y_std*self.n_bound))
        
            n1 = max(-self.vx_std*self.n_bound, min(self.vx_std*0.66*(randn()), self.vx_std*self.n_bound))

            n2 = max(-self.vy_std*self.n_bound, min(self.vy_std*0.66*(randn()), self.vy_std*self.n_bound))

        #n3 = 0.66*(randn());
            n3 = max(-self.psi_std*self.n_bound, min(self.psi_std*0.66*(randn()), self.psi_std*self.n_bound))

            n6 = max(-self.psiDot_std*self.n_bound, min(self.psiDot_std*0.66*(randn()), self.psiDot_std*self.n_bound))
        else:
            n1 = 0
            n2 = 0
            n3 = 0
            n4 = 0
            n5 = 0
            n6 = 0


        self.x      += self.dt*(cos(yaw)*vx - sin(yaw)*vy + n4)
        self.y      += self.dt*(sin(yaw)*vx + cos(yaw)*vy + n5)
        self.vx     += self.dt*(ax + psiDot*vy + n1 - dist) 
        self.vy     += self.dt*(ay - psiDot*vx + n2)

        # self.ax    = u[0]*cos(u[1]) - self.mu*vx - FyF/self.m*sin(u[1])  # front driven vehicle
        # self.ay    = u[0]*sin(u[1]) + 1.0/self.m*(FyF*cos(u[1])+FyR)
        self.ax      = u[0] - (FyF/self.m)*sin(u[1])  # front driven vehicle
        self.ay      = (1.0/self.m)*(FyF*cos(u[1])+FyR)
        self.yaw    += self.dt*(psiDot + n3)
        # self.yaw     = wrap(yaw)
        self.psiDot     += self.dt*((self.L_f*FyF*cos(u[1]) - self.L_r*FyR)/self.I_z + n6)



        self.curr_time  = rospy.get_rostime().to_sec() -self.t0
        self.time_his.append(self.curr_time)
        self.real_time_his.append(rospy.get_rostime().to_sec())
        self.noise_hist.append([n1,n2,n6,n4,n5,n3])
        self.x_his.append(x)
        self.y_his.append(y)
        self.vx_his.append(vx)
        self.vy_his.append(vy)
        self.ax_his.append(ax)
        self.ay_his.append(ay)
        self.psi_his.append(yaw)
        self.psiDot_his.append(psiDot)
        # self.noise_hist.append()
    def saveHistory(self):
        data = {'real_time':self.real_time_his,'timestamp_ms':self.time_his,'x_his':self.x_his,'y_his':self.y_his,'psi_his':self.psi_his,'vx_his':self.vx_his,'vy_his':self.vy_his,'psiDot_his':self.psiDot_his,'noise_hist':self.noise_hist}
        path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
        
        now = datetime.datetime.now()
        path = path + now.strftime("d%d_m%m_y%Y/")
        dt_string = now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")
        
        simulator_path = path + 'vehicle_simulator_his_'+ dt_string

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(simulator_path,data)

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
        self.curr_time_accel  = rospy.get_rostime().to_sec() - self.t0
        self.curr_time_steer  = rospy.get_rostime().to_sec() - self.t0

    def accel_callback(self,data):
        """Unpack message from sensor, ECU"""
        self.curr_time_accel = rospy.get_rostime().to_sec() - self.t0
        self.a  = data.data

    def steering_callback(self,data):
        self.curr_time_steer = rospy.get_rostime().to_sec() - self.t0
        self.df = data.data

def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle

if __name__ == '__main__':

    try:
        main()
    except rospy.ROSInterruptException:
        pass
