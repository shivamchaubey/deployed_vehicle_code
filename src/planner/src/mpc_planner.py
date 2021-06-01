#!/usr/bin/env python
"""
    File name: Online Planner-LPV-MPC.py
    Author: Eugenio Alcala
    Email: eugenio.alcala@upc.edu.edu
    Date: 25/07/2019
    Python Version: 2.7.12
"""

import os
import sys
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
sys.path.append(sys.path[0]+'/Utilities')
import datetime
import rospy
from trackInitialization import Map
from planner.msg import My_Planning
from sensor_fusion.msg import sensorReading
from controller.msg import  Racing_Info, states_info
from std_msgs.msg import Bool, Float32
import time
import numpy as np
from numpy import hstack
import scipy.io as sio
import pdb
import pickle
from osqp_pathplanner import Path_planner_MPC

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from scipy import signal
from math import pi, isnan

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})



class vehicle_control():
    """ Object collecting CMD command data
    Attributes:
        Input command:
            1.a 2.df
    """
    def __init__(self):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """
        rospy.Subscriber('control/accel', Float32, self.accel_callback, queue_size=1)
        rospy.Subscriber('control/steering', Float32, self.steering_callback, queue_size=1)

        # ECU measurement
        self.duty_cycle  = 0.0 #dutycyle
        self.steer = 0.0


    def accel_callback(self,data):
        """Unpack message from sensor, ECU"""
        self.duty_cycle  = data.data
        if abs(self.duty_cycle) < 0.05:
            self.duty_cycle = 0.0

    def steering_callback(self,data):
        self.steer = data.data
        # if self.steer == 0.0:
        #     self.steer = 0.001


########## current state estimation from observer #############
class EstimatorData(object):
    """Data from estimator"""
    def __init__(self):

        print "waiting for estimator information"
        rospy.wait_for_message('est_state_info', sensorReading)

        rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback)
        self.CurrentState = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print "Subscribed to observer"
    
        self.map             = Map() 
        self.states_X =0
        self.states_Y =0
        self.states_yaw =0
        self.states_vx =0
        self.states_vy =0
        self.states_omega =0
        
        self.states_ey =0
        self.states_epsi =0
        self.states_s =0

        self.states = np.array([self.states_vx, self.states_vy, self.states_omega, 0, 0]).T


    def estimator_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.states_X = msg.X
        self.states_Y = msg.Y
        self.states_yaw = (msg.yaw + pi) % (2 * pi) - pi
        self.states_vx = msg.vx
        self.states_vy = msg.vy
        self.states_omega = msg.yaw_rate

        S_realDist, ey, epsi, insideTrack = self.map.getLocalPosition(msg.X, msg.Y, self.states_yaw)

        self.states_ey = ey
        self.states_epsi = epsi
        self.states_s = S_realDist


        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, self.states_yaw]).T


        self.states = np.array([self.states_vx, self.states_vy, self.states_omega, ey, epsi]).T


########## current state of the vehicle from controller #############
class vehicle_state(object):
    """Data from estimator"""
    def __init__(self):


        rospy.Subscriber('control/states_info', states_info, self.states_callback)
        self.map             = Map() 
        self.states_X =0
        self.states_Y =0
        self.states_yaw =0
        self.states_vx =0
        self.states_vy =0
        self.states_omega =0
        self.states_ey =0
        self.states_epsi =0
        self.states_s =0
        self.states = np.array([self.states_vx, self.states_vy, self.states_omega, self.states_ey, self.states_epsi]).T
        print "Subscribed to observer"
    
    def states_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.states_X = msg.X
        self.states_Y = msg.Y
        self.states_yaw = msg.yaw
        self.states_vx = msg.vx
        self.states_vy = msg.vy
        self.states_omega = msg.omega
        self.states_ey = msg.ey
        self.states_epsi = msg.epsi
        self.states_s = msg.s
        
        S_realDist, ey, epsi, insideTrack = self.map.getLocalPosition(msg.X, msg.Y, wrap(msg.yaw))

        # self.states = np.array([self.states_vx, self.states_vy, self.states_omega, self.states_ey, self.states_epsi]).T
        self.states = np.array([self.states_vx, self.states_vy, self.states_omega, ey, epsi]).T



### wrap the angle between [-pi,pi] ###
def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def main():

    # Initializa ROS node
    rospy.init_node("MPC_Planner")

    planning_refs   = rospy.Publisher('My_Planning', My_Planning, queue_size=1)

    map             = Map()  

    refs            = My_Planning()

    refs.counter    = []    
    refs.x_d        = []
    refs.y_d        = []
    refs.psi_d      = []
    refs.vx_d       = []
    refs.vy_d       = []
    refs.omega_d    = []
    refs.epsi_d     = []
    refs.ey_d       = []
    refs.s_d        = []
    refs.curv_d     = []


    HW              = rospy.get_param("/trajectory_planner/halfWidth")
    loop_rate       = rospy.get_param("/trajectory_planner/Hz") # 20 Hz (50 ms)
    planner_dt              = 1.0/loop_rate
    rate            = rospy.Rate(loop_rate)
    
    N   = rospy.get_param("/trajectory_planner/N")
    
    loop_rate_control = 50.0
    planner_type = rospy.get_param("/trajectory_planner/Mode")
    
    if planner_type == 2:
        racing_info     = RacingDataClass()
        estimatorData   = EstimatorData()
        loop_rate_control = rospy.get_param("/control/Hz")
        control_input = vehicle_control()
    
    if planner_type == 3:
        racing_info     = RacingDataClass()
        estimatorData   = EstimatorData()
        mode            = "simulations"
        control_input = vehicle_control()

    else:   # planner_type mode
        mode            = "simulations"
    #     racing_info     = RacingDataClass()


    first_it        = 1

    Xlast           = 0.0
    Ylast           = 0.0
    Thetalast       = 0.0

    Counter         = 0

    save_reference = True
    if save_reference == True:
        ALL_LOCAL_DATA  = np.zeros((1000,7))       # [vx vy psidot ey psi udelta uaccel]
        References      = np.zeros((1000,5))
        
        refs_his        = {"planner_dt": planner_dt, "planner_N": N , "counter": [], "x_t": [], "y_t": [], "x_d": [], "y_d": [], "psi_d": [], "vx_d": [], "vy_d": [], "omega_d": [], "epsi_d": [], "ey_d": [], "s_d": [], "curv_d": []}
     
        refs_his_counter    = []
        refs_his_x_t    = []
        refs_his_y_t    = []
        refs_his_x_d    = []
        refs_his_y_d    = []
        refs_his_psi_d  = []
        refs_his_vx_d   = []
        refs_his_vy_d   = []
        refs_his_omega_d    = []
        refs_his_epsi_d = []
        refs_his_ey_d   = []
        refs_his_s_d    = []
        refs_his_curv_d = []

    ELAPSD_TIME     = np.zeros((1000,1))

    TimeCounter     = 0
    PlannerCounter  = 0
    start_LapTimer  = datetime.datetime.now()


    if rospy.get_param("/trajectory_planner/Visualization") == 1:
        fig, axtr, line_tr, line_pred, line_trs, line_cl, line_gps_cl, rec, rec_sim = InitFigure_XY( map, mode, HW )


    #####################################################################


    # planner_dt = 0.05
    Vx_ref              = rospy.get_param("/trajectory_planner/vel_ref")

    Planner  = Path_planner_MPC(N, Vx_ref, planner_dt, map)

    # Planner  = LPV_MPC_Planner(Q, R, dR, L, N, planner_dt, map, "OSQP")

    #####################################################################

    # Filter to be applied.
    b_filter, a_filter = signal.ellip(4, 0.01, 120, 0.125)  


    LPV_X_Pred      = np.zeros((N,5))
    Xref            = np.zeros(N+1)
    Yref            = np.zeros(N+1)
    Thetaref        = np.zeros(N+1)
    xp              = np.zeros(N)
    yp              = np.zeros(N)
    yaw             = np.zeros(N)    

    x_his          = []
    y_his          = []
    SS              = np.zeros(N+1,) 

    GlobalState     = np.zeros(6)
    LocalState      = np.zeros(5)    

    print "Starting iteration"
    while (not rospy.is_shutdown()):  
        
        t0 = time.time()
        
        ###################################################################################################
        # GETTING INITIAL STATE:
        ###################################################################################################   

        if (planner_type == 2) or (planner_type == 3):
            # Read Measurements
            # SS[0] = estimatorData.states_s
            # GlobalState[:] = estimatorData.CurrentState                 # The current estimated state vector [vx vy w x y psi]
            # LocalState[:]  = np.array([ GlobalState[0], GlobalState[1], GlobalState[2], estimatorData.states_ey, estimatorData.states_epsi ]) # [vx vy w ey epsi]
            # S_realDist, LocalState[4], LocalState[3], insideTrack = map.getLocalPosition(GlobalState[3], GlobalState[4], GlobalState[5])
            LocalState[:] = estimatorData.states                 # The current estimated state vector [vx vy w x y psi]
            # u = np.array([control_input.steer, control_input.duty_cycle])
            print "control_input.steer", control_input.steer, "control_input.duty_cycle", control_input.duty_cycle
            u = np.zeros([N, 2])
            u[:,0] = control_input.steer
            u[:,1] = control_input.duty_cycle
            SS[0] = estimatorData.states_s

            # if LocalState[0] < 0.1:
            #     LocalState[0] = 1.5
            # 
            # if first_it == 1:
            #     SS = np.ones(N+1,)*S_realDist

        else:
            SS[0] = SS[1]
            LocalState[:] = np.array([0.5, 0, 0, 0, 0])
            S_realDist, LocalState[4], LocalState[3], insideTrack = map.getLocalPosition(xp[0], yp[0], yaw[0])
            
        ###################################################################################################
        # OPTIMIZATION:
        ###################################################################################################             

        # if Counter == 3:
        #     break

        if first_it < 2:
        
            x0 = LocalState[:]        # Initial planning state
            print "x0",x0
            duty_cycle = 0.0
            delta = 0.0
            xx, uu = predicted_vectors_generation(N, x0, duty_cycle, delta, planner_dt)
           
            Planner.uPred = uu
            Planner.xPred = xx[:,:5]

            Planner.uminus1 = Planner.uPred[0,:] 

            first_it += 1

        else:                               


            if first_it == 2:
                

                print "MPC setup"

                A_L, B_L, C_L = Planner.LPVPrediction_setup()

                if planner_type == 1:
                    Planner.MPC_setup(A_L, B_L, Planner.uPred, Planner.xPred[1,:], Vx_ref) 

                if planner_type == 2 or planner_type == 3:

                    
                    # SS = np.zeros(N+1,)
                    
                    Planner.MPC_setup(A_L, B_L, Planner.uPred, LocalState[:], Vx_ref) 


                first_it += 1



            ### For algorithm testing or finding a shortest route for racing
            if planner_type == 1:
               
                print "Planner.uPred[0,:]", Planner.uPred[0,:]

                print "MPC update"
                A_L, B_L, C_L = Planner.LPVPrediction( Planner.xPred[1,:], SS[:] , Planner.uPred)    
            

                Planner.MPC_update(A_L, B_L, Planner.xPred[1,:]) 
                Planner.MPC_solve()


            ## real time path planner ###
            if planner_type == 2 or planner_type == 3:


                
                print "MPC update"


                # if first_it == 3:
                #     print "init"
                #     A_L, B_L, C_L = Planner.LPVPrediction( Planner.xPred[1,:], SS[:] ,  Planner.uPred )    
            
                #     Planner.MPC_update(A_L, B_L, Planner.xPred[1,:]) 
                #     Planner.MPC_solve()
                #     first_it += 1
                        
                
                A_L, B_L, C_L = Planner.LPVPrediction( LocalState[:]  , SS[:] , u)    
                
                Planner.MPC_update(A_L, B_L, LocalState[:]) 
                Planner.MPC_solve()
               
            print "time taken to solve osqp", time.time() - t0
        print "Planner.uPred[0,:]", Planner.uPred[0,:]
        print "feasible", Planner.feasible
        # print "predicted", Planner.xPred
        # print "Planner.uPred ", Planner.uPred
        Planner.uminus1 = Planner.uPred[0,:] 

        if Planner.feasible == 0:
            print "LocalState[:]", LocalState[:]  , "SS[:]"  , SS[:]
            # print "predicted", Planner.xPred
            # print "Planner.uPred ", Planner.uPred
            Planner.uPred   = np.zeros((Planner.N, Planner.nu))
            Planner.uminus1 = Planner.uPred[0,:]


        if Planner.feasible != 0 :
            ###################################################################################################
            ###################################################################################################

            # print "Ey: ", Planner.xPred[:,3] 

            
            # Saving current control actions to perform then the slew rate:
            # Planner.OldSteering.append(Planner.uPred[0,0]) 
            # Planner.OldAccelera.append(Planner.uPred[0,1])

            # pdb.set_trace()

            #####################################
            ## Getting vehicle position:
            #####################################
            Xref[0]     = Xlast
            Yref[0]     = Ylast
            Thetaref[0] = Thetalast

            # print "SS[0] = ", S_realDist

            # SS[0] = S_realDist
            # SS  = np.zeros(N+1,) 

            for j in range( 0, N ):
                PointAndTangent = map.PointAndTangent         
                
                curv            = Curvature( SS[j], PointAndTangent )

                # print "Planner.xPred[j,0]", Planner.xPred[j,0], 'Planner.xPred[j,4]', Planner.xPred[j,4], 'Planner.xPred[j,3]', Planner.xPred[j,3] 
                SS[j+1] = ( SS[j] + ( ( Planner.xPred[j,0]* np.cos(Planner.xPred[j,4])
                 - Planner.xPred[j,1]*np.sin(Planner.xPred[j,4]) ) / ( 1-Planner.xPred[j,3]*curv ) ) * planner_dt ) 

                '''Only valid if the SS[j+1] value is close to 0'''
                # if -0.001 < SS[j+1] < 0.001:
                #     SS[j+1] = 0.0

                # print "SS", SS
                # print 'ssj+1', SS[j+1]
                # print "map.getGlobalPosition( SS[j+1], 0.0 )", map.getGlobalPosition( SS[j+1], 0.0 )
                # Xref[j+1], Yref[j+1], Thetaref[j+1] = map.getGlobalPosition( SS[j+1], curr_states.states_ey )
                Xref[j+1], Yref[j+1], Thetaref[j+1] = map.getGlobalPosition( SS[j+1], 0.0 )

            # SS[0] = estimatorData.states_s
            

        
            Xlast = Xref[1]
            Ylast = Yref[1]
            Thetalast = Thetaref[1]

            for i in range(0,N):
                yaw[i]  = Thetaref[i] + Planner.xPred[i,4]
                xp[i]   = Xref[i] - Planner.xPred[i,3]*np.sin(yaw[i])
                yp[i]   = Yref[i] + Planner.xPred[i,3]*np.cos(yaw[i])        


            print "xp", xp, "PlannerCounter", PlannerCounter
            vel     = Planner.xPred[0:N,0]     
            curv    = Planner.xPred[0:N,2] / Planner.xPred[0:N,0]  




            
     


            # #####################################
            # ## Interpolating vehicle references:
            # #####################################  
            # controller_dt = 1.0/loop_rate_control
            # time50ms = np.linspace(0, N*planner_dt, num=N, endpoint=True) #planner horizon time
            # time33ms = np.linspace(0, N*planner_dt, num=np.around(N*planner_dt/controller_dt), endpoint=True)

            # # X 
            # f = interp1d(time50ms, xp, kind='cubic')
            # X_interp = f(time33ms)  

            # # Y
            # f = interp1d(time50ms, yp, kind='cubic')
            # Y_interp = f(time33ms)  

            # # Yaw
            # f = interp1d(time50ms, yaw, kind='cubic')
            # Yaw_interp = f(time33ms)  

            # # Velocity (Vx)
            # f = interp1d(time50ms, vel, kind='cubic')
            # Vx_interp = f(time33ms)

            # # Curvature (K)
            # f = interp1d(time50ms, curv, kind='cubic')
            # Curv_interp = f(time33ms)     
            # print len(Curv_interp)
            # # Curv_interp_filtered  = signal.filtfilt(b_filter, a_filter, Curv_interp, padlen=25)

            # # plt.clf()
            # # plt.figure(2)
            # # plt.plot(Curv_interp, 'k-', label='input')
            # # plt.plot(Curv_interp_filtered,  'c-', linewidth=1.5, label='pad')
            # # plt.legend(loc='best')
            # # plt.show()
            # # plt.grid()

            # # pdb.set_trace()



            # #####################################
            # ## Publishing vehicle references:
            # #####################################   
            
            # refs.counter    = PlannerCounter    
            # refs.x_d        = X_interp
            # refs.y_d        = Y_interp
            # refs.psi_d      = Yaw_interp
            # refs.vx_d       = Vx_interp
            # refs.vy_d       = []
            # refs.omega_d    = []
            # refs.epsi_d     = []
            # refs.ey_d       = []
            # refs.s_d        = []
            # refs.curv_d     = Curv_interp



            planning_refs.publish(refs)


            PlannerCounter  += 1
        
        #####################################
        ## Plotting vehicle position:
        #####################################     




        if rospy.get_param("/trajectory_planner/Visualization") == 1:
            line_trs.set_data(xp[0:N/2], yp[0:N/2])
            line_pred.set_data(xp[N/2:], yp[N/2:])
            x_his.append(xp[0])
            y_his.append(yp[0])
            line_cl.set_data(x_his, y_his)
            l = 0.4/2; w = 0.2/2
            car_sim_x, car_sim_y = getCarPosition(xp[0], yp[0], yaw[0], w, l)
            # car_sim_x, car_sim_y = getCarPosition(xp[N-1], yp[N-1], yaw[N-1], w, l)
            rec_sim.set_xy(np.array([car_sim_x, car_sim_y]).T)
            fig.canvas.draw()

            plt.show()
            plt.pause(1/2000.0)
            # plt.pause(2.0)

            StringValue = "vx = "+str(Planner.xPred[0,0]) + " epsi =" + str(Planner.xPred[0,4]) 
            axtr.set_title(StringValue)


        refs_his_counter.append(PlannerCounter)    
        refs_his_x_t.append(xp[0])
        refs_his_y_t.append(yp[0])
        refs_his_x_d.extend(xp)
        refs_his_y_d.extend(yp)
        refs_his_psi_d.extend(yaw)
        refs_his_vx_d.extend(vel)
        refs_his_vy_d.extend(Planner.xPred[0:N,1])
        refs_his_omega_d.extend(Planner.xPred[0:N,2])
        refs_his_epsi_d.extend(Planner.xPred[0:N,4])
        refs_his_ey_d.extend(Planner.xPred[0:N,3])
        refs_his_s_d.extend(SS)
        refs_his_curv_d.extend(curv)

        print "time taken = {}".format(time.time() - t0) 


        # ALL_LOCAL_DATA[Counter,:]   = np.hstack(( Planner.xPred[0,:], Planner.uPred[0,:] ))
        # References[Counter,:]       = np.hstack(( refs.x_d[0], refs.y_d[0], refs.psi_d[0], refs.vx_d[0], refs.curv_d[0] ))


        # Increase time counter and ROS sleep()
        TimeCounter     += 1
        Counter         += 1


        rate.sleep()




    #############################################################
    day         = '01_06_20'
    num_test    = 'References'

    newpath = ('/').join(__file__.split('/')[:-2]) + '/data/'+day+'/'+num_test+'/' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    print "newpath+'/References'"


    refs_his_counter    = np.array(refs_his_counter)
    refs_his_x_t    = np.array(refs_his_x_t)
    refs_his_y_t    = np.array(refs_his_y_t)
    refs_his_x_d    = np.array(refs_his_x_d)
    refs_his_y_d    = np.array(refs_his_y_d)
    refs_his_psi_d  = np.array(refs_his_psi_d)
    refs_his_vx_d   = np.array(refs_his_vx_d)
    refs_his_vy_d   = np.array(refs_his_vy_d)
    refs_his_omega_d    = np.array(refs_his_omega_d)
    refs_his_epsi_d = np.array(refs_his_epsi_d)
    refs_his_ey_d   = np.array(refs_his_ey_d)
    refs_his_s_d    = np.array(refs_his_s_d)
    refs_his_curv_d = np.array(refs_his_curv_d)

    refs_his_x_d = refs_his_x_d.reshape(Counter,len(xp))
    refs_his_y_d = refs_his_y_d.reshape(Counter, len(yp))
    refs_his_psi_d = refs_his_psi_d.reshape(Counter, len(yaw))
    refs_his_vx_d = refs_his_vx_d.reshape(Counter, len(vel))
    refs_his_vy_d = refs_his_vy_d.reshape(Counter, len((Planner.xPred[0:N,1])))
    refs_his_omega_d = refs_his_omega_d.reshape(Counter, len((Planner.xPred[0:N,2])))
    refs_his_epsi_d = refs_his_epsi_d.reshape(Counter, len((Planner.xPred[0:N,4])))
    refs_his_ey_d = refs_his_ey_d.reshape(Counter, len((Planner.xPred[0:N,3])))
    refs_his_s_d = refs_his_s_d.reshape(Counter, len(SS))
    refs_his_curv_d = refs_his_curv_d.reshape(Counter, len(curv))


    refs_his['counter'] = refs_his_counter
    refs_his['x_t']= refs_his_x_t
    refs_his['y_t']= refs_his_y_t
    refs_his['x_d']= refs_his_x_d
    refs_his['y_d']= refs_his_y_d
    refs_his['psi_d']= refs_his_psi_d
    refs_his['vx_d']= refs_his_vx_d
    refs_his['vy_d']= refs_his_vy_d
    refs_his['omega_d']= refs_his_omega_d
    refs_his['epsi_d']= refs_his_epsi_d
    refs_his['ey_d']= refs_his_ey_d
    refs_his['s_d']= refs_his_s_d
    refs_his['curv_d']= refs_his_curv_d

    # print "refs_his['x_d']",refs_his['x_d'][:-3]
    # print "refs_his_x_d",refs_his_x_d[:-3]
    # print "refs_his_counter", refs_his_counter

    np.save(newpath+'/References', refs_his)

    # np.savetxt(newpath+'/References.dat', References, fmt='%.5e')


    # #############################################################
    # # day         = '29_7_19'
    # # num_test    = 'Test_1'
    # newpath = ('/').join(__file__.split('/')[:-2]) + '/data/'+day+'/'+num_test+'/' 

    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)

    # np.savetxt(newpath+'/ALL_LOCAL_DATA.dat', ALL_LOCAL_DATA, fmt='%.5e')
    # np.savetxt(newpath+'/PREDICTED_DATA.dat', PREDICTED_DATA, fmt='%.5e')
    # np.savetxt(newpath+'/GLOBAL_DATA.dat', GLOBAL_DATA, fmt='%.5e')
    # np.savetxt(newpath+'/Complete_Vel_Vect.dat', Complete_Vel_Vect, fmt='%.5e')
    # np.savetxt(newpath+'/References.dat', References, fmt='%.5e')
    # np.savetxt(newpath+'/TLAPTIME.dat', TLAPTIME, fmt='%.5e')
    # np.savetxt(newpath+'/ELAPSD_TIME.dat', ELAPSD_TIME, fmt='%.5e')


    plt.close()

    # time50ms = np.linspace(0, (Counter-1)*dt, num=Counter-1, endpoint=True)
    # time33ms = np.linspace(0, (Counter-1)*dt, num=np.around((Counter-1)*dt/0.033), endpoint=True)
    # f = interp1d(time50ms, References[0:Counter-1,3], kind='cubic')

    # plt.figure(2)
    # plt.subplot(211)
    # plt.plot(References[0:Counter-1,3], 'o')
    # plt.legend(['Velocity'], loc='best')
    # plt.grid()
    # plt.subplot(212)
    # # plt.plot(time33ms, f(time33ms), 'o')
    # # plt.legend(['Velocity interpolated'], loc='best')    
    # plt.plot(References[0:Counter-1,4], '-')
    # plt.legend(['Curvature'], loc='best')    
    # plt.show()
    # plt.grid()
    # pdb.set_trace()


    # quit() # final del while






# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

class RacingDataClass(object):
    """ Object collecting data from racing performance """

    def __init__(self):

        rospy.Subscriber('Racing_Info', Racing_Info, self.racing_info_callback, queue_size=1)

        self.LapNumber          = 0
        self.PlannerCounter     = 0

    def racing_info_callback(self,data):
        """ ... """      
        self.LapNumber          = data.LapNumber
        self.PlannerCounter     = data.PlannerCounter




# ===============================================================================================================================
# ==================================================== END OF MAIN ==============================================================
# ===============================================================================================================================

def InitFigure_XY(map, mode, HW):
    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    axtr = plt.axes()

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, HW)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -HW)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')


    line_cl,        = axtr.plot(xdata, ydata, '-k')
    line_gps_cl,    = axtr.plot(xdata, ydata, '--ob')
    line_tr,        = axtr.plot(xdata, ydata, '-or')
    line_trs,       = axtr.plot(xdata, ydata, '-og')
    line_pred,      = axtr.plot(xdata, ydata, '-or')
    
    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])

    rec = patches.Polygon(v, alpha=0.7,closed=True, fc='r', ec='k',zorder=10)
    # axtr.add_patch(rec)

    rec_sim = patches.Polygon(v, alpha=0.7,closed=True, fc='G', ec='k',zorder=10)

    if mode == "simulations":
        axtr.add_patch(rec_sim)

    plt.show()

    return fig, axtr, line_tr, line_pred, line_trs, line_cl, line_gps_cl, rec, rec_sim



def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l*np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y



def predicted_vectors_generation(Hp, x0, accel_rate, delta, dt):

    Vx      = np.zeros((Hp+1, 1))
    Vx[0]   = x0[0]
    S       = np.zeros((Hp+1, 1))
    S[0]    = 0
    Vy      = np.zeros((Hp+1, 1))
    Vy[0]   = x0[1]
    W       = np.zeros((Hp+1, 1))
    W[0]    = x0[2]
    Ey      = np.zeros((Hp+1, 1))
    Ey[0]   = x0[3]
    Epsi    = np.zeros((Hp+1, 1))
    Epsi[0] = x0[4]

    curv    = 0

    for i in range(0, Hp): 
        Vy[i+1]      = x0[1]  
        W[i+1]       = x0[2] 
        Ey[i+1]      = x0[3] 
        Epsi[i+1]    = x0[4] 

    accel   = np.array([[accel_rate for i in range(0, Hp)]])
    delta   = np.array([[ delta for i in range(0, Hp)]])

    for i in range(0, Hp):
        Vx[i+1]    = Vx[i] + accel[0,i] * dt
        S[i+1]      = S[i] + ( (Vx[i]*np.cos(Epsi[i]) - Vy[i]*np.sin(Epsi[i])) / (1-Ey[i]*curv) ) * dt

    # print "Vx = ", Vx
    # print "Vy = ", np.transpose(Vy)
    # print "W = ", W

    # pdb.set_trace()

    xx  = np.hstack([ Vx, Vy, W, Ey, Epsi, S])    

    uu  = np.hstack([delta.transpose(),accel.transpose()])

    return xx, uu


# EA: Modified for taking also the desired velocity
def Curvature(s, PointAndTangent):
    """curvature and desired velocity computation
    s: curvilinear abscissa at which the curvature has to be evaluated
    PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
    """
    TrackLength = PointAndTangent[-1,3]+PointAndTangent[-1,4]

    # In case on a lap after the first one
    while (s > TrackLength):
        s = s - TrackLength

    # Given s \in [0, TrackLength] compute the curvature
    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)

    i = int(np.where(np.squeeze(index))[0]) #EA: this works 
    #i = np.where(np.squeeze(index))[0]     #EA: this does not work
    curvature = PointAndTangent[i, 5]

    return curvature


if __name__ == "__main__":

    try:    
        main()
        
    except rospy.ROSInterruptException:
        pass
