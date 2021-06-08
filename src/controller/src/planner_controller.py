#!/usr/bin/env python

##########################################################
##           IRI 1:10 Autonomous Car                    ##
##           ```````````````````````                    ##
##      Supervisor: Puig Cayuela Vicenc                 ##
##      Author:     Shivam Chaubey                      ##
##      Email:      shivam.chaubey1006@gmail.com        ##
##      Date:       18/05/2021                          ##
##########################################################

''' 
state estimator subscriber and MPC controller input publisher: This file
is to setup the MPC problem using the estimated states, previous input if
weights on input rate is set, and initial constraints. The vehicle has to
follow a track set on the map    

'''


import os
import datetime
import rospy
import numpy as np
from sensor_fusion.msg import sensorReading
from controller.msg import states_info
print 'states_info', states_info()
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf
from PathFollowingLPVMPC import PathFollowingLPV_MPC
from planner.msg import My_Planning
import time
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
from trackInitialization import Map
from controller.msg import mpcPrediction
from math import pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from scipy import signal
# from tqdm import tqdm
from planner.msg import My_Planning


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


########## current state estimation from observer #############
class EstimatorData(object):
    """Data from estimator"""
    def __init__(self):

        print "Waiting for observer message"
        rospy.wait_for_message("est_state_info", sensorReading)
        rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback)
        self.CurrentState = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print "Subscribed to observer"
    
    def estimator_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T

########## Full state estimation from observer #############
class planner_ref(object):
    """Data from estimator"""
    def __init__(self):

        print "Waiting for planner message"
        rospy.wait_for_message("My_Planning", My_Planning)
        rospy.Subscriber("My_Planning", My_Planning, self.planner_callback)
        print "Subscribed to planner"
    
        self.x_d    = []
        self.y_d    = []
        self.psi_d  = []
        self.vx_d   = []
        self.curv_d = []
        self.counter = 0

    def planner_callback(self, msg):
        """
        Unpack the messages from the planner
        """
        self.x_d    = msg.x_d
        self.y_d    = msg.y_d
        self.psi_d  = msg.psi_d
        self.vx_d   = msg.vx_d
        self.curv_d = msg.curv_d
        self.counter = msg.counter



### wrap the angle between [-pi,pi] ###
def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle

######## state and input vector generation for MPC initial setup #########
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

    for i in range(0, Hp):
        Vy[i+1]      = x0[1]
        W[i+1]       = x0[2]
        Ey[i+1]      = x0[3]
        Epsi[i+1]    = x0[4]

    accel   = np.array([[accel_rate for i in range(0, Hp)]])
    delta   = np.array([[ delta for i in range(0, Hp)]])

    for i in range(0, Hp):
        Vx[i+1] = Vx[i] + accel[0,i] * dt
        S[i+1]  = S[i] + Vx[i] * dt

    xx  = np.hstack([ Vx, Vy, W, Epsi ,S ,Ey]) # [vx vy omega theta_e s y_e]

    uu  = np.hstack([delta.transpose(),accel.transpose()])

    return xx, uu



class predicted_states_msg():
        
    def __init__(self):
        
        self.LPV_prediction_state_pub   = rospy.Publisher('control/LPV_prediction', Path, queue_size=1)
        self.MPC_prection_state_pub     = rospy.Publisher('control/MPC_prection', Path, queue_size=1)
        
        self.LPV_prediction_states = Path()
        self.MPC_prediction_states = Path()
        self.pose_stamp = PoseStamped()

    def LPV_msg_update(self, map, data):


        self.LPV_prediction_states.header.stamp = rospy.Time.now()
        self.LPV_prediction_states.header.frame_id = "/map" #"/lpv_predictions"

        for i in range(len(data)):
            epsi, s, ey = data[i,3], data[i,4], data[i,5]

            if abs(s) < 0.001:
                s = 0
            if abs(ey) < 0.001:
                ey = 0
            print "epsi, s, ey", epsi, s, ey
            x, y, theta  = map.getGlobalPosition(s, ey)
            print "x, y, theta", x, y, theta
            quaternion   = tf.transformations.quaternion_from_euler(0, 0, theta)
            
            self.pose_stamp.pose.position.x = x
            self.pose_stamp.pose.position.y = y
            self.pose_stamp.pose.position.z = 0.0

            self.pose_stamp.pose.orientation.x = quaternion[0]
            self.pose_stamp.pose.orientation.y = quaternion[1]
            self.pose_stamp.pose.orientation.z = quaternion[2]
            self.pose_stamp.pose.orientation.w = quaternion[3]
            self.LPV_prediction_states.poses.append(self.pose_stamp) 

        self.LPV_prediction_state_pub.publish(self.LPV_prediction_states)
    # def MPC_update(self):

def interpolate_references(planner_dt, planner_N, controller_dt, x_ref_p, y_ref_p, yaw_ref_p, vx_ref_p, vy_ref_p, omega_ref_p, epsi_ref_p,\
 ey_ref_p, s_ref_p, curv_ref_p, steer_ref_p, duty_ref_p):

    msg = "Interpolation starting"
    print msg
    ##### planner horizon time ###########
    planner_hz_time = np.linspace(0, planner_N*planner_dt, num=planner_N, endpoint=True) #planner horizon time
    
    ##### controller horizon time ###########
    controller_hz_time = np.linspace(0, planner_N*planner_dt, num=np.around(planner_N*planner_dt/controller_dt), endpoint=True)


    #### interpolation factor #####
    interp_factor = np.around(len(controller_hz_time)/len(planner_hz_time))

    msg = 'x_ref Interpolation'
    x_ref_p_interp = []
    for i in tqdm(range(len(x_ref_p))):
        f = interp1d(planner_hz_time, x_ref_p[i], kind='cubic')
        x_ref_p_interp.append(f(controller_hz_time))  
        
    x_ref_p_interp = np.array(x_ref_p_interp)
    print (x_ref_p_interp.shape)
    print 'x_ref Interpolation done'

    msg = 'y_ref Interpolation'
    y_ref_p_interp = []
    for i in tqdm(range(len(y_ref_p))):
        f = interp1d(planner_hz_time, y_ref_p[i], kind='cubic')
        y_ref_p_interp.append(f(controller_hz_time))  
        
    y_ref_p_interp = np.array(y_ref_p_interp)
    print (y_ref_p_interp.shape)
    print 'y_ref Interpolation done'

    msg = 'yaw_ref Interpolation'
    yaw_ref_p_interp = []
    for i in tqdm(range(len(yaw_ref_p))):
        f = interp1d(planner_hz_time, yaw_ref_p[i], kind='cubic')
        yaw_ref_p_interp.append(f(controller_hz_time))  
        
    yaw_ref_p_interp = np.array(yaw_ref_p_interp)
    print (yaw_ref_p_interp.shape)
    print 'yaw_ref Interpolation done'

    msg = 'vx_ref Interpolation'
    vx_ref_p_interp = []
    for i in tqdm(range(len(vx_ref_p))):
        f = interp1d(planner_hz_time, vx_ref_p[i], kind='cubic')
        vx_ref_p_interp.append(f(controller_hz_time))  
        
    vx_ref_p_interp = np.array(vx_ref_p_interp)
    print (vx_ref_p_interp.shape)
    print 'vx_ref Interpolation done'

    msg = 'vy_ref Interpolation'
    vy_ref_p_interp = []
    for i in tqdm(range(len(vy_ref_p))):
        f = interp1d(planner_hz_time, vy_ref_p[i], kind='cubic')
        vy_ref_p_interp.append(f(controller_hz_time))  
        
    vy_ref_p_interp = np.array(vy_ref_p_interp)
    print 'vy_ref Interpolation done'

    msg = 'omega_ref Interpolation'
    omega_ref_p_interp = []
    for i in tqdm(range(len(omega_ref_p))):
        f = interp1d(planner_hz_time, omega_ref_p[i], kind='cubic')
        omega_ref_p_interp.append(f(controller_hz_time))  
        
    omega_ref_p_interp = np.array(omega_ref_p_interp)
    print 'omega_ref Interpolation done'

    msg = 'epsi_ref Interpolation'
    epsi_ref_p_interp = []
    for i in tqdm(range(len(epsi_ref_p))):
        f = interp1d(planner_hz_time, epsi_ref_p[i], kind='cubic')
        epsi_ref_p_interp.append(f(controller_hz_time))  
        
    epsi_ref_p_interp = np.array(epsi_ref_p_interp)
    print 'epsi_ref Interpolation done'

    msg = 'ey_ref Interpolation'
    ey_ref_p_interp = []
    for i in tqdm(range(len(ey_ref_p))):
        f = interp1d(planner_hz_time, ey_ref_p[i], kind='cubic')
        ey_ref_p_interp.append(f(controller_hz_time))  
        
    ey_ref_p_interp = np.array(ey_ref_p_interp)
    print 'ey_ref Interpolation done'

    msg = 's_ref Interpolation'
    s_ref_p = s_ref_p[:,:s_ref_p.shape[1]-1]
    s_ref_p_interp = []
    for i in tqdm(range(len(s_ref_p))):
        f = interp1d(planner_hz_time, s_ref_p[i], kind='cubic')
        s_ref_p_interp.append(f(controller_hz_time))  
        
    s_ref_p_interp = np.array(s_ref_p_interp)
    print 's_ref Interpolation done'

    msg = 'curv_ref Interpolation'
    # Filter to be applied.
    b_filter, a_filter = signal.ellip(4, 0.01, 120, 0.125) 
    curv_ref_p_interp = []
    for i in tqdm(range(len(curv_ref_p))):
        f = interp1d(planner_hz_time, curv_ref_p[i], kind='cubic')
        Curv_interp_filtered  = signal.filtfilt(b_filter, a_filter, f(controller_hz_time), padlen=np.int(curv_ref_p.shape[1]*0.25))
        curv_ref_p_interp.append(Curv_interp_filtered)  
        
    curv_ref_p_interp = np.array(curv_ref_p_interp)
    print 'curv_ref Interpolation done'


    msg = 'steer_ref Interpolation'
    # Filter to be applied.
    b_filter, a_filter = signal.ellip(4, 0.01, 120, 0.125) 
    steer_ref_p_interp = []
    for i in tqdm(range(len(steer_ref_p))):
        f = interp1d(planner_hz_time, steer_ref_p[i], kind='cubic')
        # steer_interp_filtered  = signal.filtfilt(b_filter, a_filter, f(controller_hz_time), padlen=25)
        steer_ref_p_interp.append(f(controller_hz_time))  
        
    steer_ref_p_interp = np.array(steer_ref_p_interp)
    print 'steer_ref Interpolation done'


    msg = 'duty_ref Interpolation'
    # Filter to be applied.
    b_filter, a_filter = signal.ellip(4, 0.01, 120, 0.125) 
    duty_ref_p_interp = []
    for i in tqdm(range(len(duty_ref_p))):
        f = interp1d(planner_hz_time, duty_ref_p[i], kind='cubic')
        # duty_interp_filtered  = signal.filtfilt(b_filter, a_filter, f(controller_hz_time), padlen=25)
        duty_ref_p_interp.append(f(controller_hz_time))  
        
    duty_ref_p_interp = np.array(duty_ref_p_interp)
    print 'duty_ref Interpolation done'



    # f = interp1d(planner_hz_time, curv, kind='cubic')
    # Curv_interp = f(controller_hz_time)     
    # print len(Curv_interp)
    # # Curv_interp_filtered  = signal.filtfilt(b_filter, a_filter, Curv_interp, padlen=25)


    print "Interpolation done"


    return interp_factor, x_ref_p_interp, y_ref_p_interp, yaw_ref_p_interp, vx_ref_p_interp, vy_ref_p_interp, omega_ref_p_interp,\
     epsi_ref_p_interp, ey_ref_p_interp, s_ref_p_interp, curv_ref_p_interp, steer_ref_p_interp, duty_ref_p_interp



def main():

    rospy.init_node("LPV_MPC")
    
    ########################## Control output ##################################
    dutycycle_commands  = rospy.Publisher('control/accel', Float32, queue_size=1)
    steering_commands   = rospy.Publisher('control/steering', Float32, queue_size=1)
    

    predicted_points_pub_on = True
    if predicted_points_pub_on == True:
        # publish_predicted = predicted_states_msg()
        LPV_prediction_state_pub   = rospy.Publisher('control/LPV_prediction', mpcPrediction, queue_size=1)
        MPC_prediction_state_pub   = rospy.Publisher('control/MPC_prediction', mpcPrediction, queue_size=1)

    #     MPC_prection_state_pub     = rospy.Publisher('control/MPC_prection', Path, queue_size=1)
    

    ### same message type used for both ##
    lpvPrediction_msg = mpcPrediction()
    mpcPrediction_msg = mpcPrediction()


    dutycycle_thres     = rospy.get_param("/duty_th") # dutycycle Deadzone
    
    N                   = rospy.get_param("/control/Horizon")
    Vx_ref              = rospy.get_param("/control/vel_ref")
    loop_rate           = rospy.get_param("/control/Hz")
    save_reference      = rospy.get_param("/control/save_output")
    dt                  = 1.0/loop_rate
    rate                = rospy.Rate(loop_rate)

    cmd_servo           = Float32()
    cmd_motor           = Float32()
    
    LPV_prediction_states = Path()
    MPC_prediction_states = Path()

    rospy.sleep(1.2)    # wait for observer initialization or make a flag in that node to know if the observer started.

    map                 = Map()             # Fixed map track

    visualization = False
    if visualization == True:
        HW                  = rospy.get_param("halfWidth") 
        fig, axtr, line_tr, line_pred, line_trs, line_cl, line_gps_cl, rec, rec_sim = InitFigure_XY(map, HW)
        Xlast           = 0.0
        Ylast           = 0.0
        Thetalast       = 0.0
        Xref            = np.zeros(N+1)
        Yref            = np.zeros(N+1)
        Thetaref        = np.zeros(N+1)
        xp              = np.zeros(N)
        yp              = np.zeros(N)
        yaw             = np.zeros(N)  
        S_dist          = np.zeros(N+1,) 
        x_his           = []
        y_his           = []

    first_it            = 0

    GlobalState         = np.zeros(6)       # vehicle kinematic and dynamic states
    LocalState          = np.zeros(6)       # vehicle dynamic and error states
    LapNumber           = 0   
    LapNumber_det       = 0              
    HalfTrack           = 0

    '''References for the controller to attain this states'''
    vel_ref             = np.zeros([N,1])  
    curv_ref            = np.zeros([N,1])
    y_ref               = np.zeros([N,1])
    yaw_ref             = np.zeros([N,1])
    x_ref               = np.zeros([N,1])

    # Loop running at loop rate
    TimeCounter         = 0
    ControllerCounter   = 0
    test_gen            = rospy.get_param("planning_mode")# test generated plan 0 = offline plan, 1 = vel profile, 2 = map
    
    ###### Online planner ########
    if test_gen == 2:
            planner = planner_ref()
            curv_ref_p = planner.curv_d 
            vel_ref_p  = planner.vx_d
            y_ref      = planner.y_d
            yaw_ref    = planner.psi_d
            x_ref      = planner.x_d


    ##### offline planner #########
    if test_gen == 3:
        
        planning_refs_pub   = rospy.Publisher('Offline_Planning', My_Planning, queue_size=1)
        planning_refs_msg   = My_Planning()

        # offline_path = (('/').join(sys.path[0].split('/')[:-2])+'/planner/data/01_06_20/References/References.npy')
        offline_path = (('/').join(sys.path[0].split('/')[:-2])+'/planner/data/01_06_20/References/velocity_0.8_N40_HZ25.npy')
        # offline_path = (('/').join(sys.path[0].split('/')[:-2])+'/planner/data/01_06_20/References/velocity_0.8_N20_HZ50.npy')

        
        print "offline_path", offline_path
        planner_refs = np.load(offline_path, allow_pickle=True).item()

        planner_time  =  planner_refs['time']
        planner_dt  =  planner_refs['planner_dt']
        planner_N  =  planner_refs['planner_N']
        counter_ref  =  planner_refs['counter'] 
        xpt_ref_p      =  planner_refs['x_t']  
        ypt_ref_p      =  planner_refs['y_t']  
        x_ref_p        =  planner_refs['x_d']    
        y_ref_p        =  planner_refs['y_d']    
        yaw_ref_p      =  planner_refs['psi_d']    
        vx_ref_p     =  planner_refs['vx_d']     
        vy_ref_p     =  planner_refs['vy_d']     
        omega_ref_p  =  planner_refs['omega_d']  
        epsi_ref_p   =  planner_refs['epsi_d']   
        ey_ref_p     =  planner_refs['ey_d']     
        s_ref_p      =  planner_refs['s_d']
        curv_ref_p   =  planner_refs['curv_d']
        steer_ref_p   =  planner_refs['steer_d'] 
        duty_ref_p   =  planner_refs['duty_d']
        
        interp_factor, x_ref_interp, y_ref_interp, yaw_ref_interp, vx_ref_interp, vy_ref_interp, omega_ref_interp, epsi_ref_interp, ey_ref_interp, s_ref_interp, curv_ref_interp, \
        steer_ref_interp, duty_ref_interp= interpolate_references(planner_dt,planner_N, dt, x_ref_p, y_ref_p, yaw_ref_p, vx_ref_p, vy_ref_p, omega_ref_p, epsi_ref_p, ey_ref_p, s_ref_p,\
         curv_ref_p, steer_ref_p, duty_ref_p)

        offline_counter = 0 
        planner_index = 0   
        
        # print "calculated planner time = {}".format(np.mean(planner_time[1:] - planner_time[:-1]))       
        # print "Received planner time = {}".format(planner_dt)

    estimatorData       = EstimatorData()   # observer reading
    
    SS  = 0.0

    window_iter = 0



    if save_reference == True:
        
        ############### for controller and observer data saving used for documentation or debugging ################## 
        contr_his        = {"time": [], "controller_dt": dt, "controller_N": N , "counter": [], "x_est": [], "y_est": [], "yaw_est": [],\
         "vx_est": [], "vy_est": [], "omega_est": [], "ey_est": [], "epsi_est": [], "s_est": [], "epsi_pred_mod": [], "ey_pred_mod": [], \
         "s_pred_mod": [], "vx_pred_mod": [], "vy_pred_mod": [], "omega_pred_mod": [], "epsi_pred_cont": [], "ey_pred_cont": [],\
          "s_pred_cont": [], "vx_pred_cont": [],  "vy_pred_cont": [], "omega_pred_cont": [], "steer_pred_cont": [], "duty_pred_cont": []}
        
        time_his = time.time()
        contr_his_time    = []
        contr_his_counter    = []

        contr_his_x_est    = []
        contr_his_y_est    = []
        contr_his_yaw_est    = []
        contr_his_vx_est    = []
        contr_his_vy_est    = []
        contr_his_omega_est    = []
        contr_his_epsi_est    = []
        contr_his_ey_est  = []
        contr_his_s_est   = []
        
        contr_his_epsi_pred_mod    = []
        contr_his_ey_pred_mod  = []
        contr_his_s_pred_mod   = []
        contr_his_vx_pred_mod   = []
        contr_his_vy_pred_mod    = []
        contr_his_omega_pred_mod = []

        contr_his_epsi_pred_cont    = []
        contr_his_ey_pred_cont  = []
        contr_his_s_pred_cont   = []
        contr_his_vx_pred_cont   = []
        contr_his_vy_pred_cont    = []
        contr_his_omega_pred_cont = []
        contr_his_steer_pred_cont   = []
        contr_his_duty_pred_cont    = []




    Controller  = PathFollowingLPV_MPC(N, Vx_ref, dt, map)
    rospy.sleep(1.2)

    print "controller initialized"

    lap_counter = 1

    while (not rospy.is_shutdown()):

        startTimer = datetime.datetime.now()

        ''' 
        The estimator provides 6 states of the vehicle  [vx, vy, omega, X,
        Y, yaw] but the MPC uses error model for the optimization. We transformed 
        [X, Y, yaw] states of global states to local states [s, ey, epsi] using the function 
        map.getLocalPosition which provides error with respect to the track.
        '''

        # Read Measurements
        GlobalState[:] = estimatorData.CurrentState  # The current estimated state vector [vx vy w x y psi]
        GlobalState[5] = (GlobalState[5]  + pi) % (2 * pi) - pi
        print "GlobalState[5]", GlobalState[5]
        '''
        The above code converts the continuous yaw angle between [-pi,pi] because
        the map.getLocalPosition function work in this domain.  '''
        LocalState[:]  = estimatorData.CurrentState  # [vx vy w x y psi]


        if -0.01 < LocalState[0] < 0.01:
            LocalState[0] = 0.01


        ## This is for the map case you can add your own cases for example if you want the vehicle to follow other trajectory make another case.
        if test_gen == 1:


            ## OUT: s, ey, epsi       IN: x, y, psi
            LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
                GlobalState[3], GlobalState[4], wrap(GlobalState[5]))

            print("\n")
            print("vx vy w: ", GlobalState[:3])
            print("\n")
            print("x y yaw: ", GlobalState[3:])
            print("\n")
            print("epsi s ey: ", LocalState[3:])
            print("\n")
            curv_ref        = 0
            vel_ref         = np.ones([N,1])*Vx_ref
            # Check if the lap has finished
            if map.TrackLength*(1.0-0.1) <= LocalState[4] <= map.TrackLength*(1.0+0.1):
                HalfTrack = 1
                lap_counter += 1
                print 'the lap has finished'


        # if lap_counter == 3:
        #     print lap_counter-1, " lap has completed"

        ########################## ONLINE PLANNER #########################
        if test_gen == 2:

            # OUT: s, ey, epsi       IN: x, y, psi
            # LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
            #     GlobalState[3], GlobalState[4], wrap(GlobalState[5]))

            print "planner.counter", planner.counter
            curv_ref = planner.curv_d 
            vel_ref  = planner.vx_d 
            y_ref      = planner.y_d
            yaw_ref    = planner.psi_d
            x_ref      = planner.x_d
            
            # OUT: s, ex, ey, epsi
            LocalState[4], Xerror, LocalState[5], LocalState[3] = Body_Frame_Errors(GlobalState[3], 
                 GlobalState[4], GlobalState[5], x_ref[0], y_ref[0], yaw_ref[0], SS, LocalState[0],
                 LocalState[1], curv_ref[0], dt )   

            SS = LocalState[4] 
            


            # ey_ref   = planner.ey_d 
            # epsi_ref = planner.epsi_d     
   
            print "vel_ref",  vel_ref
            # Vx_ref   = vel_ref[1]
            # if planner.x_d < 0.5:
            #     curv_ref = curv_ref*0.0 
            #     vel_ref  = vel_ref*0.0
         
            # Check if the lap has finished
            if LocalState[4] >= 3*map.TrackLength/4:
                HalfTrack = 1
                print 'the lap has finished'
            
            window_iter += 1
            
            if window_iter == 3:
                curv_ref_p = planner.curv_d 
                vel_ref_p  = planner.vx_d
                
                window_iter = 0


        ########################## OFFLINE PLANNER #########################

        if test_gen == 3:

            # OUT: s, ey, epsi       IN: x, y, psi
            # LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
            #     GlobalState[3], GlobalState[4], wrap(GlobalState[5]))
            

            # x_ref       = x_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # y_ref       = y_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # yaw_ref     = yaw_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # vx_ref      = vx_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # vy_ref      = vy_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # omega_ref   = omega_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # epsi_ref    = epsi_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # ey_ref      = ey_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # s_ref       = s_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N]  
            # curv_ref    = curv_ref_interp[offline_counter,planner_index*N:(planner_index+1)*N] 


            x_ref       = x_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            y_ref       = y_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            yaw_ref     = yaw_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            vx_ref      = vx_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            vy_ref      = vy_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            omega_ref   = omega_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            epsi_ref    = epsi_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            ey_ref      = ey_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            s_ref       = s_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            curv_ref    = curv_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            steer_ref    = steer_ref_interp[offline_counter,planner_index:(planner_index+N)]  
            duty_ref    = duty_ref_interp[offline_counter,planner_index:(planner_index+N)]  

            print "vx_ref.shape", vx_ref.shape
            if (interp_factor -1) == planner_index: ### when the sampling factor meets then switch to new trajectory
                planner_index = 0
                offline_counter += 1

            else:
                planner_index += 1

            # planner_index += 1

            print 'planner_index', planner_index, 'offline_counter', offline_counter


            # ## OUT: s, ey, epsi       IN: x, y, psi
            # LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
            #     GlobalState[3], GlobalState[4], wrap(GlobalState[5]))

            # print "yaw_ref[0]", yaw_ref.shape
            # OUT: s, ex, ey, epsi
            # IN: (x, y, psi, xd, yd, psid, s0, vx, vy, curv, dt):
            LocalState[4], Xerror, LocalState[5], LocalState[3] = Body_Frame_Errors(GlobalState[3], 
                 GlobalState[4], GlobalState[5], x_ref[0], y_ref[0], yaw_ref[0], SS, LocalState[0],
                 LocalState[1], curv_ref[0], dt )   

            SS = LocalState[4] 

            if offline_counter == counter_ref[-1] -1 :
                " <<<<<<<<<< offline map finished >>>>>>>>>>>>>>>"

                break 
            

            # Vx_ref   = vel_ref[1]



            planning_refs_msg.counter    =      offline_counter  
            planning_refs_msg.x_pt       =      xpt_ref_p
            planning_refs_msg.y_pt       =      ypt_ref_p
            planning_refs_msg.x_d        =      x_ref_interp[offline_counter,:]#x_ref 
            planning_refs_msg.y_d        =      y_ref_interp[offline_counter,:]#y_ref 
            planning_refs_msg.psi_d      =      yaw_ref 
            planning_refs_msg.vx_d       =      vx_ref 
            planning_refs_msg.vy_d       =      vy_ref 
            planning_refs_msg.omega_d    =      omega_ref 
            planning_refs_msg.epsi_d     =      epsi_ref 
            planning_refs_msg.ey_d       =      ey_ref 
            planning_refs_msg.s_d        =      s_ref 
            planning_refs_msg.curv_d     =      curv_ref 

            planning_refs_pub.publish(planning_refs_msg)
        


        SS = LocalState[4] 

        ###################################################################################################
        ###################################################################################################

        ### END OF THE LAP.
        # PATH TRACKING:

        # if ( HalfTrack == 1 and (-0.2 <= GlobalState[3] <= 0.2) and (-0.25 <= GlobalState[4] <= 0.25) and (-0.1 <= GlobalState[5] <= 0.1)):

        if (map.TrackLength*0.98 <= LocalState[4]):#(1.0-0.1) <= LocalState[4] <= map.TrackLength*(1.0+0.1))):
            HalfTrack       = 0

            if (LapNumber_det == 0):
                LapNumber       += 1
                
            LapNumber_det += 1

            SS              = 0
            print "END OF THE LAP"
            print 'LapNumber', LapNumber 
            if LapNumber == 3:
                break

            print "LocalState[4]", LocalState[4]
        else:
            LapNumber_det = 0


        ###################################################################################################
        ###################################################################################################
        # if first_it == 6:
        #     break


        if first_it < 4:

            # Controller.planning_mode = 1
            duty_cycle  = 0.01

            delta = 0.0
            # xx, uu      = predicted_vectors_generation(N, LocalState, accel_rate, dt)
            xx, uu      = predicted_vectors_generation(N, LocalState, duty_cycle, delta, dt)
            
            curv_ref        = 0
            vel_ref         = np.ones([N,1])*Vx_ref
            Controller.uPred = uu
            Controller.xPred = xx
            
            LPV_States_Prediction = xx[:N,:]

            # first_it    = first_it + 1
            Controller.uminus1 = Controller.uPred[0,:]

        else:

            # print "Controller.uminus1", Controller.uminus1

            # print "LPV_States_Prediction", LPV_States_Prediction
            
            if first_it == 4:
                print "MPC setup"
                # print "vel_ref, curv_ref", len(vel_ref), len(curv_ref)
                

                if test_gen == 3:

                    # Vx_ref = 1.2
                    LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_setup()

                    Controller.MPC_setup(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 

                else:


                    LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_setup()

                    Controller.MPC_setup(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 

                    ######### for full MPC without warm start ##############
                    # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref)
                    # ref_qp = [Vx_ref, 0, 0, 0, 0, 0]
                    # # Controller.MPC_update(A_L, B_L, LocalState[0:6], ref_qp) 
                    # u_ref = [0,0]
                    # Controller.MPC_full(A_L, B_L, u_ref, LocalState[0:6], ref_qp)

                    # Controller.MPC_solve()

            # else:
            t1 = time.time() 

            if test_gen == 1:
                # Controller.update_q = True
                # print "len(vel_ref), len(curv_ref)",len(vel_ref), len(curv_ref)

                # [vx], [vy], [wz], [epsi], [s], [ey]
                # non_uniform_sampling 
                LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref)
                # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref,  )
            
                # publish_predicted.LPV_msg_update(map, LPV_States_Prediction)
                # publish_predicted.LPV_msg_update(map, Controller.xPred)

                # print "Controller.xPred", Controller.xPred
                print "MPC update", Vx_ref 

                ref_qp = [Vx_ref, 0, 0, 0, 0, 0]
                Controller.MPC_update(A_L, B_L, LocalState[0:6], ref_qp) 
                u_ref = [0,0]
                # Controller.MPC_full(A_L, B_L, u_ref, LocalState[0:6], ref_qp)

                Controller.MPC_solve()

            if test_gen == 3:

                Controller.update_q = True
                ref_init = [vx_ref[0] , vy_ref[0], omega_ref[0], epsi_ref[0], s_ref[0], ey_ref[0]]
                
                ref_states = [vx_ref , vy_ref, omega_ref, epsi_ref, s_ref, ey_ref, curv_ref]
                ref_u = np.vstack([steer_ref, duty_ref]).T

                print "ref_u.shape", ref_u.shape, "Controller.uPred.shape", Controller.uPred.shape, steer_ref.shape, duty_ref.shape
                # print "len(vel_ref), len(curv_ref)",len(vel_ref), len(curv_ref)
                LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_planner(ref_init, ref_u, ref_states)

                # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_planner(LocalState[0:6], ref_u, ref_states)

                # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_planner(LocalState[0:6], Controller.uPred, ref_states)

                # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_planner(LocalState[0:6], Controller.uPred, ref_lpv)
                # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref,  )


                # print "Controller.xPred", Controller.xPred
                print "MPC update", vel_ref[0], Vx_ref
                # ref_qp = [Vx_ref, 0, 0, 0, 0 , 0]
                ref_qp = [vx_ref[N-1] , 0., 0., epsi_ref[N-1], 0., ey_ref[N-1]]
                # ref_qp = [vx_ref[N-1] , vy_ref[N-1], omega_ref[N-1], epsi_ref[N-1], s_ref[N-1], ey_ref[N-1]]
                # ref_qp = [vx_ref[-1] , vy_ref[-1], omega_ref[-1], epsi_ref[-1], s_ref[-1], ey_ref[-1]]

                Controller.MPC_update(A_L, B_L, LocalState[0:6], ref_qp) 
                # ref_init = [0 , 0, 0, epsi_ref[0], s_ref[0], ey_ref[0]]

                # Controller.MPC_update(A_L, B_L, np.array(ref_init), ref_qp) 

                Controller.MPC_solve()
            
            print "feasible", Controller.feasible
            print "time taken to solve", time.time() - t1




            if predicted_points_pub_on == True:
                lpvPrediction_msg.epsi = LPV_States_Prediction[:,3]
                lpvPrediction_msg.s = LPV_States_Prediction[:,4]
                lpvPrediction_msg.ey = LPV_States_Prediction[:,5]

                mpcPrediction_msg.epsi = Controller.xPred[:,3]
                mpcPrediction_msg.s = Controller.xPred[:,4]
                mpcPrediction_msg.ey = Controller.xPred[:,5]


            Controller.uminus1 = Controller.uPred[0,:] 

        first_it    = first_it + 1


        print ('LPV_States_Prediction', LPV_States_Prediction.shape)

        if Controller.feasible == 0.0:
            Controller.uPred[0,0] = 0.0
            Controller.uPred[0,1] = 0.0

        print('control actions',"delta", Controller.uPred[0,0],"dutycycle", Controller.uPred[0,1])

        ## Publish controls ##
        cmd_servo = Controller.uPred[0,0]
        cmd_motor = Controller.uPred[0,1]
        dutycycle_commands.publish(Controller.uPred[0,1])
        steering_commands.publish(Controller.uPred[0,0])


        if predicted_points_pub_on == True:
            LPV_prediction_state_pub.publish(lpvPrediction_msg)

        if Controller.feasible == 1:
            print('control actions',"delta", Controller.uPred[0,0],"dutycycle", Controller.uPred[0,1])
            print('\n')

            # if Controller.feasible == 0.0:
            #     Controller.uPred[0,0] = 0.0
            #     Controller.uPred[0,1] = 0.0
                
            if predicted_points_pub_on == True:
                MPC_prediction_state_pub.publish(mpcPrediction_msg)


        # else:
        #     print "Not feasible"
        #     print('\n')

        #     # if Controller.feasible == 0.0:
        #     #     Controller.uPred[0,0] = 0.0
        #     #     Controller.uPred[0,1] = 0.0
                
        #     ## Publish controls ##
        #     cmd_servo = Controller.uPred[0,0]
        #     cmd_motor = Controller.uPred[0,1]
        #     dutycycle_commands.publish(0.0)
        #     steering_commands.publish(0.0)
        #     MPC_prediction_state_pub.publish([])
        #     LPV_prediction_state_pub.publish([])


        ################### visualization ########################



        if save_reference == True:

            tnow = time.time()
            contr_his_time.append(tnow - time_his)
            contr_his_counter.append(ControllerCounter)    

            contr_his_x_est.append(GlobalState[3])
            contr_his_y_est.append(GlobalState[4])
            contr_his_yaw_est.append(GlobalState[5])
            contr_his_vx_est.append(GlobalState[0])
            contr_his_vy_est.append(GlobalState[1])
            contr_his_omega_est.append(GlobalState[2])
            contr_his_ey_est.append(LocalState[5])
            contr_his_epsi_est.append(LocalState[3])
            contr_his_s_est.append(LocalState[4])
                

            contr_his_epsi_pred_mod.extend(LPV_States_Prediction[:,3])
            contr_his_ey_pred_mod.extend(LPV_States_Prediction[:,5])
            contr_his_s_pred_mod.extend(LPV_States_Prediction[:,4])
            contr_his_vx_pred_mod.extend(LPV_States_Prediction[:,0])
            contr_his_vy_pred_mod.extend(LPV_States_Prediction[:,1])
            contr_his_omega_pred_mod.extend(LPV_States_Prediction[:,2])

            contr_his_epsi_pred_cont.extend(Controller.xPred[:,3])
            contr_his_ey_pred_cont.extend(Controller.xPred[:,5])
            contr_his_s_pred_cont.extend(Controller.xPred[:,4])
            contr_his_vx_pred_cont.extend(Controller.xPred[:,0])
            contr_his_vy_pred_cont.extend(Controller.xPred[:,1])
            contr_his_omega_pred_cont.extend(Controller.xPred[:,2])
            contr_his_steer_pred_cont.extend(Controller.uPred[:,0])
            contr_his_duty_pred_cont.extend(Controller.uPred[:,1])






        ControllerCounter +=1
        rate.sleep()
    
    ### during termination send the 0.0 command to shut off the motors
    dutycycle_commands.publish(0.0)
    steering_commands.publish(0.0)
    rate.sleep()





    if save_reference == True:

        contr_his_time      = np.array(contr_his_time)
        contr_his_counter       = np.array(contr_his_counter)
        contr_his_time      = np.array(contr_his_time)
        contr_his_counter       = np.array(contr_his_counter)
        contr_his_x_est     = np.array(contr_his_x_est)
        contr_his_y_est     = np.array(contr_his_y_est)
        contr_his_yaw_est       = np.array(contr_his_yaw_est)
        contr_his_vx_est        = np.array(contr_his_vx_est)
        contr_his_vy_est        = np.array(contr_his_vy_est)
        contr_his_omega_est     = np.array(contr_his_omega_est)
        contr_his_ey_est        = np.array(contr_his_ey_est)
        contr_his_epsi_est      = np.array(contr_his_epsi_est)
        contr_his_s_est     = np.array(contr_his_s_est)

        contr_his_epsi_pred_mod     = np.array(contr_his_epsi_pred_mod)
        contr_his_ey_pred_mod       = np.array(contr_his_ey_pred_mod)
        contr_his_s_pred_mod        = np.array(contr_his_s_pred_mod)
        contr_his_vx_pred_mod       = np.array(contr_his_vx_pred_mod)
        contr_his_vy_pred_mod       = np.array(contr_his_vy_pred_mod)
        contr_his_omega_pred_mod        = np.array(contr_his_omega_pred_mod)
        contr_his_epsi_pred_cont        = np.array(contr_his_epsi_pred_cont)
        contr_his_ey_pred_cont      = np.array(contr_his_ey_pred_cont)
        contr_his_s_pred_cont       = np.array(contr_his_s_pred_cont)
        contr_his_vx_pred_cont      = np.array(contr_his_vx_pred_cont)
        contr_his_vy_pred_cont      = np.array(contr_his_vy_pred_cont)
        contr_his_omega_pred_cont       = np.array(contr_his_omega_pred_cont)
        contr_his_steer_pred_cont       = np.array(contr_his_steer_pred_cont)
        contr_his_duty_pred_cont        = np.array(contr_his_duty_pred_cont)
        



        contr_his_epsi_pred_mod     = contr_his_epsi_pred_mod.reshape(ControllerCounter, len(LPV_States_Prediction[:,3]))
        contr_his_ey_pred_mod       = contr_his_ey_pred_mod.reshape(ControllerCounter, len(LPV_States_Prediction[:,5]))
        contr_his_s_pred_mod        = contr_his_s_pred_mod.reshape(ControllerCounter, len(LPV_States_Prediction[:,4]))
        contr_his_vx_pred_mod       = contr_his_vx_pred_mod.reshape(ControllerCounter, len(LPV_States_Prediction[:,0]))
        contr_his_vy_pred_mod       = contr_his_vy_pred_mod.reshape(ControllerCounter, len(LPV_States_Prediction[:,1]))
        contr_his_omega_pred_mod        = contr_his_omega_pred_mod.reshape(ControllerCounter, len(LPV_States_Prediction[:,2]))

        contr_his_epsi_pred_cont        = contr_his_epsi_pred_cont.reshape(ControllerCounter, len(Controller.xPred[:,3]))
        contr_his_ey_pred_cont      = contr_his_ey_pred_cont.reshape(ControllerCounter, len(Controller.xPred[:,5]))
        contr_his_s_pred_cont       = contr_his_s_pred_cont.reshape(ControllerCounter, len(Controller.xPred[:,4]))
        contr_his_vx_pred_cont      = contr_his_vx_pred_cont.reshape(ControllerCounter, len(Controller.xPred[:,0]))
        contr_his_vy_pred_cont      = contr_his_vy_pred_cont.reshape(ControllerCounter, len(Controller.xPred[:,1]))
        contr_his_omega_pred_cont       = contr_his_omega_pred_cont.reshape(ControllerCounter, len(Controller.xPred[:,2]))
        contr_his_steer_pred_cont       = contr_his_steer_pred_cont.reshape(ControllerCounter, len(Controller.uPred[:,0]))
        contr_his_duty_pred_cont        = contr_his_duty_pred_cont.reshape(ControllerCounter, len(Controller.uPred[:,1]))



        contr_his["time"]       =   contr_his_time 
        contr_his["counter"]        =   contr_his_counter 
        contr_his["x_est"]      =   contr_his_x_est 
        contr_his["y_est"]      =   contr_his_y_est 
        contr_his["yaw_est"]        =   contr_his_yaw_est 
        contr_his["vx_est"]     =   contr_his_vx_est 
        contr_his["vy_est"]     =   contr_his_vy_est 
        contr_his["omega_est"]      =   contr_his_omega_est 
        contr_his["ey_est"]     =   contr_his_epsi_est 
        contr_his["epsi_est"]       =   contr_his_ey_est 
        contr_his["s_est"]      =   contr_his_s_est 
        contr_his["epsi_pred_mod"]      =   contr_his_epsi_pred_mod 
        contr_his["ey_pred_mod"]        =   contr_his_ey_pred_mod 
        contr_his["s_pred_mod"]     =   contr_his_s_pred_mod 
        contr_his["vx_pred_mod"]        =   contr_his_vx_pred_mod 
        contr_his["vy_pred_mod"]        =   contr_his_vy_pred_mod 
        contr_his["omega_pred_mod"]     =   contr_his_omega_pred_mod 
        contr_his["epsi_pred_cont"]     =   contr_his_epsi_pred_cont 
        contr_his["ey_pred_cont"]       =   contr_his_ey_pred_cont 
        contr_his["s_pred_cont"]        =   contr_his_s_pred_cont 
        contr_his["vx_pred_cont"]       =   contr_his_vx_pred_cont 
        contr_his["vy_pred_cont"]       =   contr_his_vy_pred_cont 
        contr_his["omega_pred_cont"]        =   contr_his_omega_pred_cont 
        contr_his["steer_pred_cont"]        =   contr_his_steer_pred_cont 
        contr_his["duty_pred_cont"]     =   contr_his_duty_pred_cont 



            #############################################################
        day         = '11_06_20'
        num_test    = 'controller_output'

        newpath = ('/').join(__file__.split('/')[:-2]) + '/data/'+day+'/'+num_test+'/' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        print "newpath+'/'"

        planner_name = rospy.get_param("/control/trial_name")
        np.save(newpath+'/'+planner_name, contr_his)



















    quit()


####################### VISUALIZATION #######################################################

def InitFigure_XY(map,HW):
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


    plt.show()

    return fig, axtr, line_tr, line_pred, line_trs, line_cl, line_gps_cl, rec, rec_sim


def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l*np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y


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



def Body_Frame_Errors (x, y, psi, xd, yd, psid, s0, vx, vy, curv, dt):

    ex = (x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)

    ey = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)

    epsi = wrap(psi - psid)

    #s = s0 + np.sqrt(vx*vx + vy*vy) * dt
    s = s0 + ( (vx*np.cos(epsi) - vy*np.sin(epsi)) / (1-ey*curv) ) * dt

    return s, ex, ey, epsi






if __name__ == "__main__":

    try:
        main()

    except rospy.ROSInterruptException:
        pass



