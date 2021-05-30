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
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf
from PathFollowingLPVMPC import PathFollowingLPV_MPC
import time
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
from trackInitialization import Map
from controller.msg import mpcPrediction
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

########## Full state estimation from observer #############
class EstimatorData(object):
    """Data from estimator"""
    def __init__(self):

        rospy.Subscriber("est_state_info", sensorReading, self.estimator_callback)
        self.CurrentState = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print "Subscribed to observer"
    
    def estimator_callback(self, msg):
        """
        Unpack the messages from the estimator
        """
        self.CurrentState = np.array([msg.vx, msg.vy, msg.yaw_rate, msg.X, msg.Y, msg.yaw]).T

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
    dt                  = 1.0/loop_rate
    rate                = rospy.Rate(loop_rate)

    cmd_servo           = Float32()
    cmd_motor           = Float32()
    
    LPV_prediction_states = Path()
    MPC_prediction_states = Path()

    estimatorData       = EstimatorData()   # observer reading
    rospy.sleep(1.2)    # wait for observer initialization or make a flag in that node to know if the observer started.

    map                 = Map()             # Fixed map track

    first_it            = 0

    GlobalState         = np.zeros(6)       # vehicle kinematic and dynamic states
    LocalState          = np.zeros(6)       # vehicle dynamic and error states
    LapNumber           = 0                 
    HalfTrack           = 0

    '''References for the controller to attain this states'''
    vel_ref             = np.zeros([N,1])  
    curv_ref            = np.zeros([N,1])
    y_ref               = np.zeros([N,1])
    yaw_ref             = np.zeros([N,1])
    x_ref               = np.zeros([N,1])

    # Loop running at loop rate
    TimeCounter         = 0
    PlannerCounter      = 0
    test_gen            = rospy.get_param("planning_mode")# test generated plan 0 = offline plan, 1 = vel profile, 2 = map

    SS  = 0.0


    Controller  = PathFollowingLPV_MPC(N, Vx_ref, dt, map)
    rospy.sleep(1.2)

    print "controller initialized"

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
        GlobalState[5] =  wrap(GlobalState[5] - 2 * np.pi * LapNumber) 
        '''
        The above code converts the continuous yaw angle between [-pi,pi] because
        the map.getLocalPosition function work in this domain.  '''
        LocalState[:]  = estimatorData.CurrentState  # [vx vy w x y psi]

        ## This is for the map case you can add your own cases for example if you want the vehicle to follow other trajectory make another case.
        if test_gen == 2:

            # OUT: s, ey, epsi       IN: x, y, psi
            LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
                GlobalState[3], GlobalState[4], wrap(GlobalState[5]))

            # print("\n")
            # print("vx vy w: ", GlobalState[:3])
            # print("\n")
            # print("x y yaw: ", GlobalState[3:])
            # print("\n")
            # print("epsi s ey: ", LocalState[3:])
            # print("\n")

            vel_ref         = np.ones([N,1])*Vx_ref
     
            # Check if the lap has finished
            if LocalState[4] >= 3*map.TrackLength/4:
                HalfTrack = 1
                print 'the lap has finished'


        SS = LocalState[4] 

        ###################################################################################################
        ###################################################################################################

        ### END OF THE LAP.
        # PATH TRACKING:
        if ( HalfTrack == 1 and (LocalState[4] <= map.TrackLength/4)):
            HalfTrack       = 0
            LapNumber       += 1
            SS              = 0
            print "END OF THE LAP"

        ###################################################################################################
        ###################################################################################################

        if first_it < 5:

            duty_cycle  = 0.0

            delta = 0.01
            # xx, uu      = predicted_vectors_generation(N, LocalState, accel_rate, dt)
            xx, uu      = predicted_vectors_generation(N, LocalState, duty_cycle, delta, dt)
            
            Controller.uPred = uu
            Controller.xPred = xx
            
            # first_it    = first_it + 1
            Controller.uminus1 = Controller.uPred[0,:]

        else:

            # print "Controller.uminus1", Controller.uminus1

            LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref)
            # print "LPV_States_Prediction", LPV_States_Prediction
            
            if first_it == 5:
                print "MPC setup"
                LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_setup()
                Controller.MPC_setup(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 
                

            else:
                # publish_predicted.LPV_msg_update(map, LPV_States_Prediction)
                # publish_predicted.LPV_msg_update(map, Controller.xPred)

                # print "Controller.xPred", Controller.xPred
                print "MPC update"
                t1 = time.time() 
                Controller.MPC_update(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 
                Controller.MPC_solve()
                print "time taken to solve", time.time() - t1

            lpvPrediction_msg.epsi = LPV_States_Prediction[:,3]
            lpvPrediction_msg.s = LPV_States_Prediction[:,4]
            lpvPrediction_msg.ey = LPV_States_Prediction[:,5]

            mpcPrediction_msg.epsi = Controller.xPred[:,3]
            mpcPrediction_msg.s = Controller.xPred[:,4]
            mpcPrediction_msg.ey = Controller.xPred[:,5]


            Controller.uminus1 = Controller.uPred[0,:] 

        first_it    = first_it + 1

        print('control actions',"delta", Controller.uPred[0,0],"dutycycle", Controller.uPred[0,1])
        print('\n')

        if Controller.feasible == 0.0:
            Controller.uPred[0,0] = 0.0
            Controller.uPred[0,1] = 0.0
        ## Publish controls ##
        cmd_servo = Controller.uPred[0,0]
        cmd_motor = Controller.uPred[0,1]
        dutycycle_commands.publish(Controller.uPred[0,1])
        steering_commands.publish(Controller.uPred[0,0])
        MPC_prediction_state_pub.publish(mpcPrediction_msg)
        LPV_prediction_state_pub.publish(lpvPrediction_msg)
        rate.sleep()

    quit()



if __name__ == "__main__":

    try:
        main()

    except rospy.ROSInterruptException:
        pass

