#!/usr/bin/env python

###############################################
##      IRI 1:10 Autonomous Car              ##
##      Supervisor: Puig Cayuela Vicenc      ##
##      Author: Shivam Chaubey               ##
##      Email: shivam.chaubey1006@gmail.com  ##
##      Date: 11/03/2021                     ##
###############################################


import os
import sys
import datetime
import rospy
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sensor_fusion.msg import sensorReading
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')

from std_msgs.msg import Bool, Float32
from PathFollowingLPVMPC import PathFollowingLPV_MPC
from trackInitialization import Map
from math import pi 
import time

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})






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






def main():

    rospy.init_node("LPV_MPC")
    
    ########################## Control output ##################################
    dutycycle_commands  = rospy.Publisher('control/accel', Float32, queue_size=1)
    steering_commands   = rospy.Publisher('control/steering', Float32, queue_size=1)

    dutycycle_thres     = rospy.get_param("/control/vel_th") # dutycycle Deadzone
    
    N                   = rospy.get_param("/control/Horizon")
    Vx_ref              = rospy.get_param("/control/vel_ref")

    loop_rate           = rospy.get_param("/control/Hz")
    dt                  = 1.0/loop_rate
    rate                = rospy.Rate(loop_rate)

    cmd_servo           = Float32()
    cmd_motor           = Float32()

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

        # Read Measurements
        GlobalState[:] = estimatorData.CurrentState  # The current estimated state vector [vx vy w x y psi]
        GlobalState[5] =  wrap(GlobalState[5] - 2 * np.pi * LapNumber) #(GlobalState[5] + pi) % (2 * pi) - pi
        LocalState[:]  = estimatorData.CurrentState  # [vx vy w x y psi]

        if test_gen == 2:

            # OUT: s, ey, epsi       IN: x, y, psi
            LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
                GlobalState[3], GlobalState[4], wrap(GlobalState[5]))

            print("\n")
            print("vx vy w x y yaw: ", GlobalState)
            print("\n")
            print("vx vy w epsi s ey: ", LocalState)
            print("\n")

            vel_ref         = np.ones([N,1])
     
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
            xx, uu      = predicted_vectors_generation_new(N, LocalState, duty_cycle, delta, dt)
            
            Controller.uPred = uu
            
            # first_it    = first_it + 1
            Controller.uminus1 = Controller.uPred[0,:]

        else:

            print "Controller.uminus1", Controller.uminus1

            LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref)

            if first_it == 5:
                print "MPC setup"
                # Controller.simple_MPC_setup(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 
                # Controller.simple_MPC_solve()

                LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction_setup(LocalState[0:6], Controller.uPred, vel_ref)

                Controller.MPC_integral_setup(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 
                # Controller.MPC_integral_solve()


            # if 5 < first_it < 7:
            else:
    
                t1 = time.time() 
                print "MPC update"
                Controller.MPC_integral_update(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref) 
                Controller.MPC_integral_solve()
                print "time taken to solve", time.time() - t1


            # print ("Controller.uPred shape", Controller.uPred.shape)
            # Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, Controller.uPred, vel_ref, curv_ref, A_L, B_L, C_L, first_it)


            # print "LPV_States_Prediction",LPV_States_Prediction.shape 
            # print "LocalState[0:6]",LocalState[0:6]
            # print (B_L.shape)
                # Controller.simple_MPC_update(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref)
                # Controller.simple_MPC_solve()


            # Controller.MPC_integral_solve2(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref)

                        # Solve
            # res = Controller.prob.solve()

            # print "TimeCounter",TimeCounter
            # # Check solver status
            # if res.info.status != 'solved':
            #     print ('OSQP did not solve the problem!')
            

            # Solution = res.x

            # print "controller to be applied", Solution[(N+1)*nx:(N+1)*nx + nu]
            # print "Solution shape", Solution.shape
            

            # Controller.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(nx * (N + 1))]), (N + 1, nx)))).T
            # Controller.uPred = np.squeeze(np.transpose(np.reshape((Solution[nx * (N + 1) + np.arange(nu * N)]), (N, nu)))).T

            # print "Controller.uPred", Controller.uPred
            if Controller.uPred[0,0]  == None:
                Controller.uPred[0,0] = 0.0
            if Controller.uPred[0,1]  == None:  
                Controller.uPred[0,1] = 0.0

            Controller.uminus1 = Controller.uPred[0,:] 
            print 'Controller.uminus1', Controller.uminus1, 'Controller.uPred[0,:]', Controller.uPred[0,:]            
            # LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref*0.0)

            # Controller.MPC_update(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref)


            # self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )



            # if Controller.feasible == 0:
            #     controller_Flag_msg.data = True
            #     controller_Flag.publish(controller_Flag_msg)
            # else:
            #     controller_Flag_msg.data = False
            #     controller_Flag.publish(controller_Flag_msg)

            # if first_it ==7:
            #     break
        
        first_it    = first_it + 1

        print "TimeCounter", TimeCounter
        print('control actions',"delta", Controller.uPred[0,0],"dutycycle", Controller.uPred[0,1])
        print('\n')

        # first_it    = first_it + 1
        ## Publish input simulation ##
        cmd_servo = Controller.uPred[0,0]
        cmd_motor = Controller.uPred[0,1]
        dutycycle_commands.publish(Controller.uPred[0,1])
        steering_commands.publish(Controller.uPred[0,0])


        PlannerCounter  += 1
        TimeCounter     += 1
        rate.sleep()

    quit()


# ===============================================================================================================================
# ==================================================== END OF MAIN ==============================================================
# ===============================================================================================================================


def Body_Frame_Errors (x, y, psi, xd, yd, psid, s0, vx, vy, curv, dt):

    ex = (x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)

    ey = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)

    epsi = wrap(psi - psid)

    s = s0 + ( (vx*np.cos(epsi) - vy*np.sin(epsi)) / (1-ey*curv) ) * dt

    return s, ex, ey, epsi


def predicted_vectors_generation(Hp, x0, accel_rate, dt):

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
    delta   = np.array([[ 0 for i in range(0, Hp)]])

    for i in range(0, Hp):
        Vx[i+1] = Vx[i] + accel[0,i] * dt
        S[i+1]  = S[i] + Vx[i] * dt

    xx  = np.hstack([ Vx, Vy, W, Epsi ,S ,Ey]) # [vx vy omega theta_e s y_e]

    uu  = np.hstack([delta.transpose(),accel.transpose()])

    return xx, uu

def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle

def predicted_vectors_generation_new(Hp, x0, accel_rate, delta, dt):

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

def predicted_vectors_generation_V2(Hp, x0, accel_rate, dt):

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

    Accel   = 1.0
    curv    = 0

    for i in range(0, Hp):
        Vy[i+1]      = x0[1]
        W[i+1]       = x0[2]
        Ey[i+1]      = x0[3]
        Epsi[i+1]    = x0[4]

    Accel   = Accel + np.array([ (accel_rate * i) for i in range(0, Hp)])

    for i in range(0, Hp):
        Vx[i+1]    = Vx[i] + Accel[i] * dt
        S[i+1]      = S[i] + Vx[i] * dt

    xx  = np.hstack([ Vx, Vy, W, Epsi ,S ,Ey]) # [vx vy omega theta_e s y_e]
    uu = np.zeros(( Hp, 1 ))
    return xx, uu


if __name__ == "__main__":

    try:
        main()

    except rospy.ROSInterruptException:
        pass

