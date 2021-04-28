#!/usr/bin/env python
"""
    File name: controllerMain.py (LPV-MPC controller)
    Author: Eugenio Alcala
    Email: euge2838@gmail.com
    Date: 18/02/2020
    Python Version: 2.7.12
"""


import os
import sys
import datetime
import rospy
import numpy as np
import scipy.io as sio
import pdb
import pickle
import matplotlib.pyplot as plt

sys.path.append(sys.path[0]+'/ControllerObject')
sys.path.append(sys.path[0]+'/Utilities')
sys.path.append(sys.path[0]+'/data')

from lpv_mpc.msg import control_actions
from std_msgs.msg import Bool, Float32
from utilities import wrap
from dataStructures import EstimatorData
from PathFollowingLPVMPC import PathFollowingLPV_MPC
from trackInitialization import Map


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():

    rospy.init_node("LPV-MPC")
    accel_commands     = rospy.Publisher('control/accel', Float32, queue_size=1)
    steering_commands  = rospy.Publisher('control/steering', Float32, queue_size=1)
    controller_Flag    = rospy.Publisher('controller_flag', Bool, queue_size=1)

    N                   = rospy.get_param("/control/N")
    Vx_ref              = rospy.get_param("/control/vel_ref")

    loop_rate           = rospy.get_param("/control/Hz")
    dt                  = 1.0/loop_rate
    rate                = rospy.Rate(loop_rate)
    controller_Flag_msg = Bool()

    Steering_Delay      = 0
    Velocity_Delay      = 0

    #u_steer             = 0
    #u_acc               = 0

    cmd                 = control_actions()
    cmd.servo           = 0
    cmd.motor           = 0

#########################################################
#########################################################


    # Objects initializations
    estimatorData       = EstimatorData()                                      # Map
    map                 = Map()

    first_it            = 1

    GlobalState         = np.zeros(6)
    LocalState          = np.zeros(6)
    LapNumber           = 0
    HalfTrack           = 0
    #uApplied            = np.array([0.0, 0.0])
    # oldU                = np.array([0.0, 0.0])

    vel_th              = rospy.get_param("/control/vel_th")

    vel_ref             = np.zeros([N,1])
    curv_ref            = np.zeros([N,1])
    y_ref               = np.zeros([N,1])
    yaw_ref             = np.zeros([N,1])
    x_ref               = np.zeros([N,1])

    # Loop running at loop rate
    TimeCounter         = 0
    PlannerCounter      = 0
    test_gen            = rospy.get_param("planning_mode")# test generated plan 0 = offline plan, 1 = vel profile, 2 = map
    test_type           = 1 # 1=vel cte 0=acc cte

    controller_Flag_msg.data = False
    controller_Flag.publish(controller_Flag_msg)

    rospy.sleep(1.2)    # Soluciona los problemas de inicializacion esperando a que el estimador se inicialice bien
    SS  = 0.0

    path = os.path.dirname(os.path.abspath(__file__))+'/data/UPC_ADcar_refs.mat'
    print ("path")
    CURV_Planner        = sio.loadmat(path)['CURV']
    VEL_Planner         = sio.loadmat(path)['new_Vx']
    VELY_Planner        = sio.loadmat(path)['new_Vy']
    X_Planner           = sio.loadmat(path)['new_X']
    Y_Planner           = sio.loadmat(path)['new_Y']
    PSI_Planner         = sio.loadmat(path)['new_Theta']
    PSIdot_Planner      = sio.loadmat(path)['new_Wz']

    #maximum values
    v_max               = rospy.get_param("max_vel")
    v_min               = rospy.get_param("min_vel")
    ac_max              = rospy.get_param("dutycycle_max")
    ac_min              = rospy.get_param("dutycycle_min")
    str_max             = rospy.get_param("steer_max")
    str_min             = rospy.get_param("steer_min")
    ey_max              = rospy.get_param("lat_e_max")
    etheta_max          = rospy.get_param("orient_e_max")

    dstr_max            = str_max*0.1
    dstr_min            = str_min*0.1
    dac_max             = ac_max*0.1
    dac_min             = ac_min*0.1

    vx_scale            = 1/((v_max-v_min)**2)
    acc_scale           = 1/((ac_max-ac_min)**2)
    str_scale           = 1/((str_max-str_min)**2)
    ey_scale            = 1/((ey_max+ey_max)**2)
    etheta_scale        = 1/((etheta_max+etheta_max)**2)
    dstr_scale          = 1/((dstr_max-dstr_min)**2)
    dacc_scale          = 1/((dac_max-dac_min)**2)



    # Q  = 0.9 * np.diag([0.3*vx_scale, 0.0, 0.001, 0.1*etheta_scale, 0.0, 0.6*ey_scale])
    # R  = 0.05 * np.diag([0.05*str_scale,0.1*acc_scale])     # delta, a
    # dR = 0.1 * np.array([0.01*dstr_scale,0.01*dacc_scale])  # Input rate cost u



    # xmin = np.array([v_min, -3., -3., -100, -10000., -ey_max])
    # xmax = np.array([v_max, 3., 3., 100, 10000., ey_max])

    Q  = 0.9 * np.array([0.6*vx_scale, 0.0, 0.001, 0.8*etheta_scale, 0.0, 0.1*ey_scale])
    R  = 0.05 * np.array([0.0005*str_scale,0.1*acc_scale])     # delta, a
    dR = 0.1 * np.array([0.01*dstr_scale,0.01*dacc_scale])  # Input rate cost u

    ''' Berkeley tunning:
    Q  = np.diag([100, 1, 1, 50, 0, 1000])
    R  = np.diag([1, 0.5])  # delta, a
    dR = np.array([60, 45])  # Input rate cost u
    '''

    Controller  = PathFollowingLPV_MPC(Q, R, dR, N, Vx_ref, dt, map, test_gen, "OSQP", Steering_Delay, Velocity_Delay)

###----------------------------------------------------------------###

    LPV_States_Prediction   = np.zeros((N,6))
    acc_ref = 0.25
    if test_gen == 1:
        if test_type:
            vel_ref = Vx_ref*np.ones([N,1])
            for i in range(1,N):
                x_ref[i,0] = x_ref[i-1,0] + vel_ref[i,0]*dt
        else:
            for i in range(1,N):
                vel_ref[i,0]  = vel_ref[i-1,0] + acc_ref*dt

                if vel_ref[i,0]>vel_th:
                    acc_ref = - acc_ref

                if vel_ref[i,0]<0:
                    vel_ref[i,0] = 0
                    acc_ref = - acc_ref

                x_ref[i,0] = x_ref[i-1,0] + vel_ref[i,0]*dt

    print("Controller is running")


    while (not rospy.is_shutdown()):

        startTimer = datetime.datetime.now()

        # Read Measurements
        GlobalState[:] = estimatorData.CurrentState  # The current estimated state vector [vx vy w x y psi]
        GlobalState[5] = wrap(GlobalState[5] - 2 * np.pi * LapNumber)
        LocalState[:]  = estimatorData.CurrentState  # [vx vy w x y psi]




        ## Choose the planning mode:
        if test_gen == 0:
            if (PlannerCounter + N) < len(X_Planner[0]):

                x_ref[:,:]          = X_Planner[0][PlannerCounter:PlannerCounter+N].reshape(-1,1)
                y_ref[:,:]          = Y_Planner[0][PlannerCounter:PlannerCounter+N].reshape(-1,1)
                yaw_ref[:,:]        = PSI_Planner[0][PlannerCounter:PlannerCounter+N].reshape(-1,1)
                vel_ref[:,:]        = VEL_Planner[0][PlannerCounter:PlannerCounter+N].reshape(-1,1)
                curv_ref[:,:]       = CURV_Planner[0][PlannerCounter:PlannerCounter+N].reshape(-1,1)

            else:
                print ('end of plan')
                break

            LocalState[4], Xerror, LocalState[5], LocalState[3] = Body_Frame_Errors(GlobalState[3],
                GlobalState[4], GlobalState[5], x_ref[0,0], y_ref[0,0], wrap(yaw_ref[0,0]), SS, LocalState[0],
                LocalState[1], curv_ref[0,0], dt )



        elif test_gen == 1:
            if test_type:
                for i in range(0,N):
                    vel_ref = Vx_ref*np.ones([N,1])
                    if i == N-1:
                        x_ref[i,0] = x_ref[i-1,0] + vel_ref[i,0]*dt
                    else:
                        x_ref[i,0] = x_ref[i+1,0]
            else:
                for i in range(0,N):

                    if i == N-1:
                        vel_ref[i,0]  = vel_ref[i-1,0] + acc_ref*dt
                        if vel_ref[i,0]>vel_th:
                            acc_ref = - acc_ref

                        if vel_ref[i,0]<0:
                            vel_ref[i,0] = 0
                            acc_ref = - acc_ref

                        x_ref[i,0]    = x_ref[i-1,0] + vel_ref[i,0]*dt
                    else:
                        vel_ref[i,0] = vel_ref[i+1,0]
                        x_ref[i,0]   = x_ref[i+1,0]

            # OUT: s, ex, ey, epsi
            LocalState[4], Xerror, LocalState[5], LocalState[3] = Body_Frame_Errors(GlobalState[3],
                GlobalState[4], GlobalState[5], x_ref[0,0], y_ref[0,0], wrap(yaw_ref[0,0]), SS, LocalState[0],
                LocalState[1], curv_ref[0,0], dt )



        elif test_gen == 2:

            # OUT: s, ey, epsi       IN: x, y, psi
            LocalState[4], LocalState[5], LocalState[3], insideTrack = map.getLocalPosition(
                GlobalState[3], GlobalState[4], wrap(GlobalState[5]))


            print("x:  ", GlobalState[3],"y:  ", GlobalState[4], "yaw", wrap(GlobalState[5]))
            print("\n")
            print("vx vy w epsi s ey: ", LocalState)
            print("\n")

            vel_ref         = np.ones([6,1])
            curv_ref        = np.zeros(N) # This is not used, instead the map is employed to get the road curvature

            # Check if the lap has finished
            if LocalState[4] >= 3*map.TrackLength/4:
                HalfTrack = 1

        #uApplied    = np.array([cmd.servo, cmd.motor])


        Controller.OldSteering.append(cmd.servo) # meto al final del vector
        Controller.OldAccelera.append(cmd.motor)
        Controller.OldSteering.pop(0)
        Controller.OldAccelera.pop(0)

        SS = LocalState[4] 

        ###################################################################################################
        ###################################################################################################

        ### END OF THE LAP.
        # PATH TRACKING:
        if ( HalfTrack == 1 and (LocalState[4] <= map.TrackLength/4)):
            HalfTrack       = 0
            LapNumber       += 1
            SS              = 0

        ###################################################################################################
        ###################################################################################################

        if first_it < 2:

            accel_rate  = 0.0
            delta = 0.001
            # xx, uu      = predicted_vectors_generation(N, LocalState, accel_rate, dt)
            xx, uu      = predicted_vectors_generation_new(N, LocalState, accel_rate, delta, dt)
            
            Controller.uPred = uu
            first_it    = first_it + 1

        else:

            print ("Controller.uPred shape", Controller.uPred.shape)
            LPV_States_Prediction, A_L, B_L, C_L = Controller.LPVPrediction(LocalState[0:6], Controller.uPred, vel_ref, curv_ref, test_gen)
            # Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, Controller.uPred, vel_ref, curv_ref, A_L, B_L, C_L, first_it)

            print "LPV_States_Prediction",LPV_States_Prediction.shape 
            # print (B_L.shape)
            Controller.MPC_solve(A_L, B_L, Controller.uPred, LocalState[0:6], Vx_ref)

            if Controller.feasible == 0:
                controller_Flag_msg.data = True
                controller_Flag.publish(controller_Flag_msg)
            else:
                controller_Flag_msg.data = False
                controller_Flag.publish(controller_Flag_msg)


        print('control actions',"delta", Controller.uPred[0,0],"dutycycle", Controller.uPred[0,1])
        print('\n')

        # endTimer = datetime.datetime.now()
        # deltaTimer = endTimer - startTimer
        # print("Elapsed Solver Time: ", deltaTimer.total_seconds())
        # print('\n')


        ###################################################################################################
        ###################################################################################################


        #if first_it > 10000:
        #    new_LPV_States_Prediction = LPV_States_Prediction[0, :]
        #    for i in range(1,N):
        #        new_LPV_States_Prediction = np.hstack((new_LPV_States_Prediction, LPV_States_Prediction[i,:]))
        #    PREDICTED_DATA[TimeCounter,:] = new_LPV_States_Prediction



        ## Publish input simulation IRI ##
        cmd.servo = Controller.uPred[0,0]
        cmd.motor = Controller.uPred[0,1]
        accel_commands.publish(Controller.uPred[0,1])
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

