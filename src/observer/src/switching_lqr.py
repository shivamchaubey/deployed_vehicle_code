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
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/observer/src/')

from observer_functions import plot_states, getCarPosition, append_sensor_data, append_control_data, \
     Continuous_AB_Comp, L_Computation, data_retrive, data_retrive_est, meas_retrive, load_switchingLQRgain, \
      unwrap, wrap, yaw_error_throw
from sensor_subscriber import vehicle_control, motor_encoder, IMU, fiseye_cam


def main():
    rospy.init_node('switching_lqr_state_estimation', anonymous=True)

    loop_rate   = rospy.get_param("switching_lqr_observer/publish_frequency")
    rate        = rospy.Rate(loop_rate)
    time0       = rospy.get_rostime().to_sec()
    
    counter     = 0
    record_data =    rospy.get_param("switching_lqr_observer/record_data")
    visualization  = rospy.get_param("switching_lqr_observer/visualization")

    LQR_gain, seq, sched_var = load_switchingLQRgain()

    N_enc  = rospy.get_param("switching_lqr_observer/enc_MA_window")
    N_fcam = rospy.get_param("switching_lqr_observer/fcam_MA_window")
    N_imu  = rospy.get_param("switching_lqr_observer/imu_MA_window")

    enc    = motor_encoder(time0, N_enc)
    fcam   = fiseye_cam(time0, N_fcam)
    imu    = IMU(time0, N_imu)

    time.sleep(3)
    print  "yaw_offset", fcam.yaw
    imu.yaw_offset = imu.yaw - fcam.yaw
    control_input = vehicle_control(time0)
    time.sleep(3)

    print "fcam.yaw",fcam.yaw
    
    if visualization == True:
        x_lim = 10
        y_lim = 10
        (fig, axtr, line_est, line_ol, line_meas, rec_est, rec_ol, rec_meas) = _initializeFigure_xy(x_lim,y_lim)

        ol_x_his     = []
        est_x_his    = []
        meas_x_his   = []
        ol_y_his     = []
        est_y_his    = []
        meas_y_his   = []

    # delay  = 5
    # offset = pi/2
    print ("<<<< Initializing IMU orientation >>>>")
    # imu.calibrate_imu(delay,offset)    
    # fcam.calibrate_fcam(delay,R_tf)
    print ("<<<< ORIGIN SET AND CALIBRATION DONE >>>>")


    ###### observation matrix ######
    C       =  np.array([[1, 0, 0, 0, 0, 0], 
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]]) 

    
    vy = 0.0

    yaw_curr = (imu.yaw)

    est_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, yaw_curr ]).T


    est_state_hist = [] 
    est_state_hist.append(est_state)
    
    est_state_msg = sensorReading()
    est_state_pub  = rospy.Publisher('est_state_info', sensorReading, queue_size=1)
    est_state_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}


    #### Open loop simulation ###
    ol_state = np.array([enc.vx, vy, imu.yaw_rate, fcam.X, fcam.Y, yaw_curr ]).T

    ol_state_hist = [] 
    ol_state_hist.append(ol_state)
    
    ol_state_msg = sensorReading()
    ol_state_pub  = rospy.Publisher('ol_state_info', sensorReading, queue_size=1)

    meas_state_pub  = rospy.Publisher('meas_state_info', sensorReading, queue_size=1)
    meas_state_msg = sensorReading()
    meas_state_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}


    control_data = control()
    control_hist = {'timestamp_ms_dutycycle':[],'timestamp_ms_steer':[],'steering':[], 'duty_cycle':[]}

    # enc_pub  = rospy.Publisher('encoder_fused', sensorReading, queue_size=1)
    # enc_data = sensorReading()
    # enc_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # imu_pub  = rospy.Publisher('imu_fused', sensorReading, queue_size=1)
    # imu_data = sensorReading()
    # imu_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # fcam_pub  = rospy.Publisher('fcam_fused', sensorReading, queue_size=1)
    # fcam_data = sensorReading()
    # fcam_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        

        
    # enc_MA_pub  = rospy.Publisher('encoder_MA_fused', sensorReading, queue_size=1)
    # enc_MA_data = sensorReading()
    # enc_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # imu_MA_pub  = rospy.Publisher('imu_MA_fused', sensorReading, queue_size=1)
    # imu_MA_data = sensorReading()
    # imu_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}
        
    # fcam_MA_pub  = rospy.Publisher('fcam_MA_fused', sensorReading, queue_size=1)
    # fcam_MA_data = sensorReading()
    # fcam_MA_hist = {'timestamp_ms':[], 'X':[], 'Y':[], 'roll':[], 'yaw':[], 'pitch':[], 'vx':[], 'vy':[], 'yaw_rate':[], 'ax':[], 'ay':[], 's':[], 'x':[], 'y':[]}

    curr_time = rospy.get_rostime().to_sec() - time0
     
    prev_time = curr_time 
    
    u = [0,0]
    
    #### YAW CORRECTION ####
    angle_past = imu.yaw
    

    while not (rospy.is_shutdown()):
        
        u = np.array([control_input.duty_cycle, control_input.steer]).T

        # angle_acc = imu.yaw
        angle_acc = unwrap(angle_past, imu.yaw)  

        angle_past = angle_acc
        
        # print "fcam.yaw",fcam.yaw

        curr_time = rospy.get_rostime().to_sec() - time0


        y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, angle_acc]).T 
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, fcam.yaw]).T
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X, fcam.Y, yaw_correction(fcam.yaw)]).T
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X_MA, fcam.Y_MA, imu.yaw]).T 
        # y_meas = np.array([enc.vx, imu.yaw_rate, fcam.X_MA, fcam.Y_MA, yaw_correction(imu.yaw)]).T 




        dt = curr_time - prev_time 
        
        if u[0] > 0.05:


            # yaw_trans = wrap(yaw_correction(y_meas[-1]))
            yaw_trans = (est_state[5] + pi) % (2 * pi) - pi
            # yaw_trans = est_state[5]
            # %% quadrant case
            if 0 <= yaw_trans <= pi/2:
            # % 1st quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[0], sched_var[0], seq[0])
               
            elif pi/2 < yaw_trans <= pi:
            # % 2nd quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[1], sched_var[1], seq[1])
                    
            elif -pi <= yaw_trans <= -pi/2:
            # % 3rd quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[2], sched_var[2], seq[2])
                
            elif (-pi/2 < yaw_trans < 0):
            # % 4th quadrant
                 L_gain = L_Computation(est_state[0], est_state[1], est_state[2], yaw_trans, u[1], LQR_gain[3], sched_var[3], seq[3])
                
            else:
                
                print "est theta", yaw_trans, yaw_trans*180.0/pi 

                display("ERROR Normalize the theta")



                yaw_error_throw()



            # # %% quadrant case
            # if 0 <= est_state[5] <= pi/2:
            # # % 1st quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[0], sched_var[0], seq[0])
               
            # elif pi/2 < est_state[5] <= pi:
            # # % 2nd quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[1], sched_var[1], seq[1])
                    
            # elif pi < est_state[5] <= 3*pi/2:
            # # % 3rd quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[2], sched_var[2], seq[2])
                
            # elif (3*pi/2 < est_state[5] < 2*pi):
            # # % 4th quadrant
            #      L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain[3], sched_var[3], seq[3])
                
            # else:
                
            #     print "est theta", est_state[5], 'Measured theta', yaw_correction(fcam.yaw)

            #     display("ERROR Normalize the theta")



                # yaw_error_throw()





            # print ("u",u)
            
            ####### LQR ESTIMATION ########
            A_obs, B_obs = Continuous_AB_Comp(est_state[0], est_state[1], est_state[2], est_state[5], u[1])
            # L_gain = L_Computation(est_state[0], est_state[1], est_state[2], est_state[5], u[1], LQR_gain, sched_var, seq)
            
            est_state  = est_state + ( dt * np.dot( ( A_obs - np.dot(L_gain, C) ), est_state )
                            +    dt * np.dot(B_obs, u)
                            +    dt * np.dot(L_gain, y_meas) )
            
            # print ("time taken for estimation ={}".format(rospy.get_rostime().to_sec() - time0 - curr_time))
            
            ##### OPEN LOOP SIMULATION ####
            A_sim, B_sim = Continuous_AB_Comp(ol_state[0], ol_state[1], ol_state[2], ol_state[5], u[1])
            ol_state = ol_state + dt*(np.dot(A_sim,ol_state) + np.dot(B_sim,u)) 
            # yaw_check += wrap(fcam.yaw)

        if abs(u[0]) <= 0.05:
                #     # vehicle_sim.vehicle_model(u, simulator_dt)
                    # if vehicle_sim.vx <= 0.01 :
            est_state[:-3] = 0.000001 
            ol_state[:-3] = 0.000001


        # else:
        #     if enc.vx == 0.0:
        #         est_state[0] = 0.0
        #         ol_state[0]  = 0.0
        #         est_state[1] = 0.0
        #         ol_state[1]  = 0.0

        #     if imu.yaw_rate <= 0.018:    
        #         est_state[2] = 0.0
        #         ol_state[2]  = 0.0

        print "\n <<<<<<<<< PRE WRAP >>>>>>>>>>>>>"
        print "est_state",est_state
        print "ol_state", ol_state


        # est_state[5] = wrap(est_state[5])
        # ol_state[5] = wrap(ol_state[5])
        # est_state[5] = yaw_correction(est_state[5])
        # ol_state[5] = yaw_correction(ol_state[5])
        
        print "\n <<<<<<<<< STATS >>>>>>>>>>>>>"
        print "measured states", y_meas
        print "est_state",est_state
        print "ol_state", ol_state
        print "input u", u
        print "dt", dt

        AC_sig = 0
        CC_sig = 0

        est_msg = data_retrive_est(est_state_msg, est_state, y_meas[-1], AC_sig, CC_sig)
        est_state_pub.publish(est_msg) ## remember we want to check the transformed yaw angle for debugging that's why 
                                                                                    ##publishing this information in the topic of "s" which is not used for any purpose. 
        append_sensor_data(est_state_hist, est_msg)
  

        ol_state_pub.publish(data_retrive(ol_state_msg, ol_state))
        # ol_state_hist.append(ol_state)

        meas_msg = meas_retrive(meas_state_msg, y_meas)
        meas_state_pub.publish(meas_msg)
        append_sensor_data(meas_state_hist, meas_msg)

        control_msg = control_input.data_retrive(control_data)        
        append_control_data(control_hist, control_msg)


        # enc_msg = enc.data_retrive(enc_data)
        # enc_pub.publish(enc_msg)
        # append_sensor_data(enc_hist, enc_msg)


        # enc_MA_msg = enc.data_retrive_MA(enc_MA_data)
        # enc_MA_pub.publish(enc_MA_msg)
        # append_sensor_data(enc_MA_hist, enc_MA_msg)


        # imu_msg = imu.data_retrive(imu_data)
        # imu_pub.publish(imu_msg)
        # append_sensor_data(imu_hist, imu_msg)


        # imu_MA_msg = imu.data_retrive_MA(imu_MA_data)
        # imu_MA_pub.publish(imu_MA_msg)
        # append_sensor_data(imu_MA_hist, imu_MA_msg)


        # fcam_msg = fcam.data_retrive(fcam_data)
        # fcam_pub.publish(fcam_msg)
        # append_sensor_data(fcam_hist, fcam_msg)


        # fcam_MA_msg = fcam.data_retrive_MA(fcam_MA_data)
        # fcam_MA_pub.publish(fcam_MA_msg)
        # append_sensor_data(fcam_MA_hist, fcam_MA_msg)

        prev_time = curr_time 


        if visualization == True:

            l = 0.42; w = 0.19

            (x_est , y_est , yaw_est )  = est_state[-3:]
            (x_ol  , y_ol  , yaw_ol  )  = ol_state[-3:]
            (x_meas, y_meas, yaw_meas)  = y_meas[-3:]

            est_x_his.append(x_est)
            est_y_his.append(y_est)

            ol_x_his.append(x_ol)
            ol_y_his.append(y_ol)
                        
            meas_x_his.append(x_meas)
            meas_y_his.append(y_meas)

            car_est_x, car_est_y = getCarPosition(x_est, y_est, yaw_est, w, l)
            rec_est.set_xy(np.array([car_est_x, car_est_y]).T)

            car_ol_x, car_ol_y = getCarPosition(x_ol, y_ol, yaw_ol, w, l)
            rec_ol.set_xy(np.array([car_ol_x, car_ol_y]).T)

            meas_x, car_meas_y = getCarPosition(x_meas, y_meas, yaw_meas, w, l)
            rec_meas.set_xy(np.array([meas_x, car_meas_y]).T)

            line_est.set_data(est_x_his, est_y_his)
            line_ol.set_data(ol_x_his, ol_y_his)
            line_meas.set_data(meas_x_his, meas_y_his)

            fig.canvas.draw()
            plt.show()
            plt.pause(1.0/300)


        rate.sleep()


    if record_data == True:
        path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
            
        now = datetime.datetime.now()
        # path = path + now.strftime("d%d_m%m_y%Y/")
        path = path + now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")

        if not os.path.exists(path):
            os.makedirs(path)

        est_path = path + '/est_his_resistive_estimation'
        meas_path = path + '/meas_his_resistive_estimation'


        control_path = path + '/control_his_resistive_estimation'
        
        # marvel_path = path + '/marvel_his_resistive_estimation'
        # marvel_MA_path = path + '/marvel_MA_his_resistive_estimation'

        # imu_path = path + '/imu_his_resistive_estimation'
        # imu_MA_path = path + '/imu_MA_his_resistive_estimation'
        
        # enc_path = path + '/enc_his_resistive_estimation'
        # enc_MA_path = path + '/enc_MA_his_resistive_estimation'
        
        # fcam_path = path + '/fcam_his_resistive_estimation'
        # fcam_MA_path = path + '/fcam_MA_his_resistive_estimation'

        np.save(est_path,est_state_hist)
        np.save(meas_path,meas_state_hist)


        np.save(control_path,control_hist)
        
        # np.save(imu_path,imu_hist)
        # np.save(imu_MA_path,imu_MA_hist)

        # np.save(enc_path,enc_hist)
        # np.save(enc_MA_path,enc_MA_hist)
        
        # np.save(fcam_path,fcam_hist)
        # np.save(fcam_MA_path,fcam_MA_hist)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


'''
sensorReading message >>>>>>>>>


int64 timestamp_ms
float64 X
float64 Y
float64 roll
float64 yaw
float64 pitch
float64 vx
float64 vy
float64 yaw_rate
float64 ax
float64 ay
float64 s
float64 x
float64 y

'''
