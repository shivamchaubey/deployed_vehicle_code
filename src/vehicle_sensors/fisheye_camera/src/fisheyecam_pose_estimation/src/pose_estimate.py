#!/usr/bin/env python

###############################################
##      IRI 1:10 Autonomous Car               ##
##      Supervisor: Puig Cayuela Vicenc      ##
##      Author: Shivam Chaubey               ##
##      Date: 20/03/2021                     ##
###############################################

''' The code is written for tracking position of vehicle using top camera and
aruco marker on the ceiling.  As the orientation directly from camera is very
noisy IMU information is used for the orientation.  Getting all the
information in the camera frame of reference the homogeneous transformation is
done to  estimate the position with respect to the aruco map on the ceiling
which can be further transformed for any world frame of reference in the
observer code. Camera stream is read directly by opencv library so  no need to
launch another node for publishing camera stream. '''

import rospy
import numpy as np
import cv2
import time
import datetime
import os
import cv2.aruco as aruco
# from sensor_msgs.msg import CompressedImage ## This is added in case camera stream needs to be read from another node.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import cos, sin, atan2, atan, pi, acos
from numpy.linalg import inv, norm, det
from geometry_msgs.msg import Pose


# fs = cv2.FileStorage("/home/auto/Desktop/autonomus_vehicle_project/project/development/proto/camera_development/kinect/first.yaml", cv2.FILE_STORAGE_READ)
# matrix_coefficients  = fs.getNode("K").mat()
# distortion_coefficients  = fs.getNode("D").mat()


# These matrices is obtained using camera calibration
#matrix_coefficients
mtx =  np.array([[245.58486295,   0.        , 328.57055189],
       [  0.        , 245.24846233, 229.01671809],
       [  0.        ,   0.        ,   1.        ]])   

#distortion_coefficients
dist =  np.array([[-0.32748962,  0.1208519 , -0.00152458,  0.00164202, -0.02107851]])



class ImuClass(object):
    """ Object collecting IMU data The orientation information is needed The
vehicle doesn't have absolute orientation so an offset needs to be added for
initializing otherwise at the same location at different time the orientation
reading will be different. """

    def __init__(self,t0):

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)

        self.yaw     = 0.0  
        self.yaw_offset = 0 
        self.t0     = t0
        # self.curr_time = rospy.get_rostime().to_sec() - self.t0
        # self.prev_time = self.curr_time

    def Pose_callback(self, data):
        # self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = data.orientation.x
        self.pitch  = data.orientation.y
        self.yaw    = wrap(data.orientation.z + self.yaw_offset)


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle




class image_stream():
    ''' Class to subscribe the camera stream if different node is used for
        streaming the camera information. '''
    def __init__(self):

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        self.image_np = np.zeros((1080, 1920, 3))

    def callback(self, ros_data):
        
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        self.image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:



def _initializeFigure_xy(x_lim,y_lim):
    '''
    For visualizing the tracking performance. This piece of code is only for
    debugging purpose on the development computer. This can't be used in the
    vehicle CPU  as the is no GUI in the vehicle.
    '''

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()
    line_sim,       = axtr.plot(xdata, ydata, '-k')
    line_rl,        = axtr.plot(xdata, ydata, '-b')  
    point_simc,     = axtr.plot(xdata, ydata, '-or')       
    line_SS,        = axtr.plot(xdata, ydata, 'og')
    point_rlc,      = axtr.plot(xdata, ydata, '-or')
    line_planning,  = axtr.plot(xdata, ydata, '-ok')
    line_0,        = axtr.plot(xdata, ydata, '-r')  
    line_2,        = axtr.plot(xdata, ydata, '-g')  
    line_3,        = axtr.plot(xdata, ydata, '-b')  
    line_4,        = axtr.plot(xdata, ydata, '-y')  
    line_fusion,        = axtr.plot(xdata, ydata, '-m')  
    

    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])

    marker_0 = patches.Polygon(v, alpha=0.7, closed=True, fc='r', ec='k', zorder=10,label='ID0')
    axtr.add_patch(marker_0)
    # # Vehicle:
    marker_2 = patches.Polygon(v, alpha=0.7, closed=True, fc='G', ec='k', zorder=10,label='ID2')
    axtr.add_patch(marker_2)

    marker_3 = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10,label='ID3')
    # # Vehicle:
    marker_4 = patches.Polygon(v, alpha=0.7, closed=True, fc='y', ec='k', zorder=10,label='ID4')
    
    fusion = patches.Polygon(v, alpha=0.7, closed=True, fc='m', ec='k', zorder=10,label='fusion')    

    plt.legend()
    # # Planner vehicle:
    rec_planning = patches.Polygon(v, alpha=0.7, closed=True, fc='k', ec='k', zorder=10)

    plt.show()

    return plt, fig, axtr, line_planning, point_simc, point_rlc, line_SS, line_sim, line_rl, line_0, line_2, line_3, line_4, line_fusion,\
     marker_0, marker_2, marker_3, marker_4, fusion ,rec_planning


def getCarPosition(x, y, psi, w, l):
    ''' Mapping the car coordinate from world frame of reference to the plot
        for the visualization '''
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y


def main():

    rospy.init_node('fishcam_pose', anonymous=True)
    loop_rate       = 30
    rate            = rospy.Rate(loop_rate)


    ## Parameter setting
    visual_mode     = rospy.get_param("/fisheye_tracker/visualization")
    yaw_offset      = rospy.get_param("/fisheye_tracker/yaw_offset")
    camera_stream   = rospy.get_param("/fisheye_tracker/camera_stream")
    USB_port        = rospy.get_param("/fisheye_tracker/USB_port")
    Calibrate_imu   = rospy.get_param("/fisheye_tracker/Calibrate_imu")
    record_data     = rospy.get_param("/fisheye_tracker/record_data")



    ####### ORIGIN w.r.t world
    '''
    Set according to the reference needed also remember if a orientation is different it's better to do a homogeneous 
    transformation.
    '''
    # Gx = 0.50 - (1.1 + 27.0 + 2.7/2)*10**-2
    # Gy = 1.50 - (27.0 + 4.4/2)*10**-2
    # Gz = 3.50 #from the camera focal point

    ###### ORIGIN NEAR ID2 ####################
    ## Reference set w.r.t ARUCO ID2 and later transformed in the observer part of code.
    Gx = 0
    Gy = 0
    Gz = 0 


    ###### ARUCO location setting ########

    ''' Prepare the world map for markers. This is very important part for
    robust matching of aruco. The distance between each marker needs  to be
    calculated manually and set accordingly. If the aruco location is changed
    update this information. 4 markers are used in this case placed next to each
    other with some offset. '''

    G_coor = np.array([Gx,Gy,Gz])
    cid2 = np.array([[0.0,1.1+27.0,0.0],[27.0,1.1+27.0,0.0],[27.0,1.1,0.0],[0.0,1.1,0.0]])*10**-2 + G_coor
    cid3 = cid2 + np.array([0.0,2.7+27.0,0.0])*10**-2 
    cid4 = cid2 + np.array([27.0+4.4,-1.1,0.0])*10**-2
    cid0 = cid4 + np.array([0.0,27.0+2.7,0.0])*10**-2 

    ncid2 = np.array([cid2[2],cid2[1],cid2[0],cid2[3]])

    ncid0 = np.array([cid0[0],cid0[3],cid0[2],cid0[1]])

    ncid3 = np.array([cid3[2],cid3[1],cid3[0],cid3[3]])

    ncid4 = np.array([cid4[0],cid4[3],cid4[2],cid4[1]])

    aruco_world_corners = {'0':ncid0,'2':ncid2,'3':ncid3,'4':ncid4}

    #############################################



    counter = 0

    # ##### Use for plot ####
    if visual_mode == True:
        lim = 2
        plim = 0.2
        ( plt, fig, axtr, line_planning, point_simc, point_rlc, line_SS, line_sim, line_rl, line_0, line_2, line_3, line_4, line_fusion,\
             marker_0, marker_2, marker_3, marker_4, fusion ,rec_planning) = _initializeFigure_xy(lim,lim)
        real0_x_his = []
        real0_y_his = []
        real2_x_his = []
        real2_y_his = []

        plt.plot(aruco_world_corners[str(0)][:,0],aruco_world_corners[str(0)][:,1],'r')
        plt.plot(aruco_world_corners[str(4)][:,0],aruco_world_corners[str(4)][:,1],'g')
        plt.plot(aruco_world_corners[str(2)][:,0],aruco_world_corners[str(2)][:,1],'b')
        plt.plot(aruco_world_corners[str(3)][:,0],aruco_world_corners[str(3)][:,1],'y')
        plt.gca().set_aspect('equal', adjustable='box')



    time0 = rospy.get_time()


    ## If subscriber is used for camera stream ##
    if camera_stream == 'Node':
        k_cam = image_stream()
        frame = k_cam.image_np

    ## if OpenCV 
    else:
        cap = cv2.VideoCapture(USB_port)

    ## IMU subscriber class
    imu = ImuClass(time0)
    
    ## Pure pose obtained from camera ##
    pure_cam_pose  = rospy.Publisher('pure_cam_pose', Pose, queue_size=1)
    
    ## Fused pose obtained using IMU + Camera
    fused_cam_pose  = rospy.Publisher('fused_cam_pose', Pose, queue_size=1)

    pose_cam_pure  = Pose()
    pose_cam_fused = Pose()
    
    pose_cam_pure_hist = {'timestamp_ms':[],'pos_x':[],'pos_y':[],'yaw':[]}
    pose_cam_fused_hist = {'timestamp_ms':[],'pos_x':[],'pos_y':[],'yaw':[]}

    dtheta = [] 

    calibrate = True
    rot_mat = np.identity(3) 

    x_offset = 0.0 
    x_cal = []
    y_offset = 0.0
    y_cal = []

    cal_count = 0
    pure_yaw_hist = []

    record_data = False

    while not (rospy.is_shutdown()):

        ## Uncomment this for streaming camera from another node. if else is not done to avoid extra time loss
        # frame = k_cam.image_np
        # frame=frame.astype(np.uint8)
        
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

        # detector parameters can be set here (List of detection parameters[3])
        # Best parameter is selected by experimenting on the track
        parameters = aruco.DetectorParameters_create()
        # parameters.adaptiveThreshConstant = true

        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):
            if len(ids) > 1:
                filter_ids = []
                filer_corners = []
                for i,corner in zip(ids,corners):
                    if i in [0,4,3,2]:
                        filter_ids.append(i)
                        filer_corners.append(corner)

                ids = np.array(filter_ids)
                corners = np.array(filer_corners)

                world_map = []
                for i in np.squeeze(ids,axis = 1):
                    world_map.append(aruco_world_corners[str(i)])
                world_map = np.array(world_map)

                world_map.astype(np.float32)
                corners = np.array(corners)
                new_world_map = world_map.reshape(world_map.shape[-3]*world_map.shape[-2],world_map.shape[-1])
                new_corners = corners.reshape((len(ids)*corners.shape[-2],corners.shape[-1])) 

            
            if len(corners) > 1:
                # estimate pose of each marker and return the values
                # rvet and tvec-different from camera coefficients
                arvec, atvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.27, mtx, dist)
            
        # (success, rvec, tvec) = cv2.solvePnP(new_world_map, new_corners, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                param = cv2.SOLVEPNP_ITERATIVE
                # param = cv2.SOLVEPNP_EPNP
                # param = cv2.SOLVEPNP_UPNP
                # param = cv2.SOLVEPNP_SQPNP

            
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(new_world_map, new_corners, mtx, dist,flags= param)
                # (success, rvec, tvec) = cv2.solvePnP(model_points, corners, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                (rvec-tvec).any() # get rid of that nasty numpy value array error
                # val = cv2.solvePnPRansac(new_world_map, new_corners, mtx, dist)
                # print ("val",val)


                ####### Some reference for countered bug and fixed #########
                # https://github.com/opencv/opencv/issues/8813
                # https://www.gitmemory.com/issue/opencv/opencv/8813/496945286
                # https://github.com/chili-epfl/chilitags/issues/19



                ###### Camera frame to Aruco frame transformation######
                
                # print ('rvec',rvec,'tvec',tvec)

                c_T_a = np.zeros(shape=(4,4))
                c_T_a[3,3] = 1
                rot = np.zeros(shape=(3,3))
                cv2.Rodrigues(rvec, rot)
                c_T_a[:3,:3] = rot
                c_T_a[:3,3]  = np.squeeze(tvec)

                a_T_c = np.zeros(shape=(4,4))

                a_T_c[:3,:3] = rot.T
                a_T_c[:3,3]  = np.squeeze(np.dot(-rot.T,tvec))
                        
                pos_x = a_T_c[:3,3][0]
                pos_y = a_T_c[:3,3][1]
                ypr  = cv2.RQDecomp3x3(a_T_c[:3,:3])

                # print ("cam yaw",ypr[0][0]*180/pi,"cam pitch",ypr[0][1]*180/pi,"cam roll",ypr[0][2]*180/pi)

                x_real = pos_x
                y_real = pos_y
                psi_real = ypr[0][0]

                yaw = imu.yaw 
                # print ("yaw",yaw,"yaw deg",yaw*180/pi,"imu.yaw_offset",imu.yaw_offset)
                Rz = np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0.0,0.0,1.0]])
                # Rz = np.dot(Rz,rot_mat)
                # print ("Rz",Rz)

                loc = np.array([np.squeeze(tvec)[0],np.squeeze(tvec)[1],1.0]).T
                
                # calibrated_rot = np.dot(rot_mat,-Rz)

                # yaw = np.arccos(calibrated_rot[0,0])

                pose_imu = np.dot(-Rz,loc)
                
                # print ("rot_mat",rot_mat,"corrected yaw",yaw,yaw*180/pi)
                x_imu = -pose_imu[1]
                y_imu = pose_imu[0]
                psi_imu = yaw 

                # pose_cam_pure_hist['timestamp_ms'].append(rospy.get_time()-time0)
                # pose_cam_pure_hist['pos_x'].append(x_real)
                # pose_cam_pure_hist['pos_y'].append(y_real)
                # pose_cam_pure_hist['yaw'].append(psi_real)

                # pose_cam_fused_hist['timestamp_ms'].append(rospy.get_time()-time0)
                # pose_cam_fused_hist['pos_x'].append(x_imu)
                # pose_cam_fused_hist['pos_y'].append(y_imu)
                # pose_cam_fused_hist['yaw'].append(psi_imu)


                # #### only camera used for estimation #####
                # pose_cam_pure.position.x = pos_x
                # xp = pos_x
                # pose_cam_pure.position.y = pos_y
                # yp = pos_y
                # pose_cam_pure.orientation.z = ypr[0][0]



                # pure_cam_pose.publish(pose_cam_pure)


                # #### IMU orientation used ######
                # pose_cam_fused.position.x = -pose_imu[1]
                # xf = -pose_imu[1]
                # pose_cam_fused.position.y = pose_imu[0]
                # yf = pose_imu[0]
                # pose_cam_fused.orientation.z = yaw
                # fused_cam_pose.publish(pose_cam_fused)



                # pure_cam_vec  = np.array([pos_x,pos_y]) 
                # fused_cam_vec = np.array([-pose_imu[1],pose_imu[0]])

                # theta = np.arccos(pure_cam_vec.dot(fused_cam_vec)/(norm(pure_cam_vec)*norm(fused_cam_vec)))
                # print "theta arccos", theta*180/pi
                # print ("difference between degree",psi_real-psi_imu)
                
                cal_count += 1
                if Calibrate_imu == True:

                    dtheta.append(psi_imu)   
                    pure_yaw_hist.append(ypr[0][0]) 
                    # print "psi_imu",psi_imu                    
                    
                    # calibrate = False
                    if cal_count > 20:
                        Calibrate_imu = False
                        Calibrate_pose = True
                        imu.yaw_offset = -(np.mean(psi_imu) - yaw_offset) #np.mean(pure_yaw_hist))
                        print ">>>>> Yaw Calibration Done"

                # if ((Calibrate_imu == False) and (Calibrate_pose == True)) and cal_count>25:
                    
                    # x_cal.append(pos_x - x_imu)
                    # y_cal.append(pos_y - y_imu)
                    # if cal_count > 50:
                    #     Calibrate_pose = False
                    #     Calibrate_imu = False
                    #     x_offset =  np.mean(x_cal)
                    #     y_offset =  np.mean(y_cal)

                    #     print "offset",x_offset,y_offset
                    #     print " >>>>> Pose Calibration Done"
                    #     # break
                    


                #### only camera used for estimation #####
                pose_cam_pure.position.x = pos_x
                pose_cam_pure.position.y = pos_y
                pose_cam_pure.orientation.z = ypr[0][0]
                pure_cam_pose.publish(pose_cam_pure)


                print "offset",x_offset,y_offset
                print "x_imu",x_imu, "y_imu",y_imu
                x_fused = x_imu + x_offset
                y_fused = y_imu  + y_offset
                print "x_fused",x_fused, "y_fused",y_fused
                

                #### IMU orientation used ######
                pose_cam_fused.position.x = x_fused
                pose_cam_fused.position.y = y_fused
                pose_cam_fused.orientation.z = yaw
                fused_cam_pose.publish(pose_cam_fused)

                print ("\n[PURE CAM POSE ::]","X ={}, Y = {}, Yaw = {}".format(pos_x,pos_y,ypr[0][0]*180/pi))

                print ("\n[FUSED CAM POSE ::]","X ={}, Y = {}, Yaw = {}".format(x_fused,y_fused,yaw*180/pi))


        ################### PLOTTING #############################
        
                #### Visualize detected aruco #######
        

                if visual_mode == True:
                     # font for displaying text (below)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    for i in range(0, ids.size):
                        # draw axis for the aruco markers
                        aruco.drawAxis(frame, mtx, dist, arvec[i], atvec[i], 1)


                    #draw a square around the markers
                    aruco.drawDetectedMarkers(frame, corners)

                    #code to show ids of the marker found
                    strg = ''
                    for i in range(0, ids.size):
                        strg += str(ids[i][0])+', '

                    cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.imshow('frame',np.hstack((frame,np.dstack((gray,gray,gray)))))
                    cv2.waitKey(1)

                ####################################################

                ############## For plotting the vehicle tracking ###############
            
                    # plot dimension update
                    x_p = max([abs(x_real)])
                    y_p = max([abs(y_real)])
                    # print ('x_p',x_p,'lim',lim)
                    if (lim < abs(x_p)):
                        lim = abs(x_p)
                        lim = int((1.0+plim)*lim)
                        plt.xlim([-lim,lim])
                        plt.ylim([-lim,lim])
                    if (lim < abs(y_p)):
                        lim = abs(y_p)
                        plt.xlim([-lim,lim])
                        plt.ylim([-lim,lim])
                    
                    l = 0.42; w = 0.19


                    # arucoId0_pose_wrt_cam = c_T_a[0:3,3]
                    # arucoId0_yaw_wrt_cam = psi_real
                    

                    real0_x_his.append(x_real)
                    real0_y_his.append(y_real)

                    car_real_x, car_real_y = getCarPosition(x_real, y_real, psi_real, w, l)
                    marker_0.set_xy(np.array([car_real_x, car_real_y]).T)
                    # point_rlc.set_data(x_real, y_real)
                    line_0.set_data(real0_x_his, real0_y_his)


                    real2_x_his.append(x_fused)
                    real2_y_his.append(y_fused)

                    car_real_x, car_real_y = getCarPosition(x_fused, y_fused, psi_imu, w, l)
                    marker_2.set_xy(np.array([car_real_x, car_real_y]).T)
                    # point_rlc.set_data(x_real, y_real)
                    line_2.set_data(real2_x_his, real2_y_his)


                    cv2.putText(frame, 'cam_pos x={} cm, y={} cm'.format(str(round(pos_x*10**2,2)),str(round(pos_y*10**2,2))), (50, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

                    fig.canvas.draw()


        #####################################################################


        rate.sleep()

    if record_data == True:
        print ("\n >>>>>>>>>>>>> Saving Results <<<<<<<<<<<<<<<<<<<<< \n")    
        path = ('/').join(__file__.split('/')[:-2]) + '/data/' 
            
        now = datetime.datetime.now()
        path = path + now.strftime("d%d_m%m_y%Y/")

        if not os.path.exists(path):
            os.makedirs(path)


        dt_string = now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")

        pose_cam_pure_hist_path = path + 'pose_cam_pure_hist_'+ dt_string
        pose_cam_fused_hist_path = path + 'pose_cam_fused_hist_'+ dt_string
        np.save(pose_cam_pure_hist_path,pose_cam_pure_hist)
        np.save(pose_cam_fused_hist_path,pose_cam_fused_hist)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
