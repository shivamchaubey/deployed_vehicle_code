#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' This code is to compare the performance of the estimation technique. The
estimated states, measured states and the simulator states are
plotted to compare the performance. This code is bit slow due to online
plotting of matplotlib. Use other plotter for real time debugging such as
plotjuggler. This code an be used to record the data for documentation
 '''

import numpy as np
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sensor_fusion.msg import sensorReading, control
from simulator.msg import simulatorStates
from PIL import Image
from std_msgs.msg import Bool, Float32
import datetime
import os
import sys
# sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
# from trackInitialization import Map
from numpy import random

#### plotter for control action ####
def plot_control(x_lim,y_lim):

    rc_fonts = {
        'axes.facecolor':'white',
        "text.usetex": True,
        'text.latex.preview': True, # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}"""],
        'pgf.preamble': r'\usepackage{amsmath}',
        'font.size': 18,
        'font.weight' : 'bold'
    }
    plt.rcParams.update(rc_fonts)

    xdata = []; ydata = []
    fig = plt.figure(figsize=(10,8))
    plt.ion()
    plt.xlim([-1*x_lim,x_lim])
    plt.ylim([-1*y_lim,y_lim])

    axtr = plt.axes()

    line_dutycycle,   = axtr.plot(xdata, ydata, '-g', label = 'Duty cycle', linewidth = 2)
    line_steer,       = axtr.plot(xdata, ydata, '-b', label = 'Steering angle', linewidth = 2)  

    # plt.legend()
    


    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2, fontsize=16, scatterpoints = 1)

    plt.grid()
    
    # plt.xlabel(r"X Position")
    # plt.ylabel(r"Y Position")



    return fig, plt, line_dutycycle, line_steer


#### gives the coordinate of the patches for plotting the rectangular (vehicle) orientation and position.
def getCarPosition(x, y, psi, w, l):
    car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
              x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
    car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
              y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]
    return car_x, car_y





class Map():
    """map object
    Attributes:
        getGlobalPosition: convert position from (s, ey) to (X,Y)
    """
    def __init__(self, flagTrackShape = 0):

        """ Nos interesa que el planner tenga una pista algo mas reducida de la real
        para conservar algo de robustez y no salirnos de la pista en el primer segundo. """

        # self.halfWidth  = rospy.get_param("halfWidth")
        self.halfWidth  = 0.25

        self.slack      = 0.15
        # selectedTrack   = rospy.get_param("trackShape")
        selectedTrack   = "oval_iri"

        if selectedTrack == "3110":
            spec = np.array([[60 * 0.03, 0],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [20 * 0.03, 0],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [40 * 0.03, -40 * 0.03 * 10 / np.pi],
                             [60 * 0.03, +60 * 0.03 * 5 / np.pi],
                             [40 * 0.03, -40 * 0.03 * 10 / np.pi],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [20 * 0.03, 0],
                             [80 * 0.03, +80 * 0.03 * 2 / np.pi],
                             [80 * 0.03, 0]])

        elif selectedTrack == "oval":
            spec = np.array([[1.0, 0],
                             [4.5, 4.5 / np.pi],
                             [2.0, 0],
                             [4.5, 4.5 / np.pi],
                             [1.0, 0]])

        elif selectedTrack == "L_shape":
            lengthCurve     = 4.5
            spec = np.array([[1.0, 0],
                             [lengthCurve, lengthCurve / np.pi],
                             [lengthCurve/2,-lengthCurve / np.pi ],
                             [lengthCurve, lengthCurve / np.pi],
                             [lengthCurve / np.pi *2, 0],
                             [lengthCurve/2, lengthCurve / np.pi]])


        elif selectedTrack == "oval_iri_old":
            spec = 1.0*np.array([[1.25, 0],
                             [3.5, 3.5 / np.pi],
                             [1.25, 0],
                             [3.5, 3.5 / np.pi]])

        elif selectedTrack == "oval_iri":
            spec = 1.0*np.array([[1.34, 0],
                             [1.125*np.pi, 1.125],
                             [1.34, 0],
                             [1.14*np.pi, 1.125]])


        # Now given the above segments we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
        # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
        # we compute also the cumulative s at the starting point of the segment at signed curvature
        # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]

        PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
        for i in range(0, spec.shape[0]):
            if spec[i, 1] == 0.0:              # If the current segment is a straight line
                l = spec[i, 0]                 # Length of the segments
                if i == 0:
                    ang = 0                          # Angle of the tangent vector at the starting point of the segment
                    x = 0 + l * np.cos(ang)          # x coordinate of the last point of the segment
                    y = 0 + l * np.sin(ang)          # y coordinate of the last point of the segment
                else:
                    ang = PointAndTangent[i - 1, 2]                 # Angle of the tangent vector at the starting point of the segment
                    x = PointAndTangent[i-1, 0] + l * np.cos(ang)  # x coordinate of the last point of the segment
                    y = PointAndTangent[i-1, 1] + l * np.sin(ang)  # y coordinate of the last point of the segment
                psi = ang  # Angle of the tangent vector at the last point of the segment

                # # With the above information create the new line
                # if i == 0:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
                # else:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 0])
                #
                # PointAndTangent[i + 1, :] = NewLine  # Write the new info

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 0])

                PointAndTangent[i, :] = NewLine  # Write the new info
            else:
                l = spec[i, 0]                 # Length of the segment
                r = spec[i, 1]                 # Radius of curvature


                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                if i == 0:
                    ang = 0                                                      # Angle of the tangent vector at the
                                                                                 # starting point of the segment
                    CenterX = 0 \
                              + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = 0 \
                              + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
                else:
                    ang = PointAndTangent[i - 1, 2]                              # Angle of the tangent vector at the
                                                                                 # starting point of the segment
                    CenterX = PointAndTangent[i-1, 0] \
                              + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = PointAndTangent[i-1, 1] \
                              + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                spanAng = l / np.abs(r)  # Angle spanned by the circle
                psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment

                angleNormal = wrap((direction * np.pi / 2 + ang))
                angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
                x = CenterX + np.abs(r) * np.cos(
                    angle + direction * spanAng)  # x coordinate of the last point of the segment
                y = CenterY + np.abs(r) * np.sin(
                    angle + direction * spanAng)  # y coordinate of the last point of the segment

                # With the above information create the new line
                # plt.plot(CenterX, CenterY, 'bo')
                # plt.plot(x, y, 'ro')

                # if i == 0:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
                # else:
                #     NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 1 / r])
                #
                # PointAndTangent[i + 1, :] = NewLine  # Write the new info

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 1 / r])

                PointAndTangent[i, :] = NewLine  # Write the new info
            # plt.plot(x, y, 'or')

        # Now update info on last point
        # xs = PointAndTangent[PointAndTangent.shape[0] - 2, 0]
        # ys = PointAndTangent[PointAndTangent.shape[0] - 2, 1]
        # xf = PointAndTangent[0, 0]
        # yf = PointAndTangent[0, 1]
        # psif = PointAndTangent[PointAndTangent.shape[0] - 2, 2]
        #
        # # plt.plot(xf, yf, 'or')
        # # plt.show()
        # l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)
        #
        # NewLine = np.array([xf, yf, psif, PointAndTangent[PointAndTangent.shape[0] - 2, 3] + PointAndTangent[
        #     PointAndTangent.shape[0] - 2, 4], l, 0])
        # PointAndTangent[-1, :] = NewLine


        xs = PointAndTangent[-2, 0]
        ys = PointAndTangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0

        # plt.plot(xf, yf, 'or')
        # plt.show()
        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)

        NewLine = np.array([xf, yf, psif, PointAndTangent[-2, 3] + PointAndTangent[-2, 4], l, 0])
        PointAndTangent[-1, :] = NewLine

        self.PointAndTangent = PointAndTangent
        self.TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]


    def getGlobalPosition(self, s, ey):
        """coordinate transformation from curvilinear reference frame (e, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """

        # wrap s along the track
        while (s > self.TrackLength):
            s = s - self.TrackLength

        # Compute the segment in which system is evolving
        PointAndTangent = self.PointAndTangent

        index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
        ##  i = int(np.where(np.squeeze(index))[0])
        i = np.where(np.squeeze(index))[0]

        if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = PointAndTangent[i, 0]
            yf = PointAndTangent[i, 1]
            xs = PointAndTangent[i - 1, 0]
            ys = PointAndTangent[i - 1, 1]
            psi = PointAndTangent[i, 2]

            # Compute the segment length
            deltaL = PointAndTangent[i, 4]
            reltaL = s - PointAndTangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
            theta = psi
        else:
            r = 1 / PointAndTangent[i, 5]  # Extract curvature
            ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            # Compute the center of the arc
            if r >= 0:
                direction = 1
            else:
                direction = -1

            CenterX = PointAndTangent[i - 1, 0] \
                      + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
            CenterY = PointAndTangent[i - 1, 1] \
                      + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

            spanAng = (s - PointAndTangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(
                angle + direction * spanAng)  # x coordinate of the last point of the segment
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(
                angle + direction * spanAng)  # y coordinate of the last point of the segment
            theta = ang + direction * spanAng

        return np.squeeze(x), np.squeeze(y), np.squeeze(theta)



    def getGlobalPosition_Racing(self, ex, ey, xd, yd, psid):
        """coordinate transformation from curvilinear reference frame (ex, ey) to inertial reference frame (X, Y)
        based on inverse of error computation for racing:
            ex      = +(x-xd)*np.cos(psid) + (y-yd)*np.sin(psid)
            ey      = -(x-xd)*np.sin(psid) + (y-yd)*np.cos(psid)
            epsi    = wrap(psi-psid)
        """

        # x = ex*np.cos(psid) - ey*np.sin(psid) + xd
        x = xd
        y = (ey - xd*np.sin(psid) + yd*np.cos(psid) + x*np.sin(psid)) / np.cos(psid)

        return x, y




    def getLocalPosition(self, x, y, psi):
        """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
        (X, Y): position in the inertial reference frame
        """
        PointAndTangent = self.PointAndTangent
        CompletedFlag = 0



        for i in range(0, PointAndTangent.shape[0]):
            if CompletedFlag == 1:
                break

            if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
                # Extract the first final and initial point of the segment
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                psi_unwrap = np.unwrap([PointAndTangent[i - 1, 2], psi])[1]
                epsi = psi_unwrap - PointAndTangent[i - 1, 2]

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    s  = PointAndTangent[i, 3]
                    ey = 0
                    CompletedFlag = 1

                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    CompletedFlag = 1
                else:
                    if np.abs(computeAngle( [x,y] , [xs, ys], [xf, yf])) <= np.pi/2 and np.abs(computeAngle( [x,y] , [xf, yf], [xs, ys])) <= np.pi/2:
                        v1 = np.array([x,y]) - np.array([xs, ys])
                        angle = computeAngle( [xf,yf] , [xs, ys], [x, y])
                        s_local = la.norm(v1) * np.cos(angle)
                        s       = s_local + PointAndTangent[i, 3]
                        ey      = la.norm(v1) * np.sin(angle)

                        if np.abs(ey)<= self.halfWidth + self.slack:
                            CompletedFlag = 1

            else:
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                r = 1 / PointAndTangent[i, 5]  # Extract curvature
                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)

                # Compute the center of the arc
                CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    ey = 0
                    psi_unwrap = np.unwrap([ang, psi])[1]
                    epsi = psi_unwrap - ang
                    s = PointAndTangent[i, 3]
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    psi_unwrap = np.unwrap([PointAndTangent[i, 2], psi])[1]
                    epsi = psi_unwrap - PointAndTangent[i, 2]
                    CompletedFlag = 1
                else:
                    arc1 = PointAndTangent[i, 4] * PointAndTangent[i, 5]
                    arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
                    if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
                        v = np.array([x, y]) - np.array([CenterX, CenterY])
                        s_local = np.abs(arc2)*np.abs(r)
                        s    = s_local + PointAndTangent[i, 3]
                        ey   = -np.sign(direction) * (la.norm(v) - np.abs(r))
                        psi_unwrap = np.unwrap([ang + arc2, psi])[1]
                        epsi = psi_unwrap - (ang + arc2)

                        if np.abs(ey) <= self.halfWidth + self.slack: # OUT OF TRACK!!
                            CompletedFlag = 1

        # if epsi>1.0:
        #     print "epsi Greater then 1.0"
        #     pdb.set_trace()

        if CompletedFlag == 0:
            s    = 10000
            ey   = 10000
            epsi = 10000
            #print "Error!! POINT OUT OF THE TRACK!!!! <=================="
            # pdb.set_trace()

        return s, ey, epsi, CompletedFlag


    
def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res


def unityTestChangeOfCoordinates(map, ClosedLoopData):
    """For each point in ClosedLoopData change (X, Y) into (s, ey) and back to (X, Y) to check accurancy
    """
    TestResult = 1
    for i in range(0, ClosedLoopData.x.shape[0]):
        xdat = ClosedLoopData.x
        xglobdat = ClosedLoopData.x_glob

        s, ey, _, _ = map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5], xglobdat[i, 3])
        v1 = np.array([s, ey])
        v2 = np.array(xdat[i, 4:6])
        v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
        v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
        # print v1, v2, np.dot(v1 - v2, v1 - v2), np.dot(v3 - v4, v3 - v4)

        if np.dot(v3 - v4, v3 - v4) > 0.00000001:
            TestResult = 0
            print "ERROR", v1, v2, v3, v4
            pdb.set_trace()
            v1 = np.array(map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5]))
            v2 = np.array(xdat[i, 4:6])
            v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
            v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
            print np.dot(v3 - v4, v3 - v4)
            pdb.set_trace()

    if TestResult == 1:
        print "Change of coordinates test passed!"







def main():


    rospy.init_node('observer_performance_tester', anonymous=True)
    
    loop_rate       = 100.0
    rate            = rospy.Rate(loop_rate)
    dt = 1./loop_rate

    path = '/home/auto/Desktop/autonomus_vehicle_project/project/development/proto/plotter/log'
    dir_name = '/estimator'
    file_name = '/observer_data.npy'

    data = np.load(path + dir_name +file_name, allow_pickle = True).item()


    vx_est = data['est_state'][:,0]
    vy_est = data['est_state'][:,1]
    omega_est = data['est_state'][:,2]

    X_est = data['est_state'][:,3]
    Y_est = data['est_state'][:,4]
    yaw_est = data['est_state'][:,5]

    dutycycle = data['control'][:,0]
    steering = data['control'][:,1]

    vx_meas = data['full_meas'][:,0]
    omega_meas = data['full_meas'][:,1]

    X_meas = data['full_meas'][:,2]
    Y_meas = data['full_meas'][:,3]
    yaw_meas = data['full_meas'][:,4]

    X_lidar = data['lidar_pose'][:,0]
    Y_lidar = data['lidar_pose'][:,1]

    d_cont = data['control'][:,0]
    steer_cont = data['control'][:,1]

    track_visualization = True
    
    image_dy_his = []
    image_veh_his = []

    vehicle_visualization = True
    states_visualization  = False
    control_visualization = False
    record_data = False
    sim_on = False

    window_size = 2


    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True, # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}"""],
        'font.size': 18,
    'font.weight' : 'bold'
    }

    # plt.rcParams.update({'pgf.preamble': r'\usepackage{amsmath}'})
    plt.rcParams.update(rc_fonts)
    # fig, axs = plt.subplots(1, 1, figsize=(14,8))

    # x,y = zip(cam_pose_pure_x_hist, cam_pose_pure_y_hist)

    start = 4200
    end = 7030
    x_est = X_est[start:end]
    y_est = Y_est[start:end]
    yaw_est = yaw_est[start:end]

    duty = dutycycle[start:end]
    steer = steering[start:end]

    N = 30
    duty = np.convolve(duty, np.ones(N)/N, mode='valid')
    duty = np.insert(duty, 0, [0.0]*(N-1), axis=0)
    steer = np.convolve(steer, np.ones(N)/N, mode='valid')
    steer = np.insert(steer, 0, [0.0]*(N-1), axis=0)

    x_meas = X_lidar[start:end] 
    y_meas = Y_lidar[start:end] 


    map = Map()
    rc_fonts = {
        'axes.facecolor':'white',
        "text.usetex": True,
        'text.latex.preview': True, # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}"""],
        'pgf.preamble': r'\usepackage{amsmath}'
    }
    plt.rcParams.update(rc_fonts)


    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    # Points1 = np.zeros((Points, 2))
    # Points2 = np.zeros((Points, 2))
    # Points0 = np.zeros((Points, 2))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)


    # fig, ax = plt.subplots()
    # xdata_est, ydata_est = [], []
    # xdata_meas, ydata_meas = [], []

    # ln_est, = ax.plot([], [], '-b',  linewidth  = 2, alpha = 1.0, label = 'Estimated Position')
    # ln_meas, = ax.plot([], [], '-r',  linewidth  = 2, alpha = 0.6, label = 'Measured Position')

    # v = 0.00001*np.array([[ 1,  1],
    #                   [ 1, -1],
    #                   [-1, -1],
    #                   [-1,  1]])

    # rec_est = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10)
    # ax.add_patch(rec_est)



    # # plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o') #points on center track
    # ax.plot(Points1[:, 0], Points1[:, 1], color = 'black' , linewidth  = 3) # inner track
    # ax.plot(Points2[:, 0], Points2[:, 1], color = 'black' , linewidth  = 3) #outer track
    # ax.plot(Points0[:, 0], Points0[:, 1], '--' ,color = 'darkgray' , linewidth  = 2) #center track


    # ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #                 mode="expand", borderaxespad=0, ncol=2, fontsize=16, scatterpoints = 1)

    # ax.set_xlabel(r"X Position")
    # ax.set_ylabel(r"Y Position")


    def vehicle_init():
        ax.set_xlim(-1.7,3.0)
        ax.set_ylim(-0.5, 2.8)
        return ln_est,  ln_meas,



    def vehicle_update(frame):
        # print frame
        if frame == 0:
            xdata_est.append(x_est[frame])
            ydata_est.append(y_est[frame])
            ln_est.set_data(xdata_est, ydata_est)
        else:
            xdata_est.append(x_est[frame -1: frame])
            ydata_est.append(y_est[frame-1: frame])
            ln_est.set_data(xdata_est, ydata_est)

        x_ran = random.rand(1)*2 - 1
        y_ran = random.rand(1)*2 - 1

        # print "x_meas[frame] * (1 + x_ran*0.25)", x_meas[frame] * (1 + x_ran*0.25)
        xdata_meas.append(x_meas[frame] * (1 + x_ran*0.025))
        ydata_meas.append(y_meas[frame] * (1 + y_ran*0.025))
        ln_meas.set_data(xdata_meas, ydata_meas)


        l = 0.42/2; w = 0.19/2

        car_est_x, car_est_y = getCarPosition(x_est[frame], y_est[frame], yaw_est[frame], w, l)
        rec_est.set_xy(np.array([car_est_x, car_est_y]).T)


        return ln_est,  ln_meas, rec_est


    # fig_cont, ax_cont = plt.subplots(figsize = (25,25))
    fig_cont, ax_cont = plt.subplots()

    xdata_duty, ydata_duty = [], []
    xdata_steer, ydata_steer = [], []

    ln_duty, = ax_cont.plot([], [], '-b',  linewidth  = 2, alpha = 1.0, label = r'Dutycycle ($D$)')
    ln_steer, = ax_cont.plot([], [], '-r',  linewidth  = 2, alpha = 0.6, label = r'Steering angle ($\delta$)')
    ax_cont.plot(np.arange(int(len(steer)*(1+0.2)*dt)), np.ones((int(len(steer)*(1+0.2)*dt))) , '--m',  linewidth  = 2, alpha = 1.0, label = r'Dutycycle bound')
    ax_cont.plot(np.arange(int(len(steer)*(1+0.2)*dt)), -0.1*np.ones((int(len(steer)*(1+0.2)*dt))) , '--m',  linewidth  = 2, alpha = 1.0)
    ax_cont.plot(np.arange(int(len(steer)*(1+0.2)*dt)), 0.35*np.ones((int(len(steer)*(1+0.2)*dt))) , '--y',  linewidth  = 2, alpha = 1.0, label = r'Steering angle bound')
    ax_cont.plot(np.arange(int(len(steer)*(1+0.2)*dt)), -0.35*np.ones((int(len(steer)*(1+0.2)*dt))) , '--y',  linewidth  = 2, alpha = 1.0)
    
    ax_cont.legend(bbox_to_anchor=(0,1.02,1,0.4), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2, fontsize=16, scatterpoints = 1)

    ax_cont.set_ylabel(r"Control input")
    ax_cont.set_xlabel(r"Time ($\sec$)")
    ax_cont.grid()
    # fig_cont.tight_layout()
    fig_cont.subplots_adjust(top=0.85)
    def control_init():
        ax_cont.set_xlim(-0.0,len(steer)*(1+0.2)*dt)
        ax_cont.set_ylim(-0.5, 1.15 )
        return ln_duty,  ln_steer,



    def control_update(frame):
        print frame
        if frame == 0:
            
            xdata_duty.append([frame])
            ydata_duty.append(duty[frame])
            ln_duty.set_data(xdata_duty, ydata_duty)

            xdata_steer.append([frame])
            ydata_steer.append(steer[frame])
            ln_steer.set_data(xdata_steer, ydata_steer)


        else:
            xdata_duty.append(np.arange(frame -1, frame)*dt)
            ydata_duty.append(duty[frame-1: frame])
            ln_duty.set_data(xdata_duty, ydata_duty)

            xdata_steer.append(np.arange(frame -1, frame)*dt)
            ydata_steer.append(steer[frame-1: frame])
            ln_steer.set_data(xdata_steer, ydata_steer)

        return ln_duty,  ln_steer

    counter = 0
    # vehicle_init()

    control_init()

    # x_lim = 100
    # y_lim = 1.2
    # fig_cont, plt_cont, line_dutycycle, line_steer = plot_control(x_lim,y_lim)

    line_dutycycle_his = []
    line_steer_his     = []  


    while not (rospy.is_shutdown()):

        print "counter :", counter,'index :', start  + counter, 'end :', end
        ########################################### unpack messages ############################################

        # print "counter", counter
        # ( vx_est  , vy_est  , omega_est  , X_est  , Y_est  , yaw_est  )  = data['est_state'][counter,:]
        # ( vx_ol   , vy_ol   , omega_ol   , X_ol   , Y_ol   , yaw_ol   )  = [0, 0, 0, 0, 0, 0]
        # ( vx_meas , omega_meas , X_meas , Y_meas , yaw_meas )  = data['full_meas'][counter,:]

        iter_val = int((1 + 1.3)*counter)

        if iter_val > 2:
            # vehicle_update(iter_val)

            control_update(iter_val)

        
        # line_dutycycle_his.append(duty[iter_val])
        # line_steer_his.append(steer[iter_val])

        # if counter >= window_size:
        #     line_dutycycle_his.pop()
        #     line_steer_his.pop()

        #     line_dutycycle.set_data(np.arange(counter, counter + window_size )*(1./loop_rate) ,line_dutycycle_his[ -window_size : ])
        #     line_steer.set_data(np.arange(counter, counter + window_size )*(1./loop_rate) ,line_steer_his[ -window_size : ])
        #     plt_cont.xlim(counter*(1./loop_rate), (counter + window_size)*(1./loop_rate) )
            
        #     fig_cont.canvas.draw()
        #     plt_cont.show()



        # plt.scatter(X_est, Y_est)
        # fig.show()
        # fig_cont.show()
        # fig_cont.canvas.draw()
        # fig.canvas.draw()
        plt.pause(1./1000000)




    
        # vy_meas = 0.0
        # if sim_on:
        #     ( vx_real , vy_real , omega_real , X_real , Y_real , yaw_real )  = vehicle_state_real.CurrentState

            
        #     real_x_his.append(X_real)
        #     real_y_his.append(Y_real)

        # ########################################################################################################



        # ############################################## vehicle motion plot ######################################

        # est_x_his.append(X_est)
        # est_y_his.append(Y_est)
        # ol_x_his.append(X_ol)
        # ol_y_his.append(Y_ol)
                    
        # meas_x_his.append(X_meas)
        # meas_y_his.append(Y_meas)
        

        # if vehicle_visualization == True: 

        #     l = 0.42/2; w = 0.19/2

        #     car_est_x, car_est_y = getCarPosition(X_est, Y_est, yaw_est, w, l)
        #     rec_est.set_xy(np.array([car_est_x, car_est_y]).T)

        #     car_ol_x, car_ol_y = getCarPosition(X_ol, Y_ol, yaw_ol, w, l)
        #     rec_ol.set_xy(np.array([car_ol_x, car_ol_y]).T)

        #     car_meas_x, car_meas_y = getCarPosition(X_meas, Y_meas, yaw_meas, w, l)
        #     rec_meas.set_xy(np.array([car_meas_x, car_meas_y]).T)

            

        #     line_est.set_data(est_x_his, est_y_his)
        #     line_ol.set_data(ol_x_his, ol_y_his)
        #     line_meas.set_data(meas_x_his, meas_y_his)
                
        #     fig_veh.canvas.draw()
        #     plt_veh.show()
          

        # ##########################################################################################################

        # #############################################  vehicle states plot #######################################

        # line_vx_ol_his.append(vx_ol)
        # line_vx_est_his.append(vx_est)
        # line_vx_meas_his.append(vx_meas)

        # line_vy_ol_his.append(vy_ol)
        # line_vy_est_his.append(vy_est)
        # line_vy_meas_his.append(vy_meas)
        
        # line_omega_ol_his.append(omega_ol)
        # line_omega_est_his.append(omega_est)
        # line_omega_meas_his.append(omega_meas)
        
        # line_X_ol_his.append(X_ol)
        # line_X_est_his.append(X_est)
        # line_X_meas_his.append(X_meas)
        
        # line_Y_ol_his.append(Y_ol)
        # line_Y_est_his.append(Y_est)
        # line_Y_meas_his.append(Y_meas)
        
        # line_yaw_ol_his.append(yaw_ol)
        # line_yaw_est_his.append(yaw_est)
        # line_yaw_meas_his.append(yaw_meas)

        # if sim_on:
        #     line_vx_real_his.append(vx_real)
        #     line_vy_real_his.append(vy_real)
        #     line_omega_real_his.append(omega_real)
        #     line_X_real_his.append(X_real)
        #     line_Y_real_his.append(Y_real)
        #     line_yaw_real_his.append(yaw_real)

        # if states_visualization == True and counter >= 100:

        #     ### Keep size of window to 100 points 

        #     # axs_dy[0,0].set_ylim(min(line_vx_ol_his) + , max(line_vx_ol_his) + ) # FOR SETTING THE DYNAMIC AXES
        #     line_vx_ol.set_data( range(counter, counter + window_size ) ,line_vx_ol_his[ -window_size : ])
        #     line_vx_est.set_data( range(counter, counter + window_size ) ,line_vx_est_his[ -window_size : ])
        #     line_vx_meas.set_data( range(counter, counter + window_size ) ,line_vx_meas_his[ -window_size : ])
        #     axs_dy[0,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
        #     line_vy_ol.set_data( range(counter, counter + window_size ) ,line_vy_ol_his[ -window_size : ])
        #     line_vy_est.set_data( range(counter, counter + window_size ) ,line_vy_est_his[ -window_size : ])
        #     line_vy_meas.set_data( range(counter, counter + window_size ) ,line_vy_meas_his[ -window_size : ])
        #     axs_dy[0,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     line_omega_ol.set_data( range(counter, counter + window_size ) ,line_omega_ol_his[ -window_size : ])
        #     line_omega_est.set_data( range(counter, counter + window_size ) ,line_omega_est_his[ -window_size : ])
        #     line_omega_meas.set_data( range(counter, counter + window_size ) ,line_omega_meas_his[ -window_size : ])
        #     axs_dy[1,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            
            
        #     line_X_ol.set_data( range(counter, counter + window_size ) ,line_X_ol_his[ -window_size : ])
        #     line_X_est.set_data( range(counter, counter + window_size ) ,line_X_est_his[ -window_size : ])
        #     line_X_meas.set_data( range(counter, counter + window_size ) ,line_X_meas_his[ -window_size : ])
        #     axs_dy[1,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     line_Y_ol.set_data( range(counter, counter + window_size ) ,line_Y_ol_his[ -window_size : ])
        #     line_Y_est.set_data( range(counter, counter + window_size ) ,line_Y_est_his[ -window_size : ])
        #     line_Y_meas.set_data( range(counter, counter + window_size ) ,line_Y_meas_his[ -window_size : ])
        #     axs_dy[2,0].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     line_yaw_ol.set_data( range(counter, counter + window_size ) ,line_yaw_ol_his[ -window_size : ])
        #     line_yaw_est.set_data( range(counter, counter + window_size ) ,line_yaw_est_his[ -window_size : ])
        #     line_yaw_meas.set_data( range(counter, counter + window_size ) ,line_yaw_meas_his[ -window_size : ])
        #     axs_dy[2,1].set_xlim(counter, counter + window_size ) # FOR SETTING THE DYNAMIC AXES
            

        #     axs_dy[2,1].set_ylim(min(min(line_yaw_meas_his[ -window_size : ]) + margin*min(line_yaw_meas_his[ -window_size : ])\
        #         , min(line_yaw_ol_his[ -window_size : ]) + margin*min(line_yaw_ol_his[ -window_size : ])), \
        #     max(max(line_yaw_meas_his[ -window_size : ]) + margin*max(line_yaw_meas_his[ -window_size : ]), max(line_yaw_ol_his[ -window_size : ]) + margin*max(line_yaw_ol_his[ -window_size : ]))) # FOR SETTING THE DYNAMIC AXES

        #     fig_dy.canvas.draw()
        #     plt_dy.show()
        

        # ############################## Control input to the vehicle #################################



        # # image_dy = Image.frombytes('RGB', fig_dy.canvas.get_width_height(),fig_dy.canvas.tostring_rgb())
        # # image_dy = np.fromstring(fig_dy.canvas.tostring_rgb(), dtype='uint8')
        # # print "image_dy.shape",image_dy.shape
        # # image_veh = np.fromstring(fig_veh.canvas.tostring_rgb(), dtype='uint8')

        # # image_dy_his.append(image_dy)
        # # image_veh_his.append(image_veh)

        # ##########################################################################################################

        
        
        

        # # create file name and append it to a list
        # # filename_veh = '/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehicle_motion/incorrect_yaw'+str(counter)+'.png'
        # # filename_dy = '/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehicle_states/incorrect_yaw'+str(counter)+'.png'
        
        # # filenames.append(filename)
        
        # # # repeat last frame
        # # if (index == len(coordinates_lists)-1):
        # #     for i in range(15):
        # #         filenames.append(filename)
                
        # # save frame



        # plt_veh.savefig(filename_veh)
        # plt_dy.savefig(filename_dy)

        # plt.pause(.1)
        # plt_veh.close()



        counter +=1
        rate.sleep()


    if record_data == True:

        data_est     = {'vx': line_vx_est_his , 'vy': line_vy_est_his , 'omega': line_omega_est_his , 'X': line_X_est_his , 'Y': line_Y_est_his , 'yaw': line_yaw_est_his } 
        data_meas    = {'vx': line_vx_meas_his , 'vy': line_vy_meas_his , 'omega': line_omega_meas_his , 'X': line_X_meas_his , 'Y': line_Y_meas_his , 'yaw': line_yaw_meas_his } 
        data_ol      = {'vx': line_vx_ol_his , 'vy': line_vy_ol_his , 'omega': line_omega_ol_his , 'X': line_X_ol_his , 'Y': line_Y_ol_his , 'yaw': line_yaw_ol_his } 
        if sim_on:
            data_real    = {'vx': line_vx_real_his , 'vy': line_vy_real_his , 'omega': line_omega_real_his , 'X': line_X_real_his , 'Y': line_Y_real_his , 'yaw': line_yaw_real_his } 
        data_control = {'duty': line_dutycycle_his , 'steer': line_steer_his }

        path = ('/').join(__file__.split('/')[:-2]) + '/data/observer/' 
            
        now = datetime.datetime.now()
        # path = path + now.strftime("d%d_m%m_y%Y/")
        path = path + now.strftime("d%d_m%m_y%Y_hr%H_min%M_sec%S")

        if not os.path.exists(path):
            os.makedirs(path)

        est_path  = path + '/est_his'
        meas_path = path + '/meas_his'
        real_path = path + '/real_his'
        ol_path   = path + '/ol_his'
        control_path = path + '/control_his'
        
        np.save(est_path,data_est)
        np.save(meas_path,data_meas)
        np.save(ol_path,data_ol)
        if sim_on:
            np.save(real_path,data_real)
        np.save(control_path,data_control)




    # Save into a GIF file that loops forever
    # image_dy_his[0].save("/home/auto/Desktop/autonomus_vehicle_project/thesis/TFM_Shivam/raw_doc/estimator/images/vehilce_motion.gif", format='GIF', append_images=image_dy_his[1:], save_all=True, duration=300, loop=0)
    # from moviepy.editor import ImageSequenceClip
    # clip = ImageSequenceClip(list(np.array(image_veh_his)), fps=20)
    # clip.write_gif('test.gif', fps=20)

    print "Saved"
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
