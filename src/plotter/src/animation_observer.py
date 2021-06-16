import matplotlib

import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pylab as plt
from matplotlib import animation, rc
rc('animation')    

from numpy import random



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


vx_meas = data['full_meas'][:,0]
omega_meas = data['full_meas'][:,1]

X_meas = data['full_meas'][:,2]
Y_meas = data['full_meas'][:,3]
yaw_meas = data['full_meas'][:,4]

X_lidar = data['lidar_pose'][:,0]
Y_lidar = data['lidar_pose'][:,1]

d_cont = data['control'][:,0]
steer_cont = data['control'][:,1]


rc_fonts = {
    "text.usetex": True,
    'text.latex.preview': True, # Gives correct legend alignment.
    'mathtext.default': 'regular',
    'text.latex.preamble': [r"""\usepackage{bm}"""],
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


fig, ax = plt.subplots()
xdata_est, ydata_est = [], []
xdata_meas, ydata_meas = [], []

ln_est, = plt.plot([], [], '-b',  linewidth  = 2, alpha = 1.0, label = 'Estimated Position')
ln_meas, = plt.plot([], [], '-r',  linewidth  = 2, alpha = 0.6, label = 'Measured Position')

v = 0.00001*np.array([[ 1,  1],
                  [ 1, -1],
                  [-1, -1],
                  [-1,  1]])

rec_est = patches.Polygon(v, alpha=0.7, closed=True, fc='b', ec='k', zorder=10)
ax.add_patch(rec_est)



# plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o') #points on center track
plt.plot(Points1[:, 0], Points1[:, 1], color = 'black' , linewidth  = 3) # inner track
plt.plot(Points2[:, 0], Points2[:, 1], color = 'black' , linewidth  = 3) #outer track
plt.plot(Points0[:, 0], Points0[:, 1], '--' ,color = 'darkgray' , linewidth  = 2) #center track


plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=14, scatterpoints = 1)

plt.xlabel(r"X Position")
plt.ylabel(r"Y Position")


def init():
    ax.set_xlim(-1.7,3.0)
    ax.set_ylim(-0.5, 2.8)
    return ln_est,  ln_meas,



def update(frame):
    # print frame
    xdata_est.append(x_est[frame])
    ydata_est.append(y_est[frame])
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

# ani = animation.FuncAnimation(fig, update, frames=np.array([x,y]).T,
#                     init_func=init, blit=True)

ani = animation.FuncAnimation(fig, update, frames=np.arange(0,len(x_est)), blit = True, init_func=init, interval = (1.0/50)*1000, repeat  = False)
plt.show()

# ani.save(path + dir_name + '/myAnimation.mp4', writer='imagemagick', fps=30, dpi =300)
# # saving to m4 using ffmpeg writer
# writervideo = animation.FFMpegWriter(fps=30, dpi = 300)
# ani.save(path + dir_name + 'estimator.mp4', writer=writervideo)
# plt.close()








# def animate(n):
#     if n == 10:
#         line, = plt.plot(x[:n], y[:n], color='g', label = r'Localization using visual sensor', linewidth =2 )
#         line1, = plt.plot(x1[:n], y1[:n], color='b', label = r'Localization using visual-inertial sensor', linewidth =2 )

#     else:
#         line, = plt.plot(x[n-10:n], y[n-10:n], color='g', label = r'Localization using visual sensor', linewidth =2 )
#         line1, = plt.plot(x1[n-10:n], y1[n-10:n], color='b', label = r'Localization using visual-inertial sensor', linewidth =2 )

#     # line, = plt.scatter(x[:n], y[:n])#, label = r'Localization using visual sensor' )
#     # line1, = plt.scatter(x1[:n], y1[:n])#, label = r'Localization using visual-inertial sensor')

    
#     plt.legend(loc = 'lower center')
#     return line, line1

# anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=1.0/100, blit=True, repeat = False, repeat_delay = True)
# plt.show()

# print (path+file_dir+'myAnimation.gif')
# anim.save(path+file_dir+'myAnimation.gif', writer='imagemagick', fps=10)










