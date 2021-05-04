#!/usr/bin/env python

import rospy
import numpy as np
from math import cos, sin, atan2, atan, pi, acos
from geometry_msgs.msg import Pose

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


def main():

    rospy.init_node('fishcam_pose', anonymous=True)
    loop_rate       = 30
    rate            = rospy.Rate(loop_rate)


    yaw_trans_pub  = rospy.Publisher('yaw_transformation', Pose, queue_size=1)
    yaw_msg = Pose()

    t0 = rospy.get_time()
    imu = ImuClass(t0)

    yaw_offset = pi/2

    while not (rospy.is_shutdown()):

        yaw_imu = imu.yaw

        yaw_trans =  wrap(yaw_imu + yaw_offset)
        
        yaw_msg.orientation.z = yaw_trans

        yaw_trans_pub.publish(yaw_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass