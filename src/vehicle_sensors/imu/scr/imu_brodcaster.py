#!/usr/bin/env python  
import rospy

# Because of transformations
import tf_conversions

import tf2_ros
import geometry_msgs.msg
import turtlesim.msg
from geometry_msgs.msg import Twist, Pose
import numpy as np

class ImuEncClass(object):
    """ Object collecting IMU + Encoder data
        The encoder measures the angular velocity at the DC motor output. Then
        it is transformed to wheel linear velocity and put over the message:
        /twist.linear.x
    """

    def __init__(self,t0):

        rospy.Subscriber('twist', Twist, self.Twist_callback, queue_size=1)

        rospy.Subscriber('pose', Pose, self.Pose_callback, queue_size=1)

        rospy.Subscriber('/wheel_rpm_feedback', Float32, self.RPM_callback, queue_size=1)

        self.roll    = 0.0
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.psiDot  = 0.0
        self.ax      = 0.0
        self.ay      = 0.0
        self.wheel_rpm = 0.0
        # time stamp
        self.t0     = t0

        # Time for yawDot integration
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.prev_time = self.curr_time

                #history
        self.twist_hist = {"timestamp_ms":[],"vx":[],"vy":[],"psiDot":[],"ax":[],"ay":[]}
        self.pose_hist = {"timestamp_ms":[],"roll":[],"pitch":[],"yaw":[]}
        self.wheel_rpm_hist = {"timestamp_ms":[],"wheel_rpm":[]}

    def Twist_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.vx     = self.vx+data.linear.x*(self.curr_time-self.prev_time)  # from IMU
        self.vy     = self.vy+data.linear.y*(self.curr_time-self.prev_time)
        self.ax     = data.linear.x
        self.ay     = data.linear.y
        self.psiDot = data.angular.z # from IMU

        self.twist_hist["timestamp_ms"].append(self.curr_time)
        self.twist_hist["vx"].append(self.vx)
        self.twist_hist["vy"].append(self.vy)
        self.twist_hist["ax"].append(self.ax)
        self.twist_hist["ay"].append(self.ay)
        self.twist_hist["psiDot"].append(self.psiDot)
        self.prev_time = self.curr_time

    def Pose_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0

        self.roll   = 0.0
        self.pitch  = 0.0
        self.yaw    = wrap(data.orientation.z) # from IMU

        self.pose_hist["timestamp_ms"].append(self.curr_time)
        self.pose_hist["roll"].append(self.roll)
        self.pose_hist["pitch"].append(self.pitch)
        self.pose_hist["yaw"].append(self.yaw)
        self.prev_time = self.curr_time

    def RPM_callback(self, data):
        self.curr_time = rospy.get_rostime().to_sec() - self.t0
        self.wheel_rpm = data.data

        self.wheel_rpm_hist["timestamp_ms"].append(self.curr_time)
        self.wheel_rpm_hist["wheel_rpm"].append(self.wheel_rpm)
        self.prev_time = self.curr_time
        
def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def handle_turtle_pose(msg, turtlename):

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now()
    #ImuEncClass(rospy.get_time())
    t.header.frame_id = "base_link"
    t.child_frame_id = "base_imu"
    t.transform.translation.x = 117.72 * 10**(-3)
    t.transform.translation.y = 6.1270 * 10**(-3)
    t.transform.translation.z = -0.27299 * 10**(-3)
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, 3.14/2)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('tf2_turtle_broadcaster')
    # turtlename = rospy.get_param("~base_imu")
    rospy.Subscriber('/pose',
                     Pose,
                     handle_turtle_pose,
                     "base_imu")
    rospy.spin()