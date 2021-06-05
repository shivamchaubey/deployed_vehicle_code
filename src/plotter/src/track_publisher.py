#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3, Quaternion, Pose, Twist, PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path
from plotter.msg import Track
import sys
sys.path.append(('/').join(sys.path[0].split('/')[:-2])+'/planner/src/')
from trackInitialization import Map
import numpy as np

def main():

    rospy.init_node("track")
    loop_rate       = 100
    rate            = rospy.Rate(loop_rate)

    map = Map()


    inner_path_pub_rviz = rospy.Publisher('/inner_track_rviz', Path, queue_size=10)
    outer_path_pub_rviz = rospy.Publisher('/outer_track_rviz', Path, queue_size=10)

    inner_path_pub_plot = rospy.Publisher('/inner_track_plot', Track, queue_size=10)
    outer_path_pub_plot = rospy.Publisher('/outer_track_plot', Track, queue_size=10)

    track_inner = Track()
    track_outer = Track()

    pose_inner = PoseStamped()
    path_inner = Path()

    pose_outer = PoseStamped()
    path_outer = Path()


    ################################# MAP ################################################
    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 3))
    Points2 = np.zeros((Points, 3))
    Points0 = np.zeros((Points, 3))    


    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)




    for i in range(len(Points1[:, 0])):
        pose_inner.pose.position.x = Points1[i, 0]
        pose_inner.pose.position.y = Points1[i, 1]
        pose_inner.header.frame_id = '/map'
        path_inner.poses.append(pose_inner)

        pose_outer.pose.position.x = Points2[i, 0]
        pose_outer.pose.position.y = Points2[i, 1]
        pose_outer.header.frame_id = '/map'
        path_outer.poses.append(pose_outer)



    counter_inner = 0
    counter_outer = 0
    while not (rospy.is_shutdown()):

        inner_path_pub_rviz.publish(path_inner)
        outer_path_pub_rviz.publish(path_outer)

        track_inner.X = Points1[counter_inner, 0]
        track_inner.Y = Points1[counter_inner, 1]

        track_outer.X = Points2[counter_outer, 0]
        track_outer.Y = Points2[counter_outer, 1]


        inner_path_pub_plot.publish(track_inner)
        outer_path_pub_plot.publish(track_outer)

        counter_inner += 1
        counter_outer += 1

        if counter_inner == len(Points1[:, 0]):
            counter_inner = 0

        if counter_outer == len(Points2[:, 0]):
            counter_outer = 0



        rate.sleep()


# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import PoseStamped

# path = Path()

# def odom_cb(data):
#     global path
#     path.header = data.header
#     pose = PoseStamped()
#     pose.header = data.header
#     pose.pose = data.pose.pose
#     path.poses.append(pose)
#     path_pub.publish(path)

# rospy.init_node('path_node')

# odom_sub = rospy.Subscriber('/odom', Odometry, odom_cb)
# path_pub = rospy.Publisher('/path', Path, queue_size=10)

# if __name__ == '__main__':
#     rospy.spin()







if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
