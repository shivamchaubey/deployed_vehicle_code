#!/usr/bin/env python


from geometry_msgs.msg import Pose
import rospy
import tf.transformations
from geometry_msgs.msg import PoseStamped


class lidar_pose():
    """ Object collecting hector slam pose data
    """
    def __init__(self):
        """ Initialization
        Arguments:
            t0: starting measurement time
        """
        rospy.Subscriber("/slam_out_pose", PoseStamped, self.pose_callback, queue_size=1)

        # ECU measurement
        self.X = 0.0
        self.Y = 0.0
        self.yaw = 0.0

    def pose_callback(self,msg):
        """Unpack message from lidar sensor"""
        
        self.X = msg.pose.position.x
        self.Y = msg.pose.position.y
        quat = msg.pose.orientation
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.yaw = euler[2]
    
    





def main():
    rospy.init_node('lidar_listener', anonymous=True)
    
    loop_rate   = 30
    rate        = rospy.Rate(loop_rate)
    pose_l =lidar_pose()

    while not (rospy.is_shutdown()):

        print 'pose_l.x', pose_l.X, 'pose_l.y', pose_l.Y, 'pose_l.yaw', pose_l.yaw

        rate.sleep()









if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
