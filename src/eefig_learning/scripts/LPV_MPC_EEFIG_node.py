#!/usr/bin/env python
# Tools
import os, sys
import numpy as np
import itertools

# ROS Tools:
import rospy
import message_filters
from std_msgs.msg import Float32
from eefig_learning.srv import LPVMat, LPVMatResponse

# Imports
from LPV_MPC_EEFIG import LPV_MPC_EEFIG

abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path (Floder "EEFIG_Learing")


# NOTE: This node is a implementation of the LPV_MPC_EEFIG class capable of interfacing with ROS
class LPV_MPC_EEFIG_node (LPV_MPC_EEFIG):

    def __init__ (self, settings, xk_topics, online_learning, configuration_file = None):
        super(LPV_MPC_EEFIG_node, self).__init__(settings, configuration_file)

        self.online_learning = online_learning

        # Initialize Services
        self.ser = rospy.Service('/get_LPV_matrices', LPVMat, self.callback_get_LPV_matrices)

        # Initializa Subscribers to xk
        self.ts_subs = [None] * len(xk_topics)
        for i in range(len(xk_topics)):
            self.ts_subs[i] = message_filters.Subscriber(xk_topics[i], Float32)
        
        ts = message_filters.ApproximateTimeSynchronizer(self.ts_subs, 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)


    # Time Synchronizer Callback
    def callback (self, *args):

        self.xk = np.array([]) # Empty previous values 
        for i in range(len(args)):
            self.xk = np.hstack([self.xk, args[i]])

        # Online Learning
        if self.online_learning:
            self.update(xk)


    # Service Response
    def callback_get_LPV_matrices (self, req):

        # Get LPV Matrices
        if req.xk is None:
            A, B = self.get_LPV_matrices(self.xk)
        else:
            xk = np.array(req.xk)
            A, B = self.get_LPV_matrices(xk)

        # Add to Response
        flat_A = list(itertools.chain.from_iterable(A))
        flat_B = list(itertools.chain.from_iterable(B))

        return LPVMatResponse(A=flat_A, B=flat_B)


# Null Configuration EEFIG
def get_null_settings ():

    # SETTINGS EEFIG
    settings = {}

    settings["nx"]                          = 0
    settings["nu"]                          = 0
    settings["lmbd"]                        = 0
    settings["effective_N"]                 = 0
    settings["thr_continuous_anomalies"]    = 0
    settings["separation"]                  = 0
    settings["ff"]                          = 0
    settings["nphi"]                        = 0

    return settings


# Main
if __name__ == "__main__":

    # Initialize:
    rospy.init_node("LPV_MPC_EEFIG_node")

    xk_topics = rospy.get_param("/lpv_mpc_eefig/xk_topics").split(" ")
    print("Number of 'xk_topics' = {}".format(len(xk_topics)))
    for i in range(len(xk_topics)):
        print("\ttopic {} - {}".format(i+1, xk_topics[i]))

    online_learning = rospy.get_param("/lpv_mpc_eefig/online_learning")
    print("online learning = {}".format(online_learning))

    configuration_file = rospy.get_param("/lpv_mpc_eefig/configuration_file")
    file_path = os.path.join(abspath, "saves", configuration_file)
    print("configuration file path = {}".format(file_path))

    # Start publishing:
    settings = get_null_settings()
    lpv = LPV_MPC_EEFIG_node(settings, xk_topics, online_learning, configuration_file=file_path)

    rospy.spin()