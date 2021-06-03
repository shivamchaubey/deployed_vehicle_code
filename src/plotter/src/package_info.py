#!/usr/bin/env python
import rospkg

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# list all packages, equivalent to rospack list
print "rospack.list", rospack.list() 

# get the file path for rospy_tutorials
rospack.get_path('rospy_tutorials')