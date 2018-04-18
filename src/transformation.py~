#!/usr/bin/env python  
import rospy
import math
import numpy
from geometry_msgs.msg import Transform
from sensor_msgs.msg import JointState
import tf
import tf2_ros

def listen():
    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
	dic = {}
	for i in range(0,5):
		target = '/switch' + str(i)
		try:
		    (trans,rot) = listener.lookupTransform('/hand', target, rospy.Time(0))
		    dic[target] = trans, rot
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
		    continue
    print(dic)

if __name__ == '__main__':
    rospy.init_node('transformation')
    listen()
    rospy.spin()
