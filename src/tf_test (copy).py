#!/usr/bin/env python  
import rospy
import tf
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray as vec
from sensor_msgs.msg import CameraInfo 
from sensor_msgs.msg import Image
from image_geometry import PinholeCameraModel
from gui import *
import copy

cam_model = PinholeCameraModel()
intrinsic_mat = np.zeros((3,3))
M = np.zeros((3,3))
plane = []
result_array = []
print_array = []

bridge = CvBridge()
br = tf.TransformBroadcaster()
rot = []
point = []
axis = []

def cal(xyz,plane):
    a,b,c,d = plane
    x,y,z = xyz
    k = -d/(a*x + b*y + c*z)
    return k*x,k*y,k*z
    result_array
def qv_mult(q1, v1):
    v1 = tf.transformations.unit_vector(v1)
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2), 
        tf.transformations.quaternion_conjugate(q1)
    )[:3]

def callback2(data):
    global result_array
    global plane
    global print_array
    global rot
    #rospy.loginfo("object_center %s",data.data)
    size = len(data.data)
    result_array = []
    print_array = []
    batch = 5
    for i in range(size/batch): 
    	pixel_point = data.data[i*batch:(i+1)*batch]
        index = pixel_point[0]
        xyz = cam_model.projectPixelTo3dRay(pixel_point[1:3]) #pixel points
        x,y,z = cal(xyz,plane) #calculate the 3d points
        result_array.append(index)
        result_array.append(x)
        result_array.append(y)
        result_array.append(z)

        #print(rot)  publish the transformation 
        br.sendTransform((x,y,z),(rot[0],rot[1],rot[2],rot[3]),rospy.Time.now(),"switch%d"%index,"camera_rgb_optical_frame")
        
        local_arr = []
        local_arr.append(index)
        local_arr.append(int(pixel_point[1] - pixel_point[3]/2))
        local_arr.append(int(pixel_point[2] - pixel_point[4]/2))
        local_arr.append(int(pixel_point[3]))
        local_arr.append(int(pixel_point[4]))
        print_array.append(local_arr)
    
def send_frame_based_on_other(base,x_offset,y_offset,z_offset,rot,name,base_name):
	global axis
	x_axis,y_axis,z_axis = axis
	res = base + z_axis*z_offset + x_axis*x_offset + y_axis*y_offset
	br.sendTransform((res[0],res[1],res[2])
	,(rot[0],rot[1],rot[2],rot[3]),rospy.Time.now(),name,base_name)

def callback(data):
    global intrinsic_mat
    cam_model.fromCameraInfo(data)
    intrinsic_mat = cam_model.fullIntrinsicMatrix()

def plane_func(rot,trans):
	global axis
	global M
	y_axis = np.array([0,1,0])
	x_axis = np.array([1,0,0])
	z_axis = np.array([0,0,1])
	aruco_y = qv_mult(rot, y_axis)
	aruco_x = qv_mult(rot, x_axis)
	aruco_z = qv_mult(rot, z_axis)
	a = aruco_y[0]
	b = aruco_y[1]
	c = aruco_y[2]
	x = trans[0]
	y = trans[1]
	z = trans[2]
	d = 0 - a*x - b*y - c*z
	xyz = [x,y,z]
	axis = [aruco_x,aruco_y,aruco_z]
	individual_line(xyz,axis,rot)
	return [a,b,c,d]

def img_callback(data):
	global result_array
	global print_array
	global rot
	global M
	try:
		cv_image = bridge.imgmsg_to_cv2(data,"bgr8")
		updateView(cv_image, print_array,M)
	except CvBridgeError as e: 
		print(e)
	
	
def individual_line(xyz,axis,rot):
	global point
	global M
	offset = 0.1
	try:
		x,y,z = xyz
		x_axis,y_axis,z_axis = axis
		base = np.array([x,y,z])
		points = np.array([(274, 323), (271, 190), (406, 320),(404,187)])
		dst_pts = np.float32(points[:, np.newaxis, :])
		#print(z_axis)
		center = intrinsic_mat.dot(np.array([x,y,z]))
		x_end = intrinsic_mat.dot(base + x_axis*offset)
		z_end = intrinsic_mat.dot(base + z_axis*offset)
		xz_end = intrinsic_mat.dot(base + z_axis*offset + x_axis*offset)

		center = center * ( 1.0 /center.item(0,2) )
		x_end = x_end * ( 1.0 /x_end.item(0,2) )
		z_end = z_end * ( 1.0 /z_end.item(0,2) )
		xz_end = xz_end * ( 1.0 /xz_end.item(0,2) )

		point = [ (int(round(center.item(0,0))),int(round(center.item(0,1)))),
				  (int(round(x_end.item(0,0))),int(round(x_end.item(0,1)))),
				  (int(round(z_end.item(0,0))),int(round(z_end.item(0,1)))),
				  (int(round(xz_end.item(0,0))),int(round(xz_end.item(0,1))))
				]
		p_ = np.array(point)
		src_pts = np.float32(p_[:, np.newaxis, :])
		M, mask = cv2.findHomography(src_pts, dst_pts, 0)
		#print(M)

		#-----------add--robot--frame-------------------------
		'''
		robot_x_offset = 0
		robot_y_offset = 0
		robot_z_offset = 1
		robot_location = base + z_axis*robot_z_offset + x_axis*robot_x_offset + y_axis*robot_y_offset
		br.sendTransform((robot_location[0],robot_location[1],robot_location[2]),(rot[0],rot[1],rot[2],rot[3]),rospy.Time.now(),"visual_determined_robot","camera_rgb_optical_frame")
		'''
		send_frame_based_on_other(base,0,0,1,rot,"hand","camera_rgb_optical_frame")

	except Exception as e: print(e)

if __name__ == '__main__': 
    global plane
    global rot
    print("start")
    cv2.namedWindow('interactive')
    cv2.setMouseCallback('interactive',position)
    rospy.init_node('aruco_tf_listener')
    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    rospy.Subscriber("/camera/rgb/camera_info",CameraInfo,callback)
    rospy.Subscriber("/object_center", vec, callback2)
    rospy.Subscriber("/camera/rgb/image_color", Image, img_callback)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/camera_rgb_optical_frame','/aruco_marker_frame', rospy.Time(0))
            #print(rot)
            #print(trans)
            plane = plane_func(rot,trans)
            #print(plane)
            #rospy.loginfo(plane)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()
        
