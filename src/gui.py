# -*- coding: utf-8 -*-
"""
Demonstrates a variety of uses for ROI. This class provides a user-adjustable
region of interest marker. It is possible to customize the layout and 
function of the scale/rotate handles in very flexible ways. 
"""

#import initExample ## Add path to library (just for examples; you do not need this)

import numpy as np
import time
import cv2
import sys
from cnn import *

push_pull = '/home/tommy/catkin_ws/src/compute_tf/src/push_pull.jpg'
push = '/home/tommy/catkin_ws/src/compute_tf/src/push.png'
turn = '/home/tommy/catkin_ws/src/compute_tf/src/turn.png'
icon_w = 100
icon_h = 100
chosen = -1
count = 0
imagecount = 0
down = False
record = 0

class Button(object):
	def __init__(self, category, x, y, w, h, x1,y1,x2,y2,x3,y3,x4,y4):
		self.category = category
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.x1 = x1
		self.x2 = x2
		self.x3 = x3
		self.x4 = x4
		self.y1 = y1
		self.y2 = y2
		self.y3 = y3
		self.y4 = y4
		self.setIcon(category)

	def setIcon(self,category):
		if category == 1 or category == 2:
			self.operation = cv2.imread(push_pull)
			self.iw = 2 * icon_w
			self.ih = icon_h
			self.operation = cv2.resize(self.operation, (icon_w * 2,icon_h))

		elif category == 3:
			self.operation = cv2.imread(turn)
			self.operation = cv2.resize(self.operation, (icon_w,icon_h))
			self.iw = icon_w
			self.ih = icon_h
		else:
			self.operation = cv2.imread(push)
			self.operation = cv2.resize(self.operation, (icon_w,icon_h))
			self.iw = icon_w
			self.ih = icon_h

	def within(self,px, py):
		a = (self.x2 - self.x1)*(py - self.y1) - (self.y2 - self.y1)*(px - self.x1)  
		b = (self.x4 - self.x2)*(py - self.y2) - (self.y4 - self.y2)*(px - self.x2)  
		c = (self.x3 - self.x4)*(py - self.y4) - (self.y3 - self.y4)*(px - self.x4)  
		d = (self.x1 - self.x3)*(py - self.y3) - (self.y1 - self.y3)*(px - self.x3)  
		if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0):   
			return True;   
		#if px - self.x > 0 and px - self.x < self.w and py - self.y > 0 and py - self.y < self.h:
		#	return True
		else:
			return False
	def chosen(self,bx, by):
		if bx - self.x > 0 and bx - self.x < self.w and by - self.y > 0 and by - self.y < self.h:
			return True
		else:
			return False

	def operate(self,bx, by):	
		if bx > 0 and bx < self.iw and by > 0 and by < self.ih: 
			if self.category == 3:
				s =  "turn"
			elif self.category == 4:
				s =  "push"
			else:
				if bx < self.iw / 2:
					s = "push"
				else:
					s = "pull"
		else:
			s = "empty"
		return s


class Hough(object):
	def __init__(self, category):
		self.category = category
	
	def findCircleArea(self,rois_info):
		circle_id = -1
		for i in range(len(rois_info)):
			ob = rois_info[i]
			obid = ob[0]
			if (obid == 4):
				circle_id = i
			else:
				circle_id = -1
		return circle_id


	def getTransformedPoints(self,Cid, M, print_array):

		a = np.array([[print_array[Cid][1], print_array[Cid][2]], 
			          [print_array[Cid][3], print_array[Cid][4]], 
			          [print_array[Cid][7], print_array[Cid][8]],
			          [print_array[Cid][5], print_array[Cid][6]]],dtype='float32')
		a = np.array([a])
		aN = cv2.perspectiveTransform(a, M)
		return aN

	def findHoughCircle(self,image, M, a2):
		image = cv2.warpPerspective(image,M,(640,624))

		#cv2.imshow('hough',image)
		#k = cv2.waitKey(1)

		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		size = 624,640,3
		empty = np.zeros(size, dtype=np.uint8)
		mask = np.zeros(size, dtype=np.uint8)
		
		img[:,:int(a2[0][0][0])] = 0
		#img[:int(abs(a2[0][0][1])) - 10,:] = 0
		img[:,int(abs(a2[0][1][0])) + 30:] = 0
		img[int(abs(a2[0][2][1])):,:] = 0
		img = cv2.medianBlur(img,5)
		
		circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,20,param1=50,param2=28,minRadius=20,maxRadius=50)
		#print(len(circles))
		#print(circles)
		circles = np.uint16(np.around(circles))

		return circles, empty, mask

	def combineImageWithCircle(self,empty, mask, orgin, M, color):
		empty = cv2.warpPerspective(empty,np.linalg.inv(M),(640,624))
		mask = cv2.warpPerspective(mask,np.linalg.inv(M),(640,624))
		mask[:,:,0] = mask[:,:,color]
		mask[:,:,1] = mask[:,:,color]
		mask[:,:,2] = mask[:,:,color]
		mask.astype(bool)
		#origin = orgin[~mask]
		#orgin[mask] = 0
		orgin = orgin * (~mask/255)
		#print(~mask/255) 
		image = empty + orgin
		return image 

	def chooseColor(self,info, x, y,M):
		a = np.array([[x,y]],dtype='float32')
		a = np.array([a])
		aN = cv2.perspectiveTransform(a, M)
		x = aN[0][0][0]
		y = aN[0][0][1]
		if (info[0] - x) * (info[0] - x) + (info[1] - y) * (info[1] - y) < info[2] * info[2]:
			return (0,0,255), True
		else:
			return (0,255,0), False

def individual_offset(top_left,top_right,bot_left,bot_right,center,label = 0):
	# 3 offset: 0.7,0
	# 3 new offset 0.55 0
    # 1 offset: 0.1,0
	# 2 offset: 0,0
	# 4 offset: up 0.33,0.1 down: -0.36,0.08
	off_list = [[0,0],[0.1,0],[0,0],[0.55,0],[0.33,0.1],[-0.36,0.08]]
	off1,off2 = off_list[label]
	x1,y1 = top_left
	x2,y2 = top_right
	x3,y3 = bot_left
	x4,y4 = bot_right
	xc,yc = center
	xtc = int((x2 + x1)/2)
	ytc = int((y2 + y1)/2)
	xrc = int((x2 + x4)/2)
	yrc = int((y2 + y4)/2)
	offx1 = (xtc-xc) * off1 #+ xc
	offy1 = (ytc-yc) * off1 #+ yc
	offx2 = (xrc-xc) * off2 
	offy2 = (yrc-yc) * off2 
	off_set_x = offx1+offx2
	off_set_y = offy1+offy2
	return off_set_x, off_set_y

def updateView(arr, rois_info, M, M2,model):   
	global chosen
	global count
	global ly
	global imagecount
	global record
	cmd = False
	opt = 'empty'
	b_num = 0
	image = arr
	global down
	arrx, arry, channel = arr.shape
	xc = 0
	yc = 0
	xtc = 0
	ytc = 0
	for info in rois_info:
		#print('Hello')
		c,x1,y1,x2,y2,x3,y3,x4,y4 = info
		if c == 4:
			xc = int((x1 + x2 + x3 + x4)/4)
			yc = int((y1 + y2 + y3 + y4)/4)
			xtc = int((x2 + x1)/2)
			ytc = int((y2 + y1)/2)
			break
	hg = Hough('circle')
	#print(rois_info)
	Cid = hg.findCircleArea(rois_info)

	image = np.zeros((int(arrx * 1.3), arry, channel), dtype=np.uint8)
	image[:arrx,:,:] = arr
	arr = image
	if Cid != -1: # when there are object 4,which is four circles we need
	#-------------------------------the followings are the hough stuff--------------------------------------------
		#try:
		orgin = image
		a2 = hg.getTransformedPoints(Cid, M, rois_info)
		circles, empty, mask = hg.findHoughCircle(image, M, a2)
		for i in circles[0,:]:
			# draw the outer circle
			color, circleChoose = hg.chooseColor(i, mx, my,M)
			cv2.circle(empty,(i[0],i[1]),i[2],color,5)
			# draw the center of the circle
			cv2.circle(empty,(i[0],i[1]),2,(0,0,255),3)

			cv2.circle(mask,(i[0],i[1]),i[2],(0,255,0),5)
			# draw the center of the circle
			cv2.circle(mask,(i[0],i[1]),2,(0,0,255),3)
			color, circleChoose = hg.chooseColor(i, lx, ly,M)
		
			if circleChoose and (xtc-lx)*(xc-lx) + (ytc-ly)*(yc-ly)>0:
				down = True
			#else:
			#	down = False
		arr = hg.combineImageWithCircle(empty, mask, orgin, M, 1)
		#except Exception as e: print(e)


	for info in rois_info:
		#print('Hello')
		c,x1,y1,x2,y2,x3,y3,x4,y4 = info
		x = min(x1, x2, x3, x4)
		xm = max(x1, x2, x3, x4)
		y = min(y1, y2, y3, y4)
		ym = max(y1, y2, y3, y4)
		width = xm - x
		high = ym - y
		#if x < 0 or y < 0:
		#	print ("info:", c,x,y,width,high)
		#	continue

		'''
		if c == 2 :
			imgM2 = cv2.warpPerspective(image,M2,(640,624))
			imgM2 = imgM2[70:400,170:410,:]
			#cv2.imshow('hough',imgM2)
			#k = cv2.waitKey(1)
			if record < 50:
				path = '/home/tommy/catkin_ws/src/compute_tf/data'
				cv2.imwrite(path + '/mid/mid_'+str(record + 2200) + '.png',imgM2)
				print(record)
				record += 1
		#	imagecount += 1
		'''
	
		button = Button(c,x,y,width,high,x1,y1,x2,y2,x3,y3,x4,y4)
		#x1 = x + width
		#y1 = y + high
		#print px, x, width, py, y, high
	
		xc = int((x1 + x2 + x3 + x4)/4)
		yc = int((y1 + y2 + y3 + y4)/4)
		off_set_x,off_set_y = individual_offset([x1,y1],[x2,y2],[x3,y3],[x4,y4],[xc,yc],c)
		#print(off_set_x,off_set_y)
		#off_set_x = 0
		#off_set_y = 0

		predict = 1
		if(c == 2):
			imgM2 = cv2.warpPerspective(image,M2,(640,624))
			imgM2 = imgM2[70:400,170:410,:]
			cv2.imshow('hough',imgM2)
			k = cv2.waitKey(1)

			img = cv2.resize(imgM2, (80,180), interpolation = cv2.INTER_AREA)
			#print self.path + cn + '/' + cn + '_' + str(i) + '.png'
			img = np.reshape(img, 80 * 180 * 3)
			img = np.array([img])
			#predict = classify(False,img)
			predict = model.Predict(img,0.5)
			#print(predict)
		color = (0,255,0)
		if button.within(mx,my):
			color = (255,0,0)
			# cv2.rectangle(arr, (x,y), (x1,y1),(255,0,0),2)
		elif c ==2 and predict == 0:
			color = (0,0,255)
		cv2.line(arr,(x1,y1),(x2,y2),color,2)
		cv2.line(arr,(x1,y1),(x3,y3),color,2)
		cv2.line(arr,(x4,y4),(x3,y3),color,2)
		cv2.line(arr,(x4,y4),(x2,y2),color,2)
		#cv2.circle(arr,(int(xc + off_set_x),int(yc + off_set_y)), 8, (0,255,0),8)
			# cv2.rectangle(arr, (x,y), (x1,y1),(0,255,0),2)

		
		#print bx, x, width, by, y, high
		if chosen < 0 or chosen > 0:
			if button.chosen(lx,ly):
				#print (lx, ly, "show icon")
				ppx = button.ih
				ppy = button.iw
				arr[arrx:arrx+ppx,0:ppy,:] = button.operation
				chosen = button.category

		if chosen == button.category:
			opt = button.operate(lx,ly-arrx)
			if opt != "empty":
				b_num = c
				if c == 4:
					if down:
						b_num = 5
						down = False	
				cmd = True
				print b_num, opt
				chosen = -1
					
	arrx_n, arry, channel = arr.shape
	# print (x,y,channel)
	# image = np.zeros((int(arrx * 1.2), arry, channel), dtype=np.uint8)
	image[:arrx_n,:,:] = arr
	# print (lx)
	if ly > arrx and lx > arry/2:
		robot = True
		count = 1
		ly = -1
	elif count % 50 > 0:
		robot = True
		count += 1
	else:
		robot = False
	if robot:
		image[arrx:,:,1] = 255
	# print(robot)
	cv2.imshow("interactive", image)
	k = cv2.waitKey(1)
	return robot,cmd, b_num, opt


mx = -1 
my = -1
lx = -1
ly = -1
rx = -1
ry = -1
def position(event, x, y, flags, param):
	if event == cv2.EVENT_MOUSEMOVE:
		global mx, my
		mx = x
		my= y
	if event == cv2.EVENT_LBUTTONDOWN:
		global lx, ly
		lx = x
		ly = y
		#print lx, ly
	if event == cv2.EVENT_RBUTTONDOWN:
		global rx, ry
		rx = x
		ry = y
		#print rx, ry


#cv2.namedWindow('interactive')
#cv2.setMouseCallback('interactive',position)

#cv2.setMouseCallback('choose',position)


#rois_info = [[1,200,200,100,100],[2,400,400,100,100],[3,600,600,100,100]]
#for i in range(10000):
#	arr = cv2.imread('/home/xiyacao/Desktop/red.png')
#	updateView(arr, rois_info)
	
   
