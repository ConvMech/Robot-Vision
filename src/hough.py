import cv2
import numpy as np

print("hello")
cimg = cv2.imread('/home/tommy/Pictures/1.png')
print("read")
img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)
circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,20,
					        param1=50,param2=35,minRadius=0,maxRadius=70)

print("world")
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
	# draw the outer circle
	cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	# draw the center of the circle
	cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("test_window",cimg)
cv2.waitKey(-1)
