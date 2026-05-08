import cv2

img = cv2.imread('hq720.jpg',0)

orb = cv2.ORB_create()

keypoints,desc = orb.detectAndCompute(img,None)

img2 = cv2.drawKeypoints(img,keypoints,None,color=(0,255,0))

cv2.imshow("ORB features", img2)
cv2.waitKey(0)
cv2.destroyAllWindows