import cv2
import numpy as np

url = "http://10.189.118.104:81/stream"
cap = cv2.VideoCapture(url)

# Create window
cv2.namedWindow("Trackbars")

# Create sliders
cv2.createTrackbar("LH", "Trackbars", 100, 180, lambda x: None)
cv2.createTrackbar("LS", "Trackbars", 50, 255, lambda x: None)
cv2.createTrackbar("LV", "Trackbars", 50, 255, lambda x: None)
cv2.createTrackbar("UH", "Trackbars", 130, 180, lambda x: None)
cv2.createTrackbar("US", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("UV", "Trackbars", 255, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get slider values
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()