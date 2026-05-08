import cv2

url = "http://10.189.118.104:81/stream"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # LOWER RED RANGE
    lower_red1 = (100, 30, 50)
    upper_red1 = (130, 255, 255)

    # UPPER RED RANGE
    lower_red2 = (170, 80, 50)
    upper_red2 = (180, 255, 255)

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 #+ mask2
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Color Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()