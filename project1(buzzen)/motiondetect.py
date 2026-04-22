import cv2
import serial

url = "http://10.65.36.104:81/stream"
cap = cv2.VideoCapture(url)

arduino = serial.Serial('COM8', 9600) 

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    motion_detected = False
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame1, "Motion", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
    if motion_detected:
        arduino.write(b'1')
    else:
        arduino.write(b'0')

    cv2.imshow("Motion Detected", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()