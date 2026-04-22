import cv2
import urllib.request
import numpy as np
import serial
import time

# 🔧 CHANGE THIS
arduino = serial.Serial('COM8', 9600)
time.sleep(2)

url = "http://10.36.61.104/capture"

prev_frame = None
distance = 0

while True:
    try:
       
        # 📡 Get frame from ESP32/*
        
        img_resp = urllib.request.urlopen(url, timeout=1)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        if frame is None:
            pass

        # 🎯 MOTION DETECTION
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = gray

        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        motion_detected = np.sum(thresh) > 1000000  # adjust if needed

        prev_frame = gray

        # 🔊 CONTROL BUZZER
        if motion_detected:
            arduino.write(b'1')
        else:
            arduino.write(b'0')
        
         # 📏 READ DISTANCE FROM ARDUINO
        if arduino.in_waiting:
            try:
                line = arduino.readline().decode('utf-8').strip()
                print(line)
                if line.isdigit():
                    distance = int(line)
            except:
                pass


        # 🖥️ DISPLAY DISTANCE + STATUS
        cv2.putText(frame, f"Distance: {distance} cm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        status = "MOTION DETECTED" if motion_detected else "NO MOTION"

        cv2.putText(frame, status,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # 🖼️ SHOW FRAME
        cv2.imshow("Vision + Ultrasonic", frame)

        # 🔥 IMPORTANT: slow down requests (ESP32 stability)
        time.sleep(0.2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print("Error:", e)
        time.sleep(2)

arduino.close()
cv2.destroyAllWindows()