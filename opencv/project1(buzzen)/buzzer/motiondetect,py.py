import cv2
import urllib.request
import numpy as np
import serial
import time

# 🔧 CHANGE THIS
arduino = serial.Serial('COM5', 9600)
time.sleep(2)

url = "http://YOUR_IP/capture"

distance = 0

while True:
    try:
        # 📡 Get frame
        img_resp = urllib.request.urlopen(url, timeout=5)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        if frame is None:
            continue

        # 📏 Read distance
        if arduino.in_waiting:
            line = arduino.readline().decode('utf-8').strip()
            if line.isdigit():
                distance = int(line)

        # 🖥️ Overlay text
        cv2.putText(frame, f"Distance: {distance} cm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Sensor Fusion", frame)

        time.sleep(0.2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print("Error:", e)
        time.sleep(2)

arduino.close()
cv2.destroyAllWindows()