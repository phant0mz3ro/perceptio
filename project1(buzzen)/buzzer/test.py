import cv2
import urllib.request
import numpy as np
import time

url = "http://10.85.21.104:81/capture"

cap = cv2.VideoCapture(url)

while True:
    try:
        img_resp = urllib.request.urlopen(url, timeout=5)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        if frame is None:
            continue

        cv2.imshow("Safe Feed", frame)

        time.sleep(0.3)  # 🔥 slower = stable

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception as e:
        print("Reconnect...", e)
        time.sleep(2)

cv2.destroyAllWindows()