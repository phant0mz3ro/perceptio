import cv2

url = "http://10.251.195.104:81/stream"
cap = cv2.VideoCapture(url)

prev_cx = None
smooth_cx = None
alpha =0.5

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Stream error... reconnecting")
        cap.release()
        cap = cv2.VideoCapture(url)
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # LOWER RED RANGE
    lower_red1 = (100, 50, 50)
    upper_red1 = (130, 255, 255)

    # UPPER RED RANGE
    #lower_red2 = (170, 80, 50)
    #upper_red2 = (180, 255, 255)

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 #+ mask2
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours,key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)

            cx = x + w // 2
            cy = y + h // 2

            frame_width = frame.shape[1]
            if cx < frame_width // 3:
                position = "LEFT"
            elif cx < 2 * frame_width // 3:
                position = "CENTER"
            else:
                position = "RIGHT"

            if smooth_cx is None:
                smooth_cx = cx
            else:
                smooth_cx = int(alpha * cx + (1 - alpha) * smooth_cx)
            #print(cx,smooth_cx)

            direction = "----"

            if prev_cx is not None:
                if smooth_cx - prev_cx > 10:
                    direction = "--->"
                elif prev_cx - smooth_cx > 10:
                    direction = "<---"

            # update previous position
            prev_cx = smooth_cx

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (smooth_cx, cy), 5, (0,0,255), -1)

            cv2.putText(frame, f"X:{cx} Y:{cy}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


            #cv2.putText(frame, position, (10,100),
            #cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, .6, (0,255,0), 1)
            cv2.putText(frame, f"{direction}", (10, 80),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Object Tracking", frame)
    cv2.imshow("Mask",mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()