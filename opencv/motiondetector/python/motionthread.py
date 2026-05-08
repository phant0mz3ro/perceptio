import cv2
import serial
import time
import threading
import numpy as np

# Configuration
arduino = serial.Serial('COM8', 9600, timeout=0)
time.sleep(2)

stream_url = "http://10.189.118.104:81/stream"
cap = cv2.VideoCapture(stream_url)

# Shared variables
distance = 0
motion_detected = False
latest_frame = None
frame_lock = threading.Lock()
running = True


def buzzer_control_thread():
    """INSTANT buzzer response"""
    global motion_detected, running

    last_motion_state = False

    while running:
        current_motion = motion_detected

        if current_motion != last_motion_state:
            arduino.write(b'1' if current_motion else b'0')
            last_motion_state = current_motion

        time.sleep(0)  # fastest possible


def distance_read_thread():
    """Read latest distance WITHOUT lag"""
    global distance, running

    while running:
        while arduino.in_waiting:  # flush buffer completely
            try:
                line = arduino.readline().decode().strip()
                if line.isdigit():
                    distance = int(line)
            except:
                pass

        time.sleep(0.005)


def motion_detection_thread():
    """LOW-LATENCY motion detection"""
    global motion_detected, latest_frame, running, cap

    prev_frame = None

    while running:
        # 🔥 Drop old frames (IMPORTANT)
        for _ in range(2):
            cap.grab()

        ret, frame = cap.read()

        if not ret:
            print("Reconnecting...")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(stream_url)
            continue

        # Resize small for speed
        frame = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)

            # 🔥 FASTEST motion detection
            motion_pixels = np.sum(diff > 25)
            #print(motion_pixels)

            motion_detected = motion_pixels > 150

        prev_frame = gray

        # Store latest frame safely
        with frame_lock:
            latest_frame = frame

        time.sleep(0.015)  # slightly faster loop


def display_thread_func():
    """Display only (low priority)"""
    global running, distance, motion_detected

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        frame = cv2.resize(frame, (640, 480))
        
        cv2.putText(frame, f"Distance: {distance} cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        status = "MOTION" if motion_detected else "IDLE"
        color = (0, 0, 255) if motion_detected else (0, 255, 0)

        cv2.putText(frame, status,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        cv2.imshow("ESP32 Stream", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            running = False
            break

        time.sleep(0.03)  # limit display FPS


# Start threads
print("Starting ULTRA-LOW LATENCY system...")

threads = [
    threading.Thread(target=buzzer_control_thread, daemon=True),
    threading.Thread(target=distance_read_thread, daemon=True),
    threading.Thread(target=motion_detection_thread, daemon=True),
    threading.Thread(target=display_thread_func, daemon=True),
]

for t in threads:
    t.start()

print("System READY - Minimal lag mode!")
print("Press ESC to exit")

try:
    while running:
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False

# Cleanup
cap.release()
arduino.close()
cv2.destroyAllWindows()