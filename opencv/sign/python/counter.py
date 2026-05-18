import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time,math

# Path to model
MODEL_PATH = "hand_landmarker.task"

# Create options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Create detector
detector = HandLandmarker.create_from_options(options)
history  =  deque(maxlen=5)

#time variables
candidate = None
hold_start = 0
confirmed = ""
HOLD_TIME = 0.5

url = "http://10.42.236.104:81/stream"
#cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)

gesture_dict = {

    # 👊 FIST
    "FIST": {
        "fingers": [0, 0, 0, 0],
        "thumb_index_dist": {"max": 0.3}
    },

    # ✋ OPEN PALM
    "OPEN_PALM": {
        "fingers": [1, 1, 1, 1],
        "thumb_index_dist": {"min": 0.22},
        "index_middle_dist": {"min": 0.20},
        "middle_ring_dist": {"min": 0.20},
        "ring_pinky_dist": {"min": 0.20}
    },

    # ✌ PEACE (index + middle up, others down)
    "PEACE": {
        "fingers": [1, 1, 0, 0],
        "index_middle_dist": {"min": 0.18},
        "middle_ring_dist": {"max": 0.18}
    },

    # 👍 THUMBS UP
    "THUMBS_UP": {
        "fingers": [0, 0, 0, 0],
        "thumb": [0, 1],   # vertical up
        "thumb_index_dist": {"min": 0.25}
    },

    # 👌 OK SIGN (optional but useful early sign language staple)
    "OK": {
        "fingers": [0, 1, 1, 1],
        "thumb_index_dist": {"max": 0.2}
    }
}

def match_gesture(features):
    fingers = features["fingers"]
    thumb = features["thumb"]

    for name, rules in gesture_dict.items():

        # check fingers
        if "fingers" in rules:
            if fingers != rules["fingers"]:
                continue

        # check thumb
        if "thumb" in rules:
            if thumb != rules["thumb"]:
                continue

        # check distances
        if "thumb_index_dist" in rules:
            cond = rules["thumb_index_dist"]
            val = features["thumb_index_dist"]

            if "max" in cond and val > cond["max"]:
                continue
            if "min" in cond and val < cond["min"]:
                continue

        if "index_middle_dist" in rules:
            cond = rules["index_middle_dist"]
            val = features["index_middle_dist"]

            if "min" in cond and val < cond["min"]:
                continue

        return name

    return None

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2
    )

def normalize_distance(hand, p1, p2):
    return distance(p1, p2) / distance(hand[0], hand[9])

def extract_features(hand:list,lefti: bool):
            fingers=[0,0,0,0]
            thumb = [0,0]
            """if hand[4].y < hand[3].y : thumb[1] = 1
            if lefti:
                if  hand[4].x < hand[3].x : thumb[0] = 1
            else:
                if  hand[4].x > hand[3].x : thumb[0] = 1"""
            
            thumb_tip = hand[4]
            thumb_ip = hand[3]
            thumb_mcp = hand[2]
            wrist = hand[0]
            thumb_vector_x = thumb_tip.x - wrist.x
            thumb_vector_y = thumb_tip.y - wrist.y      
            thumb = [
                1 if thumb_vector_x > 0 else 0,
                1 if thumb_vector_y < 0 else 0
            ]

            # Index finger
            if hand[8].y < hand[6].y:
                fingers[0] = 1
            # Middle finger
            if hand[12].y < hand[10].y:
                fingers[1] = 1
            # Ring finger
            if hand[16].y < hand[14].y:
                fingers[2] = 1
            # Pinky
            if hand[20].y < hand[18].y:
                fingers[3] = 1

            
            thumb_index_dist = normalize_distance(hand,hand[4],hand[8])
            index_middle_dist = normalize_distance(hand,hand[8], hand[12])
            middle_ring_dist = distance(hand[12], hand[16])
            ring_pinky_dist = distance(hand[16], hand[20])
            wrist = hand[0]
            index_wrist_dist = distance(hand[8], wrist)
            middle_wrist_dist = distance(hand[12], wrist)

            print(fingers,thumb)
            return {
                "fingers": fingers,
                "thumb": thumb,
                "thumb_index_dist": thumb_index_dist,
                "index_middle_dist": index_middle_dist,
                "middle_ring_dist": middle_ring_dist,
                "ring_pinky_dist": ring_pinky_dist,

            }

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    

    if result.hand_landmarks and result.handedness:
        for hand,hand_lbl in zip(
            result.hand_landmarks,
            result.handedness
        ):  
            labelHand = (hand_lbl[0].category_name == "Left")
            features = extract_features(hand,labelHand)
            gesture = match_gesture(features)

            if gesture is None: 
                continue

            history.append(gesture)
            current_gesture = max(set(history),key=history.count)
            if current_gesture != candidate:
                candidate = current_gesture
                hold_start = time.time()

            elif time.time() - hold_start>HOLD_TIME:
                confirmed = current_gesture

               # numValue = numbers[numIndex]

            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    else:
        history.clear()
        candidate = None
        #confirmed = ""
   
       
    cv2.putText(frame,confirmed,(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()