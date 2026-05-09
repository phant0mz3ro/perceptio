import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


url = "http://10.251.195.104:81/stream"
cap = cv2.VideoCapture(url)
#cap = cv2.VideoCapture(0)

numbers = ["zero","one","two","three","four","five"]
num_logic = [[0,0,0,0,0],[0,1,0,0,0],[0,1,1,0,0],[0,1,1,1,0],[0,1,1,1,1],[1,1,1,1,1]]

def getNumber(positions:list):
    if positions in num_logic:
        return num_logic.index(positions)
    else:
        return -1

def generate_positions(hand:list,lefti: bool):
    # Thumb (special case - compares x axis)
            fingers=[0,0,0,0,0]
            if lefti:
                fingers[0] = 1 if hand[4].x < hand[3].x else 0
            else:
                fingers[0] = 1 if hand[4].x > hand[3].x else 0

            # Index finger
            if hand[8].y < hand[6].y:
                fingers[1] = 1
            else:
                fingers[1] = 0

            # Middle finger
            if hand[12].y < hand[10].y:
                fingers[2] = 1
            else:
                fingers[2] = 0

            # Ring finger
            if hand[16].y < hand[14].y:
                fingers[3] = 1
            else:
                fingers[3] = 0

            # Pinky
            if hand[20].y < hand[18].y:
                fingers[4] = 1
            else:
                fingers[4] = 0

            print(fingers)
            return fingers

while True:
    fingers_global = None
    numValue = ""

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    

    if result.hand_landmarks and result.handedness:
        for hand,hand_lbl in zip(
            result.hand_landmarks,
            result.handedness
        ):  
            labelHand = (hand_lbl[0].category_name == "Left")
            fingers_global = generate_positions(hand,labelHand)
            numIndex = getNumber(fingers_global)
            if numIndex != -1:
                numValue = numbers[numIndex]
            else: numValue = ""

            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
   
       
    cv2.putText(frame,numValue,(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()