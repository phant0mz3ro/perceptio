import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time
import math

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "hand_landmarker.task"
HOLD_TIME   = 0.5       # seconds a gesture must be stable before confirming
HISTORY_LEN = 7         # majority-vote window
NUM_HANDS   = 2         # detect up to 2 hands
CAMERA_SRC  = 0         # 0 = webcam, or swap in your stream URL

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode    = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=NUM_HANDS
)
detector = HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  GESTURE RULES
#  All distance thresholds are normalised by
#  wrist→middle-MCP (landmark 0→9) so they are
#  camera-distance invariant.
# ─────────────────────────────────────────────
GESTURE_RULES = {
    "FIST": {
        "fingers":          [0, 0, 0, 0],
        "thumb_index_dist": {"max": 0.30},
    },
    "OPEN_PALM": {
        "fingers":          [1, 1, 1, 1],
        "thumb_index_dist": {"min": 0.22},
        "index_middle_dist":{"min": 0.20},
        "middle_ring_dist": {"min": 0.18},
        "ring_pinky_dist":  {"min": 0.18},
    },
    "PEACE": {
        "fingers":           [1, 1, 0, 0],
        "index_middle_dist": {"min": 0.18},
        "middle_ring_dist":  {"max": 0.18},
    },
    "THUMBS_UP": {
        "fingers":          [0, 0, 0, 0],
        "thumb":            [0, 1],          # thumb pointing up
        "thumb_index_dist": {"min": 0.25},
    },
    "OK": {
        "fingers":          [0, 1, 1, 1],
        "thumb_index_dist": {"max": 0.20},
    },
    # ── New gestures ──────────────────────────
    "POINTING": {
        "fingers":          [1, 0, 0, 0],   # only index up
        "thumb_index_dist": {"min": 0.20},
    },
    "THREE": {
        "fingers":          [1, 1, 1, 0],
    },
    "FOUR": {
        "fingers":          [1, 1, 1, 1],
        "thumb_index_dist": {"min": 0.20},  # thumb tucked in
    },
    "ROCK": {                                # index + pinky up (🤘)
        "fingers":          [1, 0, 0, 1],
    },
    "THUMBS_DOWN": {
        "fingers":          [0, 0, 0, 0],
        "thumb":            [0, 0],          # thumb pointing down
        "thumb_index_dist": {"min": 0.25},
    },
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def _norm(hand, p1, p2) -> float:
    """Normalise distance by wrist-to-middle-MCP span."""
    ref = _dist(hand[0], hand[9])
    return _dist(p1, p2) / ref if ref > 0 else 0.0

def extract_features(hand: list, is_left: bool) -> dict:
    """Return a feature dict from 21 MediaPipe landmarks."""
    # ── Fingers (tip.y < pip.y  →  finger is extended) ──
    fingers = [
        1 if hand[8].y  < hand[6].y  else 0,   # index
        1 if hand[12].y < hand[10].y else 0,   # middle
        1 if hand[16].y < hand[14].y else 0,   # ring
        1 if hand[20].y < hand[18].y else 0,   # pinky
    ]

    # ── Thumb direction vector from wrist ──
    tx = hand[4].x - hand[0].x
    ty = hand[4].y - hand[0].y
    thumb = [
        1 if tx > 0 else 0,   # [0]=right
        1 if ty < 0 else 0,   # [1]=up
    ]

    return {
        "fingers":           fingers,
        "thumb":             thumb,
        "thumb_index_dist":  _norm(hand, hand[4], hand[8]),
        "index_middle_dist": _norm(hand, hand[8], hand[12]),
        "middle_ring_dist":  _norm(hand, hand[12], hand[16]),   # fixed: now normalised
        "ring_pinky_dist":   _norm(hand, hand[16], hand[20]),   # fixed: now normalised
    }

def _check_dist(rules: dict, key: str, value: float) -> bool:
    """Return False if the distance value violates the rule."""
    if key not in rules:
        return True
    cond = rules[key]
    if "min" in cond and value < cond["min"]:
        return False
    if "max" in cond and value > cond["max"]:
        return False
    return True

def match_gesture(features: dict) -> str | None:
    """Rule-based gesture matching. Returns gesture name or None."""
    for name, rules in GESTURE_RULES.items():
        if "fingers" in rules and features["fingers"] != rules["fingers"]:
            continue
        if "thumb" in rules and features["thumb"] != rules["thumb"]:
            continue
        dist_keys = [
            "thumb_index_dist", "index_middle_dist",
            "middle_ring_dist", "ring_pinky_dist",
        ]
        if not all(_check_dist(rules, k, features[k]) for k in dist_keys):
            continue
        return name
    return None

# ─────────────────────────────────────────────
#  PER-HAND STATE  (supports NUM_HANDS hands)
# ─────────────────────────────────────────────
class HandState:
    def __init__(self):
        self.history   = deque(maxlen=HISTORY_LEN)
        self.candidate = None
        self.hold_start = 0.0
        self.confirmed = ""

    def update(self, gesture: str | None) -> str:
        if gesture is None:
            return self.confirmed

        self.history.append(gesture)
        dominant = max(set(self.history), key=self.history.count)

        if dominant != self.candidate:
            self.candidate  = dominant
            self.hold_start = time.time()
        elif time.time() - self.hold_start >= HOLD_TIME:
            self.confirmed = dominant

        return self.confirmed

    def reset(self):
        self.history.clear()
        self.candidate  = None
        self.hold_start = 0.0
        # intentionally keep self.confirmed so last sign stays on screen

# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────
CONNECTIONS = [
    # palm
    (0,1),(1,2),(2,3),(3,4),
    # index
    (0,5),(5,6),(6,7),(7,8),
    # middle
    (0,9),(9,10),(10,11),(11,12),
    # ring
    (0,13),(13,14),(14,15),(15,16),
    # pinky
    (0,17),(17,18),(18,19),(19,20),
    # knuckle bar
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, hand, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1)

def draw_hud(frame, states: list[HandState]):
    """Draw confirmed gestures and a simple progress bar."""
    h, w = frame.shape[:2]

    # Background bar at top
    cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)

    labels = [s.confirmed for s in states if s.confirmed]
    text   = " + ".join(labels) if labels else "—"
    cv2.putText(frame, text, (15, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 220, 255), 2)

    # Hold-time progress ring per hand
    for i, st in enumerate(states):
        if st.candidate:
            elapsed  = time.time() - st.hold_start
            progress = min(elapsed / HOLD_TIME, 1.0)
            cx, cy, r = w - 60 - i * 70, 28, 22
            cv2.ellipse(frame, (cx, cy), (r, r), -90,
                        0, int(360 * progress), (0, 200, 100), 3)
            cv2.putText(frame, st.candidate[:3], (cx - 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
cap    = cv2.VideoCapture(CAMERA_SRC)
states = [HandState() for _ in range(NUM_HANDS)]

print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Reset all states then re-populate from detections this frame
    for st in states:
        st.reset()

    if result.hand_landmarks and result.handedness:
        hand_colors = [(0, 255, 100), (100, 180, 255)]   # green / blue per hand

        for idx, (hand, hand_lbl) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            if idx >= NUM_HANDS:
                break

            is_left  = (hand_lbl[0].category_name == "Left")
            features = extract_features(hand, is_left)
            gesture  = match_gesture(features)

            states[idx].update(gesture)
            draw_hand(frame, hand, hand_colors[idx % len(hand_colors)])

    draw_hud(frame, states)

    cv2.imshow("Sign Language Decoder", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()