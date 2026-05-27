import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH       = "hand_landmarker.task"
DATA_DIR         = "asl_dataset"
CAMERA_SRC       = 0
SEQUENCE_LEN     = 30
SAMPLES_PER_SIGN = 40
CAPTURE_FPS      = 15

SIGNS = [
    "hello",
    "my",
    "name",
    "yes",
    "no",
    "please",
    "thank_you",
    "help",       # two-handed
    "finish",     # two-handed
    "want",
]

# Signs that require two hands — collector will warn if
# it only sees one hand during capture
TWO_HANDED = {"help", "finish","name"}

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,                          # always detect up to 2
)
detector = HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  DATASET STRUCTURE
#
#  asl_dataset/
#    hello/   0.npy shape=(30, 63)   ← one-handed
#    help/    0.npy shape=(30, 126)  ← two-handed
# ─────────────────────────────────────────────
def setup_dirs():
    for sign in SIGNS:
        os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)
    print(f"Dataset folder ready: {os.path.abspath(DATA_DIR)}")

def existing_sample_count(sign: str) -> int:
    path = os.path.join(DATA_DIR, sign)
    return len([f for f in os.listdir(path) if f.endswith(".npy")])

# ─────────────────────────────────────────────
#  LANDMARK EXTRACTION
#
#  Returns (126,) — left hand (63) + right hand (63)
#  If a hand is absent its 63 values are zeros.
#  Consistent left/right ordering is critical so the
#  model always knows which hand is which.
# ─────────────────────────────────────────────
def hand_to_array(hand) -> np.ndarray:
    flat = []
    for lm in hand:
        flat.extend([lm.x, lm.y, lm.z])
    return np.array(flat, dtype=np.float32)   # (63,)

def extract_landmarks(result) -> tuple[np.ndarray, int]:
    """
    Returns:
        vector  : (126,) float32 — [left_hand(63) | right_hand(63)]
        n_hands : number of hands actually detected this frame
    """
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    n_hands = 0

    if result.hand_landmarks and result.handedness:
        n_hands = len(result.hand_landmarks)
        for hand, label in zip(result.hand_landmarks, result.handedness):
            side = label[0].category_name   # "Left" or "Right"
            arr  = hand_to_array(hand)
            if side == "Left":
                left = arr
            else:
                right = arr

    return np.concatenate([left, right]), n_hands   # (126,)

# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
HAND_COLOURS = {
    "Left":  (0,  255, 100),   # green
    "Right": (100, 180, 255),  # blue
}

def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand, label in zip(result.hand_landmarks, result.handedness):
        side  = label[0].category_name
        color = HAND_COLOURS.get(side, (200, 200, 200))
        pts   = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 1)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1)
        # label which hand
        cv2.putText(frame, side, pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_overlay(frame, sign, sample_idx, state,
                 frame_idx=0, n_hands=0, two_handed=False):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 65), (15, 15, 15), -1)

    # Sign + sample counter
    cv2.putText(frame, f"Sign: {sign.upper()}", (12, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 1)
    hand_tag = "TWO-HANDED" if two_handed else "one-handed"
    cv2.putText(frame, f"Sample {sample_idx+1}/{SAMPLES_PER_SIGN}  [{hand_tag}]",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # State
    colours = {
        "READY":     (255, 200,   0),
        "COUNTDOWN": (255, 140,   0),
        "CAPTURING": (  0, 220, 100),
        "SAVED":     (100, 255, 100),
    }
    cv2.putText(frame, state, (w - 190, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.9,
                colours.get(state, (200,200,200)), 2)

    # Two-handed warning during capture
    if state == "CAPTURING" and two_handed and n_hands < 2:
        cv2.putText(frame, "⚠ SHOW BOTH HANDS", (w//2 - 130, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 60, 255), 2)

    # Progress bar
    if state == "CAPTURING":
        bar_w = int((frame_idx / SEQUENCE_LEN) * (w - 24))
        cv2.rectangle(frame, (12, h-20), (w-12, h-8), (50,50,50), -1)
        cv2.rectangle(frame, (12, h-20), (12+bar_w, h-8), (0,200,100), -1)

    # Instructions
    msgs = {
        "READY":     "SPACE to start  |  S to skip  |  Q to quit",
        "COUNTDOWN": "Get into position...",
        "CAPTURING": "Perform the sign",
        "SAVED":     "Saved!",
    }
    cv2.putText(frame, msgs.get(state,""), (12, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

# ─────────────────────────────────────────────
#  COLLECTION LOOP
# ─────────────────────────────────────────────
def collect_sign(cap, sign: str):
    start_idx  = existing_sample_count(sign)
    two_handed = sign in TWO_HANDED

    if start_idx >= SAMPLES_PER_SIGN:
        print(f"  '{sign}' already complete — skipping.")
        return

    print(f"\n── '{sign}'  {'(TWO-HANDED)' if two_handed else '(one-handed)'}  "
          f"— need {SAMPLES_PER_SIGN - start_idx} more samples ──")

    sample_idx = start_idx

    while sample_idx < SAMPLES_PER_SIGN:
        sequence   = []
        state      = "READY"
        countdown  = 0.0
        saved_time = 0.0
        n_hands    = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts     = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = detector.detect_for_video(mp_img, ts)

            landmarks, n_hands = extract_landmarks(result)

            draw_landmarks(frame, result)
            draw_overlay(frame, sign, sample_idx, state,
                         len(sequence), n_hands, two_handed)
            cv2.imshow("ASL Data Collector", frame)

            key = cv2.waitKey(1000 // CAPTURE_FPS) & 0xFF

            if state == "READY":
                if key == ord(' '):
                    state     = "COUNTDOWN"
                    countdown = time.time()
                elif key == ord('s'):
                    break
                elif key == ord('q'):
                    return

            elif state == "COUNTDOWN":
                elapsed   = time.time() - countdown
                remaining = max(0, 2.0 - elapsed)
                cv2.putText(frame, f"{remaining:.1f}",
                            (frame.shape[1]//2 - 20, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,220,255), 3)
                cv2.imshow("ASL Data Collector", frame)
                if elapsed >= 2.0:
                    state    = "CAPTURING"
                    sequence = []

            elif state == "CAPTURING":
                sequence.append(landmarks)          # (126,) per frame

                if len(sequence) == SEQUENCE_LEN:
                    arr  = np.array(sequence)       # (30, 126)
                    path = os.path.join(DATA_DIR, sign, f"{sample_idx}.npy")
                    np.save(path, arr)
                    print(f"  Saved sample {sample_idx} → shape={arr.shape}")
                    sample_idx += 1
                    state      = "SAVED"
                    saved_time = time.time()

            elif state == "SAVED":
                if time.time() - saved_time > 0.8:
                    break

            if key == ord('q'):
                return

    print(f"  '{sign}' complete.")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    setup_dirs()
    cap = cv2.VideoCapture(CAMERA_SRC)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("\n=== ASL Data Collector (two-hand aware) ===")
    print(f"Frame vector size : 126  (left 63 + right 63)")
    print(f"Sequence length   : {SEQUENCE_LEN} frames")
    print(f"Samples per sign  : {SAMPLES_PER_SIGN}")
    print(f"Signs             : {SIGNS}\n")

    for sign in SIGNS:
        collect_sign(cap, sign)

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Dataset Summary ===")
    total = 0
    for sign in SIGNS:
        count  = existing_sample_count(sign)
        status = "✓" if count >= SAMPLES_PER_SIGN else f"{count}/{SAMPLES_PER_SIGN}"
        tag    = " [two-handed]" if sign in TWO_HANDED else ""
        print(f"  {sign:<15} {status}{tag}")
        total += count
    print(f"\nTotal samples : {total}")
    print("Next step     : run train_model.py")

if __name__ == "__main__":
    main()
