import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

# ─────────────────────────────────────────────
#  CONFIG  — edit these to suit your setup
# ─────────────────────────────────────────────
MODEL_PATH      = "hand_landmarker.task"
DATA_DIR        = "asl_dataset"          # where .npy files are saved
CAMERA_SRC      = 0
SEQUENCE_LEN    = 30                     # frames captured per sample
SAMPLES_PER_SIGN = 40                    # how many samples to collect per sign
CAPTURE_FPS     = 15                     # target capture rate (ms delay = 1000/FPS)

# Signs to collect — edit/extend this list freely
# Start small (5–8 signs), verify the model works, then expand
SIGNS = [
    "hello",
    "my",
    "name",
    "yes",
    "no",
    "please",
    "thank_you",
    "help",
    "finish",
    "want",
]

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
    num_hands=2,
)
detector = HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  DATASET FOLDER STRUCTURE
#
#  asl_dataset/
#    hello/
#      0.npy   ← shape (30, 63)  — 30 frames, 21 landmarks × xyz
#      1.npy
#      ...
#    my/
#      0.npy
#      ...
# ─────────────────────────────────────────────
def setup_dirs():
    for sign in SIGNS:
        path = os.path.join(DATA_DIR, sign)
        os.makedirs(path, exist_ok=True)
    print(f"Dataset folder ready: {os.path.abspath(DATA_DIR)}")

def existing_sample_count(sign: str) -> int:
    path = os.path.join(DATA_DIR, sign)
    return len([f for f in os.listdir(path) if f.endswith(".npy")])

# ─────────────────────────────────────────────
#  LANDMARK EXTRACTION
#  Returns a flat (63,) vector for one hand,
#  or zeros if no hand is detected this frame.
#  Two-hand signs: concatenates both → (126,)
#  For now we use one hand only to keep it simple.
# ─────────────────────────────────────────────
def extract_landmarks(result) -> np.ndarray:
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]          # dominant hand
        flat = []
        for lm in hand:
            flat.extend([lm.x, lm.y, lm.z])     # 21 × 3 = 63 values
        return np.array(flat, dtype=np.float32)
    # No hand detected — return zeros so the sequence stays the same length
    return np.zeros(63, dtype=np.float32)

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hand in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 1)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 150), -1)

def draw_overlay(frame, sign: str, sample_idx: int, state: str,
                 frame_idx: int = 0, sequence_len: int = SEQUENCE_LEN):
    h, w = frame.shape[:2]

    # Dark bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)

    # Sign label
    cv2.putText(frame, f"Sign: {sign.upper()}", (12, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 1)

    # Sample counter
    cv2.putText(frame, f"Sample {sample_idx + 1}/{SAMPLES_PER_SIGN}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # State message (right side)
    colours = {
        "READY":      (255, 200,   0),
        "COUNTDOWN":  (255, 140,   0),
        "CAPTURING":  (  0, 220, 100),
        "SAVED":      (100, 255, 100),
        "DONE":       (100, 255, 100),
    }
    col = colours.get(state, (200, 200, 200))
    cv2.putText(frame, state, (w - 180, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)

    # Progress bar (only while capturing)
    if state == "CAPTURING":
        bar_w = int((frame_idx / sequence_len) * (w - 24))
        cv2.rectangle(frame, (12, h - 20), (w - 12, h - 8), (50, 50, 50), -1)
        cv2.rectangle(frame, (12, h - 20), (12 + bar_w, h - 8), (0, 200, 100), -1)

    # Instructions at bottom
    instructions = {
        "READY":     "Press SPACE to start  |  S to skip  |  Q to quit",
        "COUNTDOWN": "Get into position...",
        "CAPTURING": "Hold the sign steady",
        "SAVED":     "Saved!  Next sample incoming...",
        "DONE":      "All samples collected for this sign!",
    }
    msg = instructions.get(state, "")
    cv2.putText(frame, msg, (12, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

# ─────────────────────────────────────────────
#  COLLECTION LOOP
# ─────────────────────────────────────────────
def collect_sign(cap, sign: str):
    """
    Collect SAMPLES_PER_SIGN sequences for one sign.
    Each sequence is shape (SEQUENCE_LEN, 63).
    Saved as: DATA_DIR/sign/N.npy
    """
    start_idx = existing_sample_count(sign)
    if start_idx >= SAMPLES_PER_SIGN:
        print(f"  '{sign}' already has {start_idx} samples — skipping.")
        return

    print(f"\n── Collecting '{sign}' "
          f"(need {SAMPLES_PER_SIGN - start_idx} more) ──")

    sample_idx = start_idx

    while sample_idx < SAMPLES_PER_SIGN:
        sequence   = []
        state      = "READY"
        countdown  = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts     = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = detector.detect_for_video(mp_img, ts)

            draw_landmarks(frame, result)
            draw_overlay(frame, sign, sample_idx, state,
                         len(sequence), SEQUENCE_LEN)
            cv2.imshow("ASL Data Collector", frame)

            key = cv2.waitKey(1000 // CAPTURE_FPS) & 0xFF

            # ── State machine ──────────────────
            if state == "READY":
                if key == ord(' '):
                    state     = "COUNTDOWN"
                    countdown = time.time()
                elif key == ord('s'):
                    print(f"  Skipping sample {sample_idx}")
                    break                        # skip to next sample slot
                elif key == ord('q'):
                    return                       # exit entire sign collection

            elif state == "COUNTDOWN":
                elapsed = time.time() - countdown
                remaining = max(0, 2.0 - elapsed)   # 2-second countdown
                cv2.putText(frame, f"{remaining:.1f}", (frame.shape[1]//2 - 20, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 220, 255), 3)
                cv2.imshow("ASL Data Collector", frame)
                if elapsed >= 2.0:
                    state    = "CAPTURING"
                    sequence = []

            elif state == "CAPTURING":
                landmarks = extract_landmarks(result)
                sequence.append(landmarks)

                if len(sequence) == SEQUENCE_LEN:
                    # Save sequence
                    arr  = np.array(sequence)            # (30, 63)
                    path = os.path.join(DATA_DIR, sign, f"{sample_idx}.npy")
                    np.save(path, arr)
                    print(f"  Saved sample {sample_idx} → {path}  shape={arr.shape}")
                    sample_idx += 1
                    state = "SAVED"
                    saved_time = time.time()

            elif state == "SAVED":
                if time.time() - saved_time > 0.8:      # brief pause then loop
                    break                                # back to READY for next sample

            if key == ord('q'):
                return

    print(f"  '{sign}' complete — {SAMPLES_PER_SIGN} samples saved.")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    setup_dirs()
    cap = cv2.VideoCapture(CAMERA_SRC)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("\n=== ASL Data Collector ===")
    print(f"Signs to collect : {SIGNS}")
    print(f"Samples per sign : {SAMPLES_PER_SIGN}")
    print(f"Sequence length  : {SEQUENCE_LEN} frames")
    print(f"Saving to        : {os.path.abspath(DATA_DIR)}")
    print("\nFor each sign:")
    print("  SPACE → start 2-second countdown then capture")
    print("  S     → skip this sample slot")
    print("  Q     → quit\n")

    for sign in SIGNS:
        collect_sign(cap, sign)
        if not cap.isOpened():
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────
    print("\n=== Dataset Summary ===")
    total = 0
    for sign in SIGNS:
        count = existing_sample_count(sign)
        status = "COMPLETE" if count >= SAMPLES_PER_SIGN else f"INCOMPLETE ({count}/{SAMPLES_PER_SIGN})"
        print(f"  {sign:<15} {status}")
        total += count
    print(f"\nTotal samples: {total}")
    print(f"Each sample shape: ({SEQUENCE_LEN}, 63)  →  numpy float32")
    print("\nNext step: run train_model.py")

if __name__ == "__main__":
    main()