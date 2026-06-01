"""
ASL Live Decoder
────────────────
Opens your webcam, detects hand landmarks in real time using
the same MediaPipe setup as your extractor, feeds 30-frame
windows into asl_model.keras, and displays the predicted sign.

SETUP:
    pip install tensorflow mediapipe opencv-python numpy

REQUIRES (in same folder):
    asl_model.keras       ← produced by train_model.py
    label_map.json        ← produced by train_model.py
    hand_landmarker.task  ← same file used by extractor

CONTROLS:
    Q  → quit
    R  → reset frame buffer (clear current sequence)
"""

import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ─────────────────────────────────────────────
#  CONFIG — must match extractor + train_model
# ─────────────────────────────────────────────
MODEL_PATH    = "asl_model.keras"
LABEL_MAP     = "label_map.json"
TASK_PATH     = "hand_landmarker.task"

SEQUENCE_LEN  = 30      # must match extractor (30 frames)
FEATURES      = 126     # must match extractor (21 × 3 × 2 hands)
THRESHOLD     = 0.85    # minimum confidence to display prediction

# Motion trigger thresholds
# Watch the motion: value in the status bar while signing to tune these.
# If capture triggers too easily (e.g. just moving into frame), raise MOTION_START.
# If it misses slow signs, lower MOTION_START.
MOTION_START  = 0.005   # mean landmark movement to START capturing
MOTION_STOP   = 0.002   # mean landmark movement to consider sign FINISHED


# ─────────────────────────────────────────────
#  STEP 1 — LOAD MODEL + LABELS
# ─────────────────────────────────────────────
def load_model_and_labels():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model input shape : {model.input_shape}")

    with open(LABEL_MAP, 'r') as f:
        label_map = json.load(f)

    # Convert {"0": "hello", "1": "yes"} → ["hello", "yes", ...]
    labels = [label_map[str(i)] for i in range(len(label_map))]
    print(f"  Classes ({len(labels)}): {labels}\n")

    return model, labels


# ─────────────────────────────────────────────
#  STEP 2 — MEDIAPIPE SETUP
#
#  Must match extractor EXACTLY:
#  - Same API (Tasks API, not old solutions API)
#  - Same mode (IMAGE)
#  - Same num_hands (2)
#  - Same model file (hand_landmarker.task)
# ─────────────────────────────────────────────
def build_detector():
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_PATH),
        running_mode=VisionRunningMode.VIDEO,  # tracks across frames — more stable
        num_hands=2,
        min_hand_detection_confidence=0.3,     # lower = detects more
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
#  STEP 3 — LANDMARK EXTRACTION
#
#  Must match extractor EXACTLY:
#  - Left hand  → indices 0:63
#  - Right hand → indices 63:126
#  - Missing hand → zeros(63)
#  - Raw x,y,z — no preprocessing here
#    (normalization happens in predict())
# ─────────────────────────────────────────────
def hand_to_array(hand) -> np.ndarray:
    """Flatten 21 landmarks → (63,) array of x,y,z values."""
    flat = []
    for lm in hand:
        flat.extend([lm.x, lm.y, lm.z])
    return np.array(flat, dtype=np.float32)


def extract_frame_landmarks(result) -> np.ndarray:
    """
    Extract both hands from a MediaPipe result.
    Left hand first (0:63), right hand second (63:126).
    Fills zeros if a hand is not detected — identical to extractor.
    """
    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if result.hand_landmarks and result.handedness:
        for hand, label in zip(result.hand_landmarks, result.handedness):
            side = label[0].category_name
            arr  = hand_to_array(hand)
            if side == "Left":
                left = arr
            else:
                right = arr

    return np.concatenate([left, right])   # (126,)


# ─────────────────────────────────────────────
#  STEP 3b — DRAW HAND SKELETON
#
#  MediaPipe defines 21 landmarks per hand numbered 0-20:
#
#       8   12  16  20        ← fingertips
#       |    |   |   |
#       7   11  15  19
#       |    |   |   |
#       6   10  14  18
#       |    |   |   |
#  4    5    9  13  17
#  |    |
#  3    |
#  |    |
#  2    0 ← wrist
#  |
#  1
#
#  CONNECTIONS defines which landmarks to draw lines between.
#  Each tuple is (start_landmark_index, end_landmark_index).
# ─────────────────────────────────────────────

# MediaPipe hand connections — 21 landmarks, 0=wrist, 4/8/12/16/20=fingertips
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17), (0, 5),
]

def draw_hand_skeleton(frame, result):
    """
    Draws landmark dots and connecting lines on the frame
    for every detected hand.

    MediaPipe gives normalized coordinates (0.0 to 1.0).
    We multiply by frame width/height to get pixel positions.

    Left hand  → cyan  (255, 255, 0  in BGR)
    Right hand → magenta (255, 0, 255 in BGR)
    """
    h, w = frame.shape[:2]

    if not result.hand_landmarks or not result.handedness:
        return frame

    for hand, label in zip(result.hand_landmarks, result.handedness):
        side = label[0].category_name   # "Left" or "Right"

        # Color per hand so you can tell them apart
        bone_color = (255, 255, 0)   if side == "Left" else (255, 0, 255)
        dot_color  = (0,   255, 255) if side == "Left" else (0,   200, 255)

        # Convert normalized (x,y) → pixel coordinates
        # z is depth — not used for drawing, just for the model
        points = []
        for lm in hand:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        # Draw bones (lines between connected landmarks)
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx],
                     bone_color, 2, cv2.LINE_AA)

        # Draw landmark dots on top of bones
        for i, (px, py) in enumerate(points):
            # Fingertips (4,8,12,16,20) get a bigger dot
            radius = 6 if i in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, (px, py), radius, dot_color, -1)
            cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)  # black outline

        # Label the hand
        wrist = points[0]
        cv2.putText(frame, side,
                    (wrist[0] - 20, wrist[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bone_color, 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
#  STEP 4 — NORMALIZE
#
#  Must match train_model.py normalize() exactly.
#  Applied per-sequence before prediction.
# ─────────────────────────────────────────────
def normalize(sequence: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a single sequence.
    sequence shape: (30, 126)

    axis=(0,1) with keepdims matches the batched training version
    which used axis=(1,2) — same math, batch dimension just absent here.
    """
    mean = sequence.mean(axis=(0, 1), keepdims=True)   # shape (1, 1)
    std  = sequence.std(axis=(0, 1),  keepdims=True) + 1e-8
    return (sequence - mean) / std


# ─────────────────────────────────────────────
#  STEP 5 — PREDICT
# ─────────────────────────────────────────────
def predict(model, sequence: np.ndarray, labels: list) -> tuple:
    """
    Takes a (30, 126) sequence, normalizes it, runs inference.
    Returns:
        predicted_label  : str or None (None if below threshold)
        confidence       : float (top class probability)
        all_probs        : list of (label, probability) for all classes
                           sorted highest → lowest, used for confidence bars
    """
    seq_norm = normalize(sequence)
    inp      = np.expand_dims(seq_norm, axis=0)   # (1, 30, 126)

    probs     = model.predict(inp, verbose=0)[0]   # (n_classes,)
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]

    # All classes sorted by confidence — for the sidebar display
    all_probs = sorted(
        zip(labels, probs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    if confidence < THRESHOLD:
        return None, float(confidence), all_probs

    return labels[class_idx], float(confidence), all_probs


# ─────────────────────────────────────────────
#  STEP 6 — DRAW OVERLAY
#
#  Layout:
#  ┌─────────────────────────────┬──────────────┐
#  │  TOP BAR — prediction +     │              │
#  │  confidence                 │   SIDEBAR    │
#  │                             │  confidence  │
#  │   MAIN VIDEO FEED           │  bars for    │
#  │   with skeleton drawn       │  all classes │
#  │                             │              │
#  │  BOTTOM — progress bar      │              │
#  └─────────────────────────────┴──────────────┘
# ─────────────────────────────────────────────
SIDEBAR_W = 260   # pixels wide for the confidence sidebar

def draw_overlay(frame, prediction, confidence, buffer_size,
                 hands_detected, all_probs):
    h, w = frame.shape[:2]

    # ── top bar ──
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    if prediction:
        label_text = f"{prediction.upper()}  {confidence*100:.0f}%"
        color      = (0, 255, 120)
    else:
        label_text = f"Signing...  ({confidence*100:.0f}%)"
        color      = (100, 100, 255)

    cv2.putText(frame, label_text,
                (20, 48), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, color, 2, cv2.LINE_AA)

    # ── sidebar background ──
    sidebar_x = w - SIDEBAR_W
    cv2.rectangle(frame, (sidebar_x, 70), (w, h - 40), (15, 15, 15), -1)
    cv2.line(frame, (sidebar_x, 70), (sidebar_x, h - 40), (60, 60, 60), 1)

    # ── confidence bars (one per class) ──
    #
    # all_probs is a list of (label, probability) sorted highest first.
    # For each class we draw:
    #   - label name
    #   - a filled bar whose width = probability × max_bar_width
    #   - the % number at the end
    #
    if all_probs:
        bar_area_x  = sidebar_x + 10
        bar_max_w   = SIDEBAR_W - 70       # max width a 100% bar can be
        row_h       = (h - 110 - 40) // len(all_probs)   # height per class row
        row_h       = min(row_h, 42)       # cap at 42px so it doesn't get huge

        cv2.putText(frame, "CONFIDENCE",
                    (bar_area_x, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        for i, (lbl, prob) in enumerate(all_probs):
            y_base = 115 + i * row_h

            # Highlight the top prediction differently
            is_top    = (i == 0 and prediction is not None)
            bar_color = (0, 220, 100) if is_top else (80, 130, 200)
            txt_color = (0, 255, 120) if is_top else (200, 200, 200)

            # Label
            cv2.putText(frame, lbl.replace("_", " "),
                        (bar_area_x, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_color, 1, cv2.LINE_AA)

            # Bar background
            bar_y = y_base + 4
            cv2.rectangle(frame,
                          (bar_area_x, bar_y),
                          (bar_area_x + bar_max_w, bar_y + 10),
                          (50, 50, 50), -1)

            # Bar fill — width proportional to probability
            fill = int(bar_max_w * prob)
            if fill > 0:
                cv2.rectangle(frame,
                              (bar_area_x, bar_y),
                              (bar_area_x + fill, bar_y + 10),
                              bar_color, -1)

            # Percentage text
            cv2.putText(frame, f"{prob*100:.0f}%",
                        (bar_area_x + bar_max_w + 4, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1, cv2.LINE_AA)

    # ── buffer progress bar (bottom) ──
    bar_x  = 20
    bar_y  = h - 25
    bar_w  = (w - SIDEBAR_W) - 40
    bar_h  = 10
    fill_w = int(bar_w * buffer_size / SEQUENCE_LEN)

    cv2.putText(frame, f"Buffer {buffer_size}/{SEQUENCE_LEN}",
                (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  (0, 200, 255), -1)

    # ── hand detection dot ──
    dot_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.circle(frame, (sidebar_x - 30, 30), 8, dot_color, -1)
    cv2.putText(frame,
                "hands" if hands_detected else "no hands",
                (sidebar_x - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 1, cv2.LINE_AA)

    # ── controls hint ──
    cv2.putText(frame, "Q: quit   R: reset",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100, 100, 100), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("=== ASL Live Decoder ===\n")

    model, labels = load_model_and_labels()
    detector      = build_detector()

    # ── State machine ──
    # IDLE       → hands present but still, waiting for motion
    # CAPTURING  → motion detected, filling buffer
    # PREDICTING → buffer full, run model, show result, reset
    IDLE       = "IDLE"
    CAPTURING  = "CAPTURING"
    PREDICTING = "PREDICTING"
    state      = IDLE

    buffer        = []          # plain list — we control appending manually
    prev_landmarks = None       # previous frame landmarks for motion calculation
    still_count    = 0          # frames of stillness seen while CAPTURING
    STILL_LIMIT    = 8          # frames of stillness before we stop capturing
                                # and predict with what we have

    # Last stable prediction
    last_prediction = None
    last_confidence = 0.0
    last_all_probs  = []
    last_motion     = 0.0       # shown on screen for tuning thresholds

    import time
    fps_timer   = time.time()
    fps_counter = 0
    fps_display = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    webcam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam FPS     : {webcam_fps}")
    print(f"Capture window : {SEQUENCE_LEN / max(webcam_fps,1):.1f}s  ({SEQUENCE_LEN} frames)")
    print(f"Motion trigger : start={MOTION_START:.4f}  stop={MOTION_STOP:.4f}\n")
    print("Webcam open. Bring hands into frame and start signing.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # FPS counter
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer   = time.time()

        frame = cv2.flip(frame, 1)

        # ── extract landmarks ──
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        landmarks     = extract_frame_landmarks(result)   # (126,)
        hands_present = result.hand_landmarks is not None and len(result.hand_landmarks) > 0

        # ── draw skeleton ──
        frame = draw_hand_skeleton(frame, result)

        # ── motion score ──
        # How much did landmarks move since last frame?
        # Only meaningful when hands are present in both frames.
        motion = 0.0
        if prev_landmarks is not None and hands_present:
            if not np.all(prev_landmarks == 0):
                motion = float(np.abs(landmarks - prev_landmarks).mean())
        last_motion    = motion
        prev_landmarks = landmarks.copy()

        # ──────────────────────────────────────────
        #  STATE MACHINE
        # ──────────────────────────────────────────

        if state == IDLE:
            # Waiting for motion to start
            if hands_present and motion > MOTION_START:
                state       = CAPTURING
                buffer      = [landmarks]
                still_count = 0
                print("Motion detected → CAPTURING")

        elif state == CAPTURING:
            buffer.append(landmarks)

            if motion < MOTION_STOP:
                still_count += 1
            else:
                still_count = 0   # reset stillness counter if motion resumes

            # Predict if:
            # (a) buffer is full (30 frames collected), OR
            # (b) hands stopped moving for STILL_LIMIT frames
            #     (sign finished before 30 frames)
            enough_frames  = len(buffer) >= SEQUENCE_LEN
            sign_finished  = still_count >= STILL_LIMIT and len(buffer) >= 10

            if enough_frames or sign_finished:
                state = PREDICTING

        elif state == PREDICTING:
            # Pad to SEQUENCE_LEN if sign finished early
            while len(buffer) < SEQUENCE_LEN:
                buffer.append(np.zeros(FEATURES, dtype=np.float32))

            sequence        = np.array(buffer[:SEQUENCE_LEN], dtype=np.float32)
            non_zero_frames = np.sum(~np.all(sequence == 0, axis=1))

            if non_zero_frames >= SEQUENCE_LEN * 0.5:
                pred, conf, all_probs = predict(model, sequence, labels)
                last_all_probs = all_probs
                if pred:
                    last_prediction = pred
                    last_confidence = conf
                    print(f"Predicted: {pred}  ({conf*100:.0f}%)")

            # Reset for next sign
            buffer      = []
            still_count = 0
            state       = IDLE

        # ── status line ──
        h_f, w_f = frame.shape[:2]
        state_colors = {IDLE: (100,100,100), CAPTURING: (0,200,255), PREDICTING: (0,255,120)}
        cv2.putText(frame,
                    f"FPS:{fps_display}  motion:{last_motion:.4f}  state:{state}  buf:{len(buffer)}/{SEQUENCE_LEN}",
                    (20, h_f - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_colors[state], 1, cv2.LINE_AA)

        # ── draw overlay ──
        frame = draw_overlay(
            frame,
            last_prediction,
            last_confidence,
            len(buffer),
            hands_present,
            last_all_probs
        )

        cv2.imshow("ASL Live Decoder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            buffer      = []
            still_count = 0
            state       = IDLE
            last_prediction = None
            last_confidence = 0.0
            print("Reset → IDLE")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()