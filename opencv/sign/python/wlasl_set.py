"""
WLASL Landmark Extractor (JSON-aware version)
──────────────────────────────────────────────
Reads WLASL_v0.3.json to map numeric video IDs to sign glosses,
extracts MediaPipe landmarks from each clip, and saves (30, 126)
.npy files — identical format to data_collector.py.

SETUP:
    pip install kaggle mediapipe opencv-python numpy

    1. kaggle.com → Account → Create API Token → save kaggle.json
       to C:\\Users\\YourName\\.kaggle\\kaggle.json
    2. Run this script

WHAT GETS DOWNLOADED:
    wlasl_raw/
        WLASL_v0.3.json       ← metadata: gloss → video IDs
        videos/
            00001.mp4
            00002.mp4
            ...               ← all videos named by numeric ID
"""

import os
import json
import subprocess
import zipfile
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR        = "asl_dataset"
WLASL_DIR       = "wlasl_raw"
SEQUENCE_LEN    = 30
MAX_SAMPLES     = 50

SIGNS = [
    "hello", "my", "name", "yes", "no",
    "please", "thank you", "help", "finish", "want",
]

# Map your folder names → WLASL gloss strings
# (WLASL uses spaces, your folders use underscores)
FOLDER_NAME = {
    "thank you": "thank_you",
}

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,   # IMAGE mode — no timestamp needed
    num_hands=2,                            # since we jump frames non-sequentially
)
detector = HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────
def download_wlasl():
    os.makedirs(WLASL_DIR, exist_ok=True)

    # Check if already downloaded
    zips = [f for f in os.listdir(WLASL_DIR) if f.endswith(".zip")]
    if zips:
        print("Zip already exists — skipping download.")
        return True

    print("Downloading WLASL from Kaggle...")
    result = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "risangbaskoro/wlasl-processed",
        "-p", WLASL_DIR
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Kaggle download failed:\n", result.stderr)
        print("\nManual alternative:")
        print("  https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed")
        print(f"  Download manually and place the zip inside: {WLASL_DIR}/")
        return False

    print("Download complete.")
    return True

def extract_zip():
    marker = os.path.join(WLASL_DIR, ".extracted")
    if os.path.exists(marker):
        print("Already extracted — skipping.")
        return True

    zip_path = next(
        (os.path.join(WLASL_DIR, f)
         for f in os.listdir(WLASL_DIR) if f.endswith(".zip")),
        None
    )
    if not zip_path:
        print("No zip found in", WLASL_DIR)
        return False

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(WLASL_DIR)

    Path(marker).touch()
    print("Extraction complete.")
    return True

# ─────────────────────────────────────────────
#  STEP 2 — PARSE JSON METADATA
#
#  Builds a dict:  gloss → list of video entries
#  Each entry has: video_id, frame_start, frame_end
#
#  JSON structure:
#  [
#    {
#      "gloss": "hello",
#      "instances": [
#        {
#          "video_id": "00001",
#          "frame_start": 1,
#          "frame_end": 35,
#          ...
#        },
#        ...
#      ]
#    },
#    ...
#  ]
# ─────────────────────────────────────────────
def load_json_metadata() -> dict:
    """
    Finds WLASL JSON file and parses it into:
        { "hello": [{"video_id": "00001", "frame_start": 1, "frame_end": 35}, ...] }
    """
    # Search for the JSON file anywhere inside WLASL_DIR
    json_path = None
    for root, dirs, files in os.walk(WLASL_DIR):
        for f in files:
            if f.endswith(".json") and "WLASL" in f:
                json_path = os.path.join(root, f)
                break

    if not json_path:
        # fallback — any json file in the directory
        for root, dirs, files in os.walk(WLASL_DIR):
            for f in files:
                if f.endswith(".json"):
                    json_path = os.path.join(root, f)
                    break

    if not json_path:
        print("ERROR: WLASL JSON metadata file not found.")
        return {}

    print(f"Loading metadata from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build lookup dict
    lookup = {}
    for entry in data:
        gloss     = entry["gloss"].lower()     # normalise to lowercase
        instances = entry.get("instances", [])
        lookup[gloss] = instances

    print(f"Loaded {len(lookup)} glosses from JSON.")
    return lookup

# ─────────────────────────────────────────────
#  STEP 3 — FIND VIDEO FILE BY ID
#
#  Videos are named like: 00001.mp4, 01234.mp4
#  We search the entire WLASL_DIR for a file
#  whose name (without extension) matches video_id.
# ─────────────────────────────────────────────
def build_video_index() -> dict:
    """
    Walks wlasl_raw/ and builds:
        { "00001": "/full/path/to/00001.mp4", ... }
    One-time scan so we don't walk the filesystem per video.
    """
    print("Indexing video files...")
    index = {}
    for root, dirs, files in os.walk(WLASL_DIR):
        for f in files:
            if f.endswith((".mp4", ".avi", ".mov")):
                video_id = os.path.splitext(f)[0]   # "00001"
                index[video_id] = os.path.join(root, f)

    print(f"Found {len(index)} video files.")
    return index

# ─────────────────────────────────────────────
#  STEP 4 — EXTRACT LANDMARKS FROM ONE VIDEO
# ─────────────────────────────────────────────
def hand_to_array(hand) -> np.ndarray:
    flat = []
    for lm in hand:
        flat.extend([lm.x, lm.y, lm.z])
    return np.array(flat, dtype=np.float32)   # (63,)

def extract_frame_landmarks(result) -> np.ndarray:
    """Left hand first (0:63), right hand second (63:126). Zeros if absent."""
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

    return np.concatenate([left, right])       # (126,)

def video_to_sequence(video_path: str,
                      frame_start: int = 1,
                      frame_end:   int = -1) -> np.ndarray | None:
    """
    Extracts SEQUENCE_LEN evenly spaced frames from a video clip.

    frame_start / frame_end come from WLASL JSON metadata —
    they tell us exactly which frames contain the sign within
    the raw video. We only sample within that window.

    frame_end = -1 means "until the last frame".
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    # Clamp frame window to actual video length
    # WLASL JSON is 1-indexed, OpenCV is 0-indexed → subtract 1
    start = max(0, frame_start - 1)
    end   = total - 1 if frame_end == -1 else min(frame_end - 1, total - 1)

    if end <= start:
        end = total - 1   # fallback: use whole video

    # 30 evenly spaced indices within the sign window
    frame_indices = np.linspace(start, end, SEQUENCE_LEN, dtype=int)

    sequence  = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            sequence.append(np.zeros(126, dtype=np.float32))
            continue

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result    = detector.detect(mp_img)           # IMAGE mode — no timestamp
        landmarks = extract_frame_landmarks(result)
        sequence.append(landmarks)

    cap.release()

    arr = np.array(sequence, dtype=np.float32)   # (30, 126)

    # Reject if more than 50% of frames have no hand
    zero_frames = np.sum(np.all(arr == 0, axis=1))
    if zero_frames > SEQUENCE_LEN * 0.8:
        return None

    return arr

# ─────────────────────────────────────────────
#  STEP 5 — EXTRACT ALL SIGNS
# ─────────────────────────────────────────────
def extract_sign(sign: str, metadata: dict, video_index: dict):
    """
    Uses JSON metadata to find all video IDs for a sign,
    locates each video file using the video index,
    extracts landmarks, saves .npy files.
    """
    # Folder name for saving (underscores)
    folder = FOLDER_NAME.get(sign, sign.replace(" ", "_"))
    out_dir = os.path.join(DATA_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    # Check existing
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    if existing >= MAX_SAMPLES:
        print(f"  '{sign}' already has {existing} samples — skipping.")
        return

    # Get instances from JSON
    instances = metadata.get(sign.lower(), [])
    if not instances:
        print(f"  '{sign}' — not found in JSON metadata.")
        return

    print(f"  '{sign}' — {len(instances)} instances in JSON, "
          f"need {MAX_SAMPLES - existing} more samples...")

    saved   = existing
    skipped = 0

    for inst in instances:
        if saved >= MAX_SAMPLES:
            break

        video_id    = str(inst.get("video_id", ""))
        frame_start = inst.get("frame_start", 1)
        frame_end   = inst.get("frame_end", -1)

        # Find the actual video file
        video_path = video_index.get(video_id)
        if not video_path:
            skipped += 1
            continue   # video file missing (common with WLASL)

        seq = video_to_sequence(video_path, frame_start, frame_end)
        if seq is None:
            skipped += 1
            continue

        out_path = os.path.join(out_dir, f"{saved}.npy")
        np.save(out_path, seq)
        saved += 1

    print(f"  '{sign}' — saved {saved - existing} samples  "
          f"({skipped} skipped)")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=== WLASL Landmark Extractor ===\n")

    # Download + unzip
    if not download_wlasl():
        return
    if not extract_zip():
        return

    # Load JSON metadata and build video file index
    metadata    = load_json_metadata()
    if not metadata:
        return

    video_index = build_video_index()
    if not video_index:
        print("No video files found. Check extraction.")
        return

    # Extract landmarks for each sign
    print("\nExtracting landmarks...\n")
    for sign in SIGNS:
        extract_sign(sign, metadata, video_index)

    # Summary
    print("\n=== Dataset Summary ===")
    total = 0
    for sign in SIGNS:
        folder = FOLDER_NAME.get(sign, sign.replace(" ", "_"))
        path   = os.path.join(DATA_DIR, folder)
        count  = len([f for f in os.listdir(path) if f.endswith(".npy")]) \
                 if os.path.isdir(path) else 0
        tag    = "✓" if count >= MAX_SAMPLES else f"{count}/{MAX_SAMPLES}"
        print(f"  {sign:<15} {tag}")
        total += count

    print(f"\nTotal samples : {total}")
    print(f"Sample shape  : ({SEQUENCE_LEN}, 126) float32")
    print("Next step     : train_model.py")

if __name__ == "__main__":
    main()