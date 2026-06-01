"""
Landmark Viewer
───────────────
Replays the extracted landmark data from your .npy files
as an animated skeleton so you can verify the data quality
before and after training.

CONTROLS:
    LEFT  / RIGHT arrow  → previous / next sample
    UP    / DOWN  arrow  → previous / next sign class
    SPACE                → pause / resume animation
    R                    → restart current sample
    Q                    → quit

REQUIRES:
    asl_dataset/   ← your extracted .npy files
"""

import os
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR     = "asl_dataset"
WINDOW_W     = 1000
WINDOW_H     = 600
FRAME_DELAY  = 80    # ms between frames when playing (lower = faster)

# Canvas for skeleton (left side)
CANVAS_W     = 640
CANVAS_H     = 560

# Sidebar width
SIDEBAR_W    = WINDOW_W - CANVAS_W

# ─────────────────────────────────────────────
#  HAND CONNECTIONS — same as live_decoder
# ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (9, 10), (10, 11), (11, 12),           # middle
    (13, 14), (14, 15), (15, 16),          # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17), (0, 5),    # palm
]

FINGERTIPS = {4, 8, 12, 16, 20}


# ─────────────────────────────────────────────
#  LOAD DATASET INDEX
# ─────────────────────────────────────────────
def load_index(data_dir: str) -> dict:
    """
    Builds a dict:
        { "hello": ["path/0.npy", "path/1.npy", ...], ... }
    """
    index = {}
    sign_folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for sign in sign_folders:
        sign_path = os.path.join(data_dir, sign)
        files = sorted([
            os.path.join(sign_path, f)
            for f in os.listdir(sign_path)
            if f.endswith(".npy")
        ])
        if files:
            index[sign] = files

    return index


# ─────────────────────────────────────────────
#  DRAW ONE HAND FROM RAW LANDMARK ARRAY
#
#  The .npy landmarks are stored as normalized
#  x, y, z coordinates (0.0 to 1.0).
#  We scale them to the canvas pixel size.
#
#  Layout of the 126 values per frame:
#  [0:63]   = left hand  (21 landmarks × xyz)
#  [63:126] = right hand (21 landmarks × xyz)
# ─────────────────────────────────────────────
def parse_hand(flat: np.ndarray) -> list:
    """
    Convert flat (63,) array → list of 21 (x, y) pixel coords.
    Returns empty list if hand is all zeros (not detected).
    """
    if np.all(flat == 0):
        return []   # hand was not detected in this frame

    points = []
    for i in range(21):
        x = flat[i * 3]       # normalized 0-1
        y = flat[i * 3 + 1]   # normalized 0-1
        # z = flat[i*3 + 2]   # depth — not used for 2D drawing

        px = int(x * CANVAS_W * 0.8 + CANVAS_W * 0.1)   # add 10% padding
        py = int(y * CANVAS_H * 0.8 + CANVAS_H * 0.1)
        points.append((px, py))

    return points


def draw_hand(canvas, points: list, color_bone, color_dot):
    """Draw skeleton lines and landmark dots for one hand."""
    if not points:
        return

    # Bones
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(canvas, points[start], points[end],
                     color_bone, 2, cv2.LINE_AA)

    # Dots
    for i, (px, py) in enumerate(points):
        radius = 7 if i in FINGERTIPS else 5
        cv2.circle(canvas, (px, py), radius, color_dot, -1)
        cv2.circle(canvas, (px, py), radius, (0, 0, 0), 1)


def render_frame(sequence: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Renders one frame of a sequence as a skeleton on a dark canvas.

    sequence shape: (30, 126)
    frame_idx:      0-29
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 35)   # dark blue-grey background

    frame = sequence[frame_idx]   # (126,)

    left_flat  = frame[0:63]
    right_flat = frame[63:126]

    left_pts  = parse_hand(left_flat)
    right_pts = parse_hand(right_flat)

    # Left hand = cyan, Right hand = magenta (same as live_decoder)
    draw_hand(canvas, left_pts,  (255, 220, 0),   (0, 255, 255))
    draw_hand(canvas, right_pts, (255, 50,  255),  (200, 100, 255))

    # Legend
    cv2.circle(canvas, (20, CANVAS_H - 40), 6, (0, 255, 255), -1)
    cv2.putText(canvas, "Left hand",  (32, CANVAS_H - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.circle(canvas, (20, CANVAS_H - 20), 6, (200, 100, 255), -1)
    cv2.putText(canvas, "Right hand", (32, CANVAS_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 255), 1)

    # Frame number
    cv2.putText(canvas, f"Frame {frame_idx + 1}/30",
                (CANVAS_W - 110, CANVAS_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Zero frame warning
    both_zero = np.all(left_flat == 0) and np.all(right_flat == 0)
    if both_zero:
        cv2.putText(canvas, "NO HANDS DETECTED",
                    (CANVAS_W // 2 - 120, CANVAS_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 200), 2)

    return canvas


# ─────────────────────────────────────────────
#  DRAW SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(sign: str, sample_idx: int, total_samples: int,
                   sign_idx: int, total_signs: int,
                   sequence: np.ndarray, frame_idx: int,
                   paused: bool) -> np.ndarray:
    """
    Right panel showing:
    - Current sign name + sample number
    - Navigation hints
    - Per-frame quality stats (how many frames have hands)
    - Frame strip — 30 tiny indicators showing which frames
      have landmark data and which are zeros
    """
    sidebar = np.zeros((WINDOW_H, SIDEBAR_W, 3), dtype=np.uint8)
    sidebar[:] = (18, 18, 28)

    # ── Sign name ──
    cv2.putText(sidebar, sign.upper().replace("_", " "),
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 220, 255), 2, cv2.LINE_AA)

    cv2.putText(sidebar, f"Class {sign_idx + 1} of {total_signs}",
                (15, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120, 120, 120), 1)

    # ── Sample number ──
    cv2.putText(sidebar, f"Sample  {sample_idx + 1} / {total_samples}",
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    # ── Playback status ──
    status_color = (0, 180, 255) if paused else (0, 255, 120)
    status_text  = "PAUSED" if paused else "PLAYING"
    cv2.putText(sidebar, status_text,
                (15, 125), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, status_color, 1)

    # ── Data quality stats ──
    #
    # Count how many of the 30 frames have actual hand data
    # vs zeros (hand not detected in that frame)
    #
    left_detected  = 0
    right_detected = 0
    both_detected  = 0
    neither        = 0

    for f in sequence:
        l_zero = np.all(f[0:63]  == 0)
        r_zero = np.all(f[63:126] == 0)

        if not l_zero and not r_zero:
            both_detected += 1
        elif not l_zero:
            left_detected += 1
        elif not r_zero:
            right_detected += 1
        else:
            neither += 1

    cv2.putText(sidebar, "DATA QUALITY", (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    quality_rows = [
        (f"Both hands:   {both_detected}/30",  (0, 220, 120)),
        (f"Left only:    {left_detected}/30",   (0, 220, 255)),
        (f"Right only:   {right_detected}/30",  (180, 100, 255)),
        (f"No hands:     {neither}/30",         (80, 80, 200)),
    ]
    for i, (txt, col) in enumerate(quality_rows):
        cv2.putText(sidebar, txt, (15, 188 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    # Quality score
    valid = both_detected + left_detected + right_detected
    quality_pct = valid / 30 * 100
    q_color = (0, 220, 120) if quality_pct >= 80 else \
              (0, 200, 255) if quality_pct >= 50 else \
              (0, 80,  200)
    cv2.putText(sidebar, f"Valid frames: {quality_pct:.0f}%",
                (15, 278), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, q_color, 1)

    # ── Frame strip ──
    #
    # 30 small squares, one per frame.
    # Color shows what was detected in that frame:
    #   Green   = both hands
    #   Cyan    = left hand only
    #   Purple  = right hand only
    #   Dark    = no hands (zeros)
    #   Outline = current frame position
    #
    cv2.putText(sidebar, "FRAME STRIP  (30 frames)",
                (15, 315), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (150, 150, 150), 1)

    strip_x = 15
    strip_y = 325
    sq_w    = (SIDEBAR_W - 30) // 30   # width of each square
    sq_h    = 18

    for fi in range(30):
        f      = sequence[fi]
        l_zero = np.all(f[0:63]   == 0)
        r_zero = np.all(f[63:126] == 0)

        if not l_zero and not r_zero:
            col = (0, 180, 80)      # green — both hands
        elif not l_zero:
            col = (180, 180, 0)     # cyan — left only
        elif not r_zero:
            col = (140, 0, 180)     # purple — right only
        else:
            col = (40, 40, 50)      # dark — nothing

        x1 = strip_x + fi * sq_w
        x2 = x1 + sq_w - 1
        cv2.rectangle(sidebar, (x1, strip_y), (x2, strip_y + sq_h), col, -1)

        # Highlight current frame with white outline
        if fi == frame_idx:
            cv2.rectangle(sidebar, (x1 - 1, strip_y - 1),
                          (x2 + 1, strip_y + sq_h + 1), (255, 255, 255), 1)

    # Strip legend
    legend_y = strip_y + sq_h + 16
    legends = [
        ((0, 180, 80),   "both"),
        ((180, 180, 0),  "left"),
        ((140, 0, 180),  "right"),
        ((40, 40, 50),   "none"),
    ]
    lx = 15
    for col, txt in legends:
        cv2.rectangle(sidebar, (lx, legend_y), (lx + 10, legend_y + 10), col, -1)
        cv2.putText(sidebar, txt, (lx + 13, legend_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        lx += 52

    # ── Raw coordinate readout for current frame ──
    #
    # Shows the actual x,y,z values of the wrist landmark (index 0)
    # for each hand — just to confirm real numbers are in the array
    #
    cv2.putText(sidebar, "RAW VALUES  (wrist landmark)",
                (15, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (150, 150, 150), 1)

    curr_frame = sequence[frame_idx]
    lx_raw = curr_frame[0:3]    # left wrist x,y,z
    rx_raw = curr_frame[63:66]  # right wrist x,y,z

    if not np.all(lx_raw == 0):
        cv2.putText(sidebar,
                    f"L wrist: ({lx_raw[0]:.3f}, {lx_raw[1]:.3f}, {lx_raw[2]:.3f})",
                    (15, legend_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
    else:
        cv2.putText(sidebar, "L wrist: not detected",
                    (15, legend_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    if not np.all(rx_raw == 0):
        cv2.putText(sidebar,
                    f"R wrist: ({rx_raw[0]:.3f}, {rx_raw[1]:.3f}, {rx_raw[2]:.3f})",
                    (15, legend_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 100, 255), 1)
    else:
        cv2.putText(sidebar, "R wrist: not detected",
                    (15, legend_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    # ── Navigation controls ──
    cv2.putText(sidebar, "CONTROLS",
                (15, WINDOW_H - 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    controls = [
        "LEFT / RIGHT  : prev/next sample",
        "UP   / DOWN   : prev/next sign",
        "SPACE         : pause / play",
        "R             : restart",
        "Q             : quit",
    ]
    for i, line in enumerate(controls):
        cv2.putText(sidebar, line,
                    (15, WINDOW_H - 95 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

    return sidebar


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=== Landmark Viewer ===\n")

    index = load_index(DATA_DIR)
    if not index:
        print(f"No data found in {DATA_DIR}/")
        return

    signs      = list(index.keys())
    sign_idx   = 0
    sample_idx = 0
    frame_idx  = 0
    paused     = False

    print(f"Loaded {len(signs)} signs:")
    for s, files in index.items():
        print(f"  {s:<15} {len(files)} samples")
    print("\nOpening viewer...\n")

    cv2.namedWindow("Landmark Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmark Viewer", WINDOW_W, WINDOW_H)

    while True:
        sign     = signs[sign_idx]
        files    = index[sign]
        filepath = files[sample_idx]
        sequence = np.load(filepath)   # (30, 126)

        # ── render skeleton canvas ──
        skeleton = render_frame(sequence, frame_idx)

        # ── render sidebar ──
        sidebar = render_sidebar(
            sign, sample_idx, len(files),
            sign_idx, len(signs),
            sequence, frame_idx, paused
        )

        # ── combine into one window ──
        # Pad skeleton height to match window height if needed
        pad_h = WINDOW_H - CANVAS_H
        if pad_h > 0:
            pad = np.zeros((pad_h, CANVAS_W, 3), dtype=np.uint8)
            pad[:] = (18, 18, 28)
            skeleton = np.vstack([skeleton, pad])

        frame_display = np.hstack([skeleton, sidebar])
        cv2.imshow("Landmark Viewer", frame_display)

        # ── keyboard + timing ──
        key = cv2.waitKey(1 if paused else FRAME_DELAY) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = not paused

        elif key == ord('r'):
            frame_idx = 0

        elif key == 81 or key == ord('a'):   # LEFT arrow
            sample_idx = (sample_idx - 1) % len(files)
            frame_idx  = 0

        elif key == 83 or key == ord('d'):   # RIGHT arrow
            sample_idx = (sample_idx + 1) % len(files)
            frame_idx  = 0

        elif key == 82 or key == ord('w'):   # UP arrow
            sign_idx   = (sign_idx - 1) % len(signs)
            sample_idx = 0
            frame_idx  = 0

        elif key == 84 or key == ord('s'):   # DOWN arrow
            sign_idx   = (sign_idx + 1) % len(signs)
            sample_idx = 0
            frame_idx  = 0

        else:
            # Advance frame when playing
            if not paused:
                frame_idx += 1
                if frame_idx >= 30:
                    frame_idx = 0   # loop the animation

    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == "__main__":
    main()