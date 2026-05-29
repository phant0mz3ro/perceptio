import numpy as np
import os
import random

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR         = "asl_dataset"
TARGET_SAMPLES   = 40        # how many samples you want per sign after augmentation
REAL_SAMPLES_MIN = 3         # minimum real samples needed before augmenting

SIGNS = [
    "hello", "my", "name", "yes", "no",
    "please", "thank_you", "help", "finish", "want",
]

# ─────────────────────────────────────────────
#  AUGMENTATION FUNCTIONS
#
#  Every function takes a sequence (30, 126) and
#  returns a new (30, 126) with a realistic variation.
#  126 = left hand (63) + right hand (63)
#  Each hand = 21 landmarks × (x, y, z)
# ─────────────────────────────────────────────

def add_noise(seq: np.ndarray, std: float = 0.005) -> np.ndarray:
    """
    Add tiny Gaussian noise to every landmark coordinate.
    Simulates natural hand tremor and minor detection jitter.
    std=0.005 means ±0.5% of frame width — invisible to the eye
    but enough to make each sample unique.
    Only adds noise where the hand is actually present (non-zero rows).
    """
    seq = seq.copy()
    for i, frame in enumerate(seq):
        if np.any(frame != 0):                        # hand present this frame
            noise = np.random.normal(0, std, frame.shape).astype(np.float32)
            seq[i] = np.clip(frame + noise, 0.0, 1.0) # keep coords in [0,1]
    return seq


def scale(seq: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    """
    Scale the hand size around its centroid.
    Simulates the hand being slightly closer or further from the camera.
    We scale x,y only (not z) to keep depth realistic.
    """
    seq    = seq.copy()
    factor = random.uniform(*factor_range)

    for i, frame in enumerate(seq):
        # process each hand slot separately (left=0:63, right=63:126)
        for start in (0, 63):
            chunk = frame[start:start+63]
            if not np.any(chunk):        # hand absent — skip
                continue
            xyz = chunk.reshape(21, 3)

            # centroid of x,y
            cx = xyz[:, 0].mean()
            cy = xyz[:, 1].mean()

            # scale around centroid
            xyz[:, 0] = cx + (xyz[:, 0] - cx) * factor
            xyz[:, 1] = cy + (xyz[:, 1] - cy) * factor
            xyz        = np.clip(xyz, 0.0, 1.0)

            seq[i, start:start+63] = xyz.flatten()

    return seq


def translate(seq: np.ndarray, max_shift: float = 0.05) -> np.ndarray:
    """
    Shift the entire hand slightly in x and/or y.
    Simulates the signer not being perfectly centred in frame.
    Same random shift applied to every frame so the motion
    path shape is preserved — only position changes.
    """
    seq = seq.copy()
    dx  = random.uniform(-max_shift, max_shift)
    dy  = random.uniform(-max_shift, max_shift)

    for i, frame in enumerate(seq):
        for start in (0, 63):
            chunk = frame[start:start+63]
            if not np.any(chunk):
                continue
            xyz        = chunk.reshape(21, 3)
            xyz[:, 0]  = np.clip(xyz[:, 0] + dx, 0.0, 1.0)
            xyz[:, 1]  = np.clip(xyz[:, 1] + dy, 0.0, 1.0)
            seq[i, start:start+63] = xyz.flatten()

    return seq


def time_warp(seq: np.ndarray, speed_range=(0.8, 1.2)) -> np.ndarray:
    """
    Stretch or compress the sequence in time using linear interpolation.
    speed > 1.0 → sign performed faster (fewer source frames used)
    speed < 1.0 → sign performed slower (source frames stretched out)
    Output is always SEQUENCE_LEN frames so shape stays (30, 126).
    Simulates natural variation in signing speed.
    """
    n      = len(seq)                           # 30
    speed  = random.uniform(*speed_range)
    # source indices we sample from (may be fractional)
    src_len    = int(n * speed)
    src_len    = max(2, min(src_len, n))        # clamp to valid range
    src_idx    = np.linspace(0, src_len - 1, n) # 30 evenly spaced source points
    warped     = np.zeros_like(seq)

    for i, si in enumerate(src_idx):
        lo  = int(si)
        hi  = min(lo + 1, src_len - 1)
        t   = si - lo                           # fractional part
        # linear interpolation between adjacent frames
        src_lo = seq[min(lo, n-1)]
        src_hi = seq[min(hi, n-1)]
        warped[i] = src_lo * (1 - t) + src_hi * t

    return warped.astype(np.float32)


def rotate_2d(seq: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """
    Rotate hand landmarks in the x-y plane around the hand centroid.
    Simulates the signer's wrist being slightly rotated.
    Only rotates x,y — z (depth) is left unchanged.
    """
    seq   = seq.copy()
    angle = random.uniform(-max_angle_deg, max_angle_deg)
    rad   = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    for i, frame in enumerate(seq):
        for start in (0, 63):
            chunk = frame[start:start+63]
            if not np.any(chunk):
                continue
            xyz = chunk.reshape(21, 3)
            cx  = xyz[:, 0].mean()
            cy  = xyz[:, 1].mean()

            x_centered = xyz[:, 0] - cx
            y_centered = xyz[:, 1] - cy

            xyz[:, 0] = np.clip(cx + x_centered * cos_a - y_centered * sin_a, 0, 1)
            xyz[:, 1] = np.clip(cy + x_centered * sin_a + y_centered * cos_a, 0, 1)

            seq[i, start:start+63] = xyz.flatten()

    return seq


# ─────────────────────────────────────────────
#  AUGMENTATION PIPELINE
#  Randomly chains 2–4 augmentations together.
#  Chaining means each synthetic sample is a unique
#  combination — noise+scale, rotate+translate+warp, etc.
# ─────────────────────────────────────────────
ALL_AUGMENTATIONS = [add_noise, scale, translate, time_warp, rotate_2d]

def augment_sample(seq: np.ndarray) -> np.ndarray:
    """Apply a random chain of 2–4 augmentations to one sequence."""
    n_augs = random.randint(2, 4)
    chosen = random.sample(ALL_AUGMENTATIONS, n_augs)
    for fn in chosen:
        seq = fn(seq)
    return seq

# ─────────────────────────────────────────────
#  MAIN AUGMENTATION LOOP
# ─────────────────────────────────────────────
def augment_sign(sign: str):
    path = os.path.join(DATA_DIR, sign)

    # Load existing real samples
    real_files = sorted([f for f in os.listdir(path) if f.endswith(".npy")])
    n_real     = len(real_files)

    if n_real == 0:
        print(f"  '{sign}' — no samples found, skipping.")
        return

    if n_real < REAL_SAMPLES_MIN:
        print(f"  '{sign}' — only {n_real} real samples "
              f"(need at least {REAL_SAMPLES_MIN}). Record more first.")
        return

    if n_real >= TARGET_SAMPLES:
        print(f"  '{sign}' — already has {n_real} samples, nothing to do.")
        return

    real_seqs = [np.load(os.path.join(path, f)) for f in real_files]
    needed    = TARGET_SAMPLES - n_real
    print(f"  '{sign}' — {n_real} real samples → generating {needed} synthetic ones...")

    generated = 0
    idx       = n_real    # start numbering from after real samples

    while generated < needed:
        # pick a random real sample as the base
        base   = random.choice(real_seqs)
        synth  = augment_sample(base)
        save_p = os.path.join(path, f"{idx}.npy")
        np.save(save_p, synth)
        idx       += 1
        generated += 1

    print(f"  '{sign}' — done. Total samples now: {TARGET_SAMPLES}")


def main():
    print("=== ASL Dataset Augmenter ===")
    print(f"Target samples per sign : {TARGET_SAMPLES}")
    print(f"Min real samples needed : {REAL_SAMPLES_MIN}\n")

    for sign in SIGNS:
        augment_sign(sign)

    # ── Summary ───────────────────────────────
    print("\n=== Final Dataset Summary ===")
    total = 0
    for sign in SIGNS:
        path  = os.path.join(DATA_DIR, sign)
        count = len([f for f in os.listdir(path) if f.endswith(".npy")])
        real  = len([f for f in os.listdir(path)
                     if f.endswith(".npy") and int(f[:-4]) < REAL_SAMPLES_MIN])
        tag   = "✓" if count >= TARGET_SAMPLES else f"{count}/{TARGET_SAMPLES}"
        print(f"  {sign:<15} {tag}  (real≈{real}  synthetic≈{count-real})")
        total += count

    print(f"\nTotal samples : {total}")
    print("Next step     : run train_model.py")


if __name__ == "__main__":
    main()