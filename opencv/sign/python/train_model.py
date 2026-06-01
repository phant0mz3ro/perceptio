"""
ASL Sign Language LSTM Trainer  (improved)
───────────────────────────────────────────
Loads asl_dataset/, trains an LSTM classifier,
saves model as asl_model.keras

SETUP:
    pip install tensorflow scikit-learn matplotlib numpy

DATASET STRUCTURE:
    asl_dataset/
        hello/   0.npy ... 39.npy   shape (30, 126)
        yes/     0.npy ... 39.npy
        ...

OUTPUTS:
    asl_model.keras       ← trained model
    label_map.json        ← maps class index → sign name
    training_plot.png     ← accuracy/loss curves
    confusion_matrix.png  ← per-class breakdown
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR      = "asl_dataset"
MODEL_PATH    = "asl_model.keras"
LABEL_MAP     = "label_map.json"
PLOT_PATH     = "training_plot.png"

SEQUENCE_LEN  = 30       # frames per sample — must match extractor
FEATURES      = 126      # 21 landmarks × xyz × 2 hands
EPOCHS        = 100
BATCH_SIZE    = 16       # FIX: smaller batch for small dataset (was 32)
LEARNING_RATE = 0.001
TEST_SPLIT    = 0.2

# Augmentation strength (set to 0.0 to disable)
NOISE_STD     = 0.005    # Gaussian noise on landmark coords
SCALE_RANGE   = (0.9, 1.1)  # random scale per sequence


# ─────────────────────────────────────────────
#  STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset(data_dir: str) -> tuple:
    """
    Walks data_dir and loads all .npy files.
    Returns:
        X      : (N, 30, 126)  float32
        y      : (N,)          int
        labels : list of sign names in class-index order
    """
    sequences, class_labels, labels = [], [], []

    sign_folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print(f"Found {len(sign_folders)} sign classes: {sign_folders}\n")

    for class_idx, sign in enumerate(sign_folders):
        labels.append(sign)
        sign_path = os.path.join(data_dir, sign)
        npy_files = sorted([f for f in os.listdir(sign_path) if f.endswith(".npy")])

        loaded = 0
        for f in npy_files:
            seq = np.load(os.path.join(sign_path, f))
            if seq.shape != (SEQUENCE_LEN, FEATURES):
                print(f"  WARNING: {sign}/{f} has shape {seq.shape} — skipping")
                continue
            sequences.append(seq)
            class_labels.append(class_idx)
            loaded += 1

        print(f"  {sign:<15} class={class_idx}  samples={loaded}")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(class_labels, dtype=np.int32)

    print(f"\nTotal samples : {len(X)}")
    print(f"X shape       : {X.shape}")
    print(f"Classes       : {len(labels)}")
    return X, y, labels


# ─────────────────────────────────────────────
#  STEP 2 — NORMALIZE
#
#  Raw landmark coords vary by hand size, distance
#  from camera, and position in frame. Normalizing
#  each sequence independently makes the model learn
#  MOTION PATTERNS, not absolute positions.
#
#  We use per-sequence z-score: subtract mean, divide
#  by std. This centers every sequence around zero
#  with unit variance.
# ─────────────────────────────────────────────
def normalize(X: np.ndarray) -> np.ndarray:
    """Z-score normalization per sequence."""
    mean = X.mean(axis=(1, 2), keepdims=True)   # shape (N, 1, 1)
    std  = X.std(axis=(1, 2),  keepdims=True) + 1e-8
    return (X - mean) / std


# ─────────────────────────────────────────────
#  STEP 3 — AUGMENTATION
#
#  With only 40 samples per sign, the model WILL
#  memorize training data without augmentation.
#
#  We apply two transforms to training data only:
#
#  1. Gaussian noise  — simulates hand tremor /
#     imprecise landmark detection
#
#  2. Random scale    — simulates hand size
#     variation between different people
#
#  Augmentation runs on the training split only,
#  never on validation.
# ─────────────────────────────────────────────
def augment(X: np.ndarray) -> np.ndarray:
    """Apply noise + random scaling to a batch of sequences."""
    X = X.copy()

    # 1. Add Gaussian noise
    if NOISE_STD > 0:
        X += np.random.normal(0, NOISE_STD, X.shape).astype(np.float32)

    # 2. Random per-sequence scale factor
    lo, hi = SCALE_RANGE
    scales = np.random.uniform(lo, hi, size=(len(X), 1, 1)).astype(np.float32)
    X *= scales

    return X


# ─────────────────────────────────────────────
#  STEP 4 — PREPROCESS + AUGMENT
# ─────────────────────────────────────────────
def preprocess(X: np.ndarray, y: np.ndarray, n_classes: int):
    # Normalize first
    X = normalize(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=y
    )

    # Augment training set only
    X_train = augment(X_train)

    y_train = to_categorical(y_train, num_classes=n_classes)
    y_val   = to_categorical(y_val,   num_classes=n_classes)

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────
#  STEP 5 — BUILD MODEL
#
#  Input (30, 126)
#      ↓
#  LSTM(128, return_sequences=True)
#      ↓
#  Dropout(0.4)                  ← FIX: increased from 0.3
#      ↓
#  LSTM(64, return_sequences=False)
#      ↓
#  BatchNormalization
#      ↓
#  Dense(64, relu)
#      ↓
#  Dropout(0.4)
#      ↓
#  Dense(n_classes, softmax)
# ─────────────────────────────────────────────
def build_model(n_classes: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(SEQUENCE_LEN, FEATURES)),
        Dropout(0.4),                              # FIX: 0.3 → 0.4

        LSTM(64, return_sequences=False),
        BatchNormalization(),

        Dense(64, activation='relu'),
        Dropout(0.4),                              # FIX: 0.3 → 0.4

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ─────────────────────────────────────────────
#  STEP 6 — CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks() -> list:
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,          # FIX: reduced from 15 (small dataset overfits fast)
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,           # FIX: reduced from 7
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]


# ─────────────────────────────────────────────
#  STEP 7 — PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_training(history, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True)

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plot saved → {save_path}")


# ─────────────────────────────────────────────
#  STEP 8 — CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion(model, X_val, y_val_onehot, labels):
    y_pred     = model.predict(X_val, verbose=0)
    y_pred_idx = np.argmax(y_pred,       axis=1)
    y_true_idx = np.argmax(y_val_onehot, axis=1)

    cm   = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # FIX: explicit labels

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved → confusion_matrix.png")

    # Also print per-class accuracy to console
    print("\nPer-class accuracy:")
    for i, label in enumerate(labels):
        row   = cm[i]
        acc   = cm[i, i] / row.sum() if row.sum() > 0 else 0
        print(f"  {label:<15} {acc * 100:.0f}%  ({cm[i,i]}/{row.sum()})")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=== ASL LSTM Trainer ===\n")

    # 1 — Load
    X, y, labels = load_dataset(DATA_DIR)
    n_classes    = len(labels)

    # Save label map
    label_map = {str(i): label for i, label in enumerate(labels)}
    with open(LABEL_MAP, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"\nLabel map saved → {LABEL_MAP}")
    print(f"  {label_map}\n")

    # 2 — Preprocess + augment
    X_train, X_val, y_train, y_val = preprocess(X, y, n_classes)

    # 3 — Build
    model = build_model(n_classes)
    model.summary()

    # 4 — Train
    print("\nTraining...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 5 — Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal val accuracy : {val_acc * 100:.1f}%")
    print(f"Final val loss     : {val_loss:.4f}")

    # 6 — Plots
    plot_training(history, PLOT_PATH)
    plot_confusion(model, X_val, y_val, labels)

    print(f"\nModel saved → {MODEL_PATH}")
    print("Next step   : run live_decoder.py")


if __name__ == "__main__":
    main()