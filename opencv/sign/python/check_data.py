import numpy as np
import os

DATA_DIR = "asl_dataset"

print("=== Dataset Inspection ===\n")

for sign in sorted(os.listdir(DATA_DIR)):
    sign_path = os.path.join(DATA_DIR, sign)
    if not os.path.isdir(sign_path):
        continue

    files = sorted([f for f in os.listdir(sign_path) if f.endswith(".npy")])
    zero_samples   = 0
    weak_samples   = 0
    good_samples   = 0

    for f in files:
        arr      = np.load(os.path.join(sign_path, f))
        non_zero = np.sum(~np.all(arr == 0, axis=1))

        if non_zero == 0:
            zero_samples += 1
        elif non_zero < 5:
            weak_samples += 1
        else:
            good_samples += 1

    print(f"  {sign:<15} total={len(files)}  "
          f"good={good_samples}  weak={weak_samples}  all_zeros={zero_samples}")