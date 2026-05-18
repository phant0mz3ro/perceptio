import numpy as np
import os

DATA_DIR = "asl_dataset"
sign = "hello"

samples = []
path = os.path.join(DATA_DIR, sign)

for f in sorted(os.listdir(path)):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(path, f))
        samples.append(arr)
        print(f"{f}  shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  zeros={np.all(arr==0)}")

print(f"\nTotal samples: {len(samples)}")
print(f"Any shape mismatches: {len(set(s.shape for s in samples)) > 1}")