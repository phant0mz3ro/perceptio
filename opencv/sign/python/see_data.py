# Run this separately — paste in terminal as a quick check
import numpy as np

seq = np.load("asl_dataset/hello/5.npy")
print(f"RAW   mean: {seq.mean():.4f}  std: {seq.std():.4f}  min: {seq.min():.4f}  max: {seq.max():.4f}")
print(f"Left  wrist (frame 0): {seq[0, 0:3]}")
print(f"Right wrist (frame 0): {seq[0, 63:66]}")
print(f"ALL {seq}")