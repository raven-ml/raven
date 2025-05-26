#!/usr/bin/env python3
"""Generate test data for Nx I/O example"""

import numpy as np
import os

output_dir = "./test_data"
os.makedirs(output_dir, exist_ok=True)

# Generate sample .npy file
arr = np.arange(12, dtype=np.float32).reshape((3, 4))
np.save(os.path.join(output_dir, "python_array.npy"), arr)
print(f"Generated python_array.npy: shape={arr.shape}, dtype={arr.dtype}")

# Generate sample .npz archive
data1 = np.linspace(0, 1, 6).reshape((2, 3))
data2 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
np.savez(os.path.join(output_dir, "python_archive.npz"), 
         floats=data1, integers=data2)
print("Generated python_archive.npz with 'floats' and 'integers' arrays")