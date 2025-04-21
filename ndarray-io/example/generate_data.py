import numpy as np
import h5py
import os

output_dir = "./test_data"
os.makedirs(output_dir, exist_ok=True)
print(f"Generating test data in '{output_dir}'...")

# --- 1. Data for .npy example ---

npy_array_f32 = np.arange(12, dtype=np.float32).reshape((3, 4))
npy_path_f32 = os.path.join(output_dir, "array_f32.npy")
np.save(npy_path_f32, npy_array_f32)
print(f"Saved: {npy_path_f32} (Shape: {npy_array_f32.shape}, Dtype: {npy_array_f32.dtype})")
npy_array_i64 = np.array([-10, 0, 10, 20, 30], dtype=np.int64)
npy_path_i64 = os.path.join(output_dir, "array_i64.npy")
np.save(npy_path_i64, npy_array_i64)
print(f"Saved: {npy_path_i64} (Shape: {npy_array_i64.shape}, Dtype: {npy_array_i64.dtype})")

# --- 2. Data for .npz example ---

npz_floats = np.linspace(0.0, 1.0, 6, dtype=np.float64).reshape((2, 3))
npz_ints = np.random.randint(-100, 100, size=(4,), dtype=np.int32)
npz_bools = np.array([[True, False], [False, True]], dtype=np.bool_) # Note: bool might not map directly
npz_path = os.path.join(output_dir, "archive.npz")
np.savez(npz_path, my_floats=npz_floats, my_integers=npz_ints, my_flags=npz_bools)
print(f"Saved: {npz_path}")
print(f"  - 'my_floats': Shape {npz_floats.shape}, Dtype {npz_floats.dtype}")
print(f"  - 'my_integers': Shape {npz_ints.shape}, Dtype {npz_ints.dtype}")
print(f"  - 'my_flags': Shape {npz_bools.shape}, Dtype {npz_bools.dtype}")

# --- 3. Data for HDF5 example ---

hdf5_path = os.path.join(output_dir, "data.hdf5")
hdf5_data_c64 = np.array([1+2j, 3-4j, 5+0j], dtype=np.complex64)
hdf5_data_u8 = np.arange(6, dtype=np.uint8).reshape((3, 2))
with h5py.File(hdf5_path, 'w') as f:
    f.create_dataset("/complex_numbers", data=hdf5_data_c64)
    print(f"Saved dataset 'complex_numbers' to {hdf5_path}")
    grp = f.create_group("/images/set1")
    grp.create_dataset("matrix_u8", data=hdf5_data_u8)
    print(f"Saved dataset '/images/set1/matrix_u8' to {hdf5_path}")

print("Data generation complete.")