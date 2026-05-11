import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Define path to the output folder where .h5 files are stored
output_folder = r'C:\Users\Nitro v15\Desktop\Minor Project\archive\ValidData\img'

# # Get the first .h5 file
# h5_files = sorted(glob.glob(os.path.join(output_folder, "*.h5")))
# if not h5_files:
#     print("No .h5 files found in the directory.")
#     exit()

# h5_path = h5_files[0]  # Select the first .h5 file

# # Open the HDF5 file
# with h5py.File(h5_path, "r") as f:
#     # List available datasets
#     print("Available datasets:", list(f.keys()))

#     # Check if "images" dataset exists
#     if "image" not in f:
#         print("Dataset 'images' not found in the file.")
#         exit()

    # Load a sample image (first image)
sample = r'C:\Users\Nitro v15\Desktop\Minor Project\archive\ValidData\img\image_1.h5'

# Plot the RGB channels
plt.imshow(sample[:, :, :3])  # Show RGB channels
plt.title("Sample Training Image")
plt.axis("off")
plt.show()
