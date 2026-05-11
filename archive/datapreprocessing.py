import os
import cv2
import numpy as np
import h5py
from glob import glob

# Define input/output paths
input_folder = r'C:\Users\Nitro v15\Desktop\Minor Project\archive\check input'
output_folder = r'C:\Users\Nitro v15\Desktop\Minor Project\archive\check output'
os.makedirs(output_folder, exist_ok=True)

# HDF5 File Name
h5_filename = os.path.join(output_folder, "img001.h5")

# Create an HDF5 file
with h5py.File(h5_filename, "w") as h5f:
    dataset = []  # List to store patches

    # Process each image
    for img_path in glob(os.path.join(input_folder, "*.png")):  # Change to *.jpg if needed
        # Load image (RGB)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img.astype(np.float32) / 255.0  # Normalize

        # Create additional synthetic bands (grayscale and edges)
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        edges = (cv2.Canny((img * 255).astype(np.uint8), 100, 200) / 255.0).astype(np.float32)

        # Stack channels to get a 6-band image
        img_six_band = np.dstack([img, gray, edges])

        # Crop into 128x128 patches
        h, w, c = img_six_band.shape
        for i in range(0, h - 128 + 1, 128):
            for j in range(0, w - 128 + 1, 128):
                patch = img_six_band[i:i+128, j:j+128, :]
                dataset.append(patch)

    # Convert to numpy array and save to HDF5
    if dataset:
        dataset = np.array(dataset, dtype=np.float32)
        h5f.create_dataset("images", data=dataset)
        print(f"Saved {len(dataset)} patches of shape (128,128,6) to {h5_filename}")
    else:
        print("No valid patches found. HDF5 file was not created.")
