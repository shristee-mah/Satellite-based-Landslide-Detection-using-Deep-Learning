import os
import glob
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Custom Metrics
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Change to the desired directory (Optional)
archive_path = r'C:\Users\Nitro v15\Desktop\Minor Project\archive'
try:
    os.chdir(archive_path)
except Exception as e:
    print(f"Error changing directory: {e}")

# Load the model
model = load_model("best_model.keras", custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})

# Load validation images
validation_url = os.path.join(archive_path, "TestData","img", "*.h5")
img_val = sorted(glob.glob(validation_url)) #files

# Initialize VAL_XX
num_samples = len(img_val)
VAL_XX = np.zeros((num_samples, 128, 128, 6))
mask_name = []

# Processing images
eps = 1e-6  # Small constant to avoid division by zero
for i, img in enumerate(img_val):
    print(f"Processing: {img} ({i}/{num_samples})")
    mask_name.append(os.path.basename(img).replace('image', 'mask'))

    with h5py.File(img, "r") as hdf:
        keys = list(hdf.keys())
        print(f"Keys in {img}: {keys}")
        
        data = np.array(hdf.get("img"), dtype=np.float32)  # Ensure data is in float32
        
        if data.ndim == 3:
            # Handle NaN values
            data[np.isnan(data)] = eps

            # Normalization
            mid_rgb = data[:, :, 1:4].max() / 2.0
            mid_slope = data[:, :, 12].max() / 2.0
            mid_elevation = data[:, :, 13].max() / 2.0

            # NDVI calculation (Fixed potential division by zero)
            data_red = data[:, :, 3]
            data_nir = data[:, :, 7]
            data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red) + eps)

            # Populate VAL_XX
            VAL_XX[i, :, :, 0] = 1 - data[:, :, 3] / mid_rgb  # RED
            VAL_XX[i, :, :, 1] = 1 - data[:, :, 2] / mid_rgb  # GREEN
            VAL_XX[i, :, :, 2] = 1 - data[:, :, 1] / mid_rgb  # BLUE
            VAL_XX[i, :, :, 3] = data_ndvi  # NDVI
            VAL_XX[i, :, :, 4] = 1 - data[:, :, 12] / mid_slope  # SLOPE (Fixed)
            VAL_XX[i, :, :, 5] = 1 - data[:, :, 13] / mid_elevation  # ELEVATION (Fixed)
        else:
            print(f"Skipping {img} due to unexpected shape: {data.shape}")

# Predict
threshold = 0.6
pred_img = model.predict(VAL_XX)
pred_img = (pred_img > threshold).astype(np.uint8)

# Ensure image index is valid
img = min(1, VAL_XX.shape[0] - 1)

# Check for landslide
is_landslide = np.any(pred_img[img, :, :, 0] == 1)
label_text = "Landslide Detected" if is_landslide else "No Landslide detected"

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
ax1.imshow(pred_img[img, :, :, 0], cmap="gray")
ax1.set_title("Prediction")

ax2.imshow(VAL_XX[img, :, :, 0:3])
ax2.set_title("Image")

fig.suptitle(label_text, fontsize=16, fontweight="bold", color="red")
plt.show()
