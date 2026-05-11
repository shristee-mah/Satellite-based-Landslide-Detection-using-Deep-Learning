import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import h5py
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
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

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return (2. * intersection + smooth) / (union + smooth)


# Load the model with custom metrics
model = load_model(
    "best_model.keras", 
    custom_objects={
        "f1_m": f1_m, 
        "precision_m": precision_m, 
        "recall_m": recall_m, 
        "dice_coefficient": dice_coefficient
    }
)
# Set page config
st.set_page_config(page_title="Landslide Detection", page_icon="🌍", layout="wide")

# Set custom CSS for design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Indie+Flower&display=swap');

    .stApp {
        background-color: #1E201E;
        padding: 20px;
        color: #FBFFE4;
    }
    .header {
        font-family: 'Playfair Display', serif;
        color: #ECDFCC;
        font-size: 7em;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        font-family: 'Poppins', sans-serif;
        color: #697565;
        font-size: 1.5em;
        text-align: center;
        font-weight: 600;
        margin-bottom: 30px;
    }
    .fact-box {
        font-family: 'Poppins', sans-serif;
        color: #1E201E;
        font-size: 1.4em;
        font-weight: bold;
        background-color: #ECDFCC;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        line-height: 1.6;
    }
    .button {
        font-family: 'Poppins', sans-serif;
        background-color: #1E201E;
        color: white;
        padding: 12px 25px;
        border-radius: 8px;
        font-size: 1.2em;
        transition: background-color 0.3s;
        display: block;
        margin: 30px auto;
        text-align: center;
        font-weight: 600;
    }
    .button:hover {
        background-color: #ECDFCC;
    }
    footer {
        font-family: 'Poppins', sans-serif;
        text-align: center;
        color: #697565;
        font-size: 1.2em;
        margin-top: 40px;
        padding-top: 20px;
        font-weight: 400;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Function to make rounded images
def make_rounded(image_path, radius=30):
    img = Image.open(image_path).convert("RGBA")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), img.size], radius, fill=255)
    img.putalpha(mask)
    return img

# App title
st.markdown("<div class='header'>पहिरो पहिचान </div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Satellite-Based Landslide Detection</div>", unsafe_allow_html=True)

# List of landslide facts
landslide_facts = [
    "🌍 Mapping the Future of Safety: Landslide detection at your Fingertips.",
    # "⛰️ The deadliest landslide in history was in Venezuela (1999), killing over 30,000 people.",
    # "🌧️ Heavy rainfall is one of the primary causes of landslides.",
    # "🚧 Deforestation increases the risk of landslides due to reduced soil stability.",
    # "🔍 Landslides can move as fast as 200 mph in extreme cases.",
    # "🏔️ The Himalayas are one of the most landslide-prone regions in the world.",
    # "💧 Earthquakes can trigger landslides by loosening soil and rock layers.",
    # "🌲 Planting trees on slopes can help prevent landslides by holding soil together.",
    # "📊 Landslides cause billions of dollars in damage worldwide every year."
]

# Get the absolute path of the current directory
base_path = os.path.dirname(__file__)  # Get the current script location

# Define the path to the images folder (update this to your actual folder if needed)
image_folder = os.path.join(base_path, "images")

# List of image filenames (make sure these files exist in your "images" folder)
landslide_images = [
    os.path.join(image_folder, "landslide1.jpg"),
    os.path.join(image_folder, "landslide2.jpg"),
    os.path.join(image_folder, "landslide3.jpg"),
    os.path.join(image_folder, "landslide4.jpg"),
    os.path.join(image_folder, "landslide5.jpg"),
    os.path.join(image_folder, "landslide6.jpg"),
]

# Display placeholders for the facts and images
fact_placeholder = st.empty()
image_placeholder = st.empty()

def update_fact_and_image():
    fact = random.choice(landslide_facts)
    image_path = random.choice(landslide_images)

    fact_placeholder.markdown(f"<div class='fact-box'>{fact}</div>", unsafe_allow_html=True)

    if os.path.exists(image_path):
        rounded_img = make_rounded(image_path)
        image_placeholder.image(rounded_img, use_column_width=True, caption="Landslide Event")

# Display facts and images
update_fact_and_image()

uploaded_files = st.file_uploader("Upload .h5 files", type=["h5"], accept_multiple_files=True)

if uploaded_files:
    temp_folder = "temp_folder"
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        files.append(file_path)

    st.write(f"Uploaded {len(files)} .h5 files successfully!")

    num_samples = len(files)
    VAL_XX = np.zeros((num_samples, 128, 128, 6))
    mask_name = []
    eps = 1e-6

    for i, img in enumerate(files):
        try:
            with h5py.File(img, "r") as hdf:
                if "img" not in hdf:
                    continue

                data = np.array(hdf.get("img"), dtype=np.float32)
                data[np.isnan(data)] = eps

                data_red = data[:, :, 3]
                data_nir = data[:, :, 7]
                data_ndvi = (data_nir - data_red) / (data_nir + data_red + eps)

                VAL_XX[i, :, :, 0] = 1 - data[:, :, 3] / max(1, data[:, :, 1:4].max() / 2.0)
                VAL_XX[i, :, :, 1] = 1 - data[:, :, 2] / max(1, data[:, :, 1:4].max() / 2.0)
                VAL_XX[i, :, :, 2] = 1 - data[:, :, 1] / max(1, data[:, :, 1:4].max() / 2.0)
                VAL_XX[i, :, :, 3] = data_ndvi
                VAL_XX[i, :, :, 4] = 1 - data[:, :, 12] / max(1, data[:, :, 12].max() / 2.0)
                VAL_XX[i, :, :, 5] = 1 - data[:, :, 13] / max(1, data[:, :, 13].max() / 2.0)
        except Exception as e:
            print(f"Error processing {img}: {e}")

    pred_img = model.predict(VAL_XX)
    pred_img = (pred_img > 0.5).astype(np.uint8)

    st.write(f"Total images processed: {num_samples}")

    landslide_count = 0
    non_landslide_count = 0
    mask_threshold = 50  # Minimum number of pixels to consider a landslide

    if st.button("Show the Predictions"):
        for i in range(num_samples):
            landslide_pixels = np.sum(pred_img[i, :, :, 0] == 1)
            is_landslide = landslide_pixels > mask_threshold
            
            if is_landslide:
                landslide_count += 1
                label = "Landslide Detected"
            else:
                non_landslide_count += 1
                label = "No Landslide Detected"
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(pred_img[i, :, :, 0], cmap='gray')
            ax1.set_title(f"Prediction: {label}")

            ax2.imshow(VAL_XX[i, :, :, 0:3])
            ax2.set_title("Original Image")

            st.pyplot(fig)

        st.write(f"Total Landslide Detected: {landslide_count}")
        st.write(f"Total Non-Landslide Detected: {non_landslide_count}")