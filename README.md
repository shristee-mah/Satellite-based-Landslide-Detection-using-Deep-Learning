# Landslide Detection System

A deep learning-based landslide detection system using satellite imagery and machine learning techniques. This project uses the Landslide4Sense dataset from Kaggle to train models that can identify landslide-prone areas from satellite data.

## 📊 Dataset

**Source:** [Landslide4Sense Dataset](https://www.kaggle.com/datasets/ritesh2000/landslide4sense) from Kaggle

The Landslide4Sense dataset contains:
- **Training Data:** 7,598 samples with multi-spectral satellite imagery
- **Validation Data:** 490 samples for model validation
- **Test Data:** 800 samples for final evaluation
- **Image Format:** HDF5 (.h5) files containing 14-band satellite imagery
- **Resolution:** 128x128 pixel patches
- **Labels:** Binary masks indicating landslide presence

### Data Features
- **Multi-spectral Bands:** 14 channels including RGB, NIR, and elevation data
- **NDVI:** Normalized Difference Vegetation Index calculated from NIR and Red bands
- **Topographical Data:** Slope and elevation information
- **Preprocessing:** Normalized and processed for optimal model performance

## 🏗️ Project Structure

```
Minor Project/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── utils.py                           # Utility functions for data processing
├── landslidedetection.ipynb           # Main training notebook
├── archive/                           # Main project directory
│   ├── deploy.py                      # Main Streamlit deployment application
│   ├── check.py                       # Model testing and validation script
│   ├── hahahaha.py                    # Simplified Streamlit app
│   ├── miyamura.py                    # Data preprocessing utility
│   ├── best_model.keras               # Trained model weights
│   ├── model_save.keras               # Alternative model save
│   ├── images/                        # UI images for Streamlit
│   │   ├── landslide1.jpg
│   │   ├── landslide2.jpg
│   │   └── ...
│   ├── TestData/                      # Test dataset
│   │   ├── img/                       # Test images (.h5 files)
│   │   └── mask/                      # Test masks
│   ├── TrainData/                     # Training dataset
│   │   ├── img/                       # Training images (.h5 files)
│   │   └── mask/                      # Training masks
│   ├── ValidData/                     # Validation dataset
│   │   ├── img/                       # Validation images (.h5 files)
│   │   └── mask/                      # Validation masks
│   ├── check input/                   # Input folder for preprocessing
│   ├── check output/                  # Output folder for preprocessing
│   └── temp_folder/                   # Temporary upload folder
└── __pycache__/                       # Python cache files
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import streamlit, tensorflow, h5py, matplotlib, PIL; print('All packages installed successfully')"
   ```

## 📁 Required Files for Development

### Core Python Files
- **`deploy.py`** - Main Streamlit web application with full UI
- **`check.py`** - Model testing and validation script
- **`hahahaha.py`** - Simplified Streamlit application
- **`miyamura.py`** - Data preprocessing utility
- **`utils.py`** - General utility functions
- **`landslidedetection.ipynb`** - Main training and experimentation notebook

### Model Files
- **`best_model.keras`** - Trained deep learning model
- **`model_save.keras`** - Alternative model checkpoint

### Data Directories
- **`archive/TrainData/`** - Training dataset (.h5 files)
- **`archive/ValidData/`** - Validation dataset (.h5 files)
- **`archive/TestData/`** - Test dataset (.h5 files)
- **`archive/images/`** - UI images for Streamlit application

### Configuration Files
- **`requirements.txt`** - Python package dependencies

## 🎯 Key Features

### Model Architecture
- **Input:** 128x128x6 (6-channel satellite imagery)
- **Channels:** RGB, NDVI, Slope, Elevation
- **Output:** Binary segmentation mask (landslide/no landslide)
- **Metrics:** F1-score, Precision, Recall, Dice Coefficient

### Web Application Features
- **File Upload:** Support for multiple .h5 files
- **Real-time Prediction:** Instant landslide detection
- **Visualization:** Side-by-side comparison of original and predicted images
- **Statistics:** Summary of landslide detection results
- **Responsive Design:** Modern UI with custom styling

### Data Processing
- **Multi-spectral Analysis:** Utilizes 14-band satellite data
- **NDVI Calculation:** Vegetation index computation
- **Topographical Features:** Slope and elevation integration
- **Normalization:** Proper data preprocessing for model input

## 🖥️ Usage

### Running the Web Application
```bash
cd archive
streamlit run deploy.py
```

### Running Model Tests
```bash
cd archive
python check.py
```

### Data Preprocessing
```bash
cd archive
python miyamura.py
```

## 📊 Model Performance

The trained model achieves:
- **High Accuracy:** Effective landslide detection on satellite imagery
- **Multi-spectral Analysis:** Utilizes various spectral bands for better detection
- **Real-time Processing:** Fast inference for web deployment

## 🔧 Technical Stack

- **Deep Learning:** TensorFlow/Keras
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, HDF5, OpenCV
- **Visualization:** Matplotlib, PIL
- **Scientific Computing:** Scikit-learn, Pandas

## 📝 Development Notes

### Model Training
- Use `landslidedetection.ipynb` for training experiments
- Model checkpoints saved as `.keras` files
- Custom metrics implemented for evaluation

### Web Deployment
- Streamlit provides easy web deployment
- Supports multiple file uploads
- Real-time prediction and visualization

### Data Format
- Input files must be in HDF5 (.h5) format
- Images should contain 14 spectral bands
- Expected input shape: (128, 128, 14)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses the Landslide4Sense dataset. Please refer to the dataset's license terms on Kaggle for usage restrictions.

## 📞 Contact

For questions or issues related to this project, please refer to the dataset documentation and code comments.

---

**Note:** This project is designed for educational and research purposes in landslide detection using satellite imagery and deep learning techniques.
