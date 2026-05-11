# Project Files Documentation

## 🎯 Essential Files for Project Development

### 1. Core Application Files

#### `deploy.py` (Main Web Application)
- **Purpose:** Primary Streamlit web application for landslide detection
- **Features:** 
  - Full UI with custom styling
  - Multiple file upload support
  - Real-time prediction and visualization
  - Landslide statistics and results display
- **Dependencies:** streamlit, tensorflow, h5py, matplotlib, PIL
- **Usage:** `streamlit run deploy.py`

#### `check.py` (Model Validation)
- **Purpose:** Test and validate the trained model on test data
- **Features:**
  - Load and process test images
  - Generate predictions with confidence scores
  - Visual comparison of results
- **Dependencies:** tensorflow, h5py, numpy, matplotlib
- **Usage:** `python check.py`

#### `hahahaha.py` (Simplified Web App)
- **Purpose:** Lightweight version of the web application
- **Features:**
  - Basic Streamlit interface
  - Essential prediction functionality
  - Minimal UI design
- **Dependencies:** streamlit, tensorflow, h5py, matplotlib
- **Usage:** `streamlit run hahahaha.py`

### 2. Data Processing Files

#### `miyamura.py` (Data Preprocessing)
- **Purpose:** Convert raw images to HDF5 format for model input
- **Features:**
  - Image loading and conversion
  - Multi-band image generation
  - HDF5 file creation
- **Dependencies:** opencv-python, numpy, h5py
- **Usage:** `python miyamura.py`

#### `utils.py` (Utility Functions)
- **Purpose:** Common utility functions for data handling
- **Features:**
  - Data loading helpers
  - Visualization utilities
  - Common preprocessing functions
- **Dependencies:** h5py, matplotlib, numpy, glob, os

### 3. Training and Development

#### `landslidedetection.ipynb` (Training Notebook)
- **Purpose:** Main training and experimentation environment
- **Features:**
  - Model architecture definition
  - Training pipeline
  - Performance evaluation
  - Hyperparameter tuning
- **Dependencies:** tensorflow, numpy, matplotlib, h5py
- **Usage:** Open in Jupyter Notebook

### 4. Model Files

#### `best_model.keras` (Primary Model)
- **Purpose:** Trained deep learning model weights
- **Architecture:** U-Net based segmentation model
- **Input Shape:** (128, 128, 6)
- **Output:** Binary segmentation mask
- **Custom Metrics:** F1-score, Precision, Recall, Dice Coefficient

#### `model_save.keras` (Alternative Model)
- **Purpose:** Backup model checkpoint
- **Usage:** Alternative model for testing/comparison

### 5. Data Directories

#### `archive/TrainData/`
- **Contents:** Training images and masks
- **Format:** HDF5 (.h5) files
- **Count:** 7,598 samples
- **Structure:** `img/` and `mask/` subdirectories

#### `archive/ValidData/`
- **Contents:** Validation images and masks
- **Format:** HDF5 (.h5) files
- **Count:** 490 samples
- **Structure:** `img/` and `mask/` subdirectories

#### `archive/TestData/`
- **Contents:** Test images for final evaluation
- **Format:** HDF5 (.h5) files
- **Count:** 800 samples
- **Structure:** `img/` and `mask/` subdirectories

#### `archive/images/`
- **Contents:** UI images for Streamlit application
- **Format:** JPEG files
- **Purpose:** Visual elements in web interface

### 6. Configuration Files

#### `requirements.txt`
- **Purpose:** Python package dependencies
- **Usage:** `pip install -r requirements.txt`
- **Includes:** All necessary packages for the project

#### `README.md`
- **Purpose:** Project documentation and setup guide
- **Contents:** Installation, usage, and project structure

## 🔄 Development Workflow

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, tensorflow, h5py, matplotlib, PIL; print('Ready')"
```

### 2. Model Training
```bash
# Open training notebook
jupyter notebook landslidedetection.ipynb
```

### 3. Web Application Development
```bash
# Run main application
cd archive
streamlit run deploy.py

# Or run simplified version
streamlit run hahahaha.py
```

### 4. Model Testing
```bash
# Test model on validation data
python check.py
```

### 5. Data Preprocessing
```bash
# Process new images
python miyamura.py
```

## 📋 File Dependencies

### Core Dependencies
- **Python 3.8+** - Base programming language
- **TensorFlow 2.21.0** - Deep learning framework
- **Streamlit 1.57.0** - Web application framework
- **HDF5 3.14.0** - Data format handling
- **NumPy 2.4.4** - Numerical computing
- **Matplotlib 3.10.9** - Visualization
- **PIL 12.2.0** - Image processing

### Optional Dependencies
- **OpenCV 4.10.0.84** - Advanced image processing
- **Pandas 3.0.2** - Data manipulation
- **Scikit-learn 1.6.1** - Machine learning utilities

## 🚨 Important Notes

1. **Model Files:** The `.keras` files are essential for prediction
2. **Data Format:** All input images must be in HDF5 format
3. **Directory Structure:** Maintain the archive/ folder structure
4. **Custom Metrics:** Model loading requires custom metric functions
5. **Memory Usage:** Large datasets require sufficient RAM

## 🔧 File Maintenance

### Regular Updates
- Update `requirements.txt` when adding new packages
- Keep model files backed up
- Maintain data directory structure
- Update documentation when changing features

### Backup Strategy
- Backup `best_model.keras` regularly
- Save training checkpoints
- Document model versions
- Maintain data backups

---

This documentation provides a comprehensive overview of all necessary files for the landslide detection project development.
