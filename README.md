# Face Anti-Spoofing Detection Project

A comprehensive YOLO-based face anti-spoofing detection system that can distinguish between live (genuine) and spoof (fake) faces in images and videos.

## 🎯 Project Overview

This project implements a deep learning-based solution for face anti-spoofing detection using YOLO (You Only Look Once) object detection. The system is trained to classify faces into two categories:
- **Live**: Genuine, real faces
- **Spoof**: Fake faces (printed photos, digital screens, masks, etc.)

## 📁 Project Structure

### Main Repository: `spoof-object-detect`
```
spoof-object-detect/
├── assets/                     # Demo images and videos
├── data/                       # Dataset directory
│   ├── raw/                    # Original dataset
│   │   └── datasets/           # Celeb dataset with train/val/test splits
│   └── processed/              # Preprocessed dataset ready for training
│       └── 1/                  # Processed dataset version
│           ├── data.yaml       # YOLO dataset configuration
│           ├── images/         # Resized images (224x224)
│           └── labels/         # YOLO format labels
├── deploy_models/              # Deployment configurations
│   ├── cloud-deploy_spoof-detect/  # Local copy of cloud deployment
│   │   ├── app/
│   │   │   ├── main.py        # FastAPI application
│   │   │   └── model/
│   │   │       └── best.pt    # Trained model
│   │   ├── Dockerfile         # Docker configuration
│   │   ├── render.yaml        # Render deployment config
│   │   └── requirements.txt   # Python dependencies
│   ├── streamlit_spoof-detect/     # Streamlit GUI deployment
│   │   ├── app.py             # Streamlit main app
│   │   ├── spoof_detection.py # Detection logic
│   │   └── requirements.txt   # Python dependencies
│   └── spoof-detect_v1.1_1750604360.pt  # Trained model file
├── outputs/                    # Training and inference results
│   ├── models/                 # Trained model outputs
│   │   └── Spoof-Detect-1-1/  # Model training results
│   │       ├── weights/        # Model weights (best.pt, last.pt)
│   │       ├── plots/          # Training plots and metrics
│   │       ├── images/         # Training sample images
│   │       ├── args.yaml       # Training hyperparameters
│   │       └── results.csv     # Training metrics by epoch
│   └── predict/                # Inference results
│       └── Spoof-Detect-1-1/  # Prediction outputs
│           ├── labels/         # Predicted labels
│           └── *.jpg           # Images with bounding boxes
├── src/                        # Source code utilities
│   ├── utils.py               # Data processing and visualization utilities
│   └── yolo_detect.py         # YOLO detection script for local inference
├── config.yaml                # Project configuration
├── object_detection.ipynb     # Main Jupyter notebook for training
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── yolo11n.pt                # Pre-trained YOLO model
```

### Cloud Deployment Repository: `cloud-deploy_spoof-detect`
- **GitHub**: https://github.com/EASONTAN03/cloud-deploy_spoof-detect
- **Purpose**: FastAPI cloud deployment on Render.com
- **Auto-deploy**: Enabled from GitHub repository

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Training the Model**:
   ```bash
   # Open the main notebook
   jupyter notebook object_detection.ipynb
   ```

2. **Local Inference** (Command Line):
   ```bash
   python src/yolo_detect.py --model outputs/models/Spoof-Detect-1-1/weights/best.pt --source path/to/image.jpg
   ```

3. **Streamlit GUI**:
   ```bash
   cd deploy_models/streamlit_spoof-detect
   streamlit run app.py
   ```

4. **Cloud Deployment** (FastAPI):
   ```bash
   cd deploy_models/cloud-deploy_spoof-detect
   docker build -t spoof-detect-api .
   docker run -p 8000:8000 spoof-detect-api
   ```

## 📊 Dataset

The project uses a custom face anti-spoofing dataset with the following structure:

- **Classes**: 2 (live, spoof)
- **Image Size**: 224x224 pixels (resized from original)
- **Format**: YOLO format with normalized bounding boxes
- **Splits**: Train (77 images), Validation (20 images), Test (20 images)

### Dataset Statistics
- **Total Images**: 117
- **Live Faces**: Class 0
- **Spoof Faces**: Class 1
- **Label Format**: YOLO normalized coordinates (class x_center y_center width height)

## 🔧 Configuration

The project uses `config.yaml` for centralized configuration:

```yaml
data:
  raw: data/raw
  processed: data/processed

outputs:
  models: outputs/models
  predict: outputs/predict

configs:
  dataset: datasets
  model_name: Spoof-Detect
  class: ['live', 'spoof']

params:
  target_size: 224
  epochs: 30
  batch_size: 16
  conf: 0.45
  iou: 0.8
```

## 🎯 Model Performance

The trained model achieves:
- **Precision**: High precision for both live and spoof detection
- **Recall**: Good recall across different spoofing techniques
- **F1-Score**: Balanced performance between precision and recall
- **Inference Speed**: Real-time capable on modern hardware

## 🛠️ Utilities

### `src/utils.py`
Contains essential utilities for:
- **Data Processing**: Image resizing, label adjustment
- **Visualization**: Bounding box drawing, grid display
- **Configuration**: YAML parsing, path management
- **Dataset Creation**: YOLO format data.yaml generation

### `src/yolo_detect.py`
Command-line inference tool supporting:
- **Multiple Sources**: Images, folders, videos, webcams
- **Real-time Processing**: USB cameras, PiCamera
- **Recording**: Video output with custom resolution
- **Flexible Parameters**: Confidence, IoU thresholds

## 🚀 Deployment Options

### 1. Streamlit GUI (Local)
- **Location**: `deploy_models/streamlit_spoof-detect/`
- **Features**: Interactive web interface, file upload, real-time prediction
- **Usage**: `streamlit run app.py`

### 2. FastAPI Cloud Deployment
- **Repository**: [cloud-deploy_spoof-detect](https://github.com/EASONTAN03/cloud-deploy_spoof-detect) (Separate repo)
- **Features**: REST API, Docker containerization, cloud-ready
- **Endpoints**: 
  - `GET /`: Health check
  - `POST /predict/`: Image prediction
- **Deployment**: Render.com (Auto-deployed)
- **Local Development**: `deploy_models/cloud-deploy_spoof-detect/`

### 3. Command Line Interface
- **Script**: `src/yolo_detect.py`
- **Features**: Batch processing, video support, real-time inference
- **Usage**: See examples below

## 📝 Usage Examples

### Training
```python
# In object_detection.ipynb
from src.utils import get_config
datasets_path, processed_data_path, models_path, predict_path, model_name, data_class, prepare_benchmark, model_benchmark, params = get_config('config.yaml')
```

### Inference
```bash
# Single image
python src/yolo_detect.py --model outputs/models/Spoof-Detect-1-1/weights/best.pt --source test.jpg

# Video file
python src/yolo_detect.py --model outputs/models/Spoof-Detect-1-1/weights/best.pt --source video.mp4 --record --resolution 640x480

# Webcam
python src/yolo_detect.py --model outputs/models/Spoof-Detect-1-1/weights/best.pt --source usb0

# Batch folder
python src/yolo_detect.py --model outputs/models/Spoof-Detect-1-1/weights/best.pt --source images/ --save_predict
```

### API Usage
```python
import requests

# FastAPI endpoint
url = "http://localhost:8000/predict/"
files = {"file": open("test.jpg", "rb")}
response = requests.post(url, files=files)
predictions = response.json()
```

## 🔍 Model Architecture

- **Base Model**: YOLO (You Only Look Once)
- **Input Size**: 224x224 pixels
- **Classes**: 2 (live, spoof)
- **Output**: Bounding boxes with confidence scores
- **Post-processing**: Non-maximum suppression (NMS)

## 📈 Training Results

Training outputs are stored in `outputs/models/Spoof-Detect-1-1/`:
- **Weights**: `best.pt` (best model), `last.pt` (last epoch)
- **Metrics**: `results.csv` (epoch-wise metrics)
- **Plots**: Confusion matrix, precision-recall curves, F1 curves
- **Samples**: Training and validation batch images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO community for the excellent object detection framework
- Dataset contributors for the face anti-spoofing dataset
- Open source community for various supporting libraries

## 🔄 Repository Workflow

### Development Workflow
1. **Main Project**: Develop and train models in `spoof-object-detect`
2. **Model Updates**: Copy best model to cloud deployment
3. **Cloud Deployment**: Push changes to `cloud-deploy_spoof-detect` repo
4. **Auto-deploy**: Render automatically deploys the updated model

### Updating the Cloud Deployment
```bash
# Copy the latest model to cloud deployment
cp outputs/models/Spoof-Detect-1-1/weights/best.pt deploy_models/cloud-deploy_spoof-detect/app/model/

# Update the cloud deployment repository
cd deploy_models/cloud-deploy_spoof-detect
git add app/model/best.pt
git commit -m "Update model to latest version"
git push origin main
```

## 📞 Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Provide reproduction steps if applicable

**Repository-specific issues:**
- **Main project issues**: Create in `spoof-object-detect` repository
- **Cloud deployment issues**: Create in `cloud-deploy_spoof-detect` repository

---

**Note**: This project is designed for educational and research purposes. For production use, ensure proper validation and testing on your specific use case.
