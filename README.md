# YOLO 12 Animal Detection Model Training

This project trains a YOLO 12 object detection model for animal classification using the Ultralytics framework. The model can detect 9 different animal classes: Cat, Dog, Pig, bird, cow, duck, hen, horse, and sheep.

## Features

- 🚀 YOLO 12 model training with data augmentation
- 📊 Comprehensive training pipeline with validation metrics
- 📦 Model export to multiple formats (ONNX, CoreML, TensorFlow Lite)
- 🔧 Easy setup with `uv` package manager
- 📓 Jupyter notebook for interactive training

## Project Structure

```
modelo-animales/
├── dataset/              # Dataset folder (see Dataset section below)
│   └── data.yaml         # Dataset configuration file
├── train_yolo12.ipynb    # Main training notebook
├── pyproject.toml        # Project dependencies
├── uv.lock              # Dependency lock file
└── README.md            # This file
```

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- GPU (optional, but recommended for faster training)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd modelo-animales
```

2. Install dependencies using `uv`:
```bash
uv sync
```

This will create a virtual environment and install all required packages.

## Dataset

The dataset should be placed in the `dataset/` folder with the following structure:

```
dataset/
├── data.yaml            # Dataset configuration (included in repo)
├── train/
│   ├── images/          # Training images
│   └── labels/          # Training labels (YOLO format)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels (YOLO format)
└── test/
    ├── images/          # Test images
    └── labels/          # Test labels (YOLO format)
```

**Note:** The dataset files (images and labels) are not included in this repository as they may be versioned separately on another platform. The `data.yaml` configuration file is included to show the expected dataset structure and class definitions.

### Dataset Configuration

The `dataset/data.yaml` file contains:
- Paths to train/validation/test splits
- Number of classes (9)
- Class names: ['Cat', 'Dog', 'Pig', 'bird', 'cow', 'duck', 'hen', 'horse', 'sheep']

## Usage

### Training the Model

1. Open the Jupyter notebook:
```bash
uv run jupyter notebook train_yolo12.ipynb
```

Or use JupyterLab:
```bash
uv run jupyter lab train_yolo12.ipynb
```

2. Run the cells sequentially:
   - **Cell 1**: Import libraries and check device
   - **Cell 2**: Verify dataset configuration
   - **Cell 3**: Initialize YOLO 12 model
   - **Cell 4**: Configure data augmentation
   - **Cell 5**: Train the model (100 epochs by default)
   - **Cell 6**: Evaluate the model
   - **Cell 7**: Export the model to various formats

### Training Parameters

The notebook is configured with the following default parameters:
- **Model**: YOLOv12n (nano) - can be changed to yolov12s, yolov12m, yolov12l, yolov12x
- **Epochs**: 100
- **Image Size**: 640x640
- **Batch Size**: 16 (adjust based on GPU memory)
- **Optimizer**: AdamW
- **Data Augmentation**: Enabled with mosaic, mixup, copy-paste, and various geometric transformations

### Model Export

The trained model can be exported to:
- **ONNX**: Recommended for cross-platform deployment (works on all platforms)
- **CoreML**: For Apple devices (macOS/Linux only)
- **TensorFlow Lite**: For mobile devices (may have compatibility issues on Windows)

**Note:** ONNX is the recommended format for deployment as it works reliably across all platforms.

## Training Results

After training, the model weights and training artifacts are saved in:
```
runs/detect/yolo12_animal_detection/
├── weights/
│   ├── best.pt          # Best model weights
│   ├── last.pt          # Last epoch weights
│   └── best.onnx        # Exported ONNX model
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
└── ...
```

## Model Performance

The model achieves the following metrics on the validation set:
- **mAP50**: 0.995
- **mAP50-95**: 0.936
- **Precision**: 0.936
- **Recall**: 0.926

## Dependencies

Key dependencies include:
- `ultralytics>=8.4.7` - YOLO framework
- `torch` - PyTorch (automatically installed with ultralytics)
- `onnx`, `onnxruntime` - For ONNX export and inference
- `jupyter`, `ipykernel` - For notebook support
- `opencv-python`, `pillow` - Image processing
- `matplotlib`, `seaborn`, `pandas` - Visualization and data analysis

See `pyproject.toml` for the complete list of dependencies.

## Troubleshooting

### ONNX Export Issues
If you encounter issues with ONNX export, ensure all dependencies are installed:
```bash
uv sync
```

### TensorFlow Lite Export
TensorFlow Lite export may fail on Windows due to compatibility issues. Use ONNX format instead, which works on all platforms.

### CoreML Export
CoreML export is only supported on macOS and Linux. On Windows, this export is automatically skipped.

### GPU Not Detected
If you have a GPU but it's not being used, ensure:
- CUDA is properly installed
- PyTorch with CUDA support is installed
- The device is set to 'cuda' in the notebook

## Contributing

This repository is set up for training new models. To train a model:
1. Ensure your dataset is in the `dataset/` folder
2. Update `dataset/data.yaml` if your class names or structure differ
3. Run the training notebook
4. Trained models will be saved in `runs/detect/` (not tracked in git)

## License

[Add your license here]

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- Dataset may be sourced from [Roboflow](https://roboflow.com) or other platforms

