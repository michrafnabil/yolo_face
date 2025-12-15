# YOLO Face Detection 🎯

A production-ready YOLO-based face detection system with training and inference capabilities. Supports both Google Colab and local environments.

## 🚀 Features

- **Easy Training**: Train custom YOLOv8 models for face detection
- **Iterative Training**: Support for multiple training loops with automatic model updates
- **Face Cropping**: Extract and crop detected faces with configurable padding
- **Batch Prediction**: Run predictions on multiple images with bounding box visualization
- **Kaggle Integration**: Seamless dataset downloading from Kaggle
- **Flexible Deployment**: Works on Google Colab, Linux, and Windows

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- See `requirements.txt` for all dependencies

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo_face_final.git
cd yolo_face_final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Training

#### Using Kaggle Dataset (Google Colab)

```bash
python train.py --use-kaggle --epochs-per-loop 50 --workers 8
```

#### Using Local Dataset

```bash
python train.py \
    --train-path /path/to/train \
    --val-path /path/to/val \
    --epochs-per-loop 50 \
    --workers 8 \
    --save-dir ./trained_model
```

**Training Arguments:**
- `--model`: YOLO model to use (default: yolov8n.pt)
- `--n-loops`: Number of training loops (default: 1)
- `--epochs-per-loop`: Epochs per training loop (default: 50)
- `--workers`: Number of data loading workers (default: 8)
- `--save-dir`: Directory to save final model
- `--use-kaggle`: Download dataset from Kaggle
- `--train-path`: Path to training images (if not using Kaggle)
- `--val-path`: Path to validation images (if not using Kaggle)

### Prediction

#### Predict with Bounding Boxes

```bash
python predict.py \
    --input-dir /path/to/images \
    --output-dir ./predictions \
    --mode predict \
    --model-path ./trained_model/best.pt
```

#### Crop Faces

```bash
python predict.py \
    --input-dir /path/to/images \
    --output-dir ./crops \
    --mode crop \
    --padding 30 \
    --save-crops
```

#### Both Predict and Crop

```bash
python predict.py \
    --input-dir /path/to/images \
    --output-dir ./results \
    --mode both \
    --save-crops
```

**Prediction Arguments:**
- `--input-dir`: Directory containing input images (required)
- `--output-dir`: Directory to save predictions (default: ./predictions)
- `--model-path`: Path to model file (default: trained_model/best.pt)
- `--num-images`: Number of images to process (default: all)
- `--mode`: Prediction mode - `predict`, `crop`, or `both`
- `--padding`: Padding pixels for face cropping (default: 30)
- `--save-crops`: Save cropped face images

## 📁 Project Structure

```
yolo_face_final/
├── config.py           # Configuration settings
├── train.py            # Training script
├── predict.py          # Prediction/inference script
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── notebook.ipynb     # Original Jupyter notebook
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Dataset**: Kaggle dataset identifier, number of classes
- **Training**: Model type, epochs, workers, image size
- **Inference**: Padding, confidence threshold

## 🎯 Model

This project uses YOLOv8 for face detection. Available models:
- `yolov8n.pt` - Nano (fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## 📊 Training Process

The training script implements an iterative training approach:

1. Downloads/loads the dataset
2. Creates YAML configuration
3. Loads the YOLO model
4. Trains for specified epochs
5. Saves the best model
6. Repeats for multiple loops (if configured)
7. Final model is saved to the specified directory

## 🖼️ Inference Modes

### Predict Mode
- Draws bounding boxes on detected faces
- Saves annotated images to output directory
- Useful for visualization and validation

### Crop Mode
- Extracts detected faces with configurable padding
- Optionally saves cropped faces
- Useful for face recognition preprocessing

### Both Mode
- Combines predict and crop modes
- Saves both annotated images and cropped faces

## 🌐 Google Colab Usage

1. Upload the notebook to Google Colab
2. Run the first cell to download the dataset from Kaggle
3. Run subsequent cells to train and test the model
4. Download the trained model from `/content/trained_yolo_face/best.pt`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [KaggleHub](https://github.com/Kaggle/kagglehub) for dataset distribution
- Face Detection Dataset contributors

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔄 Quick Start Examples

### Example 1: Train on Kaggle Dataset (Colab)
```bash
python train.py --use-kaggle --epochs-per-loop 100
```

### Example 2: Train on Local Dataset
```bash
python train.py \
    --train-path C:/dataset/train \
    --val-path C:/dataset/val \
    --epochs-per-loop 50 \
    --save-dir ./my_model
```

### Example 3: Predict and Crop
```bash
python predict.py \
    --input-dir ./test_images \
    --output-dir ./results \
    --mode both \
    --save-crops \
    --model-path ./my_model/best.pt
```

## 📈 Performance Tips

1. **Training**: Use a GPU with at least 8GB VRAM for faster training
2. **Workers**: Adjust `--workers` based on your CPU cores (typically 4-16)
3. **Batch Size**: Modify in `config.py` for optimal GPU utilization
4. **Model Size**: Start with `yolov8n.pt` for quick tests, use larger models for better accuracy

## 🐛 Troubleshooting

### NumPy Compatibility Error
```bash
pip install "numpy<2"
```

### CUDA Out of Memory
- Reduce batch size in training configuration
- Use a smaller model (yolov8n.pt instead of yolov8m.pt)
- Reduce image size

### Dataset Not Found
- Ensure Kaggle credentials are configured
- Check dataset paths are correct
- Verify internet connection for Kaggle downloads
