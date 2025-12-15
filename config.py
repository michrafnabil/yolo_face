"""Configuration file for YOLO face detection."""

import os

# Dataset configuration
DATASET_CONFIG = {
    'kaggle_dataset': 'fareselmenshawii/face-detection-dataset',
    'nc': 1,
    'names': ['Human Face']
}

# Training configuration
TRAINING_CONFIG = {
    'model_name': 'yolov8n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    'project_name': 'face_detection',
    'n_loops': 1,
    'epochs_per_loop': 50,
    'workers': 8,
    'imgsz': 640,
    'save_dir': '/content/trained_yolo_face'  # For Colab, change for local
}

# Inference configuration
INFERENCE_CONFIG = {
    'padding_pixels': 30,
    'confidence_threshold': 0.5
}

# Paths (will be set dynamically)
PATHS = {
    'train': None,
    'val': None,
    'model_save': None
}
