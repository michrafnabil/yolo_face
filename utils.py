"""Utility functions for YOLO face detection."""

import os
import cv2
import numpy as np
import yaml


def create_data_yaml(train_path, val_path, nc=1, names=None, output_path='data.yaml'):
    """Create YAML configuration file for YOLO training.
    
    Args:
        train_path (str): Path to training images
        val_path (str): Path to validation images
        nc (int): Number of classes
        names (list): List of class names
        output_path (str): Path to save the YAML file
    """
    if names is None:
        names = ['Human Face']
    
    data_yaml = {
        'train': train_path,
        'val': val_path,
        'nc': nc,
        'names': names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at: {output_path}")
    with open(output_path, 'r') as f:
        print(f.read())


def crop_face(model, image_path, padding_pixels=25):
    """Crop the best detected face from an image with padding.
    
    Args:
        model: YOLO model instance
        image_path (str): Path to the input image
        padding_pixels (int): Padding around the detected face box
        
    Returns:
        numpy.ndarray: Cropped face image or None if no face detected
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Cannot read image from path: {image_path}")
        return None

    results = model(image)
    detections = results[0].boxes.data.cpu().tolist()

    if not detections:
        print(f"⚠️ No face detected in image: {os.path.basename(image_path)}")
        return None

    # Select detection with highest confidence
    best_detection = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2, score = map(int, best_detection[:5])
    
    print(f"Image '{os.path.basename(image_path)}': Selected box with score {score:.2f}")

    # Get image dimensions
    img_h, img_w, _ = image.shape

    # Add padding
    padded_x1 = x1 - padding_pixels
    padded_y1 = y1 - padding_pixels
    padded_x2 = x2 + padding_pixels
    padded_y2 = y2 + padding_pixels

    # Clip to image boundaries
    final_x1 = int(np.clip(padded_x1, 0, img_w))
    final_y1 = int(np.clip(padded_y1, 0, img_h))
    final_x2 = int(np.clip(padded_x2, 0, img_w))
    final_y2 = int(np.clip(padded_y2, 0, img_h))
    
    cropped_face = image[final_y1:final_y2, final_x1:final_x2]
    return cropped_face


def get_image_files(folder_path, num_files=None, extensions=('.jpg', '.jpeg', '.png')):
    """Get image files from a folder.
    
    Args:
        folder_path (str): Path to the image folder
        num_files (int, optional): Maximum number of files to retrieve
        extensions (tuple): Valid image file extensions
        
    Returns:
        list: List of image file paths
    """
    images = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return images
    
    with os.scandir(folder_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(extensions):
                images.append(entry.path)
                if num_files and len(images) == num_files:
                    break
    
    return images


def setup_kaggle_dataset(dataset_name):
    """Download and setup Kaggle dataset.
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        
    Returns:
        tuple: (train_path, val_path)
    """
    import kagglehub
    
    print(f"📥 Downloading dataset from Kaggle: {dataset_name}")
    dataset_path = kagglehub.dataset_download(dataset_name)
    
    train_path = f"{dataset_path}/images/train"
    val_path = f"{dataset_path}/images/val"
    
    print(f"✅ Dataset downloaded to: {dataset_path}")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    
    return train_path, val_path
