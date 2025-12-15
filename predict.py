"""YOLO face detection prediction script."""

import os
import argparse
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import Image, display

from config import TRAINING_CONFIG, INFERENCE_CONFIG
from utils import crop_face, get_image_files


def predict_and_save(model, image_paths, output_dir):
    """Run predictions on images and save results.
    
    Args:
        model: YOLO model instance
        image_paths (list): List of image file paths
        output_dir (str): Directory to save prediction results
    
    Returns:
        list: List of paths to predicted images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        results = model.predict(
            source=img_path,
            save=True,
            project=output_dir,
            name="results",
            exist_ok=True
        )
    
    pred_dir = os.path.join(output_dir, "results")
    
    if not os.path.exists(pred_dir):
        print(f"⚠️ Prediction directory not created: {pred_dir}")
        return []
    
    predicted_imgs = [
        os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    return predicted_imgs


def crop_and_display(model, image_paths, padding_pixels=30, save_crops=False, output_dir=None):
    """Crop faces from images and display them.
    
    Args:
        model: YOLO model instance
        image_paths (list): List of image file paths
        padding_pixels (int): Padding around detected faces
        save_crops (bool): Whether to save cropped images
        output_dir (str): Directory to save cropped images
    """
    if save_crops and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        cropped_image = crop_face(model, img_path, padding_pixels=padding_pixels)
        
        if cropped_image is not None:
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            
            # Display
            plt.imshow(cropped_image_rgb)
            plt.axis("off")
            plt.title(f"Cropped face: {os.path.basename(img_path)}")
            plt.show()
            
            # Save if requested
            if save_crops and output_dir:
                crop_filename = f"crop_{os.path.basename(img_path)}"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_image)
                print(f"💾 Saved crop to: {crop_path}")
        else:
            print(f"⚠️ No face detected in: {os.path.basename(img_path)}")


def load_model(model_path=None):
    """Load YOLO model from local path.
    
    Args:
        model_path (str): Local model path (default: trained model from config)
        
    Returns:
        YOLO: Loaded YOLO model
    """
    if model_path is None:
        model_path = os.path.join(TRAINING_CONFIG['save_dir'], 'best.pt')
    
    if os.path.exists(model_path):
        print(f"📥 Loading model from: {model_path}")
        return YOLO(model_path)
    else:
        print(f"⚠️ Model not found at: {model_path}")
        print(f"Loading default YOLO model: {TRAINING_CONFIG['model_name']}")
        return YOLO(TRAINING_CONFIG['model_name'])


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Run YOLO face detection predictions')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model file (default: trained_model/best.pt)')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to process (default: all)')
    parser.add_argument('--mode', type=str, choices=['predict', 'crop', 'both'], default='predict',
                        help='Prediction mode: predict (save with boxes), crop (crop faces), or both')
    parser.add_argument('--padding', type=int, default=INFERENCE_CONFIG['padding_pixels'],
                        help='Padding pixels for face cropping')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save cropped face images')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(model_path=args.model_path)
    
    # Get image files
    image_paths = get_image_files(args.input_dir, num_files=args.num_images)
    
    if not image_paths:
        print(f"❌ No images found in: {args.input_dir}")
        return
    
    print(f"📁 Found {len(image_paths)} images")
    
    if args.mode in ['predict', 'both']:
        # Run predictions and save
        predicted_imgs = predict_and_save(model, image_paths, args.output_dir)
        print(f"\n✅ Predictions saved to: {args.output_dir}")
        print(f"📊 Processed {len(predicted_imgs)} images")
        
    if args.mode in ['crop', 'both']:
        # Crop and display faces
        crop_output = os.path.join(args.output_dir, 'crops') if args.save_crops else None
        crop_and_display(
            model,
            image_paths,
            padding_pixels=args.padding,
            save_crops=args.save_crops,
            output_dir=crop_output
        )


if __name__ == "__main__":
    main()
