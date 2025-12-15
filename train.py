"""YOLO face detection training script."""

import os
import shutil
import datetime
import argparse
from ultralytics import YOLO

from config import DATASET_CONFIG, TRAINING_CONFIG
from utils import create_data_yaml, setup_kaggle_dataset


def yolo_train(model_name, data_yaml, project_name,
               save_dir="/content/trained_yolo_face",
               n_loops=1, epochs_per_loop=5, workers=8, imgsz=640):
    """Train YOLO model with iterative loops.
    
    Args:
        model_name (str): YOLO model name (e.g., 'yolov8n.pt', 'yolov8s.pt')
        data_yaml (str): Path to data YAML file
        project_name (str): Name of the training project
        save_dir (str): Directory to save the final model
        n_loops (int): Number of training loops
        epochs_per_loop (int): Epochs per training loop
        workers (int): Number of data loading workers
        imgsz (int): Input image size
    
    Returns:
        str: Path to the trained model
    """
    # Load YOLO model (will download automatically if not present)
    print(f"📥 Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    for i in range(1, n_loops + 1):
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        run_name = f"{project_name}_run_{timestamp_str}"

        print(f"\n🚀 Training Loop {i}/{n_loops} → {run_name}")
        
        # Train model
        model.train(
            data=data_yaml,
            epochs=epochs_per_loop,
            imgsz=imgsz,
            project=project_name,
            name=run_name,
            workers=workers
        )

        # Path to best model from this run
        run_dir = os.path.join(project_name, run_name)
        weights_dir = os.path.join(run_dir, "weights")
        best_model_path = os.path.join(weights_dir, "best.pt")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"❌ best.pt not found in {weights_dir}")

        print(f"📌 Best model saved at: {best_model_path}")

        # Copy best model to final directory
        final_model_path = os.path.join(save_dir, "best.pt")
        shutil.copy(best_model_path, final_model_path)
        print(f"💾 Best model updated → {final_model_path}")

        # Load the best model for the next training loop
        model = YOLO(best_model_path)

    total_epochs = n_loops * epochs_per_loop
    print("\n🎉 Training completed!")
    print(f"➡ Total epochs: {total_epochs}")
    print(f"🏁 Final model saved at: {final_model_path}")
    
    return final_model_path


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train YOLO face detection model')
    parser.add_argument('--model', type=str, default=TRAINING_CONFIG['model_name'],
                        help='YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)')
    parser.add_argument('--n-loops', type=int, default=TRAINING_CONFIG['n_loops'],
                        help='Number of training loops')
    parser.add_argument('--epochs-per-loop', type=int, default=TRAINING_CONFIG['epochs_per_loop'],
                        help='Epochs per training loop')
    parser.add_argument('--workers', type=int, default=TRAINING_CONFIG['workers'],
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default=TRAINING_CONFIG['save_dir'],
                        help='Directory to save final model')
    parser.add_argument('--use-kaggle', action='store_true',
                        help='Download dataset from Kaggle')
    parser.add_argument('--train-path', type=str, default=None,
                        help='Path to training images (if not using Kaggle)')
    parser.add_argument('--val-path', type=str, default=None,
                        help='Path to validation images (if not using Kaggle)')
    
    args = parser.parse_args()
    
    # Setup dataset paths
    if args.use_kaggle:
        print("📦 Using Kaggle dataset...")
        train_path, val_path = setup_kaggle_dataset(DATASET_CONFIG['kaggle_dataset'])
    else:
        if not args.train_path or not args.val_path:
            raise ValueError("Must provide --train-path and --val-path or use --use-kaggle")
        train_path = args.train_path
        val_path = args.val_path
    
    # Create data.yaml file
    create_data_yaml(
        train_path=train_path,
        val_path=val_path,
        nc=DATASET_CONFIG['nc'],
        names=DATASET_CONFIG['names'],
        output_path='data.yaml'
    )
    
    # Start training
    yolo_train(
        model_name=args.model,
        data_yaml='data.yaml',
        project_name=TRAINING_CONFIG['project_name'],
        n_loops=args.n_loops,
        epochs_per_loop=args.epochs_per_loop,
        workers=args.workers,
        imgsz=TRAINING_CONFIG['imgsz'],
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
