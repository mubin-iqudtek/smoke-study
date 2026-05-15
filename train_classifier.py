import os
from ultralytics import YOLO

def train_smoke_model():
    # 1. Load a pre-trained YOLOv8 classification model (smallest/fastest version)
    model = YOLO('yolov8n-cls.pt')

    # 2. Train the model using the dataset we generated
    # We include heavy augmentation (flip, rotation, brightness) for robustness
    results = model.train(
        data='dataset',          # Path to our dataset folder
        epochs=50,               # Number of learning cycles
        imgsz=224,               # Image size
        batch=16,                # Number of images processed at once
        project='smoke_model',   # Folder to save results
        name='v1',               # Name of this training run
        
        # --- AUGMENTATION FOR ROBUSTNESS ---
        augment=True,
        flipud=0.5,              # 50% chance to flip upside down (handles camera angles)
        fliplr=0.5,              # 50% chance to flip left-right
        degrees=15,              # Random rotation up to 15 degrees
        hsv_h=0.015,             # Random color shifts (handles lighting)
        hsv_s=0.7,
        hsv_v=0.4
    )

    print("\nTraining Complete!")
    print(f"Your model is saved in: smoke_model/v1/weights/best.pt")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("Error: 'dataset' folder not found. Please run dataset_generator.py first.")
    else:
        train_smoke_model()
