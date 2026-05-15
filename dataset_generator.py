import cv2
import os
import numpy as np
from detector import detect_smoke
from turbulence import calculate_turbulence
from stagnation import detect_stagnation
from recovery import calculate_recovery
from config import VIDEO_DIR, RECOVERY_THRESHOLD

# -----------------------------------
# CONFIGURATION
# -----------------------------------
TRAINING_VIDEO_DIR = "training-video"
DATASET_DIR = "dataset"
TRAIN_RATIO = 0.8  # 80% for training, 20% for validation
FRAME_INTERVAL = 1.0  # Extract 1 frame per second

# Classes
CLASSES = ["PASS", "FAIL"]

def setup_folders():
    """Creates the dataset folder structure."""
    for split in ["train", "val"]:
        for cls in CLASSES:
            path = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(path, exist_ok=True)
    print(f"Dataset structure created at: {DATASET_DIR}")

def get_video_files():
    """Lists all video files in the training-video directory."""
    if not os.path.exists(TRAINING_VIDEO_DIR):
        print(f"Error: {TRAINING_VIDEO_DIR} directory not found.")
        return []
    
    extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in os.listdir(TRAINING_VIDEO_DIR)]

def extract_and_label():
    """Extracts frames and uses current logic to auto-label them."""
    video_files = get_video_files()
    if not video_files:
        print("No videos found to process.")
        return

    setup_folders()
    
    total_extracted = 0
    
    for video_name in video_files:
        video_path = os.path.join(TRAINING_VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            print(f"Skipping {video_name}: Could not read FPS.")
            continue

        print(f"Processing: {video_name} ({frame_count} frames, {fps} FPS)")
        
        frame_no = 0
        extracted_from_this_video = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at intervals
            if frame_no % int(fps * FRAME_INTERVAL) == 0:
                # --- AUTO LABELING LOGIC ---
                # We use the existing logic to decide if this frame is a PASS or FAIL
                smoke_mask, flow, magnitude = detect_smoke(frame)
                
                status = "PASS" # Default
                
                if magnitude is not None:
                    turbulence_score = calculate_turbulence(magnitude)
                    stagnation = detect_stagnation(magnitude, 0.01) # Dummy coverage
                    
                    # Criteria for FAIL (matching main.py thresholds)
                    if turbulence_score >= 3.0 or stagnation:
                        status = "FAIL"
                    elif np.mean(magnitude) < 0.4 and np.sum(smoke_mask) > 0:
                        status = "FAIL"
                
                # --- SAVE FRAME ---
                # Decide split (random or simple ratio)
                split = "train" if (total_extracted % 10) < (TRAIN_RATIO * 10) else "val"
                
                filename = f"{os.path.splitext(video_name)[0]}_f{frame_no}.jpg"
                save_path = os.path.join(DATASET_DIR, split, status, filename)
                
                cv2.imwrite(save_path, frame)
                extracted_from_this_video += 1
                total_extracted += 1
                
            frame_no += 1
            
        cap.release()
        print(f"Extracted {extracted_from_this_video} frames from {video_name}.")

    print(f"\nProcessing Complete. Total frames extracted: {total_extracted}")
    print("Please check the 'dataset/' folder to verify the auto-labels.")

if __name__ == "__main__":
    extract_and_label()
