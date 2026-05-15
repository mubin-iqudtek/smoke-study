# -----------------------------------------------------------------------------
# DETECTOR AI MODULE
# -----------------------------------------------------------------------------
# This file is the "AI Brain" of the system.
# It handles:
# 1. Loading the trained YOLOv8 Classification Model (best.pt).
# 2. Receiving video frames from the main analysis loop.
# 3. Running the AI inference to classify the frame as PASS or FAIL.
# 4. Providing a "Confidence Score" for each prediction.
# -----------------------------------------------------------------------------

from ultralytics import YOLO
import cv2
import os

# Load the trained model
MODEL_PATH = "runs/classify/smoke_model/v1-2/weights/best.pt"

# Fallback to v1 if v1-2 doesn't exist (safety)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "runs/classify/smoke_model/v1/weights/best.pt"

model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)

def classify_frame(frame):
    """
    Uses the trained YOLO model to classify the frame as PASS or FAIL.
    Returns: status, confidence
    """
    if model is None:
        return "PASS", 0.0 # Default if no model
    
    results = model(frame, verbose=False)
    
    # Get the class with the highest probability
    probs = results[0].probs
    class_id = probs.top1
    confidence = float(probs.top1conf)
    
    status = "PASS" if class_id == 0 else "FAIL" # Based on directory sorting (F comes before P? No, P comes before F? F, P. F=0, P=1?)
    
    # YOLO sorts classes alphabetically: FAIL=0, PASS=1
    # Let's verify class names
    names = results[0].names
    status = names[class_id]
    
    return status, confidence
