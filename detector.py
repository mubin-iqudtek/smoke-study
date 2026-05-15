# -----------------------------------------------------------------------------
# SMOKE DETECTOR MODULE (TRADITIONAL CV)
# -----------------------------------------------------------------------------
# This file is responsible for the "Physical Analysis" of the smoke.
# It handles:
# 1. Image Pre-processing (Resizing, Gray-scaling, Blurring).
# 2. Optical Flow Calculation (Farneback Algorithm) to track movement.
# 3. Generating a "Smoke Mask" based on motion magnitude.
# 4. Providing raw flow data (speed and direction) for further diagnosis.
# -----------------------------------------------------------------------------

import cv2
import numpy as np

prev_gray = None


def detect_smoke(frame):

    global prev_gray

    # Resize for performance
    frame = cv2.resize(
        frame,
        (640, 360)
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    if prev_gray is None:

        prev_gray = gray

        return None, None, None

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        2,
        15,
        2,
        5,
        1.1,
        0
    )

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    smoke_mask = magnitude > 1.5

    prev_gray = gray

    return smoke_mask, flow, magnitude