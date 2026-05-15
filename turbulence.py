# -----------------------------------------------------------------------------
# TURBULENCE CALCULATION MODULE
# -----------------------------------------------------------------------------
# This file measures how "chaotic" the smoke flow is.
# Logic:
# 1. It takes the flow vectors (directions) from Optical Flow.
# 2. It calculates the "Variance" of the angles.
# 3. High variance means smoke is moving in many different directions (Turbulent).
# 4. Low variance means smoke is moving in a uniform direction (Laminar).
# -----------------------------------------------------------------------------

import numpy as np

def calculate_turbulence(flow):

    if flow is None:
        return 0

    fx = flow[..., 0]
    fy = flow[..., 1]

    # Convert vectors to angles
    angles = np.arctan2(fy, fx)

    # Calculate how much the angles vary
    variance = np.var(angles)

    # Normalize the score (max 1.0)
    score = min(variance / 10, 1.0)

    return score