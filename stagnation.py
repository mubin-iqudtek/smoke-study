import numpy as np

def detect_stagnation(magnitude):

    if magnitude is None:
        return False

    low_motion = np.mean(magnitude)

    return low_motion < 0.5