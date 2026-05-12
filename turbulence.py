import numpy as np

def calculate_turbulence(flow):

    if flow is None:
        return 0

    fx = flow[..., 0]
    fy = flow[..., 1]

    angles = np.arctan2(fy, fx)

    variance = np.var(angles)

    score = min(variance / 10, 1.0)

    return score