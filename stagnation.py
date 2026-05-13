import numpy as np


def detect_stagnation(magnitude, smoke_coverage):

    if magnitude is None:
        return False

    mean_flow = np.mean(magnitude)

    # Smoke exists but airflow almost stopped
    if smoke_coverage > 0.01 and mean_flow < 0.4:
        return True

    return False