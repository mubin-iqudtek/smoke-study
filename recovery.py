import time
import numpy as np

recovery_start = None

def calculate_recovery(smoke_mask):

    global recovery_start

    if smoke_mask is None:
        return None

    density = np.mean(smoke_mask)

    if density > 0.2 and recovery_start is None:
        recovery_start = time.time()

    if density < 0.05 and recovery_start is not None:
        elapsed = time.time() - recovery_start
        recovery_start = None
        return elapsed

    return None