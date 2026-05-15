# -----------------------------------------------------------------------------
# RECOVERY TIME CALCULATION MODULE
# -----------------------------------------------------------------------------
# This file measures how long it takes for smoke to clear the area.
# Logic:
# 1. Start timer when smoke density exceeds a "dense" threshold.
# 2. Stop timer when smoke density falls below a "clear" threshold.
# 3. The difference is the Recovery Time (how long the room takes to clean itself).
# -----------------------------------------------------------------------------

import time
import numpy as np

recovery_start = None

def calculate_recovery(smoke_mask):

    global recovery_start

    if smoke_mask is None:
        return None

    # Calculate average density of smoke in the frame
    density = np.mean(smoke_mask)

    # Start timer when smoke becomes thick
    if density > 0.2 and recovery_start is None:
        recovery_start = time.time()

    # Stop timer when smoke clears
    if density < 0.05 and recovery_start is not None:
        elapsed = time.time() - recovery_start
        recovery_start = None
        return elapsed

    return None