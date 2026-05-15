# -----------------------------------------------------------------------------
# STAGNATION DETECTION MODULE
# -----------------------------------------------------------------------------
# This file detects if smoke is "stuck" or not moving fast enough.
# Logic:
# 1. It checks the "Magnitude" (speed) of the smoke flow.
# 2. If smoke is present but its speed is below the STAGNATION_THRESHOLD,
#    it flags the area as a "Dead Zone."
# -----------------------------------------------------------------------------

import numpy as np

def detect_stagnation(magnitude, smoke_coverage):

    if magnitude is None:
        return False

    mean_flow = np.mean(magnitude)

    # If smoke exists (coverage > 1%) but airflow is almost stopped (speed < 0.4)
    if smoke_coverage > 0.01 and mean_flow < 0.4:
        return True

    return False