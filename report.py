from config import LOG_FILE

def log_result(
    frame_no,
    turbulence,
    stagnation,
    recovery,
    status
):

    log = f"""
==================================================
Frame: {frame_no}
Turbulence Score: {turbulence:.2f}
Stagnation: {stagnation}
Recovery Time: {recovery}
STATUS: {status}
==================================================
"""

    print(log)

    with open(LOG_FILE, "a") as f:
        f.write(log)