# -----------------------------------------------------------------------------
# REPORTING & LOGGING MODULE
# -----------------------------------------------------------------------------
# This file handles the output of the analysis results to the console and files.
# Logic:
# 1. It only logs "FAIL" events to keep the logs clean.
# 2. It limits logging to 5 entries per second to avoid flooding the text file.
# 3. It writes formatted reports into "output/logs.txt".
# -----------------------------------------------------------------------------

from config import LOG_FILE

# STORE FAIL COUNTS PER SECOND (To prevent log flooding)
time_log_counter = {}

def log_result(
    frame_no,
    timestamp,
    formatted_time,
    formatted_total,
    turbulence,
    stagnation,
    recovery,
    status,
    observation
):

    # ONLY LOG FAIL STATUS
    if status != "FAIL":
        return

    # LIMIT TO 5 LOGS PER SECOND
    if formatted_time not in time_log_counter:

        time_log_counter[formatted_time] = 0

    if time_log_counter[formatted_time] >= 5:
        return

    time_log_counter[formatted_time] += 1

    log = f"""

##################################################
################### FAIL #########################
##################################################

Frame Number      : {frame_no}

Current Time      : {formatted_time}

Video Progress    : {timestamp:.2f} sec / {formatted_total}

Turbulence Score  : {turbulence:.2f}

Stagnation        : {stagnation}

Recovery Time     : {recovery}

Observation       : {observation}

FINAL STATUS      : {status}

==================================================

"""

    # PRINT FAIL LOG TO CONSOLE
    print(log)

    # SAVE FAIL LOG TO FILE
    with open(LOG_FILE, "a") as f:

        f.write(log)