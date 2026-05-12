from config import LOG_FILE

# STORE FAIL COUNTS PER SECOND
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

    # PRINT FAIL LOG
    print(log)

    # SAVE FAIL LOG
    with open(LOG_FILE, "a") as f:

        f.write(log)