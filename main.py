import cv2
import argparse
import numpy as np

import os
import sys

from downloader import download_video
from detector import detect_smoke
from turbulence import calculate_turbulence
from stagnation import detect_stagnation
from recovery import calculate_recovery
from annotator import annotate_frame, mark_failure_area
from report import log_result

from config import (
    FAILURE_SCREENSHOTS_PER_SECOND,
    OUTPUT_VIDEO,
    RECOVERY_THRESHOLD,
    SCREENSHOTS_DIR,
    VIDEO_DIR
)

# -----------------------------------
# CREATE REQUIRED DIRECTORIES
# -----------------------------------
os.makedirs(
    os.path.dirname(OUTPUT_VIDEO),
    exist_ok=True
)

os.makedirs(
    SCREENSHOTS_DIR,
    exist_ok=True
)

# -----------------------------------
# UNIQUE ANALYSIS FOLDER
# -----------------------------------
analysis_no = 1

while os.path.exists(
    os.path.join(
        SCREENSHOTS_DIR,
        f"analysis{analysis_no}"
    )
):
    analysis_no += 1

current_screenshots_dir = os.path.join(
    SCREENSHOTS_DIR,
    f"analysis{analysis_no}"
)

os.makedirs(
    current_screenshots_dir,
    exist_ok=True
)

print(
    f"\nStoring screenshots in: {current_screenshots_dir}\n"
)

# -----------------------------------
# ARGUMENTS
# -----------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    required=False,
    help="Video URL"
)

parser.add_argument(
    "--video",
    required=False,
    help="Path to local video file"
)

args = parser.parse_args()

video_path = None

downloaded_video_path = None

# -----------------------------------
# LOCAL VIDEO
# -----------------------------------
if args.video:

    if os.path.exists(args.video):

        video_path = args.video

    else:

        print(
            f"Error: Video file not found at {args.video}"
        )

        sys.exit(1)

# -----------------------------------
# DOWNLOAD VIDEO
# -----------------------------------
elif args.url:

    print("\nDownloading Video...\n")

    try:

        video_path = download_video(args.url)

        downloaded_video_path = video_path

        print(f"Video Saved: {video_path}")

    except Exception as e:

        print(f"Error downloading video: {e}")

        sys.exit(1)

# -----------------------------------
# DEFAULT VIDEO
# -----------------------------------
else:

    if os.path.exists(VIDEO_DIR):

        video_files = [
            f for f in os.listdir(VIDEO_DIR)
            if f.lower().endswith(
                ('.mp4', '.avi', '.mov', '.mkv')
            )
        ]

        if video_files:

            video_path = os.path.join(
                VIDEO_DIR,
                video_files[0]
            )

            print(
                f"\nNo source provided. Using stored video: {video_path}\n"
            )

        else:

            print(
                f"\nError: No video files found in '{VIDEO_DIR}' folder."
            )

            sys.exit(1)

    else:

        print(
            f"\nError: '{VIDEO_DIR}' folder not found."
        )

        sys.exit(1)

# -----------------------------------
# VIDEO SETUP
# -----------------------------------
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_duration = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
)

print(
    f"\nVideo Duration: {video_duration} sec\n"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# -----------------------------------
# VARIABLES
# -----------------------------------
frame_no = 0

turbulence_history = []

smoke_release_detected = False

smoke_flow_started = False

smoke_start_frame = 0

last_failure_screenshot_at = None

first_gray = None

# NEW:
missing_smoke_frames = 0

# -----------------------------------
# MAIN LOOP
# -----------------------------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    abs_timestamp = frame_no / fps

    # -----------------------------------
    # SMOKE RELEASE DETECTION
    # -----------------------------------
    if not smoke_release_detected:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )

        if first_gray is None:

            first_gray = gray

            frame_no += 1

            continue

        diff = cv2.absdiff(
            first_gray,
            gray
        )

        _, thresh = cv2.threshold(
            diff,
            25,
            255,
            cv2.THRESH_BINARY
        )

        motion_level = (
            np.sum(thresh)
            /
            (thresh.shape[0] * thresh.shape[1])
            /
            255
        )

        if motion_level > 0.01:

            smoke_release_detected = True

            smoke_start_frame = frame_no

            print(
                f"\nSmoke Release Detected at {abs_timestamp:.2f}s..."
            )

            print(
                "Waiting for smoke flow to enter analysis zone...\n"
            )

    # -----------------------------------
    # WAITING SCREEN
    # -----------------------------------
    if not smoke_release_detected:

        waiting_frame = frame.copy()

        min_abs = int(abs_timestamp // 60)
        sec_abs = int(abs_timestamp % 60)

        cv2.putText(
            waiting_frame,
            f"WAITING FOR SMOKE RELEASE ({min_abs:02d}:{sec_abs:02d} / {video_duration//60:02d}:{video_duration%60:02d})",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Smoke Study Analysis",
            waiting_frame
        )

        out.write(waiting_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_no += 1

        continue

    # -----------------------------------
    # SMOKE DETECTION
    # -----------------------------------
    smoke_mask, flow, magnitude = detect_smoke(frame)

    if magnitude is None:

        waiting_frame = frame.copy()

        cv2.putText(
            waiting_frame,
            "WAITING FOR SMOKE FLOW",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Smoke Study Analysis",
            waiting_frame
        )

        out.write(waiting_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_no += 1

        continue

    # -----------------------------------
    # SMOKE COVERAGE
    # -----------------------------------
    if smoke_mask is not None:

        smoke_pixels = np.sum(smoke_mask > 0)

        total_pixels = (
            smoke_mask.shape[0]
            *
            smoke_mask.shape[1]
        )

        smoke_coverage = (
            smoke_pixels / total_pixels
        )

    else:

        smoke_coverage = 0

    # -----------------------------------
    # SMOKE PRESENCE
    # -----------------------------------
    smoke_present = smoke_coverage > 0.003

    if smoke_present:

        if not smoke_flow_started:

            smoke_flow_started = True

            smoke_start_frame = frame_no

            print(
                f"Smoke Flow Detected at {abs_timestamp:.2f}s. Starting status evaluation...\n"
            )

        missing_smoke_frames = 0

    elif smoke_flow_started:

        missing_smoke_frames += 1

    # -----------------------------------
    # ANALYSIS
    # -----------------------------------
    turbulence_score = calculate_turbulence(
        magnitude
    )

    stagnation = detect_stagnation(
        magnitude,
        smoke_coverage
    )

    recovery_time = calculate_recovery(
        smoke_mask
    )

    # -----------------------------------
    # TURBULENCE SMOOTHING
    # -----------------------------------
    turbulence_history.append(
        turbulence_score
    )

    if len(turbulence_history) > 10:

        turbulence_history.pop(0)

    avg_turbulence = np.mean(
        turbulence_history
    )

    mean_flow = np.mean(magnitude)

    # -----------------------------------
    # WAIT FOR FLOW BEFORE STATUS
    # -----------------------------------
    if not smoke_flow_started:

        elapsed_time = int(abs_timestamp)

        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        formatted_time = (
            f"{minutes:02d}:{seconds:02d}"
        )

        total_minutes = (
            int(video_duration) // 60
        )

        total_seconds = (
            int(video_duration) % 60
        )

        formatted_total = (
            f"{total_minutes:02d}:{total_seconds:02d}"
        )

        annotated = annotate_frame(
            frame,
            smoke_mask,
            magnitude,
            avg_turbulence,
            stagnation,
            recovery_time,
            smoke_coverage,
            "WAITING",
            "Smoke flow not found yet",
            formatted_time,
            formatted_total
        )

        out.write(annotated)

        cv2.imshow(
            "Smoke Study Analysis",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_no += 1

        continue

    # -----------------------------------
    # DEFAULT STATUS
    # -----------------------------------
    status = "PASS"

    observation = (
        "Uniform airflow observed"
    )

    # -----------------------------------
    # SMOKE MISSING
    # -----------------------------------
    if (
        smoke_flow_started
        and
        not smoke_present
    ):

        status = "WAITING"

        observation = (
            "Smoke not found in analysis zone"
        )

    # -----------------------------------
    # SMOKE DISAPPEARED
    # -----------------------------------
    elif missing_smoke_frames > fps * 2:

        status = "FAIL"

        observation = (
            "Smoke disappeared from analysis zone"
        )

    # -----------------------------------
    # WARNING
    # -----------------------------------
    elif (
        avg_turbulence >= 1.2
        and
        avg_turbulence < 3.0
    ):

        status = "WARNING"

        observation = (
            "Minor airflow disturbance detected"
        )

    # -----------------------------------
    # TURBULENCE FAIL
    # -----------------------------------
    elif avg_turbulence >= 3.0:

        status = "FAIL"

        observation = (
            "Severe turbulence detected"
        )

    # -----------------------------------
    # STAGNATION FAIL
    # -----------------------------------
    elif stagnation:

        status = "FAIL"

        observation = (
            "Smoke stagnation detected"
        )

    # -----------------------------------
    # DEAD AIRFLOW FAIL
    # -----------------------------------
    elif (
        smoke_coverage > 0.01
        and
        mean_flow < 0.4
    ):

        status = "FAIL"

        observation = (
            "Dead airflow detected"
        )

    # -----------------------------------
    # RECOVERY FAIL
    # -----------------------------------
    elif (
        recovery_time is not None
        and
        recovery_time > RECOVERY_THRESHOLD
    ):

        status = "FAIL"

        observation = (
            "Recovery time exceeded acceptable limit"
        )

    # -----------------------------------
    # TIME FORMATTING
    # -----------------------------------
    elapsed_time = int(abs_timestamp)

    minutes = elapsed_time // 60
    seconds = elapsed_time % 60

    formatted_time = (
        f"{minutes:02d}:{seconds:02d}"
    )

    total_minutes = (
        int(video_duration) // 60
    )

    total_seconds = (
        int(video_duration) % 60
    )

    formatted_total = (
        f"{total_minutes:02d}:{total_seconds:02d}"
    )

    # -----------------------------------
    # LOGGING
    # -----------------------------------
    if observation != "Smoke not found in analysis zone":

        log_result(
            frame_no,
            abs_timestamp,
            formatted_time,
            formatted_total,
            avg_turbulence,
            stagnation,
            recovery_time,
            status,
            observation
        )

    # -----------------------------------
    # FRAME ANNOTATION
    # -----------------------------------
    annotated = annotate_frame(
        frame,
        smoke_mask,
        magnitude,
        avg_turbulence,
        stagnation,
        recovery_time,
        smoke_coverage,
        status,
        observation,
        formatted_time,
        formatted_total
    )

    # -----------------------------------
    # SAVE VIDEO
    # -----------------------------------
    out.write(annotated)

    # -----------------------------------
    # SAVE FAILURE SCREENSHOTS
    # -----------------------------------
    screenshot_eligible_failure = (
        status == "FAIL"
        and
        observation != "Smoke not found in analysis zone"
    )

    if not screenshot_eligible_failure:

        last_failure_screenshot_at = None

    else:

        screenshot_interval = 1 / FAILURE_SCREENSHOTS_PER_SECOND

        if (
            last_failure_screenshot_at is None
            or
            abs_timestamp - last_failure_screenshot_at >= screenshot_interval
        ):

            screenshot_filename = (
                f"failure_{frame_no}_{formatted_time.replace(':', '-')}.jpg"
            )

            screenshot_path = os.path.join(
                current_screenshots_dir,
                screenshot_filename
            )

            failure_screenshot = mark_failure_area(
                annotated,
                smoke_mask,
                observation
            )

            saved = cv2.imwrite(
                screenshot_path,
                failure_screenshot
            )

            if saved:

                print(
                    f"Failure Screenshot Saved: {screenshot_path}"
                )

                last_failure_screenshot_at = abs_timestamp

            else:

                print(
                    f"Failed to save failure screenshot: {screenshot_path}"
                )

    # -----------------------------------
    # LIVE PREVIEW
    # -----------------------------------
    cv2.imshow(
        "Smoke Study Analysis",
        annotated
    )

    # EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_no += 1

# -----------------------------------
# CLEANUP
# -----------------------------------
cap.release()

out.release()

cv2.destroyAllWindows()

print("\nSmoke Study Completed.\n")

print(
    f"Annotated Video Saved: {OUTPUT_VIDEO}"
)

if (
    downloaded_video_path
    and
    os.path.exists(downloaded_video_path)
):

    try:

        os.remove(downloaded_video_path)

        print(
            f"Deleted cached video: {downloaded_video_path}"
        )

    except OSError as e:

        print(
            f"Could not delete cached video {downloaded_video_path}: {e}"
        )
