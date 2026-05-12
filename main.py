import cv2
import argparse
import numpy as np

from downloader import download_video
from detector import detect_smoke
from turbulence import calculate_turbulence
from stagnation import detect_stagnation
from recovery import calculate_recovery
from annotator import annotate_frame
from report import log_result

from config import (
    OUTPUT_VIDEO,
    RECOVERY_THRESHOLD
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    required=True,
    help="Video URL"
)

args = parser.parse_args()

print("\nDownloading Video...\n")

video_path = download_video(args.url)

print(f"Video Saved: {video_path}")

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_duration = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
)

print(f"\nVideo Duration: {video_duration} sec\n")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_no = 0

# TURBULENCE HISTORY
turbulence_history = []

# SMOKE START FLAG
smoke_started = False

# SMOKE START FRAME
smoke_start_frame = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    smoke_mask, flow, magnitude = detect_smoke(frame)

    # HANDLE EMPTY DETECTION
    if magnitude is None:

        frame_no += 1

        continue

    # SAFE SMOKE COVERAGE
    if smoke_mask is not None:

        smoke_pixels = np.sum(smoke_mask > 0)

        total_pixels = (
            smoke_mask.shape[0] *
            smoke_mask.shape[1]
        )

        smoke_coverage = smoke_pixels / total_pixels

    else:

        smoke_coverage = 0

    # DETECT SMOKE RELEASE
    if not smoke_started and smoke_coverage > 0.01:

        smoke_started = True

        smoke_start_frame = frame_no

        print("\nSmoke Release Detected...")
        print("Starting Smoke Study Measurement...\n")

    # BEFORE SMOKE RELEASE
    if not smoke_started:

        waiting_frame = frame.copy()

        cv2.putText(
            waiting_frame,
            "WAITING FOR SMOKE RELEASE",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
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

    # ANALYSIS STARTS HERE
    turbulence_score = calculate_turbulence(magnitude)

    stagnation = detect_stagnation(magnitude)

    recovery_time = calculate_recovery(smoke_mask)

    # SMOOTH TURBULENCE
    turbulence_history.append(turbulence_score)

    if len(turbulence_history) > 10:
        turbulence_history.pop(0)

    avg_turbulence = np.mean(turbulence_history)

    # DEFAULT STATUS
    status = "PASS"

    observation = "Stable unidirectional airflow observed"

    # PASS CONDITION
    if avg_turbulence < 1.2 and not stagnation:

        status = "PASS"

        observation = "Uniform airflow observed"

    # WARNING CONDITION
    elif avg_turbulence >= 1.2 and avg_turbulence < 3.0:

        status = "WARNING"

        observation = "Minor airflow disturbance detected"

    # FAIL CONDITION
    elif avg_turbulence >= 3.0:

        status = "FAIL"

        observation = "Severe turbulence detected"

    # STAGNATION CHECK
    if stagnation and smoke_coverage > 0.02:

        status = "FAIL"

        observation = "Stagnant airflow zone detected"

    # RECOVERY CHECK
    if recovery_time:

        if recovery_time > RECOVERY_THRESHOLD:

            status = "FAIL"

            observation = "Recovery time exceeded acceptable limit"

    # RELATIVE TIMESTAMP
    relative_frame = frame_no - smoke_start_frame

    timestamp = relative_frame / fps

    elapsed_time = int(timestamp)

    minutes = elapsed_time // 60
    seconds = elapsed_time % 60

    formatted_time = f"{minutes:02d}:{seconds:02d}"

    total_minutes = int(video_duration) // 60
    total_seconds = int(video_duration) % 60

    formatted_total = f"{total_minutes:02d}:{total_seconds:02d}"

    # LOGGING
    log_result(
        frame_no,
        timestamp,
        formatted_time,
        formatted_total,
        avg_turbulence,
        stagnation,
        recovery_time,
        status,
        observation
    )

    # FRAME ANNOTATION
    annotated = annotate_frame(
        frame,
        smoke_mask,
        avg_turbulence,
        stagnation,
        status,
        formatted_time,
        formatted_total
    )

    # SAVE VIDEO
    out.write(annotated)

    # LIVE PREVIEW
    cv2.imshow(
        "Smoke Study Analysis",
        annotated
    )

    # EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_no += 1

cap.release()

out.release()

cv2.destroyAllWindows()

print("\nSmoke Study Completed.\n")
print(f"Annotated Video Saved: {OUTPUT_VIDEO}")