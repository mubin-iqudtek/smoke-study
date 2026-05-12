import cv2
import os
import argparse

from downloader import download_video
from detector import detect_smoke
from turbulence import calculate_turbulence
from stagnation import detect_stagnation
from recovery import calculate_recovery
from annotator import annotate_frame
from report import log_result

from config import (
    TURBULENCE_THRESHOLD,
    RECOVERY_THRESHOLD,
    OUTPUT_VIDEO
)

os.makedirs("output", exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    required=True,
    help="Smoke study video URL"
)

args = parser.parse_args()

print("\nDownloading Video...\n")

video_path = download_video(args.url)

print(f"Video Saved: {video_path}")

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_no = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    smoke_mask, flow, magnitude = detect_smoke(frame)

    turbulence_score = calculate_turbulence(flow)

    stagnation = detect_stagnation(magnitude)

    recovery_time = calculate_recovery(smoke_mask)

    status = "PASS"

    if turbulence_score > TURBULENCE_THRESHOLD:
        status = "FAIL"

    if stagnation:
        status = "FAIL"

    if recovery_time:

        if recovery_time > RECOVERY_THRESHOLD:
            status = "FAIL"

    annotated = annotate_frame(
        frame,
        smoke_mask,
        turbulence_score,
        stagnation,
        status
    )

    writer.write(annotated)

    log_result(
        frame_no,
        turbulence_score,
        stagnation,
        recovery_time,
        status
    )

    cv2.imshow("Smoke Study", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_no += 1

cap.release()

writer.release()

cv2.destroyAllWindows()

print("\nAnalysis Complete")

print(f"\nAnnotated Video Saved: {OUTPUT_VIDEO}")