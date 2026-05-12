import cv2
import numpy as np


def annotate_frame(
    frame,
    smoke_mask,
    turbulence_score,
    stagnation,
    status,
    formatted_time,
    formatted_total
):

    overlay = frame.copy()

    if smoke_mask is not None:

        resized_mask = cv2.resize(
            smoke_mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0])
        )

        resized_mask = resized_mask.astype(bool)

        overlay[resized_mask] = (0, 255, 0)

    cv2.putText(
        overlay,
        f"Turbulence: {turbulence_score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        overlay,
        f"Stagnation: {stagnation}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        overlay,
        f"STATUS: {status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        f"TIME: {formatted_time} / {formatted_total}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return overlay