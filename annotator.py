# -----------------------------------------------------------------------------
# ANNOTATOR MODULE
# -----------------------------------------------------------------------------
# This file is responsible for all the "Visual Work" in the project.
# It handles:
# 1. Drawing the Status HUD (PASS/FAIL, Turbulence, Time).
# 2. Painting flow vectors (arrows) on the screen to show smoke direction.
# 3. Drawing red boxes around "Failure Areas" for screenshots.
# 4. Creating the final annotated video frames.
# -----------------------------------------------------------------------------

import cv2
import numpy as np


def mark_failure_area(frame, smoke_mask, label="DEFECT AREA"):
    # Creates a copy of the frame to draw on, leaving the original untouched.
    highlighted = frame.copy()

    # If there is no smoke mask, draw a simple red rectangle across the whole image.
    if smoke_mask is None:

        cv2.rectangle(
            highlighted,
            (8, 8),
            (highlighted.shape[1] - 8, highlighted.shape[0] - 8),
            (0, 0, 255),
            2
        )

        cv2.putText(
            highlighted,
            label,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        return highlighted

    resized_mask = cv2.resize(
        smoke_mask.astype(np.uint8),
        (highlighted.shape[1], highlighted.shape[0])
    )

    resized_mask = (
        resized_mask > 0
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (9, 9),
        np.uint8
    )

    resized_mask = cv2.morphologyEx(
        resized_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        resized_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = (
        highlighted.shape[0]
        *
        highlighted.shape[1]
        *
        0.001
    )

    issue_found = False

    for contour in contours:

        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        padding = 12

        x1 = max(
            x - padding,
            0
        )
        y1 = max(
            y - padding,
            0
        )
        x2 = min(
            x + w + padding,
            highlighted.shape[1] - 1
        )
        y2 = min(
            y + h + padding,
            highlighted.shape[0] - 1
        )

        cv2.rectangle(
            highlighted,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        label_y = max(
            y1 - 12,
            30
        )

        cv2.putText(
            highlighted,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

        issue_found = True

    if not issue_found:

        cv2.rectangle(
            highlighted,
            (8, 8),
            (highlighted.shape[1] - 8, highlighted.shape[0] - 8),
            (0, 0, 255),
            2
        )

        cv2.putText(
            highlighted,
            label,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    return highlighted


def draw_heatmap_border(frame, issue_mask, color=(0, 0, 255)):

    marked = frame.copy()

    issue_mask = (
        issue_mask > 0
    ).astype(np.uint8) * 255

    kernel = np.ones(
        (13, 13),
        np.uint8
    )

    issue_mask = cv2.morphologyEx(
        issue_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    issue_mask = cv2.morphologyEx(
        issue_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        issue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = (
        frame.shape[0]
        *
        frame.shape[1]
        *
        0.001
    )

    visible_mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    for contour in contours:

        if cv2.contourArea(contour) < min_area:
            continue

        cv2.drawContours(
            visible_mask,
            [contour],
            -1,
            255,
            -1
        )

        cv2.drawContours(
            marked,
            [contour],
            -1,
            color,
            2
        )

    marked[visible_mask > 0] = cv2.addWeighted(
        marked,
        0.65,
        np.full_like(marked, color),
        0.35,
        0
    )[visible_mask > 0]

    return marked


def annotate_frame(
    frame,
    smoke_mask,
    magnitude,
    turbulence_score,
    stagnation,
    recovery_time,
    smoke_coverage,
    status,
    observation,
    formatted_time,
    formatted_total
):

    overlay = frame.copy()

    # GREEN SMOKE OVERLAY
    if smoke_mask is not None:

        resized_mask = cv2.resize(
            smoke_mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0])
        )

        resized_mask = resized_mask.astype(bool)

        overlay[resized_mask] = (0, 255, 0)

    if (
        stagnation
        and
        smoke_mask is not None
        and
        magnitude is not None
    ):

        smoke_context = (
            smoke_mask > 0
        ).astype(np.uint8) * 255

        smoke_context = cv2.dilate(
            smoke_context,
            np.ones(
                (35, 35),
                np.uint8
            ),
            iterations=1
        )

        low_flow_mask = (
            (magnitude < 0.4).astype(np.uint8) * 255
        )

        stagnation_mask = cv2.bitwise_and(
            low_flow_mask,
            smoke_context
        )

        if cv2.countNonZero(stagnation_mask) == 0:
            stagnation_mask = smoke_context

        stagnation_mask = cv2.resize(
            stagnation_mask,
            (frame.shape[1], frame.shape[0])
        )

        overlay = draw_heatmap_border(
            overlay,
            stagnation_mask
        )

    # TURBULENCE
    cv2.putText(
        overlay,
        f"Turbulence: {turbulence_score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # STAGNATION
    cv2.putText(
        overlay,
        f"Stagnation: {stagnation}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

    # SMOKE COVERAGE
    cv2.putText(
        overlay,
        f"Smoke Coverage: {smoke_coverage:.3f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    # RECOVERY TIME
    if recovery_time is not None:

        recovery_color = (
            (0, 0, 255)
            if recovery_time > 10
            else (0, 255, 0)
        )

        cv2.putText(
            overlay,
            f"Recovery: {recovery_time:.1f}s",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            recovery_color,
            2
        )

    # STATUS
    status_color = {
        "WAITING": (0, 255, 255),
        "PASS": (0, 255, 0),
        "WARNING": (0, 255, 255),
        "FAIL": (0, 0, 255)
    }.get(status, (255, 255, 255))

    cv2.putText(
        overlay,
        f"STATUS: {status}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3
    )

    if observation:

        cv2.putText(
            overlay,
            f"Reason: {observation}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2
        )

    # TIME
    cv2.putText(
        overlay,
        f"TIME: {formatted_time} / {formatted_total}",
        (20, 295),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    return overlay
