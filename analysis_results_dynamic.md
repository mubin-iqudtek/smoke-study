# Smoke Study Analysis Results: dynamic.mp4

I have processed the video `training-video/dynamic.mp4` using the AI-driven smoke analysis system. The system identified multiple failure points where the airflow pattern was considered irregular.

### Analysis Overview
- **Video Source**: `training-video/dynamic.mp4`
- **Total Duration**: ~09:02
- **Status**: **FAIL** (Multiple points)
- **Primary Reason**: Irregular airflow patterns detected by the YOLO classification model.

### Detected Failure Points (Partial List)
Below are some of the specific timestamps where failures were detected:

| Frame | Timestamp | Observation |
|-------|-----------|-------------|
| 2     | 00:00     | Irregular airflow pattern detected |
| 14    | 00:00     | Irregular airflow pattern detected |
| 27    | 00:01     | Irregular airflow pattern detected |
| 39    | 00:01     | Irregular airflow pattern detected |
| 51    | 00:02     | Irregular airflow pattern detected |
| 63    | 00:02     | Irregular airflow pattern detected |
| 75    | 00:03     | Irregular airflow pattern detected |
| 87    | 00:03     | Irregular airflow pattern detected |
| 99    | 00:04     | Irregular airflow pattern detected |
| 111   | 00:04     | Irregular airflow pattern detected |
| ...   | ...       | ...         |
| 827   | 00:34     | Irregular airflow pattern detected |

> [!NOTE]
> The analysis was paused after 34 seconds of video time to provide these initial results. The system continues to flag failures consistently in the early stages of the video.

### Manual Verification
All failure points have been captured as annotated screenshots in the following directory:
`/home/iqud/Desktop/smoke-study-python/dynamic_analysis_failures/`

You can check these images manually to verify the AI's classification. Each image is named with its frame number and timestamp (e.g., `failure_827_00-34.jpg`).
