# -----------------------------------
# ANALYSIS THRESHOLDS
# -----------------------------------
# Sensitivity for turbulence detection (higher = more sensitive)
TURBULENCE_THRESHOLD = 0.7

# Max seconds allowed for smoke to clear the area
RECOVERY_THRESHOLD = 30

# Min speed magnitude; if smoke is present but moving slower than this, it's "stagnant"
STAGNATION_THRESHOLD = 0.05

# -----------------------------------
# FILE PATHS
# -----------------------------------
# Where the final annotated video is saved
OUTPUT_VIDEO = "output/annotated/annotated.mp4"

# Path to the text log file
LOG_FILE = "output/logs.txt"

# Directory where failure screenshots are stored
SCREENSHOTS_DIR = "output/screenshots"

# Default directory to look for input videos
VIDEO_DIR = "video"

# -----------------------------------
# SCREENSHOT SETTINGS
# -----------------------------------
# Minimum time (in seconds) to wait before taking another failure screenshot
SCREENSHOT_COOLDOWN = 0.5

# How many screenshots to capture per second during a FAIL event
FAILURE_SCREENSHOTS_PER_SECOND = 2
