Smoke Study Analysis System — Setup & Run Guide

1. Clone Repository
git clone repo-url
cd smoke-study-new

3. Install Python 3.11
Ubuntu/Linux:
sudo apt update
sudo apt install python3.11 python3.11-venv ffmpeg -y

Verify:
python3.11 --version

Expected:
Python 3.11.x

3. Create Virtual Environment
python3.11 -m venv venv

4. Activate Virtual Environment
Linux/macOS:
source venv/bin/activate

Windows:
venv\Scripts\activate

Expected:
(venv)

5. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

6. Verify FFmpeg Installation
ffmpeg -version

7. Run Smoke Study Analysis
Using YouTube URL:
python main.py --url "https://youtu.be/lhQ6XNXNwdY"
Using direct MP4 URL:
python main.py --url "https://example.com/video.mp4"

8. Output Files
Generated automatically inside:
output/
Files:
File	Description
annotated.mp4	Annotated smoke analysis video
logs.txt	Smoke-study analysis logs

9. Features
Smoke flow detection
Optical flow analysis
Turbulence detection
Stagnation zone detection
Recovery time analysis
Automated observations
PASS/FAIL evaluation
Annotated output video
YouTube + direct URL support

11. Common Fixes
yt-dlp Error
Run:
pip install -U yt-dlp


Final Run:
python main.py --url "https://youtu.be/lhQ6XNXNwdY"
