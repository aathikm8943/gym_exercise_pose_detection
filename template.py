import os 
import sys
from pathlib import Path

list_of_files = [
    "main.py",
    "configs/rules_config.yaml",  # Configuration for rules
    "config.py",
    "src/detector/mediapipe_detector.py", # Pose estimation logic
    "src/detector/__init__.py",
    "src/config/__init__.py",
    "src/config/load_config.py",  # Configuration for the application
    "src/rules/base_rule.py",
    "src/rules/bicep_curl_rule.py",
    "src/rules/lateral_raise_rule.py",
    "src/rules/__init__.py",
    "src/utils/__init__.py",
    "src/utils/extract_zip.py",
    "src/utils/pose_utils.py",
    "src/utils/draw_feedback.py",
    "src/loggingInfo/loggingFile.py",
    "src/loggingInfo/__init__.py",
    "src/app/__init__.py",
    "src/app/streamlit_app.py",
    "data/input_video",
    ".gitignore",
    "experiments/experiment.ipynb",
    "requirements.txt",
    "Readme.md"
]

for file in list_of_files:
    fileFullPath = Path(file)
    fileExt, fileName = os.path.split(fileFullPath)
    
    if fileExt != "":
        os.makedirs(fileExt, exist_ok=True)
    
    if not (os.path.exists(fileFullPath)):
        
        with open(fileFullPath, "wb") as f:
            pass
    