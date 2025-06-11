# 🎥 Exercise Movement Detection with Rule-Based Evaluation

## 📚 Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [System Architecture](#system-architecture)
* [Folder Structure](#folder-structure)
* [Setup & How to Run](#setup--how-to-run)
* [Sample Output](#sample-output)
* [Notes](#notes)

---

## Project Overview

A robust pose-aware exercise movement analysis system that:

* Extracts and preprocesses video frames.
* Classifies gym exercises such as **Bicep Curl** and **Lateral Raise**.
* Applies **custom rule-based checks** (e.g., joint angles) on pose data to validate **correct form**.
* Displays **real-time feedback** via a Streamlit app.

---

## Objectives

* Detect exercise type and verify if performed **with correct form**.
* Highlight real-time feedback using **joint-based rules** (e.g., elbow angle must be > 90°).
* Support modular expansion for additional exercises and rules.
* Offer an intuitive web interface to **display rep count, rule violations, and live accuracy**.

---

## System Architecture

```text
Video Input (.mp4)
     |
[ Frame Extraction ]
     ↓
[ Pose Detection ] → PoseLandmarks (OpenCV + MediaPipe)
     ↓
[ Rule Evaluation Engine ]
     ↓
[ Rep Counter + Rule Violation Logger ]
     ↓
[ Streamlit Live UI ]
```

---

## Folder Structure

```text
gym_exercise_pose_detection/
├── main.py                      # Entry point (real-time feedback loop)
├── template.py                  # Master file to create folder structure and files
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── app/
│   │   └── streamlit_app.py     # Streamlit frontend interface
│   ├── config/
│   │   └── load_config.py       # Function for loading the config.yaml files
│   ├── rules/
│   │   ├── base_rules.py        # Abstract class
│   │   ├── bicep_curl_rule.py   # Contains the Bicep Curl rules
│   │   └── lateral_raise.py     # Contains the Lateral Raise rules
│   ├── utils/
│   │   ├── draw_feedback.py
│   │   └── pose_utils.py
│   └── loggingInfo/
│       └── loggingFile.py       # Log configuration file
```

---

## Setup & How to Run

### 1. Install Dependencies

Ensure Python 3.11+ is installed, then:

```bash
pip install -r requirements.txt
```

---

### 2. Launch Streamlit Live App

Run the web interface:

```bash
streamlit run src/app/streamlit_app.py
```

Features:

* Real-time pose detection
* Rule validation feedback
* Rep count
* Accuracy progress bar

---

## Sample Output

Output video: [Link](https://drive.google.com/drive/folders/19miin2IzUx6KrV4sDLU38nECgYaB6rE-?usp=sharing)

Real-time app feedback:

* Reps counted successfully when rules are satisfied
* Warnings shown for broken posture (e.g., wrong elbow angle)
* Pass ratio bar with live accuracy
* Uploaded video displayed frame-by-frame

---

## Notes

* Pose detection uses **MediaPipe**.
* Frame validation rules are **editable via `config/rules_description.yaml`**.
* Extend support to other exercises by:
* Works on CPU, but GPU strongly recommended for smooth pose estimation.
