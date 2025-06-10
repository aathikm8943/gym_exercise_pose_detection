import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.config.load_config import load_rule_evaluators
from src.utils.pose_utils import get_pose_landmarks_dict
from src.utils.draw_feedback import draw_feedback
from src.loggingInfo.loggingFile import logging

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
evaluators = load_rule_evaluators("configs/rules_config.yaml")

st.set_page_config(page_title="AI Exercise Form Checker", layout="wide")
st.title("🏋️‍♀️ Real-Time or Uploaded Video Pose Form Feedback")

mode = st.radio("Choose input mode", ["Upload Video", "Webcam"])
exercise_type = st.selectbox("Choose Exercise", list(evaluators.keys()))

# Right-hand sidebar simulation
main_col, right_sidebar = st.columns([3, 1])
feedback_container = right_sidebar.empty()

# Tracking counters
total_passed = 0
total_rules = 0

def render_feedback(feedback):
    rep_count = feedback["rep_count"]
    seen_messages = set()
    unique_msgs = []
    passed = 0
    rule_count = 0

    for rule in feedback["details"]:
        rule_count += 1
        if rule["passed"]:
            passed += 1
        elif rule["message"] not in seen_messages:
            unique_msgs.append(rule["message"])
            seen_messages.add(rule["message"])

    return rep_count, unique_msgs, passed, rule_count

def update_sidebar(rep_count, rule_msgs, total_passed, total_rules):
    pass_ratio = (total_passed / total_rules) * 100 if total_rules > 0 else 0

    with feedback_container.container():
        st.markdown("### 🔎 Live Frame Feedback")
        if rule_msgs:
            for msg in rule_msgs:
                st.warning(f"❌ {msg}", icon="⚠️")
        else:
            st.success("✅ All rules passed!", icon="✅")

        st.markdown("---")
        st.markdown("### 📊 Overall Pass Ratio")
        st.progress(pass_ratio / 100)
        st.metric(label="Pass Ratio", value=f"{pass_ratio:.2f}%")
        st.markdown(f"**💪 Reps Count:** `{rep_count}`")

# ---------------- Webcam Mode ----------------
if mode == "Webcam":
    with main_col:
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        st.caption("Click 'Stop' to end webcam feed.")

        stop_btn = st.button("Stop")
        last_rep_count = 0

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                landmarks = get_pose_landmarks_dict(result.pose_landmarks)
                feedback = evaluators[exercise_type].evaluate_all(landmarks)
                frame = draw_feedback(frame, feedback)
                last_rep_count, last_rule_msg, passed, rule_count = render_feedback(feedback)

                total_passed += passed
                total_rules += rule_count

                update_sidebar(last_rep_count, last_rule_msg, total_passed, total_rules)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.write("Webcam session ended.")

# ---------------- Video Upload Mode ----------------
else:
    with main_col:
        uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            last_rep_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)

                if result.pose_landmarks:
                    landmarks = get_pose_landmarks_dict(result.pose_landmarks)
                    feedback = evaluators[exercise_type].evaluate_all(landmarks)
                    frame = draw_feedback(frame, feedback)
                    logging.info(f"Feedback for frame: {feedback}")

                    last_rep_count, last_rule_msg, passed, rule_count = render_feedback(feedback)

                    total_passed += passed
                    total_rules += rule_count

                    update_sidebar(last_rep_count, last_rule_msg, total_passed, total_rules)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            st.success("✅ Video processing complete!")