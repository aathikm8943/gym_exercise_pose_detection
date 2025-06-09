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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
evaluators = load_rule_evaluators("configs/rules_config.yaml")

st.set_page_config(page_title="AI Exercise Form Checker", layout="centered")
st.title("🏋️‍♀️ Real-Time or Uploaded Video Pose Form Feedback")

mode = st.radio("Choose input mode", [ "Upload Video", "Webcam"])
exercise_type = st.selectbox("Choose Exercise", list(evaluators.keys()))

if mode == "Webcam":
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    st.caption("Click 'Stop' to end webcam feed.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            landmarks = get_pose_landmarks_dict(result.pose_landmarks)
            feedback = evaluators[exercise_type].evaluate_all(landmarks)
            result = evaluators[exercise_type].evaluate_all(landmarks)
            rep_count = evaluators[exercise_type].count_reps(landmarks)

            st.write(f"Total Reps: {rep_count}")
            for detail in result["details"]:
                st.markdown(f"- **{detail['rule']}**: {detail['message']}")
            frame = draw_feedback(frame, feedback)
            mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if st.button("Stop"):
            break

    cap.release()
    st.write("Webcam session ended.")

else:
    uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

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
                mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                result = evaluators[exercise_type].evaluate_all(landmarks)
                rep_count = evaluators[exercise_type].count_reps(landmarks)

                st.write(f"Total Reps: {rep_count}")
                for detail in result["details"]:
                    st.markdown(f"- **{detail['rule']}**: {detail['message']}")

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        st.success("Video processing complete!")