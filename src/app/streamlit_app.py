import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import sys

# --- Add root project path to import custom modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- Custom Imports ---
from src.config.load_config import load_rule_evaluators, load_rules_description_config
from src.utils.pose_utils import get_pose_landmarks_dict
from src.utils.draw_feedback import draw_feedback
from src.loggingInfo.loggingFile import logging

# -----------------------------
# Initialize MediaPipe and Rules
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
evaluators = load_rule_evaluators("configs/rules_config.yaml")

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="AI Exercise Form Checker", layout="wide")
st.title("🏋️‍♀️ Real-Time or Uploaded Video Pose Form Feedback")

mode = st.radio("Choose input mode", ["Upload Video", "Webcam"])
exercise_type = st.selectbox("Choose Exercise", list(evaluators.keys()))

main_col, right_sidebar = st.columns([3, 1])
feedback_container = right_sidebar.empty()

rules_config = load_rules_description_config()

exercise_key = exercise_type.lower().replace(" ", "_")  # e.g., "Bicep Curl" → "bicep_curl"
exercise_rules = rules_config.get(exercise_key, {})

with right_sidebar:

    # Create reusable containers for each section
    feedback_title = st.empty()
    warning_container = st.empty()
    progress_container = st.empty()
    metric_container = st.empty()
    rep_container = st.empty()


    # Mentioned the exercise rules
    if exercise_rules:
        st.markdown(f"### {exercise_rules['title']}")
        for rule in exercise_rules["rules"]:
            st.info(f"- {rule}")
    else:
        st.warning("No rules found for the selected exercise.")

    # Sidebar purpose and rules
    st.markdown("## 📌 Purpose & Rules")
    
    st.info("""
    This sidebar shows real-time feedback on your exercise form.

    **Detection Logic**:
    - If the rep count stays 0, your movement is **not matching** the required motion pattern.
    - Even if accuracy is > 0%, a 0 count indicates form is **invalid**.

    Improve your form until rep count increases and rule violations disappear.
    """)
    


# -----------------------------
# Helper Functions
# -----------------------------
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
    unique_msgs = list(set(rule_msgs)) if rule_msgs else []

    feedback_title.markdown("### 🔍 Live Frame Feedback")

    if rep_count == 0 and unique_msgs:
        warning_container.error("Exercise movement not recognized. Please adjust your form.", icon="❗")
    elif unique_msgs:
        warning_container.markdown("")  # Clear any old errors
        for msg in unique_msgs:
            warning_container.warning(msg, icon="⚠️")
    else:
        warning_container.success("All rules passed!", icon="✅")

    progress_container.markdown("### 📊 Accuracy Estimate")
    progress_container.progress(pass_ratio / 100)
    metric_container.metric(label="Pass Ratio", value=f"{pass_ratio:.2f}%")
    rep_container.markdown(f"**💪 Reps Count:** `{rep_count}`")


# def update_sidebar(rep_count, rule_msgs, total_passed, total_rules):
#     pass_ratio = (total_passed / total_rules) * 100 if total_rules > 0 else 0

#     feedback_container.markdown("### 🔍 Live Frame Feedback")

#     if rep_count == 0 and rule_msgs:
#         feedback_container.error("Exercise movement not recognized. Please adjust your form.", icon="❗")
#     elif rule_msgs:
#         for msg in rule_msgs:
#             feedback_container.warning(f"{msg}", icon="⚠️")
#     else:
#         feedback_container.success("All rules passed!", icon="✅")

#     feedback_container.markdown("---")
#     feedback_container.markdown("### 📊 Accuracy Estimate")
#     feedback_container.progress(pass_ratio / 100)
#     feedback_container.metric(label="Pass Ratio", value=f"{pass_ratio:.2f}%")
#     feedback_container.markdown(f"**💪 Reps Count:** {rep_count}")


def draw_landmarks(frame, pose_landmarks, passed=True):
    landmark_color = (0, 255, 0) if passed else (0, 0, 255)  # Green or Red
    edge_color = (100, 200, 255)

    h, w, _ = frame.shape

    for landmark in pose_landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(frame, (cx, cy), 5, landmark_color, -1)

    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start = pose_landmarks.landmark[start_idx]
        end = pose_landmarks.landmark[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        if all(0 <= v < w for v in [start_point[0], end_point[0]]) and all(0 <= v < h for v in [start_point[1], end_point[1]]):
            cv2.line(frame, start_point, end_point, edge_color, 2)

    return frame

def process_frame(frame, exercise_type):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        landmarks = get_pose_landmarks_dict(result.pose_landmarks)
        feedback = evaluators[exercise_type].evaluate_all(landmarks)
        frame = draw_feedback(frame, feedback)
        frame = draw_landmarks(frame, result.pose_landmarks, passed=feedback["rep_count"] > 0)
        logging.info(f"Feedback: {feedback}")
        return frame, feedback
    else:
        return frame, None

# -----------------------------
# Webcam Mode
# -----------------------------
if mode == "Webcam":
    with main_col:
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        st.caption("Click 'Stop' to end webcam feed.")
        stop_btn = st.button("Stop")
        last_rep_count = 0
        total_passed = 0
        total_rules = 0

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame, feedback = process_frame(frame, exercise_type)

            if feedback:
                rep_count, rule_msgs, passed, rule_count = render_feedback(feedback)
                total_passed += passed
                total_rules += rule_count
                
                update_sidebar(rep_count, rule_msgs, total_passed, total_rules)
            else:
                update_sidebar(0, ["Pose not detected. Please stay in frame."], 0, 1)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.write("Webcam session ended.")

# -----------------------------
# Uploaded Video Mode
# -----------------------------
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
            total_passed = 0
            total_rules = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame, feedback = process_frame(frame, exercise_type)

                if feedback:
                    rep_count, rule_msgs, passed, rule_count = render_feedback(feedback)

                    # Use only current frame stats (not accumulated)
                    update_sidebar(rep_count, rule_msgs, passed, rule_count)
                else:
                    update_sidebar(0, ["Pose not detected. Please stay in frame."], 0, 1)


                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            st.success("✅ Video processing complete!")