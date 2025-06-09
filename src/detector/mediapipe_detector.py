import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.loggingInfo.loggingFile import logging

class MediaPipePoseDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        logging.info("MediaPipePoseDetector initialized")

    def process_video(self, video_path: str, display: bool = False) -> List[Dict]:
        """
        Process a video to extract pose landmarks for each frame.

        Args:
            video_path (str): Path to the input video file.
            display (bool): Whether to show annotated video during processing.

        Returns:
            List[Dict]: A list of dictionaries per frame with 'frame_index' and 'landmarks'.
        """
        logging.info(f"Starting video processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return []

        frame_data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video reached or failed to read frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            landmarks = {}
            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    name = self.mp_pose.PoseLandmark(idx).name
                    # Store x, y, z, visibility normalized coords
                    landmarks[name] = (lm.x, lm.y, lm.z, lm.visibility)

                frame_data.append({
                    "frame_index": frame_idx,
                    "landmarks": landmarks
                })

                if display:
                    annotated_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )
                    cv2.imshow("Pose Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested exit from video display")
                        break
            else:
                logging.debug(f"No pose landmarks detected in frame {frame_idx}")

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Video processing completed. Total frames processed: {frame_idx}")
        return frame_data
