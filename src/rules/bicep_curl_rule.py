import numpy as np
import sys
import os

# Ensure correct import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.rules.base_rules import BaseRuleSet
from src.loggingInfo.loggingFile import logging


def get_z_safe(point):
    """Utility to safely access Z-coordinate."""
    return point[2] if len(point) > 2 else None


class BicepCurlRules(BaseRuleSet):
    def __init__(self):
        self.prev_phase = None
        self.rep_started = False
        self.rep_count = 0

    def elbow_angle(self, landmarks):
        try:
            def compute_angle(a, b, c):
                ba = a - b
                bc = c - b
                cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

            left_angle = compute_angle(np.array(landmarks['left_shoulder']),
                                       np.array(landmarks['left_elbow']),
                                       np.array(landmarks['left_wrist']))

            right_angle = compute_angle(np.array(landmarks['right_shoulder']),
                                        np.array(landmarks['right_elbow']),
                                        np.array(landmarks['right_wrist']))

            passed = 30 < left_angle < 160 and 30 < right_angle < 160
            message = "Elbow angle is correct." if passed else " Maintain elbows at ~90° during curl."
            return passed, message
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def wrist_below_elbow(self, landmarks):
        try:
            left = landmarks['left_wrist'][1] > landmarks['left_elbow'][1]
            right = landmarks['right_wrist'][1] > landmarks['right_elbow'][1]
            passed = left and right
            message = "Wrist position is correct." if passed else " Lower your wrists below elbows at bottom of curl."
            return passed, message
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def shoulder_stability(self, landmarks):
        try:
            y_diff = abs(landmarks['left_shoulder'][1] - landmarks['right_shoulder'][1])
            passed = y_diff < 15
            message = "Shoulders are stable." if passed else " Keep shoulders steady and level during movement."
            return passed, message
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def upper_arm_vertical(self, landmarks):
        """Ensures the upper arm stays roughly vertical (i.e., elbow directly under shoulder)."""
        try:
            shoulder = np.array(landmarks['left_shoulder'])
            elbow = np.array(landmarks['left_elbow'])
            dx = abs(shoulder[0] - elbow[0])
            passed = dx < 40  # elbow under shoulder
            msg = "Upper arm is vertical." if passed else " Keep your upper arm vertical during curl."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def count_reps(self, landmarks, rule_passed: bool, threshold: int = 15):
        try:
            if not rule_passed:
                return self.rep_count  # Skip rep count if form is incorrect

            left_elbow_y = landmarks["left_elbow"][1]
            left_wrist_y = landmarks["left_wrist"][1]

            if left_wrist_y > left_elbow_y + threshold:
                current_phase = "down"
            elif left_wrist_y < left_elbow_y - threshold:
                current_phase = "up"
            else:
                current_phase = self.prev_phase

            if self.prev_phase == "down" and current_phase == "up":
                self.rep_started = True
            elif self.prev_phase == "up" and current_phase == "down" and self.rep_started:
                self.rep_count += 1
                self.rep_started = False

            self.prev_phase = current_phase
            return self.rep_count

        except KeyError as e:
            logging.error(f"Missing keypoint for rep count: {e}")
            return self.rep_count


    def evaluate_all(self, landmarks):
        results = []

        for rule_fn in [
            self.elbow_angle,
            self.wrist_below_elbow,
            self.shoulder_stability,
            self.upper_arm_vertical
        ]:
            passed, msg = rule_fn(landmarks)
            results.append({
                "rule": rule_fn.__name__,
                "passed": passed,
                "message": msg
            })

        passed_count = sum(1 for r in results if r["passed"])
        overall = passed_count >= 3
        # overall = all(r["passed"] for r in results)
        logging.info(f"Bicep Curl Rule Evaluation: Overall Passed: {overall}, Details: {results}")
        self.count_reps(landmarks, rule_passed=overall)

        return {
            "overall_passed": overall,
            "rep_count": self.rep_count,
            "details": results
        }
