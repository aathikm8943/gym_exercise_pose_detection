import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.rules.base_rules import BaseRuleSet
from src.loggingInfo.loggingFile import logging

class LateralRaiseRules(BaseRuleSet):
    def __init__(self):
        self.prev_phase = None
        self.rep_started = False
        self.rep_count = 0

    def arm_parallel_to_ground(self, landmarks):
        """
        Checks if both arms are raised roughly parallel to the ground.
        """
        try:
            left_angle = abs(landmarks['left_shoulder'][1] - landmarks['left_wrist'][1])
            right_angle = abs(landmarks['right_shoulder'][1] - landmarks['right_wrist'][1])
            passed = left_angle < 30 and right_angle < 30
            msg = "Arms are roughly parallel to the ground." if passed else " - Raise your arms to shoulder level."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def elbow_straight(self, landmarks):
        """
        Checks if the elbows are straight by comparing upper and lower arm segment lengths.
        """
        try:
            def distance(p1, p2):
                return np.linalg.norm(np.array(p1) - np.array(p2))

            left_upper = distance(landmarks['left_shoulder'], landmarks['left_elbow'])
            left_lower = distance(landmarks['left_elbow'], landmarks['left_wrist'])
            passed = abs(left_upper - left_lower) < 20  # roughly straight
            msg = "Elbows are straight." if passed else " Try to straighten your elbows."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def shoulders_aligned_during_raise(self, landmarks):
        """
        Checks if both shoulders are aligned during lateral raise.
        """
        try:
            left_shoulder_y = landmarks["left_shoulder"][1]
            right_shoulder_y = landmarks["right_shoulder"][1]
            passed = abs(left_shoulder_y - right_shoulder_y) < 20  # within acceptable range
            msg = "Shoulders are level." if passed else " Keep your shoulders level."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"
        
    def arm_symmetric_lift(self, landmarks):
        """
        Ensures both arms lift together (symmetric height).
        """
        try:
            diff = abs(landmarks['left_wrist'][1] - landmarks['right_wrist'][1])
            passed = diff < 20
            msg = "Both arms lifting symmetrically." if passed else "Raise both arms equally."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def count_reps(self, landmarks, rule_passed: bool):
        try:
            if not rule_passed:
                return self.rep_count  # Don't count bad reps

            left_wrist_y = landmarks["left_wrist"][1]
            left_shoulder_y = landmarks["left_shoulder"][1]

            if left_wrist_y > left_shoulder_y + 30:
                phase = "down"
            elif left_wrist_y < left_shoulder_y - 30:
                phase = "up"
            else:
                phase = self.prev_phase

            if self.prev_phase == "down" and phase == "up":
                self.rep_started = True
            elif self.prev_phase == "up" and phase == "down" and self.rep_started:
                self.rep_count += 1
                self.rep_started = False

            self.prev_phase = phase
            return self.rep_count
        except KeyError as e:
            logging.error(f"Missing keypoint for rep count: {e}")
            return self.rep_count


    def evaluate_all(self, landmarks):
        results = []

        for rule_fn in [
            self.arm_parallel_to_ground,
            self.elbow_straight,
            self.shoulders_aligned_during_raise,
            self.arm_symmetric_lift
        ]:
            passed, msg = rule_fn(landmarks)
            results.append({
                "rule": rule_fn.__name__,
                "passed": passed,
                "message": msg
            })

        passed_count = sum(1 for r in results if r["passed"])
        overall = passed_count >= 3
        
        self.count_reps(landmarks, rule_passed=overall)

        return {
            "overall_passed": overall,
            "rep_count": self.rep_count,
            "details": results
        }

