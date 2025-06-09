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
        try:
            left_angle = abs(landmarks['left_shoulder'][1] - landmarks['left_wrist'][1])
            right_angle = abs(landmarks['right_shoulder'][1] - landmarks['right_wrist'][1])
            passed = left_angle < 30 and right_angle < 30
            msg = "Arms are roughly parallel to the ground." if passed else "Raise your arms to shoulder level."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def elbow_straight(self, landmarks):
        try:
            def distance(p1, p2):
                return np.linalg.norm(np.array(p1) - np.array(p2))

            left_upper = distance(landmarks['left_shoulder'], landmarks['left_elbow'])
            left_lower = distance(landmarks['left_elbow'], landmarks['left_wrist'])
            passed = abs(left_upper - left_lower) < 20  # roughly straight
            msg = "Elbows are straight." if passed else "Try to straighten your elbows."
            return passed, msg
        except KeyError as e:
            logging.error(f"Missing keypoint: {e}")
            return False, f"Missing keypoint: {e}"

    def count_reps(self, landmarks):
        try:
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

        for rule_fn in [self.arm_parallel_to_ground, self.elbow_straight]:
            passed, msg = rule_fn(landmarks)
            results.append({
                "rule": rule_fn.__name__,
                "passed": passed,
                "message": msg
            })

        overall = all(r["passed"] for r in results)
        return {
            "overall_passed": overall,
            "rep_count": self.count_reps(landmarks),
            "details": results
        }


# import numpy as np
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from src.loggingInfo.loggingFile import logging
# from src.rules.base_rules import BaseRuleSet

# class LateralRaiseRules(BaseRuleSet):
#     def __init__(self):
#        pass

#     def wrist_shoulder_alignment(self, landmarks):
#         try:
#             left_wrist = landmarks['left_wrist']
#             left_shoulder = landmarks['left_shoulder']
#             right_wrist = landmarks['right_wrist']
#             right_shoulder = landmarks['right_shoulder']

#             left_diff = abs(left_wrist[1] - left_shoulder[1])
#             right_diff = abs(right_wrist[1] - right_shoulder[1])
#             result = left_diff < 20 and right_diff < 20
#             logging.debug(f"Wrist-Shoulder Alignment: {result} (Left diff: {left_diff}, Right diff: {right_diff})")
#             return result
#         except KeyError as e:
#             logging.error(f"Missing keypoint: {e}")
#             return False

#     def elbow_level_with_wrist(self, landmarks):
#         try:
#             left_elbow = landmarks['left_elbow']
#             left_wrist = landmarks['left_wrist']
#             right_elbow = landmarks['right_elbow']
#             right_wrist = landmarks['right_wrist']

#             left_diff = abs(left_elbow[1] - left_wrist[1])
#             right_diff = abs(right_elbow[1] - right_wrist[1])
#             result = left_diff < 15 and right_diff < 15
#             logging.debug(f"Elbow Level With Wrist: {result} (Left diff: {left_diff}, Right diff: {right_diff})")
#             return result
#         except KeyError as e:
#             logging.error(f"Missing keypoint: {e}")
#             return False

#     def arm_abduction_angle(self, landmarks):
#         try:
#             def angle(a, b, c):
#                 ba = a - b
#                 bc = c - b
#                 cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#                 return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

#             left_angle = angle(np.array(landmarks['left_elbow']),
#                                np.array(landmarks['left_shoulder']),
#                                np.array(landmarks['left_hip']))
#             right_angle = angle(np.array(landmarks['right_elbow']),
#                                 np.array(landmarks['right_shoulder']),
#                                 np.array(landmarks['right_hip']))

#             result = 80 < left_angle < 110 and 80 < right_angle < 110
#             logging.debug(f"Arm Abduction Angle: {result} (Left: {left_angle:.2f}, Right: {right_angle:.2f})")
#             return result
#         except KeyError as e:
#             logging.error(f"Missing keypoint: {e}")
#             return False
    
#     def evaluate_all(self, landmarks):
#         results = {
#             "wrist_shoulder_alignment": self.wrist_shoulder_alignment(landmarks),
#             "elbow_level_with_wrist": self.elbow_level_with_wrist(landmarks),
#             "arm_abduction_angle": self.arm_abduction_angle(landmarks)
#         }
#         logging.info(f"Evaluation Results: {results}")
#         return results
