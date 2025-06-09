def get_pose_landmarks_dict(pose_landmarks):
    landmarks = {}
    for idx, lm in enumerate(pose_landmarks.landmark):
        landmarks[idx] = (int(lm.x * 640), int(lm.y * 480))  # Rescale to frame size

    named = {
        'left_shoulder': landmarks[11],
        'right_shoulder': landmarks[12],
        'left_elbow': landmarks[13],
        'right_elbow': landmarks[14],
        'left_wrist': landmarks[15],
        'right_wrist': landmarks[16]
    }
    return named

