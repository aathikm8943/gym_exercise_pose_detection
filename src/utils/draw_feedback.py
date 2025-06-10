import cv2

def draw_feedback(frame, feedback):
    """
    Draws rule evaluation feedback and rep count on the video frame.

    Args:
        frame (np.array): The video frame from OpenCV.
        feedback (dict): Output from the evaluate_all() method, containing rule details and rep count.

    Returns:
        np.array: The annotated video frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Increase for larger text
    font_color = (0, 0, 225)
    thickness = 2
    y0 = 30
    dy = 40

    for i, rule in enumerate(feedback["details"]):
        text = f"{rule['rule']}: {'Yes' if rule['passed'] else 'No' + rule['message']}"
        y = y0 + i * dy
        cv2.putText(frame, text, (10, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Rep count
    rep_text = f"Reps: {feedback['rep_count']}"
    cv2.putText(frame, rep_text, (10, y + dy), font, font_scale, (225, 225, 225), thickness, cv2.LINE_AA)

    return frame
