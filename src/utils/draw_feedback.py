import cv2

def draw_feedback(frame, feedback: dict):
    y = 30
    for rule, passed in feedback.items():
        color = (0, 255, 0) if passed else (0, 0, 255)
        text = f"{rule}: {'Yes' if passed else 'No'}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
    return frame

